"""Three-class Simple-BEV wrapper for free/occupied/unknown logits.

The upstream Simple-BEV submodule stays untouched. This module reuses its
encoder, lifting, BEV compressor, and decoder trunk, then replaces the final
segmentation projection with one N-channel softmax-ready head.

**출력 채널 수는 `num_classes`로 정한다** (기본 3 = free/occupied/unknown).
`num_classes=2`는 (D) binary 정식화(free / not-free)이고 나머지는 전부 같다 --
클래스 이름은 역사적이지만 채널 수는 호출부가 명시하고 학습 배너에 찍힌다.
근거는 `docs/finetune_overfitting_diagnosis.md` §15.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.basic as utils_basic  # noqa: E402  (`_grid_on_cpu`가 이 모듈의 함수를 감싼다)
from nets.segnet import Decoder, Segnet  # noqa: E402

from projects.models.encoder_stride4 import (  # noqa: E402
    STRIDE4_ENCODER_TYPE,
    Encoder_res101_stride4,
)

NUM_CLASSES = 3

# 원본 `Decoder`가 만들지만 3-class 학습이 쓰지 않는 head들. nuScenes instance segmentation용이라
# 이 태스크에는 대응하는 라벨도 loss도 없다. loss에 안 들어가니 gradient는 0이지만
# `Decoder.forward`가 매 스텝 계산은 한다 -- BEV 전 해상도에서 Conv3x3(128->128)을 세 번 도는
# 것이라 공짜가 아니다. 실측(2026-08-18): decoder forward가 pretrain 240x240 bs16에서
# 68.3ms -> 38.3ms(-43.9%), fine-tune 120x120 bs8에서 10.0ms -> 6.7ms(-33.1%).
# 파라미터도 decoder 3,831,046개 중 459,267개(12.0%)가 여기 묶여 있어 임베디드 배포에 그대로 실린다.
DISCARDED_HEADS = ("feat_head", "instance_center_head", "instance_offset_head")


def build_three_class_head(channels: int, num_classes: int = NUM_CLASSES) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, num_classes, kernel_size=1, padding=0),
    )


class ThreeClassDecoder(Decoder):
    """Simple-BEV decoder trunk with a 3-class segmentation projection.

    `Decoder.forward`를 그대로 쓰지 않고 다시 구현한다 -- 원본은 안 쓰는 head 세 개를 항상
    계산하기 때문이다(`DISCARDED_HEADS` 주석 참고). trunk 부분은 원본과 동일하며,
    `tests/models/test_simplebev_three_class.py`가 원본 trunk와 같은 값을 내는지 대조한다.
    """

    def __init__(self, in_channels: int, predict_future_flow: bool = False,
                 num_classes: int = NUM_CLASSES):
        super().__init__(in_channels=in_channels, n_classes=1, predict_future_flow=predict_future_flow)
        self.segmentation_head = build_three_class_head(in_channels, num_classes)
        for name in DISCARDED_HEADS:
            delattr(self, name)

    def forward(self, x, bev_flip_indices=None):
        # trunk -- `nets/segnet.py`의 `Decoder.forward`와 같은 순서다.
        skip_x = {"1": x}
        x = self.relu(self.bn1(self.first_conv(x)))
        x = self.layer1(x)
        skip_x["2"] = x
        x = self.layer2(x)
        skip_x["3"] = x
        x = self.layer3(x)
        x = self.up3_skip(x, skip_x["3"])
        x = self.up2_skip(x, skip_x["2"])
        x = self.up1_skip(x, skip_x["1"])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            # [-3]이 아니라 [-2]인 것은 원본과 같다 -- 이 시점에는 Y축이 이미 사라졌다.
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        segmentation = self.segmentation_head(x)
        # `Segnet.forward`가 5-튜플로 언패킹하므로 키는 그대로 유지하고 값만 None으로 둔다.
        return {
            "raw_feat": x,
            "feat": None,
            "segmentation": segmentation,
            "three_class": segmentation,
            "instance_center": None,
            "instance_offset": None,
            "instance_flow": None,
        }


@contextmanager
def _grid_on_cpu():
    """`Segnet.__init__`의 4x4 역행렬을 CPU에서 계산하게 한다 -- **Jetson 이식성 문제 하나.**

    `Segnet.__init__`(segnet.py:371)이 `gridcloud3d(...)` -> `vox_util.Mem2Ref(...)`를 부르고,
    그 안에서 `mem_T_ref.inverse()`(vox.py:108)가 돈다. `gridcloud3d`의 `device` 기본값이
    **`'cuda'`**라 그 역행렬이 CUDA에서 계산되고, **CUDA 경로는 cuSOLVER를 탄다.**

    Jetson에서 torch wheel과 JetPack의 CUDA 버전이 어긋나면 여기서 죽는다(실측, 2026-08-26):

        RuntimeError: Error in dlopen: .../libtorch_cuda_linalg.so:
        undefined symbol: cusolverDnXsyevBatched_bufferSize, version libcusolver.so.11

    **CPU에서 계산해도 결과가 같고 부작용이 없다.** 근거 셋:

    1. **생성 시 단 한 번**이다 -- 학습·추론 루프에 없으므로 성능과 무관하다.
    2. `Segnet.forward`가 `self.xyz_camA.to(feat_camXs_.device)`로 **어차피 다시 옮긴다**
       (segnet.py:417). 즉 이 텐서가 어느 장치에 있어도 forward는 정상 동작한다.
    3. forward가 쓰는 다른 역행렬 `utils.geom.safe_inverse`는 **transpose와 matmul뿐**이라
       cuSOLVER를 타지 않는다. 즉 **막히는 곳은 이 한 지점뿐**이다.

    `third_party/`는 수정하지 않으므로(`AGENTS.md`) 여기서 감싼다. 4x4 역행렬 하나의
    수치가 CPU와 CUDA에서 갈릴 여지는 float32 반올림 수준이고, 그 뒤 `grid_sample`의
    표본 좌표로 들어가므로 §18.3이 다룬 규약 오차(1/3 특징픽셀)보다 몇 자릿수 작다.
    """
    original = utils_basic.gridcloud3d

    def on_cpu(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original(*args, **kwargs)

    utils_basic.gridcloud3d = on_cpu
    try:
        yield
    finally:
        utils_basic.gridcloud3d = original


class ThreeClassSegnet(Segnet):
    """`Segnet` variant whose segmentation output has free/occupied/unknown logits.

    `encoder_type="res101_s4"`면 lifting이 표본하는 2D 특징맵이 stride 8 -> 4가 된다
    (진단 문서 §29). upstream `Segnet.__init__`은 `assert encoder_type in [...]`로
    모르는 문자열에 죽으므로 **super()에는 `"res101"`을 넘겨 통과시키고, 생성된 뒤
    `self.encoder`만 교체한다** -- 이 문자열은 우리 래퍼가 소비하고 upstream에는 가지 않는다.
    나머지(격자·lifting·decoder)는 전부 불변이다.
    """

    def __init__(self, *args, num_classes: int = NUM_CLASSES, **kwargs):
        latent_dim = kwargs.get("latent_dim", 128)
        stride4 = kwargs.get("encoder_type") == STRIDE4_ENCODER_TYPE
        if stride4:
            kwargs["encoder_type"] = "res101"
        # `Segnet(Z, Y, X, vox_util)`에서 `Y`와 `vox_util.Y`가 어긋나면 lifting이 만든 볼륨과
        # `bev_compressor`가 기대하는 채널 수가 달라진다. 형상 에러는 나지만 한참 뒤 forward에서
        # 나므로 원인이 안 보인다. **호출부 20곳이 각자 리터럴 `Y`를 들고 있었으므로** 여기서
        # 즉시 잡는다. `Y`의 단일 출처는 `projects.datasets.simplebev_vox.vox_dims`다.
        vox_util_arg = args[3] if len(args) > 3 else kwargs.get("vox_util")
        if vox_util_arg is not None and len(args) >= 3 and args[1] != vox_util_arg.Y:
            raise ValueError(
                f"Segnet의 Y={args[1]}와 vox_util.Y={vox_util_arg.Y}가 다르다 -- "
                f"둘 다 vox_dims(grid_spec, height_bins)에서 받아야 한다")
        with _grid_on_cpu():
            super().__init__(*args, **kwargs)
        # `_grid_on_cpu`가 CPU에 만든 `xyz_camA`를 원래 장치로 되돌린다. 안 되돌리면
        # `Segnet.forward`(segnet.py:417)의 `.to(device)`가 매 forward마다 실제 복사가 되어
        # **지연 측정에 들어간다** -- 173 KB이지만 벤치마크 숫자를 오염시킬 이유가 없다.
        if getattr(self, "xyz_camA", None) is not None and torch.cuda.is_available():
            self.xyz_camA = self.xyz_camA.cuda()
        if stride4:
            # super()가 만든 stride-8 encoder는 여기서 버려진다. ImageNet 가중치를 두 번
            # 읽는 낭비지만 생성 1회뿐이고, 이렇게 해야 `third_party/`를 안 건드린다.
            self.encoder = Encoder_res101_stride4(self.feat2d_dim)
            # 체크포인트를 다시 얹을 때 이 값으로 encoder를 고르므로 실제 값을 남긴다.
            self.encoder_type = STRIDE4_ENCODER_TYPE
        self.decoder = ThreeClassDecoder(
            in_channels=latent_dim, predict_future_flow=False, num_classes=num_classes
        )


def height_bins_from_state_dict(state_dict) -> int:
    """체크포인트가 **자기 `Y`를 스스로 말하게 한다.**

    `bev_compressor[0]`이 `Conv2d(latent_dim*Y -> latent_dim, 3x3)`이므로 가중치 형상이
    `(latent_dim, latent_dim*Y, 3, 3)`이고, 두 값의 비가 곧 `Y`다.

    왜 이렇게 하나: `saverloader.save`(third_party, 수정 금지)가 state_dict만 저장하고
    config 메타데이터를 남기지 않는다. 평가 도구가 `Y`를 인자로 받게 하면 **사람이 매번
    맞춰 줘야 하고 틀리면 조용히 다른 아키텍처를 만든다.** 형상에서 되읽으면 학습과 평가가
    구조적으로 어긋날 수 없다.

    `Y=1`인 옛 체크포인트는 그대로 1을 돌려주므로 하위 호환이 유지된다.
    """
    weight = state_dict.get("bev_compressor.0.weight")
    if weight is None:
        raise KeyError("bev_compressor.0.weight가 없다 -- Simple-BEV 체크포인트가 맞는가?")
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    if in_channels % out_channels:
        raise ValueError(
            f"bev_compressor 입력 채널 {in_channels}이 출력 {out_channels}의 배수가 아니다")
    return in_channels // out_channels


def num_classes_from_state_dict(state_dict) -> int:
    """체크포인트가 **자기 출력 채널 수를 스스로 말하게 한다.**

    `height_bins_from_state_dict`와 같은 이유다 -- `saverloader.save`가 메타데이터를 남기지
    않으므로, 도구가 `--formulation`을 사람에게서 받으면 틀렸을 때 `strict=True`가 죽는다.
    형상에서 되읽으면 그런 실수가 아예 불가능하다. binary=2, three_class=3.
    """
    weight = state_dict.get("decoder.segmentation_head.3.weight")
    if weight is None:
        raise KeyError("decoder.segmentation_head.3.weight가 없다 -- 이 저장소의 체크포인트가 맞는가?")
    return int(weight.shape[0])


def unexpected_skips(skipped) -> list:
    """전이되지 않은 키 중 **설명되지 않는 것**만 남긴다.

    설명되는 skip은 둘이다:
    - `segmentation_head`: 출력 head. 형상이 다른 체크포인트에서는 안 넘어오는 게 정상이고,
      그때는 랜덤 초기화로 시작한다.
    - `DISCARDED_HEADS`: 이 모델이 아예 갖고 있지 않은 head. 제거 이전에 만들어진
      체크포인트에는 남아 있으므로 skip되는 것이 정상이다.

    그 외의 skip은 trunk가 안 붙었다는 뜻이라 학습을 시작하면 안 된다.
    """
    return [
        name for name in skipped
        if "segmentation_head" not in name
        and not any(head in name for head in DISCARDED_HEADS)
    ]


def head_was_transferred(skipped) -> bool:
    """출력 head까지 전이됐는가. 3-class -> 3-class 전이에서만 True다."""
    return not any("segmentation_head" in name for name in skipped)


def load_trunk_weights(model, checkpoint_path, device) -> dict:
    """체크포인트에서 형상이 맞는 키만 명시적으로 복사한다.

    `strict=False`로 조용히 넘기지 않고 하나씩 대조하는 이유: 전이되지 않은 키가
    반환값의 `skipped`에 그대로 남아야 "출력 head가 랜덤 초기화로 시작했다"와
    "trunk가 통째로 안 붙었다"를 구분할 수 있다. 판정은 `unexpected_skips`가 한다.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()

    transfer, skipped = {}, []
    for name, tensor in source.items():
        if name in target and target[name].shape == tensor.shape:
            transfer[name] = tensor
        else:
            skipped.append(name)

    target.update(transfer)
    model.load_state_dict(target)
    model.to(device)
    return {"loaded": len(transfer), "skipped": skipped}
