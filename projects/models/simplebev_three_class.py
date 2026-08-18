"""Three-class Simple-BEV wrapper for free/occupied/unknown logits.

The upstream Simple-BEV submodule stays untouched. This module reuses its
encoder, lifting, BEV compressor, and decoder trunk, then replaces the final
segmentation projection with one 3-channel softmax-ready head.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

from nets.segnet import Decoder, Segnet  # noqa: E402

NUM_CLASSES = 3

# 원본 `Decoder`가 만들지만 3-class 학습이 쓰지 않는 head들. nuScenes instance segmentation용이라
# 이 태스크에는 대응하는 라벨도 loss도 없다. loss에 안 들어가니 gradient는 0이지만
# `Decoder.forward`가 매 스텝 계산은 한다 -- BEV 전 해상도에서 Conv3x3(128->128)을 세 번 도는
# 것이라 공짜가 아니다. 실측(2026-08-18): decoder forward가 pretrain 240x240 bs16에서
# 68.3ms -> 38.3ms(-43.9%), fine-tune 120x120 bs8에서 10.0ms -> 6.7ms(-33.1%).
# 파라미터도 decoder 3,831,046개 중 459,267개(12.0%)가 여기 묶여 있어 임베디드 배포에 그대로 실린다.
DISCARDED_HEADS = ("feat_head", "instance_center_head", "instance_offset_head")


def build_three_class_head(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, NUM_CLASSES, kernel_size=1, padding=0),
    )


class ThreeClassDecoder(Decoder):
    """Simple-BEV decoder trunk with a 3-class segmentation projection.

    `Decoder.forward`를 그대로 쓰지 않고 다시 구현한다 -- 원본은 안 쓰는 head 세 개를 항상
    계산하기 때문이다(`DISCARDED_HEADS` 주석 참고). trunk 부분은 원본과 동일하며,
    `tests/models/test_simplebev_three_class.py`가 원본 trunk와 같은 값을 내는지 대조한다.
    """

    def __init__(self, in_channels: int, predict_future_flow: bool = False):
        super().__init__(in_channels=in_channels, n_classes=1, predict_future_flow=predict_future_flow)
        self.segmentation_head = build_three_class_head(in_channels)
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


class ThreeClassSegnet(Segnet):
    """`Segnet` variant whose segmentation output has free/occupied/unknown logits."""

    def __init__(self, *args, **kwargs):
        latent_dim = kwargs.get("latent_dim", 128)
        super().__init__(*args, **kwargs)
        self.decoder = ThreeClassDecoder(in_channels=latent_dim, predict_future_flow=False)


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
