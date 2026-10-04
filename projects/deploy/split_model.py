"""학습된 `ThreeClassSegnet`을 **인코더 단계**와 **BEV 단계**로 나눈다 -- Jetson 배포 측정용.

왜 나누나 (2026-10-04, 사용자 결정: "encoder only TensorRT"):

- **인코더**(ResNet-101 stride 8)는 이미지 한 장 -> 특징맵인 평범한 2D CNN이라 ONNX -> TensorRT로
  그대로 변환된다. 옛 Orin 측정에서 지연의 75~82 %가 여기였다.
- **BEV 단계**(특징 집어 오기 `grid_sample` + 카메라 평균 + BEV 압축 + 디코더)는 PyTorch로 둔다.

**투영은 상수다.** 리그가 고정이라 "BEV 복셀마다 특징맵의 어느 위치를 읽을지"(`grid`)와 유효
마스크(`valid`)는 이미지와 무관하다. 여기서 fp32로 한 번 계산해 버퍼로 박으므로, 배포 단계에는
어안 투영 계산이 아예 없다(`DoubleSphereVoxUtil.sampling_grid`).

**계약: 두 단계를 이은 결과는 원래 `Segnet.forward`와 같다.** `tests/deploy/test_split_model.py`가
대조한다. 연산 순서도 원본(segnet.py `forward`, `DoubleSphereVoxUtil.unproject_image_to_mem`)을
그대로 따른다 -- 순서를 바꾸면 float 반올림이 달라진다.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.basic  # noqa: E402
import utils.geom  # noqa: E402


class EncoderStage(nn.Module):
    """`(N, 3, H, W)` RGB `[0, 1]` -> `(N, C, H/8, W/8)` 특징맵. N = 프레임 수 × 카메라 수.

    정규화(ImageNet mean/std)를 안에 넣는다 -- TensorRT 엔진의 입력이 곧 `[0, 1]` 이미지가 된다.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.register_buffer("mean", model.mean.detach().clone().reshape(1, 3, 1, 1))
        self.register_buffer("std", model.std.detach().clone().reshape(1, 3, 1, 1))

    def forward(self, rgb01):
        # 원본과 **같은 연산 순서**로 둔다 -- `(x - 0.5) + 0.5`는 float에서 `x`와 1 ulp 다를 수 있다.
        x = rgb01 - 0.5
        return self.encoder((x + 0.5 - self.mean) / self.std)


class BevStage(nn.Module):
    """`(S, C, Hf, Wf)` 한 프레임의 카메라별 특징맵 -> `(1, H, W)` `p(free)`.

    배포 단위는 **한 프레임**(배치 1)이다. `grid`·`valid`가 카메라 S개분만 박혀 있다.
    """

    def __init__(self, model, grid, valid):
        super().__init__()
        self.Z, self.Y, self.X = model.Z, model.Y, model.X
        self.feat2d_dim = model.feat2d_dim
        self.bev_compressor = model.bev_compressor
        self.decoder = model.decoder
        self.register_buffer("grid", grid.detach().clone())    # (S, Z, Y, X, 3)
        self.register_buffer("valid", valid.detach().clone())  # (S, 1, Z, Y, X)

    def forward(self, feat):
        S, C = feat.shape[0], feat.shape[1]
        Z, Y, X = self.Z, self.Y, self.X
        values = F.grid_sample(feat.unsqueeze(2), self.grid, align_corners=False)
        values = torch.reshape(values, (S, C, Z, Y, X)) * self.valid
        feat_mems = values.unsqueeze(0)                          # (1, S, C, Z, Y, X)
        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)
        feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(1, self.feat2d_dim * Y, Z, X)
        feat_bev = self.bev_compressor(feat_bev_)
        logits = self.decoder(feat_bev)["segmentation"]
        return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def constant_sampling_grid(model, vox_util, pix_T_cams, cam0_T_camXs, image_hw):
    """한 프레임의 카메라 보정 `(S, 4, 4)` 둘에서 `(grid, valid)`를 fp32로 낸다.

    `Segnet.forward`가 `unproject_image_to_mem`에 넘기는 인자를 그대로 재현한다. 특징맵 크기는
    인코더 stride(8)에서 정해진다 -- 원본이 실제 특징맵 형상에서 읽으므로 여기서도 형상으로 받는다.
    """
    S = cam0_T_camXs.shape[0]
    H, W = image_hw
    device = cam0_T_camXs.device
    feat = model.encoder(torch.zeros(1, 3, H, W, device=device))
    Hf, Wf = feat.shape[-2:]
    camXs_T_cam0 = utils.geom.safe_inverse(cam0_T_camXs)
    xyz_camA = (model.xyz_camA.to(device).repeat(S, 1, 1)
                if model.xyz_camA is not None else None)
    del pix_T_cams  # DS 투영은 pix_T_cams를 쓰지 않는다(보정은 vox_util 안에 있다)
    grid, valid = vox_util.sampling_grid(camXs_T_cam0, model.Z, model.Y, model.X, Hf, Wf,
                                         xyz_camA=xyz_camA)
    return grid, valid


def split_model(model, vox_util, pix_T_cams, cam0_T_camXs, image_hw):
    """`(EncoderStage, BevStage)`. 보정 텐서는 **한 프레임분** `(S, 4, 4)`를 넘긴다."""
    grid, valid = constant_sampling_grid(model, vox_util, pix_T_cams, cam0_T_camXs, image_hw)
    return EncoderStage(model).eval(), BevStage(model, grid, valid).eval()
