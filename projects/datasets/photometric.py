"""카메라 이미지용 광도(photometric) augmentation.

기하는 전혀 건드리지 않는다. 밝기/대비/채도/감마/노이즈만 바꾸므로 `pix_T_cams`나 fisheye
`radial_poly` 캘리브레이션과 무관하고, BEV GT도 그대로 유효하다. 이미지 공간의 기하 변환
(flip/crop/rotation)은 calibration을 같이 고치지 않으면 projection이 어긋나므로 여기서 다루지
않는다 -- `docs/archive/training_improvement_plan.md` Step 4 참고.

한 샘플 안의 4개 카메라에는 **같은 파라미터**를 적용한다. 카메라마다 다른 색보정을 걸면 실제
리그에는 없는 카메라 간 색차를 학습하게 된다. 카메라별 노출 차이를 모사하고 싶다면 그건 별도
실험으로 분리하는 편이 해석이 깨끗하다.
"""
from dataclasses import dataclass

import torch
from torchvision.transforms.v2 import functional as TF

# 보수적인 시작 범위. 세게 걸면 SynWoodScape의 합성 렌더 특유의 색분포가 무너져서
# occupancy 경계 자체가 흐려질 수 있으므로, 효과가 확인되면 그때 넓힌다.
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
SATURATION_RANGE = (0.8, 1.2)
GAMMA_RANGE = (0.8, 1.25)
NOISE_STD_RANGE = (0.0, 0.02)


@dataclass(frozen=True)
class PhotometricParams:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    noise_std: float = 0.0


IDENTITY_PHOTOMETRIC = PhotometricParams()


def _uniform(low: float, high: float, generator) -> float:
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())


def sample_photometric_params(generator=None) -> PhotometricParams:
    return PhotometricParams(
        brightness=_uniform(*BRIGHTNESS_RANGE, generator=generator),
        contrast=_uniform(*CONTRAST_RANGE, generator=generator),
        saturation=_uniform(*SATURATION_RANGE, generator=generator),
        gamma=_uniform(*GAMMA_RANGE, generator=generator),
        noise_std=_uniform(*NOISE_STD_RANGE, generator=generator),
    )


def apply_photometric(rgb, params: PhotometricParams, generator=None):
    """`rgb`: (S, 3, H, W), 값 범위 [0, 1]. 같은 shape을 돌려준다.

    감마를 먼저 적용한다 -- 실제 카메라 파이프라인에서 감마는 센서 응답 쪽에 있고, 밝기/대비는
    그 뒤의 보정에 해당하기 때문이다.
    """
    if rgb.numel() == 0:
        return rgb

    out = rgb
    if params.gamma != 1.0:
        out = TF.adjust_gamma(out, params.gamma)
    if params.brightness != 1.0:
        out = TF.adjust_brightness(out, params.brightness)
    if params.contrast != 1.0:
        out = TF.adjust_contrast(out, params.contrast)
    if params.saturation != 1.0:
        out = TF.adjust_saturation(out, params.saturation)
    if params.noise_std > 0.0:
        noise = torch.empty_like(out).normal_(0.0, params.noise_std, generator=generator)
        out = out + noise
    return out.clamp(0.0, 1.0)
