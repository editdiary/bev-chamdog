"""특징맵 표본 좌표의 **규약** -- `grid_sample` 정규화와 상수 오프셋.

근거: `docs/finetune_overfitting_diagnosis.md` §18.3.

upstream Simple-BEV(`utils/vox.py:337`)와 우리 두 래퍼(`double_sphere_vox`,
`fisheye_vox`)는 표본 좌표를 `utils.basic.normalize_grid2d`로 정규화한 뒤
`F.grid_sample(align_corners=False)`로 표본한다. **두 함수의 규약이 다르다.**

- `normalize_grid2d`: 픽셀 **인덱스** 규약. `g = 2x/(W-1) - 1`. 즉 `x=0`과 `x=W-1`이
  각각 `-1`, `+1`로 간다 -- 이것은 `align_corners=True`의 규약이다.
- `grid_sample(align_corners=False)`: 픽셀 **가장자리** 규약. `g=-1`이 텐서의 왼쪽
  가장자리(픽셀 0의 중심보다 0.5px 왼쪽), `g=+1`이 오른쪽 가장자리다.

두 규약을 합성하면 실제 표본 위치는

    x' = x * W/(W-1) - 0.5

가 된다. **상수 편이가 아니라 배율 오차(W/(W-1))와 편이가 섞여 있어서**, 오프셋
하나로는 없앨 수 없다. W=64에서 실측한 값(§18.3의 표):

    목표 x |  8     16     32     48     63
    실제   |  7.63  15.75  32.01  48.26  63.5 (-> zero-padding과 섞여 값이 절반)

중심에서 0이고 가장자리로 갈수록 커지는 계통 오차다.

## 이 모듈이 나누는 두 가지

1. **정규화 규약**(`convention`) -- 위의 배율 오차. `pixel_center`가 옳고, 그러면
   `x' = x`가 **정확히** 성립한다(`tests/models/test_pixel_grid.py`가 고정한다).
   `legacy_index`는 기존 동작을 비트 단위로 보존하는 **대조군**이다.

2. **상수 오프셋**(`pixel_offset`, 특징픽셀 단위) -- 규약을 고쳐도 남는 부분.
   native -> 입력 리사이즈 규약과 stride 8 특징맵의 수용영역 중심이 어디인지는
   유도로 확정되지 않는다. 두 극단을 계산해 보면

   - 리사이즈가 half-pixel center(PIL/cv2 기본)이고 stride 8 특징 `j`의 중심이
     입력 픽셀 `8j+3.5`라면  ->  **-0.475** 특징픽셀
   - 같은 리사이즈에 중심이 `8j`(torchvision ResNet의 stride-2 conv는 좌상단을
     집는다)라면  ->  **-0.0375** 특징픽셀

   여기에 `Encoder_res101`의 `upsampling_layer`(stride16 -> stride8 보간)가 자기
   규약을 더한다. 그래서 **유도로 정하지 않고 스윕으로 실측한다**
   (`configs/sweep_pixel_offset.sh`).

   **오프셋은 이 모듈이 아니라 `unproject_image_to_mem`이 적용한다** -- native -> 특징
   좌표 변환 직후, 좌우 반전(`mirror_x`)과 유효 영역 판정보다 **앞**이어야 하기 때문이다.
   반전 뒤에 더하면 x축에서 부호가 뒤집히고(`(W-1) - (x+o)` != `(W-1) - x + o`),
   유효 영역도 실제 표본 위치가 아닌 좌표로 판정된다.

## 기본값을 왜 아직 `legacy_index`로 두는가

`pixel_center`가 정규화 단계만 놓고 보면 명백히 옳지만, 그것이 **val을 올리는지는
아직 측정 전이다.** 기본값을 먼저 바꾸면 기존 런 전부와 비교 불가능해진다. 스윕
결과가 나온 뒤 근거와 함께 기본값을 옮긴다.
"""
import sys
from pathlib import Path

import torch

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.basic  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)

LEGACY_INDEX = "legacy_index"
PIXEL_CENTER = "pixel_center"
CONVENTIONS = (LEGACY_INDEX, PIXEL_CENTER)

DEFAULT_CONVENTION = LEGACY_INDEX


def normalize_pixel_grid2d(y, x, H, W, convention=DEFAULT_CONVENTION):
    """픽셀 좌표 `(y, x)`를 `grid_sample(align_corners=False)`용 `[-1, 1]` 좌표로 보낸다.

    반환은 `utils.basic.normalize_grid2d`와 같은 `(grid_y, grid_x)` 순서다.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention은 {CONVENTIONS} 중 하나여야 한다: {convention!r}")

    if convention == LEGACY_INDEX:
        # 기존 동작 그대로. 비트 단위 보존을 위해 upstream 함수를 그대로 부른다.
        return utils.basic.normalize_grid2d(y, x, H, W)

    # 픽셀 **중심** 규약: `x = 0`이 픽셀 0의 중심을 가리키고, `align_corners=False`의
    # 역변환 `x' = ((g+1)*W - 1)/2`에 넣으면 `x' = x`가 정확히 나온다.
    grid_x = (2.0 * x + 1.0) / float(W) - 1.0
    grid_y = (2.0 * y + 1.0) / float(H) - 1.0

    # upstream과 같은 clamp -- 화각 밖으로 크게 튄 투영이 grid_sample에 inf로 들어가는 것을
    # 막는다. 유효 영역 판정은 호출자(`unproject_image_to_mem`)가 clamp 이전 좌표로 한다.
    grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
    return grid_y, grid_x
