"""`projects/bev_gt` grid_spec -> Simple-BEV `Vox_util`/`Segnet(Z, Y, X)` 설정.

Simple-BEV의 참조("ref"/"cam0") 프레임은 축 이름(Z/Y/X) 자체에 물리적 의미를 두지 않지만,
코드 관례상 두 가지는 고정돼 있다 (`nets/segnet.py`):
- `Y`축은 `bev_compressor`가 채널로 접어버리는 축이다 → 우리에게는 **높이**여야 한다
  (occupancy GT가 지면 2D뿐이므로 `Y=1`).
- `(Z, X)`가 decoder 출력의 (row, col) 두 공간축이 된다.

`projects/geometry`가 검증한 ego 프레임(X=전방, Y=좌측, Z=상방, 오른손계)은 이 축 이름과
다르므로, **회전만으로** 축을 맞바꾸는 고정 행렬 `EGO_TO_REF_ROTATION`을 둔다(평행이동 없음 —
두 프레임 모두 원점이 차량 중심·지면 높이로 같다). 부호는 `projects.bev_gt.grid`의 관례
(`cell_centers_m`: row 0 = 최전방, col 0 = 차량 좌측)가 `Vox_util`의 mem index 증가 방향
(각 축 MIN→MAX일 때 index 0→N-1)과 그대로 맞도록 다음처럼 정했다 — 그래서 occupancy GT
배열을 따로 뒤집을 필요가 없다:

    ref.x = -ego.y   (ego.y=+half_width(좌측)일 때 ref.x=XMIN → mem_x=0 = col 0 = 좌측)
    ref.y =  ego.z   (높이. Y=1 bin이라 방향 무관)
    ref.z = -ego.x   (ego.x=+front_m(최전방)일 때 ref.z=ZMIN → mem_z=0 = row 0 = 최전방)

수치 검증(`tests/datasets/test_simplebev_vox.py`)이 이 부호가 실제로 `cell_centers_m`과
일치함을 `Vox_util.Ref2Mem`으로 직접 확인한다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.vox  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)

from projects.bev_gt.grid import OccupancyGridSpec  # noqa: E402

EGO_TO_REF_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ]
)


def ref_T_ego_4x4() -> np.ndarray:
    ref_T_ego = np.eye(4, dtype=np.float64)
    ref_T_ego[:3, :3] = EGO_TO_REF_ROTATION
    return ref_T_ego


def ref_T_cam_from_ego_T_cam(ego_T_cam: np.ndarray) -> np.ndarray:
    """`ego_T_cam`(camera -> ego) -> `ref_T_cam`(camera -> Simple-BEV ref frame).

    `Segnet.forward`에 `cam0_T_camXs`로 그대로 넘긴다(변수명은 "cam0"이지만 실제로는 이
    ref 프레임 — Segnet은 그 이름에 특별한 의미를 두지 않는다).
    """
    return ref_T_ego_4x4() @ ego_T_cam


def vox_dims(grid_spec: OccupancyGridSpec) -> tuple:
    """(Z, Y, X) - `Segnet(Z, Y, X, ...)` 생성자에 그대로 넘긴다."""
    return grid_spec.n_rows, 1, grid_spec.n_cols


def vox_bounds(grid_spec: OccupancyGridSpec, height_margin_m: float = 0.25) -> tuple:
    """(XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX) - `Vox_util(bounds=...)`에 그대로 넘긴다."""
    return (
        -grid_spec.half_width_m, grid_spec.half_width_m,
        -height_margin_m, height_margin_m,
        -grid_spec.front_m, grid_spec.rear_m,
    )


def build_vox_util(grid_spec: OccupancyGridSpec, height_margin_m: float = 0.25, device="cpu"):
    Z, Y, X = vox_dims(grid_spec)
    bounds = vox_bounds(grid_spec, height_margin_m)
    scene_centroid = torch.zeros(1, 3, dtype=torch.float32, device=device)
    return utils.vox.Vox_util(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds, assert_cube=False)
