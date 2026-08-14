"""BEV 셀이 카메라 이미지 안에 담기는지 판정 — 자체 리그(Double Sphere)용.

`projects/bev_gt/visibility.py`와 목적이 다르다. 그쪽은 depth map으로 **가림(occlusion)**까지
판정하는 SynWoodScape 전용이고, 여기는 depth 없이 **화각 커버리지만** 본다.

왜 이게 따로 필요한가: 자체 데이터셋의 visibility 라벨은 수동 occupancy에서 2D raycast로
재생성한 것이라 "다른 장애물에 가렸는가"만 담고 있고, "애초에 카메라가 이 지점을 담는가"는
빠져 있다. 그런데 이 리그는 카메라 3대가 모두 **수평**으로 달려 있어 지면 근처가 크게
비는데(반경 약 0.5 m는 어떤 광선도 닿지 않음), 그 셀들이 라벨에는 visible로 찍혀 있다.
그대로 학습하면 이미지에 근거가 없는 셀에서 occupancy 예측을 요구하게 된다.

`common/self_mask.png`(ego 테이블)가 이 문제의 나머지 절반을 덮지만, 그 마스크는 이미지
픽셀을 지면으로 투영해 만든 것이라 **광선이 아예 닿지 않는 안쪽 원반에는 구멍이 뚫려 있다**
(투영될 픽셀이 없으므로). 그래서 두 마스크는 상호 보완이며 둘 다 필요하다.
"""
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, cell_centers_m
from projects.geometry.double_sphere import se3_inverse

# occupancy grid 자체가 지면 투영이므로 grid cell도 지면 점으로 취급한다.
# ego 프레임 원점이 이미 지면 높이라 z=0이다(`projects.geometry.double_sphere` §2).
GROUND_HEIGHT_M = 0.0


def compute_per_camera_coverage(grid_spec: OccupancyGridSpec, cameras: dict,
                                ego_T_cams: dict) -> dict:
    """카메라 이름 -> (n_rows, n_cols) bool. True = 그 셀의 지면 점이 이미지 안에 찍힌다.

    `cameras`/`ego_T_cams`는 같은 이름 집합을 키로 가져야 한다. 방향·좌표 규약은
    `cell_centers_m`을 공유해 occupancy/visibility 라벨과 자동으로 맞춘다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    forward_grid, lateral_grid = np.meshgrid(forward_m, lateral_m, indexing="ij")
    points_ego = np.stack(
        [
            forward_grid.ravel(),
            lateral_grid.ravel(),
            np.full(forward_grid.size, GROUND_HEIGHT_M),
            np.ones(forward_grid.size),
        ],
        axis=1,
    )

    per_camera = {}
    for name, camera in cameras.items():
        points_cam = (se3_inverse(ego_T_cams[name]) @ points_ego.T).T[:, :3]
        u, v, valid = camera.project(points_cam)
        inside = valid & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        per_camera[name] = inside.reshape(grid_spec.n_rows, grid_spec.n_cols)
    return per_camera


def compute_camera_coverage_mask(grid_spec: OccupancyGridSpec, cameras: dict,
                                 ego_T_cams: dict) -> np.ndarray:
    """(n_rows, n_cols) bool. True = 카메라 중 최소 한 대가 그 셀을 이미지에 담는다."""
    per_camera = compute_per_camera_coverage(grid_spec, cameras, ego_T_cams)
    coverage = np.zeros((grid_spec.n_rows, grid_spec.n_cols), dtype=bool)
    for mask in per_camera.values():
        coverage |= mask
    return coverage
