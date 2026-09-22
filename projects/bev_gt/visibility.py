"""BEV grid cell별 4-cam 가시성(occlusion) 판정 — "ignore" 버전 GT용.

`tools/build_occupancy_gt.py`가 만드는 binary occupancy는 항상 값이 있다 — top-down
`_BEV.png` 라벨을 그대로 읽으므로, 실제로 ego의 어안 카메라가 그 지점을 봤는지와 무관하다.
이 모듈은 각 grid cell을 지면(z=0) 점으로 보고 어안 카메라들에 투영해, 그 중 한 대에서라도
실제로(가려지지 않고) 보이는지를 판정한다. 어느 카메라에서도 안 보이는 cell은 학습 loss에서
제외(ignore)하기 위한 별도 mask로 쓴다.
"""
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, cell_centers_m
from projects.geometry.reprojection import project_points_to_image, visibility_mask

# occupancy grid 자체가 지면 투영이므로, grid cell도 지면(z=0) 점으로 일관되게 취급한다.
GROUND_HEIGHT_M = 0.0


def compute_per_camera_visible_masks(grid_spec: OccupancyGridSpec, cameras: dict, depth_maps: dict) -> dict:
    """카메라 이름 -> (n_rows, n_cols) bool 배열. `compute_visible_mask`가 OR로 합치기 전,
    카메라별 판정을 그대로 보고 싶을 때(디버깅/시각화) 쓴다. 방향·좌표 규약은
    `compute_visible_mask`와 동일 — `cell_centers_m`을 공유해서 보장한다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    forward_grid, lateral_grid = np.meshgrid(forward_m, lateral_m, indexing="ij")
    points_ego = np.stack(
        [forward_grid.ravel(), lateral_grid.ravel(), np.full(forward_grid.size, GROUND_HEIGHT_M)],
        axis=1,
    )

    per_camera = {}
    for camera_name, camera in cameras.items():
        visible = np.zeros(points_ego.shape[0], dtype=bool)
        projected = project_points_to_image(camera, points_ego)
        if len(projected):
            seen = visibility_mask(projected, depth_maps[camera_name])
            visible[projected.index[seen]] = True
        per_camera[camera_name] = visible.reshape(grid_spec.n_rows, grid_spec.n_cols)

    return per_camera


def compute_visible_mask(grid_spec: OccupancyGridSpec, cameras: dict, depth_maps: dict) -> np.ndarray:
    """(n_rows, n_cols) bool 배열. True = 카메라 중 최소 한 대에서 실제로(가려지지 않고) 보임.

    `cameras`/`depth_maps`는 카메라 이름(예: "FV") -> `Camera`/depth map(np.ndarray) 매핑이며,
    같은 카메라 이름 집합을 키로 가져야 한다.
    """
    per_camera = compute_per_camera_visible_masks(grid_spec, cameras, depth_maps)
    visible = np.zeros((grid_spec.n_rows, grid_spec.n_cols), dtype=bool)
    for mask in per_camera.values():
        visible |= mask
    return visible


# gather-column 가시성의 높이 표본. 0.0~0.8 m를 33점으로 훑는다 -- 아래 재현 검증에서
# 17점(0.9395) → 33점(0.9406) → 65점(0.9411)로 포화하므로 33점이면 충분하다.
COLUMN_HEIGHTS_M = tuple(float(h) for h in np.linspace(0.0, 0.8, 33))


def compute_column_visible_mask(grid_spec: OccupancyGridSpec, cameras: dict, depth_maps: dict,
                                heights_m=COLUMN_HEIGHTS_M) -> np.ndarray:
    """**gather-column 가시성**: 셀 위 기둥(z=0~0.8 m) 중 한 점이라도 보이면 visible.

    `compute_visible_mask`(지면 z=0 한 점)와 다른 규칙이다. 포장된 라벨
    `dataset/synwoodscape_2head_roi_8_4_6_h08/*_visible.npy`가 이 규칙으로 만들어졌다 --
    리프팅이 높이 축을 여러 개 표본하므로, 지면만 보고 판정하면 실제로 특징을 모을 수 있는
    셀을 ignore로 버리게 된다.

    **재현 검증(2026-09-22).** 원 생성 스크립트(`work_dirs/build_visibility_h08_roi_8_4_6.py`)는
    저장소에 남아 있지 않아 문서에 적힌 규칙으로 다시 구현했다. ROI 8/4/±6에서 기록된
    `visibility_h08/raw_visible/`과 대조하면 **셀 단위 일치율 99.0 %**다(7샘플). 남는 1 %는
    occlusion 허용치 처리 차이로 보이며, 평균 가시 비율은 0.9406 대 기록본 0.9486이다.
    **그러므로 이 함수로 만든 마스크를 기록본과 한 표에 섞지 않는다** -- 새 그리드용으로만 쓴다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    forward_grid, lateral_grid = np.meshgrid(forward_m, lateral_m, indexing="ij")
    visible = np.zeros(forward_grid.size, dtype=bool)
    for height in heights_m:
        points_ego = np.stack(
            [forward_grid.ravel(), lateral_grid.ravel(),
             np.full(forward_grid.size, float(height))], axis=1)
        for camera_name, camera in cameras.items():
            projected = project_points_to_image(camera, points_ego)
            if not len(projected):
                continue
            seen = visibility_mask(projected, depth_maps[camera_name])
            visible[projected.index[seen]] = True
    return visible.reshape(grid_spec.n_rows, grid_spec.n_cols)
