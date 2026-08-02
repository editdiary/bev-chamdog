import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, remap_semantic_to_occupancy

# Task 8(tools/calibrate_bev_scale.py) 실행 결과로 확정되는 값. 초기값은 샘플 00000
# 단일 계측 기반(design doc §4.3)이며, Task 8을 실행한 뒤 평균값으로 갱신한다.
BEV_METERS_PER_PIXEL = 0.0284
BEV_ORIGIN_PX = (511.5, 511.5)
SIGN_FORWARD = -1
SIGN_LATERAL = -1


def crop_bev_occupancy(
    semantic_bev_image: np.ndarray,
    grid_spec: OccupancyGridSpec,
    meters_per_pixel: float = BEV_METERS_PER_PIXEL,
    origin_px=BEV_ORIGIN_PX,
    sign_forward: int = SIGN_FORWARD,
    sign_lateral: int = SIGN_LATERAL,
) -> np.ndarray:
    """Crop and remap a semantic BEV image into an ego-relative occupancy grid.

    Row axis of `semantic_bev_image` corresponds to forward/backward (ego X),
    column axis corresponds to left/right (ego Y) — confirmed empirically in
    docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md §4.3.
    """
    origin_col, origin_row = origin_px
    row_offsets_m = (np.arange(grid_spec.n_rows) + 0.5) * grid_spec.cell_m - grid_spec.rear_m
    col_offsets_m = (np.arange(grid_spec.n_cols) + 0.5) * grid_spec.cell_m - grid_spec.half_width_m

    rows = np.round(origin_row + sign_forward * row_offsets_m / meters_per_pixel).astype(int)
    cols = np.round(origin_col + sign_lateral * col_offsets_m / meters_per_pixel).astype(int)

    height, width = semantic_bev_image.shape
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)

    sampled_labels = semantic_bev_image[np.ix_(rows, cols)]
    return remap_semantic_to_occupancy(sampled_labels)
