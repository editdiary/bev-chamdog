import numpy as np

from projects.bev_gt.bev_crop import crop_bev_occupancy
from projects.bev_gt.grid import OccupancyGridSpec


def test_crop_bev_occupancy_extracts_expected_region_and_remaps():
    # 30x20 synthetic semantic image, 1m/px. Ego sits at pixel row 10.5 (a half-integer,
    # so that the grid's half-integer row offsets land exactly on integer pixel rows —
    # avoids numpy's round-half-to-even ambiguity at exact .5 boundaries).
    # Rows 0..10 are "sidewalk"(8, non-drivable); rows 11+ are "road"(7, drivable).
    semantic_image = np.full((30, 20), 8, dtype=np.uint8)
    semantic_image[11:, :] = 7

    grid_spec = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=1.0)
    occupancy = crop_bev_occupancy(
        semantic_image, grid_spec, meters_per_pixel=1.0, origin_px=(10.5, 10.5),
        sign_forward=1, sign_lateral=1,
    )

    assert occupancy.shape == (grid_spec.n_rows, grid_spec.n_cols)
    # front-most row (image row 13) -> road -> drivable; rear-most row (image row 10) -> sidewalk -> non-drivable
    assert occupancy[-1].all()
    assert not occupancy[0].any()
