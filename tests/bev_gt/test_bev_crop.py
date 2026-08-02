from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from projects.bev_gt.bev_crop import (
    BEV_METERS_PER_PIXEL,
    BEV_ORIGIN_PX,
    SIGN_FORWARD,
    SIGN_LATERAL,
    crop_bev_occupancy,
    ego_to_bev_pixel,
)
from projects.bev_gt.grid import SYNWOODSCAPE_PRETRAIN_GRID_SPEC, OccupancyGridSpec

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT / "semantic_annotations" / "gtLabels").exists(),
    reason="SynWoodScape dataset not available locally",
)
requires_depth = pytest.mark.skipif(
    not (DATASET_ROOT / "depth_maps" / "raw_data").exists(),
    reason="SynWoodScape depth maps not available locally",
)

# Ground-level SynWoodScape classes: road line(6), road(7), sidewalk(8), ground(14), terrain(20)
GROUND_CLASS_IDS = (6, 7, 8, 14, 20)
CAMERAS = ("FV", "MVL", "MVR", "RV")


def test_bev_meters_per_pixel_is_the_analytic_15_over_512():
    # 15 m camera height, 90 deg FOV, 1024 px wide (512 px half-width)
    # -> the full image covers exactly 30.0 m on the ground plane.
    assert BEV_METERS_PER_PIXEL == pytest.approx(0.029296875)
    assert BEV_METERS_PER_PIXEL * 1024 == pytest.approx(30.0)


def test_ego_to_bev_pixel_maps_forward_up_and_left_left():
    """Source-image convention: +x(forward) -> smaller row, +y(left) -> smaller col."""
    row_front, _ = ego_to_bev_pixel(5.0, 0.0)
    row_rear, _ = ego_to_bev_pixel(-5.0, 0.0)
    _, col_left = ego_to_bev_pixel(0.0, 5.0)
    _, col_right = ego_to_bev_pixel(0.0, -5.0)

    assert row_front < BEV_ORIGIN_PX[1] < row_rear
    assert col_left < BEV_ORIGIN_PX[0] < col_right
    # 5 m forward at 15/512 m/px is 5 / 0.029296875 = 170.67 px from the centre
    assert BEV_ORIGIN_PX[1] - row_front == pytest.approx(5.0 / BEV_METERS_PER_PIXEL)


def test_crop_bev_occupancy_row0_is_frontmost():
    """Row axis: output row 0 must be the front-most cell (image top when saved directly).

    Synthetic 30x20 image at 1 m/px with ego at pixel row 10.5 and the real sign
    convention (SIGN_FORWARD=-1 -> forward is toward *smaller* source rows).
    Source rows <= 10 are road(7, drivable) = in front of the ego,
    source rows >= 11 are sidewalk(8, non-drivable) = behind it.
    """
    semantic_image = np.full((30, 20), 7, dtype=np.uint8)
    semantic_image[11:, :] = 8

    grid_spec = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=1.0)
    occupancy = crop_bev_occupancy(
        semantic_image, grid_spec, meters_per_pixel=1.0, origin_px=(10.5, 10.5)
    )

    assert occupancy.shape == (grid_spec.n_rows, grid_spec.n_cols)
    assert occupancy[0].all(), "row 0 must be the front-most cell (road here)"
    assert not occupancy[-1].any(), "row -1 must be the rear-most cell (sidewalk here)"


def test_crop_bev_occupancy_col0_is_vehicle_left():
    """Column axis: output col 0 must be the vehicle's left (image left when saved directly).

    This is the case the original test never covered — its synthetic rows were uniform
    across all columns, so a left/right mirror in the output was invisible.
    SIGN_LATERAL=-1 -> the vehicle's left (+y) is toward *smaller* source columns, so
    source cols <= 10 (road) are on the vehicle's left and cols >= 11 (sidewalk) on its right.
    """
    semantic_image = np.full((30, 20), 7, dtype=np.uint8)
    semantic_image[:, 11:] = 8

    grid_spec = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=3.0, cell_m=1.0)
    occupancy = crop_bev_occupancy(
        semantic_image, grid_spec, meters_per_pixel=1.0, origin_px=(10.5, 10.5)
    )

    assert occupancy.shape == (grid_spec.n_rows, grid_spec.n_cols)
    half = grid_spec.n_cols // 2
    assert occupancy[:, :half].all(), "left half of the output must be the vehicle's left (road)"
    assert not occupancy[:, half:].any(), "right half must be the vehicle's right (sidewalk)"


def test_crop_bev_occupancy_orientation_is_independent_of_source_sign_convention():
    """Output orientation is defined by ego geometry, not by the source pixel ordering.

    Feeding the opposite source signs must still yield row 0 = front, col 0 = left.
    """
    # forward (+x) toward larger rows, left (+y) toward larger cols
    semantic_image = np.full((30, 30), 8, dtype=np.uint8)
    semantic_image[11:, 11:] = 7  # front-left quadrant is drivable

    grid_spec = OccupancyGridSpec(front_m=3.0, rear_m=3.0, half_width_m=3.0, cell_m=1.0)
    occupancy = crop_bev_occupancy(
        semantic_image, grid_spec, meters_per_pixel=1.0, origin_px=(10.5, 10.5),
        sign_forward=1, sign_lateral=1,
    )

    n_rows, n_cols = occupancy.shape
    assert occupancy[: n_rows // 2, : n_cols // 2].all(), "front-left quadrant -> top-left of array"
    assert not occupancy[n_rows // 2 :, :].any()
    assert not occupancy[:, n_cols // 2 :].any()


@requires_dataset
def test_crop_bev_occupancy_matches_a_plain_slice_of_the_real_source_image():
    """Real data: the crop must be the same sub-window of `_BEV.png`, with no flips.

    Regression guard for the left-right mirror bug: before the fix, the output's
    drivable fraction per half was swapped relative to the source window.
    """
    semantic_bev = np.array(
        Image.open(DATASET_ROOT / "semantic_annotations/gtLabels/00000_BEV.png")
    )
    spec = SYNWOODSCAPE_PRETRAIN_GRID_SPEC
    occupancy = crop_bev_occupancy(semantic_bev, spec)

    row_front, _ = ego_to_bev_pixel(spec.front_m, 0.0)
    row_rear, _ = ego_to_bev_pixel(-spec.rear_m, 0.0)
    _, col_left = ego_to_bev_pixel(0.0, spec.half_width_m)
    _, col_right = ego_to_bev_pixel(0.0, -spec.half_width_m)
    window = semantic_bev[
        int(round(row_front)) : int(round(row_rear)),
        int(round(col_left)) : int(round(col_right)),
    ]
    drivable_window = np.isin(window, (6, 7))

    n_rows, n_cols = occupancy.shape
    w_rows, w_cols = drivable_window.shape
    # Same orientation: front half of the source window is the top half of the output, etc.
    assert occupancy[: n_rows // 2].mean() == pytest.approx(
        drivable_window[: w_rows // 2].mean(), abs=0.02
    )
    assert occupancy[:, : n_cols // 2].mean() == pytest.approx(
        drivable_window[:, : w_cols // 2].mean(), abs=0.02
    )
    # ...and the two halves really do differ, so the assertions above have teeth.
    assert abs(
        drivable_window[:, : w_cols // 2].mean() - drivable_window[:, w_cols // 2 :].mean()
    ) > 0.05


@requires_dataset
@requires_depth
def test_bev_scale_beats_the_biased_vehicle_silhouette_estimate_on_ground_points():
    """Real data: reproject ground-plane pixels via the Phase 1 fisheye pipeline and check
    that `BEV_METERS_PER_PIXEL` (15/512) matches the BEV semantic labels better than the
    old vehicle-silhouette estimate (0.0284), which is biased low by pinhole magnification.
    """
    from projects.geometry.fisheye import load_camera
    from projects.geometry.reprojection import unproject_depth_to_ego

    sample_idx = "00000"
    bev_semantic = np.array(
        Image.open(DATASET_ROOT / f"semantic_annotations/gtLabels/{sample_idx}_BEV.png")
    )

    def match_rate(meters_per_pixel):
        matches = total = 0
        for camera_name in CAMERAS:
            camera = load_camera(DATASET_ROOT / "calibration_data" / f"{camera_name}.json")
            depth = np.load(
                DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{camera_name}.npy"
            )
            semantic = np.array(
                Image.open(
                    DATASET_ROOT / f"semantic_annotations/gtLabels/{sample_idx}_{camera_name}.png"
                )
            )
            rows, cols = np.nonzero(
                np.isin(semantic, GROUND_CLASS_IDS) & (depth > 0) & (depth < 40)
            )
            step = max(1, len(rows) // 4000)  # subsample: keep the test fast
            rows, cols = rows[::step], cols[::step]
            points_ego = unproject_depth_to_ego(camera, rows, cols, depth)
            labels = semantic[rows, cols]

            radius = np.linalg.norm(points_ego[:, :2], axis=1)
            keep = (np.abs(points_ego[:, 2]) < 0.15) & (radius > 2.5) & (radius < 13.0)
            points_ego, labels = points_ego[keep], labels[keep]
            if not len(points_ego):
                continue

            row_px, col_px = ego_to_bev_pixel(
                points_ego[:, 0], points_ego[:, 1], meters_per_pixel=meters_per_pixel
            )
            row_i = np.round(row_px).astype(int)
            col_i = np.round(col_px).astype(int)
            height, width = bev_semantic.shape
            inside = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
            matches += int((bev_semantic[row_i[inside], col_i[inside]] == labels[inside]).sum())
            total += int(inside.sum())
        assert total > 1000, f"too few ground points to judge ({total})"
        return matches / total

    rate_correct = match_rate(BEV_METERS_PER_PIXEL)
    rate_silhouette = match_rate(0.0284)

    assert rate_correct > 0.95, f"15/512 should agree closely, got {rate_correct:.3f}"
    assert rate_correct > rate_silhouette + 0.02, (
        f"15/512 ({rate_correct:.3f}) must clearly beat the biased silhouette estimate "
        f"0.0284 ({rate_silhouette:.3f})"
    )


@requires_dataset
@requires_depth
def test_bev_sign_convention_beats_all_three_alternatives_on_ground_points():
    """Real data: (SIGN_FORWARD, SIGN_LATERAL) = (-1, -1) must beat the other 3 candidates.

    Guards the lateral sign in particular — the axis that carried the mirror bug.
    """
    from projects.geometry.fisheye import load_camera
    from projects.geometry.reprojection import unproject_depth_to_ego

    sample_idx = "00000"
    bev_semantic = np.array(
        Image.open(DATASET_ROOT / f"semantic_annotations/gtLabels/{sample_idx}_BEV.png")
    )

    collected = []
    for camera_name in CAMERAS:
        camera = load_camera(DATASET_ROOT / "calibration_data" / f"{camera_name}.json")
        depth = np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{camera_name}.npy")
        semantic = np.array(
            Image.open(
                DATASET_ROOT / f"semantic_annotations/gtLabels/{sample_idx}_{camera_name}.png"
            )
        )
        rows, cols = np.nonzero(np.isin(semantic, GROUND_CLASS_IDS) & (depth > 0) & (depth < 40))
        step = max(1, len(rows) // 4000)
        rows, cols = rows[::step], cols[::step]
        points_ego = unproject_depth_to_ego(camera, rows, cols, depth)
        radius = np.linalg.norm(points_ego[:, :2], axis=1)
        keep = (np.abs(points_ego[:, 2]) < 0.15) & (radius > 2.5) & (radius < 13.0)
        collected.append((points_ego[keep], semantic[rows, cols][keep]))

    def match_rate(sign_forward, sign_lateral):
        matches = total = 0
        for points_ego, labels in collected:
            if not len(points_ego):
                continue
            row_px, col_px = ego_to_bev_pixel(
                points_ego[:, 0], points_ego[:, 1],
                sign_forward=sign_forward, sign_lateral=sign_lateral,
            )
            row_i = np.round(row_px).astype(int)
            col_i = np.round(col_px).astype(int)
            height, width = bev_semantic.shape
            inside = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
            matches += int((bev_semantic[row_i[inside], col_i[inside]] == labels[inside]).sum())
            total += int(inside.sum())
        return matches / max(total, 1)

    rates = {
        (sf, sl): match_rate(sf, sl) for sf in (1, -1) for sl in (1, -1)
    }
    best = max(rates, key=rates.get)
    assert best == (SIGN_FORWARD, SIGN_LATERAL), f"best signs were {best}, rates={rates}"
    runner_up = max(rate for signs, rate in rates.items() if signs != best)
    assert rates[best] > runner_up + 0.02, f"signs not clearly separated: {rates}"
