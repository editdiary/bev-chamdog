from pathlib import Path

import numpy as np
import pytest

from projects.bev_gt.bev_calibration import (
    box_3d_footprint_meters,
    ego_vehicle_bbox_center,
    estimate_scale_m_per_px,
    instance_pixel_bbox,
    load_box_3d_annotations,
    load_instance_bev_image,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="SynWoodScape dataset not available locally"
)


def test_instance_pixel_bbox_on_synthetic_image():
    instance_image = np.zeros((10, 10), dtype=np.uint8)
    instance_image[2:5, 3:6] = 7  # rows 2..4, cols 3..5

    bbox = instance_pixel_bbox(instance_image, 7)

    assert bbox == (3, 5, 2, 4)


def test_instance_pixel_bbox_returns_none_when_absent():
    instance_image = np.zeros((10, 10), dtype=np.uint8)

    assert instance_pixel_bbox(instance_image, 7) is None


def test_box_3d_footprint_meters_on_synthetic_corners():
    corners = np.array(
        [[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0], [2.0, 3.0, 1.5, 1.0]]
    )

    width_m, length_m = box_3d_footprint_meters(corners)

    assert width_m == pytest.approx(2.0)
    assert length_m == pytest.approx(3.0)


@requires_dataset
def test_estimate_scale_for_ego_vehicle_instance_in_sample_00000():
    instance_image = load_instance_bev_image(
        DATASET_ROOT / "_unuserd/instance_annotations/gtLabels/00000_BEV.png"
    )
    box_3d_by_instance = load_box_3d_annotations(
        DATASET_ROOT / "_unuserd/box_3d_annotations/00000.pkl"
    )

    scale_x, scale_y = estimate_scale_m_per_px(instance_image, box_3d_by_instance, instance_id=24)

    assert scale_x == pytest.approx(0.0288, abs=0.001)
    assert scale_y == pytest.approx(0.0285, abs=0.001)


@requires_dataset
def test_ego_vehicle_bbox_center_is_image_center_for_sample_00000():
    semantic_bev = load_instance_bev_image(
        DATASET_ROOT / "semantic_annotations/gtLabels/00000_BEV.png"
    )

    center = ego_vehicle_bbox_center(semantic_bev)

    assert center == pytest.approx((511.5, 511.5), abs=0.5)
