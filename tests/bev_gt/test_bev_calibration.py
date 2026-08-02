from pathlib import Path

import numpy as np
import pytest

from projects.bev_gt.bev_calibration import (
    box_3d_footprint_meters,
    box_corners_to_ego,
    ego_vehicle_bbox_center,
    instance_pixel_bbox,
    load_box_3d_annotations,
    load_instance_bev_image,
    silhouette_scale_m_per_px,
)
from projects.bev_gt.bev_crop import BEV_METERS_PER_PIXEL
from projects.geometry.frames import parse_vehicle_transform

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


def _ego_silhouette_scale(sample_idx):
    instance_image = load_instance_bev_image(
        DATASET_ROOT / f"_unuserd/instance_annotations/gtLabels/{sample_idx}_BEV.png"
    )
    box_3d_by_instance = load_box_3d_annotations(
        DATASET_ROOT / f"_unuserd/box_3d_annotations/{sample_idx}.pkl"
    )
    world_T_ego = parse_vehicle_transform(
        DATASET_ROOT / f"vehicle_data/rgb_images/{sample_idx}.txt"
    )
    corners_ego = box_corners_to_ego(box_3d_by_instance, 24, world_T_ego)
    return silhouette_scale_m_per_px(instance_image, corners_ego, instance_id=24)


@requires_dataset
def test_ego_frame_box_extent_is_yaw_free_across_samples():
    """world 프레임 코너를 ego로 되돌리면 ego 박스 치수가 샘플(=world yaw)과 무관해진다."""
    ego_extents, world_extents = [], []
    for sample_idx in ("00000", "00100", "00300"):  # world yaw가 서로 크게 다른 샘플들
        box_3d_by_instance = load_box_3d_annotations(
            DATASET_ROOT / f"_unuserd/box_3d_annotations/{sample_idx}.pkl"
        )
        world_T_ego = parse_vehicle_transform(
            DATASET_ROOT / f"vehicle_data/rgb_images/{sample_idx}.txt"
        )
        corners_ego = box_corners_to_ego(box_3d_by_instance, 24, world_T_ego)
        ego_extents.append(box_3d_footprint_meters(corners_ego))
        world_extents.append(box_3d_footprint_meters(np.asarray(box_3d_by_instance[24])))

    for forward_m, lateral_m in ego_extents:
        assert forward_m == pytest.approx(3.705, abs=0.01)  # ego x = 전후 길이
        assert lateral_m == pytest.approx(1.789, abs=0.01)  # ego y = 좌우 폭

    # 대조: 변환하지 않은 world 코너의 extent는 샘플(=yaw)마다 크게 달라진다.
    world_forward = [extent[0] for extent in world_extents]
    assert max(world_forward) - min(world_forward) > 1.0


@requires_dataset
def test_silhouette_scale_is_biased_low_relative_to_the_true_ground_scale():
    """실루엣 추정치는 항상 참값(15/512)보다 작다 — 핀홀 높이 확대 때문.

    이 테스트는 "실루엣 값으로 BEV_METERS_PER_PIXEL을 되돌리면 안 된다"는 사실 자체를
    회귀로 고정한다.
    """
    # ego 박스 z 최상단 1.556 m → 확대율 15/(15-1.556) = 1.1157
    lower_bound = BEV_METERS_PER_PIXEL / (15.0 / (15.0 - 1.556))

    for sample_idx in ("00000", "00100", "00300"):
        lateral, forward = _ego_silhouette_scale(sample_idx)
        assert lateral == pytest.approx(0.02710, abs=0.0006)
        assert forward == pytest.approx(0.02807, abs=0.0006)
        for estimate in (lateral, forward):
            assert lower_bound < estimate < BEV_METERS_PER_PIXEL, (
                f"{sample_idx}: silhouette estimate {estimate:.5f} should sit strictly between "
                f"the roof-magnified scale {lower_bound:.5f} and the ground scale "
                f"{BEV_METERS_PER_PIXEL:.5f}"
            )


@requires_dataset
def test_ego_vehicle_bbox_center_is_image_center_for_sample_00000():
    semantic_bev = load_instance_bev_image(
        DATASET_ROOT / "semantic_annotations/gtLabels/00000_BEV.png"
    )

    center = ego_vehicle_bbox_center(semantic_bev)

    assert center == pytest.approx((511.5, 511.5), abs=0.5)
