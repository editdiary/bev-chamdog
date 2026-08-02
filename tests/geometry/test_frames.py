import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from projects.geometry.frames import (
    LIDAR_MOUNT_OFFSET_EGO,
    apply_4x4,
    invert_4x4,
    parse_vehicle_location,
    world_points_to_ego,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="SynWoodScape dataset not available locally"
)


def test_apply_4x4_translation_only():
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    result = apply_4x4(transform, points)

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])


def test_invert_4x4_round_trip():
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [4.0, -1.0, 2.5]
    points = rng.normal(size=(5, 3))

    round_tripped = apply_4x4(invert_4x4(transform), apply_4x4(transform, points))

    np.testing.assert_allclose(round_tripped, points, atol=1e-10)


def test_world_points_to_ego_maps_lidar_origin_to_mount_offset():
    world_T_lidar = np.eye(4)
    world_T_lidar[:3, 3] = [10.0, -5.0, 3.0]
    points_world = np.array([[10.0, -5.0, 3.0]])  # the LiDAR sensor's own origin, in world frame

    result = world_points_to_ego(points_world, world_T_lidar)

    np.testing.assert_allclose(result, [[0.0, 0.0, 2.0]], atol=1e-10)


@requires_dataset
def test_lidar_transform_matches_vehicle_pose_for_sample_00000():
    vehicle_location = parse_vehicle_location(
        DATASET_ROOT / "vehicle_data" / "rgb_images" / "00000.txt"
    )
    with open(DATASET_ROOT / "lidar_data" / "00000.pkl", "rb") as f:
        lidar_record = pickle.load(f)
    lidar_translation = np.asarray(lidar_record["transform"])[:3, 3]

    np.testing.assert_allclose(lidar_translation[:2], vehicle_location[:2], atol=0.01)
    z_diff = lidar_translation[2] - vehicle_location[2]
    assert abs(z_diff - LIDAR_MOUNT_OFFSET_EGO[2]) < 0.05
