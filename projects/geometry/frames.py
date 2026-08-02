import re
from pathlib import Path

import numpy as np


def apply_4x4(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = points_h @ np.asarray(transform, dtype=np.float64).T
    return transformed[:, :3]


def invert_4x4(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid-body homogeneous transform."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


LIDAR_MOUNT_OFFSET_EGO = np.array([0.0, 0.0, 2.0])  # readme.txt: LiDAR is mounted at ego (x=0, y=0, z=2.0)


def world_points_to_ego(points_world: np.ndarray, world_T_lidar: np.ndarray) -> np.ndarray:
    """Convert LiDAR points (CARLA world frame) into the ego-vehicle frame.

    `world_T_lidar` is the `transform` field stored in each lidar_data/*.pkl
    (confirmed to be the LiDAR-sensor-to-world pose by cross-referencing
    vehicle_data — see docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md §3.3).
    """
    points_lidar = apply_4x4(invert_4x4(world_T_lidar), points_world)
    return points_lidar + LIDAR_MOUNT_OFFSET_EGO


def parse_vehicle_location(vehicle_data_txt_path) -> np.ndarray:
    """Parse the ego world Location(x=..., y=..., z=...) out of a vehicle_data/rgb_images/*.txt file."""
    text = Path(vehicle_data_txt_path).read_text()
    match = re.search(r"Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\)", text)
    return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
