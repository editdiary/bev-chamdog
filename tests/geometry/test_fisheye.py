from pathlib import Path

import numpy as np
import pytest

from projects.geometry.fisheye import load_camera

CALIB_DIR = Path("dataset/synwoodscape/SynWoodScape_V0.1.0/calibration_data")
requires_dataset = pytest.mark.skipif(
    not CALIB_DIR.exists(), reason="SynWoodScape dataset not available locally"
)


@requires_dataset
def test_load_camera_parses_fv_json():
    cam = load_camera(CALIB_DIR / "FV.json")

    assert cam.width == 1280
    assert cam.height == 966
    assert abs(cam.aspect_ratio - 1.00021) < 1e-6
    np.testing.assert_allclose(cam.translation, [1.92, 0.0, 0.9])


@requires_dataset
def test_project_point_on_optical_axis_lands_on_principal_point():
    cam = load_camera(CALIB_DIR / "FV.json")
    point_in_camera_frame = np.array([0.0, 0.0, 5.0])  # 5m straight ahead of the lens
    point_in_ego_frame = cam.translation + cam.rotation @ point_in_camera_frame

    pixel = cam.project_3d_to_2d(point_in_ego_frame[np.newaxis, :])

    np.testing.assert_allclose(pixel[0], [cam.cx, cam.cy], atol=1e-6)


@requires_dataset
def test_project_then_unproject_round_trip():
    cam = load_camera(CALIB_DIR / "FV.json")
    rng = np.random.default_rng(1)
    offsets_in_camera_frame = rng.uniform(-1.0, 1.0, size=(20, 2))
    depths = rng.uniform(5.0, 15.0, size=20)
    points_in_camera_frame = np.column_stack([offsets_in_camera_frame, depths])
    points_in_ego_frame = (cam.rotation @ points_in_camera_frame.T).T + cam.translation

    pixels = cam.project_3d_to_2d(points_in_ego_frame)
    norms = np.linalg.norm(points_in_ego_frame - cam.translation, axis=1)
    recovered = cam.project_2d_to_3d(pixels, norms)

    np.testing.assert_allclose(recovered, points_in_ego_frame, atol=1e-4)
