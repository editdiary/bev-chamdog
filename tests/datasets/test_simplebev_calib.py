from pathlib import Path

import numpy as np
import pytest

from projects.datasets.simplebev_calib import (
    ego_T_cam_from_camera,
    pinhole_pix_T_cam_from_camera,
)
from projects.geometry.fisheye import load_camera

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
CALIB_DIR = DATASET_ROOT / "calibration_data"
requires_dataset = pytest.mark.skipif(
    not CALIB_DIR.exists(), reason="SynWoodScape dataset not available locally"
)


@requires_dataset
def test_ego_T_cam_matches_camera_rotation_and_translation():
    camera = load_camera(CALIB_DIR / "FV.json")
    ego_T_cam = ego_T_cam_from_camera(camera)

    np.testing.assert_allclose(ego_T_cam[:3, :3], camera.rotation)
    np.testing.assert_allclose(ego_T_cam[:3, 3], camera.translation)
    np.testing.assert_allclose(ego_T_cam[3, :], [0, 0, 0, 1])


@requires_dataset
def test_ego_T_cam_rotation_is_orthonormal():
    camera = load_camera(CALIB_DIR / "FV.json")
    R = ego_T_cam_from_camera(camera)[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


@requires_dataset
def test_pinhole_pix_T_cam_at_native_resolution_uses_k1_as_focal_length():
    camera = load_camera(CALIB_DIR / "FV.json")
    pix_T_cam = pinhole_pix_T_cam_from_camera(camera, camera.width, camera.height)

    k1 = camera.lens.coefficients[0]
    assert pix_T_cam[0, 0] == pytest.approx(k1)
    assert pix_T_cam[1, 1] == pytest.approx(k1 * camera.aspect_ratio)
    assert pix_T_cam[0, 2] == pytest.approx(camera.cx)
    assert pix_T_cam[1, 2] == pytest.approx(camera.cy)
    assert pix_T_cam[2, 2] == 1.0
    assert pix_T_cam[3, 3] == 1.0


@requires_dataset
def test_pinhole_pix_T_cam_scales_with_output_resolution():
    camera = load_camera(CALIB_DIR / "FV.json")
    native = pinhole_pix_T_cam_from_camera(camera, camera.width, camera.height)
    half = pinhole_pix_T_cam_from_camera(camera, camera.width // 2, camera.height // 2)

    np.testing.assert_allclose(half[0, 0], native[0, 0] / 2, rtol=1e-6)
    np.testing.assert_allclose(half[1, 1], native[1, 1] / 2, rtol=1e-6)
    np.testing.assert_allclose(half[0, 2], native[0, 2] / 2, rtol=1e-6)
    np.testing.assert_allclose(half[1, 2], native[1, 2] / 2, rtol=1e-6)
