import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from projects.geometry.double_sphere import (
    FINETUNE_CAMERA_NAMES,
    GROUND_Z_IN_LIDAR_M,
    DoubleSphereCamera,
    load_camera_index_names,
    load_cameras,
    load_ego_T_cams,
    se3_inverse,
)

COMMON_DIR = Path("dataset/sj_datasets/common")
CALIB_PATH = COMMON_DIR / "calibration/calib.yaml"
ORIENTATION_PATH = COMMON_DIR / "calibration/orientation.json"
REFERENCE_DS_MODEL = COMMON_DIR / "calibration/ds_model.py"

requires_calib = pytest.mark.skipif(
    not CALIB_PATH.exists(), reason="self-collected calibration not available locally"
)
requires_reference = pytest.mark.skipif(
    not REFERENCE_DS_MODEL.exists(), reason="reference ds_model.py not available locally"
)


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("_reference_ds_model", REFERENCE_DS_MODEL)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass`가 cls.__module__을 sys.modules에서 되찾으므로 exec 전에 등록해야 한다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_points(n=4000, seed=0):
    """리그의 실제 화각(약 175°)을 덮도록 광축 뒤쪽(z<0) 점도 포함해서 뽑는다."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * rng.uniform(0.3, 20.0, (n, 1))


@requires_calib
@requires_reference
def test_project_matches_reference_implementation():
    """우리 DS 구현이 라벨을 만든 원본 `ds_model.py`와 일치하는지 — 라벨과 같은 기하여야 한다."""
    reference = _load_reference_module()
    cameras = load_cameras(CALIB_PATH)
    points = _sample_points()

    for name, camera in cameras.items():
        ref_camera = reference.DoubleSphereCamera(
            camera.xi, camera.alpha, camera.fx, camera.fy,
            camera.cx, camera.cy, camera.width, camera.height, name,
        )
        expected_u, expected_v, expected_valid = ref_camera.project(points)
        u, v, valid = camera.project(points)

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_allclose(u[valid], expected_u[valid], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(v[valid], expected_v[valid], rtol=1e-12, atol=1e-9)


@requires_calib
@requires_reference
def test_unproject_matches_reference_implementation():
    reference = _load_reference_module()
    camera = load_cameras(CALIB_PATH)["front"]
    ref_camera = reference.DoubleSphereCamera(
        camera.xi, camera.alpha, camera.fx, camera.fy,
        camera.cx, camera.cy, camera.width, camera.height, "front",
    )
    u, v = np.meshgrid(np.linspace(0, camera.width - 1, 60), np.linspace(0, camera.height - 1, 40))

    dirs, valid = camera.unproject(u.ravel(), v.ravel())
    expected_dirs, expected_valid = ref_camera.unproject(u.ravel(), v.ravel())

    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_allclose(dirs[valid], expected_dirs[valid], rtol=1e-12, atol=1e-9)


@requires_calib
def test_project_unproject_round_trip():
    camera = load_cameras(CALIB_PATH)["front"]
    points = _sample_points(n=2000, seed=1)
    u, v, valid = camera.project(points)
    inside = valid & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

    dirs, ok = camera.unproject(u[inside], v[inside])
    expected = points[inside] / np.linalg.norm(points[inside], axis=1, keepdims=True)

    assert ok.all()
    np.testing.assert_allclose(dirs, expected, atol=1e-6)


@requires_calib
def test_ds_model_covers_beyond_180_degrees():
    """DS는 화각이 180°를 넘으므로 광축 뒤쪽(z<0) 점도 유효해야 한다.

    `unproject_image_to_mem`에서 radial_poly 판처럼 `z > 0`으로 자르면 화각 주변부가
    통째로 날아간다 — 이 리그는 카메라 3대의 주변부로 후방을 덮으므로 치명적이다.
    """
    camera = load_cameras(CALIB_PATH)["front"]
    behind = np.array([[3.0, 0.0, -0.2], [-3.0, 0.0, -0.2]])
    u, v, valid = camera.project(behind)

    assert valid.all()
    assert ((u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)).any()


@requires_calib
def test_ego_T_cams_place_cameras_at_measured_height_and_bearings():
    """extrinsic 체인이 실측 리그와 맞는지 — 높이 0.87 m, yaw 0 / +90 / -90."""
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    assert set(FINETUNE_CAMERA_NAMES) <= set(ego_T_cams)

    expected_yaw_deg = {"front": 0.0, "left": 90.0, "right": -90.0, "rear": 180.0}
    for name, ego_T_cam in ego_T_cams.items():
        position = ego_T_cam[:3, 3]
        optical_axis = ego_T_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])

        # 렌즈 중앙 실측 높이 0.87 m(지면 기준). LiDAR가 카메라보다 약간 위에 있다.
        assert 0.80 < position[2] < 0.90, f"{name}: z={position[2]}"
        assert np.linalg.norm(position[:2]) < 0.10, f"{name}: 리그는 LiDAR 기준 10 cm 안"

        yaw = np.degrees(np.arctan2(optical_axis[1], optical_axis[0]))
        delta = (yaw - expected_yaw_deg[name] + 180.0) % 360.0 - 180.0
        assert abs(delta) < 5.0, f"{name}: yaw={yaw}"
        # 모든 카메라가 수평을 본다(사각지대가 커진 원인).
        assert abs(np.degrees(np.arcsin(-optical_axis[2]))) < 5.0, f"{name}: pitch"


@requires_calib
def test_ground_plane_lies_at_z_zero_in_ego_frame():
    """ego 원점을 지면으로 내린 게 맞는지 — LiDAR 프레임 지면 z와 정확히 상쇄돼야 한다."""
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    raw_T_cams = load_ego_T_cams(CALIB_PATH, ground_z_in_lidar_m=0.0)

    for name, ego_T_cam in ego_T_cams.items():
        shift = ego_T_cam[:3, 3] - raw_T_cams[name][:3, 3]
        np.testing.assert_allclose(shift, [0.0, 0.0, -GROUND_Z_IN_LIDAR_M], atol=1e-12)
        np.testing.assert_allclose(ego_T_cam[:3, :3], raw_T_cams[name][:3, :3], atol=1e-12)


@requires_calib
def test_ground_point_ahead_projects_below_image_centre():
    """지면 위 점은 수평을 보는 카메라에서 principal point보다 아래(v가 큼)에 찍혀야 한다."""
    camera = load_cameras(CALIB_PATH)["front"]
    ego_T_cam = load_ego_T_cams(CALIB_PATH)["front"]
    cam_T_ego = se3_inverse(ego_T_cam)

    ground_ahead = np.array([[2.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    points_cam = (cam_T_ego @ ground_ahead.T).T[:, :3]
    u, v, valid = camera.project(points_cam)

    assert valid.all()
    assert (v > camera.cy).all()
    # 더 가까운 점일수록 더 아래에 찍힌다.
    assert v[1] > v[0]


@requires_calib
def test_load_camera_index_names():
    assert load_camera_index_names(ORIENTATION_PATH) == {0: "front", 1: "right", 2: "rear", 3: "left"}


def test_domain_cos_limit_handles_both_alpha_branches():
    low = DoubleSphereCamera(xi=0.1, alpha=0.3, fx=1, fy=1, cx=0, cy=0, width=2, height=2)
    high = DoubleSphereCamera(xi=0.1, alpha=0.7, fx=1, fy=1, cx=0, cy=0, width=2, height=2)
    assert low.domain_cos_limit > 0
    assert high.domain_cos_limit > 0
