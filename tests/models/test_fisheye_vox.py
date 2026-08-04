from pathlib import Path

import numpy as np
import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.geometry.fisheye import load_camera
from projects.models.fisheye_vox import build_fisheye_vox_util, radial_poly_pixel_coords

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
CALIB_DIR = DATASET_ROOT / "calibration_data"
requires_dataset = pytest.mark.skipif(
    not CALIB_DIR.exists(), reason="SynWoodScape dataset not available locally"
)


@requires_dataset
def test_radial_poly_pixel_coords_matches_woodscape_reference():
    """torch 재구현이 이미 검증된 WoodScape numpy `RadialPolyCamProjection`과 일치하는지."""
    camera = load_camera(CALIB_DIR / "FV.json")

    rng = np.random.default_rng(0)
    n = 2000
    # 카메라 앞쪽(z>0), 어안 FOV 안쪽으로 그럴듯한 범위의 점들.
    x = rng.uniform(-8, 8, n)
    y = rng.uniform(-8, 8, n)
    z = rng.uniform(0.5, 20, n)
    points_cam = np.stack([x, y, z], axis=1)

    lens_points = camera.lens.project_3d_to_2d(points_cam)
    expected = lens_points * np.array([1.0, camera.aspect_ratio]) + np.array([camera.cx, camera.cy])

    k = torch.tensor(camera.lens.coefficients[:4], dtype=torch.float64)
    u, v = radial_poly_pixel_coords(
        torch.tensor(x), torch.tensor(y), torch.tensor(z),
        k, float(camera.cx), float(camera.cy), float(camera.aspect_ratio),
    )

    np.testing.assert_allclose(u.numpy(), expected[:, 0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(v.numpy(), expected[:, 1], atol=1e-6, rtol=1e-6)


@requires_dataset
def test_fisheye_coverage_beats_pinhole_approximation_on_the_bev_grid():
    """실제 어안 투영은 핀홀 근사보다 그리드를 더 넓게 봐야 한다(핀홀은 화각을 과소평가).

    Phase 3.1의 핀홀 근사는 화각 중심부만 맞고, 주변부는 화각이 유한한 픽셀로 안 접혀서
    실제보다 좁게(적은 grid 커버리지) 나온다 — 그 반대(어안이 더 좁게 나옴)면 구현에 부호나
    스케일 버그가 있다는 뜻이다.
    """
    from projects.datasets.simplebev_calib import ego_T_cam_from_camera, pinhole_pix_T_cam_from_camera
    from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam

    spec = OccupancyGridSpec(front_m=5.0, rear_m=3.0, half_width_m=4.0, cell_m=0.1)  # 가벼운 grid
    camera = load_camera(CALIB_DIR / "FV.json")
    ego_T_cam = ego_T_cam_from_camera(camera)
    cam_T_ego = np.linalg.inv(ego_T_cam)

    forward_m = spec.front_m - (np.arange(spec.n_rows) + 0.5) * spec.cell_m
    lateral_m = spec.half_width_m - (np.arange(spec.n_cols) + 0.5) * spec.cell_m
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    pts_ego = np.stack([fg.ravel(), lg.ravel(), np.zeros(fg.size)], axis=1)
    pts_h = np.concatenate([pts_ego, np.ones((pts_ego.shape[0], 1))], axis=1)
    pts_cam = (cam_T_ego @ pts_h.T).T[:, :3]

    # 핀홀 근사 커버리지 (네이티브 해상도 기준)
    pix_T_cam = pinhole_pix_T_cam_from_camera(camera, camera.width, camera.height)
    pix = (pix_T_cam[:3, :3] @ pts_cam.T).T
    uv_pinhole = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    infront = pts_cam[:, 2] > 0
    pinhole_valid = (
        infront
        & (uv_pinhole[:, 0] >= 0) & (uv_pinhole[:, 0] < camera.width)
        & (uv_pinhole[:, 1] >= 0) & (uv_pinhole[:, 1] < camera.height)
    )

    # 어안 커버리지 (torch 함수, 네이티브 해상도 기준)
    k = torch.tensor(camera.lens.coefficients[:4], dtype=torch.float64)
    u, v = radial_poly_pixel_coords(
        torch.tensor(pts_cam[:, 0]), torch.tensor(pts_cam[:, 1]), torch.tensor(pts_cam[:, 2]),
        k, float(camera.cx), float(camera.cy), float(camera.aspect_ratio),
    )
    fisheye_valid = (
        infront
        & (u.numpy() >= 0) & (u.numpy() < camera.width)
        & (v.numpy() >= 0) & (v.numpy() < camera.height)
    )

    assert fisheye_valid.mean() > pinhole_valid.mean()


@requires_dataset
def test_fisheye_vox_util_unproject_runs_end_to_end_without_nan():
    from projects.datasets.simplebev_calib import ego_T_cam_from_camera
    from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam

    spec = OccupancyGridSpec(front_m=2.0, rear_m=1.0, half_width_m=1.5, cell_m=0.5)
    cameras = [load_camera(CALIB_DIR / f"{name}.json") for name in ("FV", "MVL", "MVR", "RV")]
    vox_util = build_fisheye_vox_util(spec, cameras)

    B, S, C, H, W = 2, len(cameras), 8, 12, 16
    rgb_camXs_packed = torch.rand(B * S, C, H, W)
    cam0_T_camXs = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cam_from_camera(cam)) for cam in cameras])
    camXs_T_cam0 = np.linalg.inv(cam0_T_camXs)
    camB_T_camA = torch.from_numpy(np.tile(camXs_T_cam0, (B, 1, 1))).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    values = vox_util.unproject_image_to_mem(rgb_camXs_packed, camB_T_camA, camB_T_camA, Z, Y, X)

    assert values.shape == (B * S, C, Z, Y, X)
    assert not torch.isnan(values).any()
    assert values.abs().sum() > 0  # 최소 일부 voxel은 실제로 카메라에 투영돼야 한다
