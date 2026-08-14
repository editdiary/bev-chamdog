from pathlib import Path

import numpy as np
import pytest
import torch

from projects.bev_gt.grid import ROBOT_GRID_SPEC, OccupancyGridSpec, cell_centers_m
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.double_sphere import (
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
    se3_inverse,
)
from projects.models.double_sphere_vox import (
    build_double_sphere_vox_util,
    double_sphere_pixel_coords,
)

CALIB_PATH = Path("dataset/sj_datasets/common/calibration/calib.yaml")
requires_calib = pytest.mark.skipif(
    not CALIB_PATH.exists(), reason="self-collected calibration not available locally"
)


def _torch_project(camera, points):
    u, v, valid = double_sphere_pixel_coords(
        torch.tensor(points[:, 0]), torch.tensor(points[:, 1]), torch.tensor(points[:, 2]),
        torch.tensor(camera.xi, dtype=torch.float64),
        torch.tensor(camera.alpha, dtype=torch.float64),
        torch.tensor(camera.fx, dtype=torch.float64),
        torch.tensor(camera.fy, dtype=torch.float64),
        torch.tensor(camera.cx, dtype=torch.float64),
        torch.tensor(camera.cy, dtype=torch.float64),
        torch.tensor(camera.domain_cos_limit, dtype=torch.float64),
    )
    return u.numpy(), v.numpy(), valid.numpy()


@requires_calib
def test_torch_pixel_coords_match_numpy_camera():
    """torch 재구현이 numpy `DoubleSphereCamera.project`(참조 구현과 대조 완료)와 일치하는지."""
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(3000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = direction * rng.uniform(0.3, 20.0, (3000, 1))

    for camera in load_cameras(CALIB_PATH).values():
        expected_u, expected_v, expected_valid = camera.project(points)
        u, v, valid = _torch_project(camera, points)

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_allclose(u[valid], expected_u[valid], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(v[valid], expected_v[valid], rtol=1e-12, atol=1e-9)


@requires_calib
def test_unproject_runs_end_to_end_without_nan():
    spec = OccupancyGridSpec(front_m=2.0, rear_m=1.0, half_width_m=1.5, cell_m=0.5)
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]
    vox_util = build_double_sphere_vox_util(spec, cameras)

    B, S, C, H, W = 2, len(cameras), 8, 12, 16
    rgb_packed = torch.rand(B * S, C, H, W)
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.tile(np.linalg.inv(ref_T_cams), (B, 1, 1))).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    values = vox_util.unproject_image_to_mem(rgb_packed, camB_T_camA, camB_T_camA, Z, Y, X)

    assert values.shape == (B * S, C, Z, Y, X)
    assert not torch.isnan(values).any()
    assert values.abs().sum() > 0


@requires_calib
def test_grid_coverage_matches_independent_numpy_projection():
    """voxel -> 픽셀 배관 전체(Mem2Ref, extrinsic 체인, 해상도 스케일)를 numpy로 독립 검증.

    카메라 순서·좌표 규약이 어긋나면 여기서 잡힌다 -- 학습 중에는 loss가 조용히 나빠질 뿐
    드러나지 않는 종류의 버그다.
    """
    spec = ROBOT_GRID_SPEC
    names = FINETUNE_CAMERA_NAMES
    cameras = [load_cameras(CALIB_PATH)[name] for name in names]
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    vox_util = build_double_sphere_vox_util(spec, cameras)

    H, W = 720, 1280  # native 해상도 -> 스케일 인자가 1이라 경계 판정이 정확히 대응한다
    Z, Y, X = spec.n_rows, 1, spec.n_cols
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in names])
    camB_T_camA = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    ones = torch.ones(len(names), 1, H, W)
    values = vox_util.unproject_image_to_mem(ones, camB_T_camA, camB_T_camA, Z, Y, X)
    covered = (values[:, 0, :, 0, :] > 0.5).numpy()  # (S, rows, cols)

    forward_m, lateral_m = cell_centers_m(spec)
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    points_ego = np.stack([fg.ravel(), lg.ravel(), np.zeros(fg.size), np.ones(fg.size)], axis=1)

    for i, name in enumerate(names):
        points_cam = (se3_inverse(ego_T_cams[name]) @ points_ego.T).T[:, :3]
        u, v, valid = cameras[i].project(points_cam)
        expected = (valid & (u > -0.5) & (u < W - 0.5) & (v > -0.5) & (v < H - 0.5))
        expected = expected.reshape(spec.n_rows, spec.n_cols)

        agreement = (covered[i] == expected).mean()
        assert agreement > 0.99, f"{name}: {agreement:.4f} 일치"
        assert expected.any(), f"{name}: 그리드를 전혀 못 본다 -- extrinsic 체인 의심"


@requires_calib
def test_camera_order_is_respected_across_the_packed_batch():
    """`pack_seqdim` 순서(b 바깥, S 안쪽)와 캘리브레이션 repeat이 맞물리는지."""
    spec = OccupancyGridSpec(front_m=2.0, rear_m=1.0, half_width_m=1.5, cell_m=0.25)
    names = FINETUNE_CAMERA_NAMES
    cameras = [load_cameras(CALIB_PATH)[name] for name in names]
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in names])
    cam_T_ref = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    H, W = 72, 128
    rgb = torch.rand(len(names), 3, H, W)

    multi = build_double_sphere_vox_util(spec, cameras)
    batched = multi.unproject_image_to_mem(
        torch.cat([rgb, rgb]), torch.cat([cam_T_ref, cam_T_ref]),
        torch.cat([cam_T_ref, cam_T_ref]), Z, Y, X,
    )
    # 배치를 두 번 반복했으므로 두 절반이 같아야 한다.
    torch.testing.assert_close(batched[: len(names)], batched[len(names):])

    for i, name in enumerate(names):
        single = build_double_sphere_vox_util(spec, [cameras[i]])
        expected = single.unproject_image_to_mem(
            rgb[i: i + 1], cam_T_ref[i: i + 1], cam_T_ref[i: i + 1], Z, Y, X
        )
        torch.testing.assert_close(batched[i: i + 1], expected)


@requires_calib
def test_rear_of_the_grid_is_covered_by_the_side_cameras():
    """3-cam(front/left/right)만으로 후방 그리드가 덮이는지 -- 화각이 약 175°라 덮여야 한다.

    `z_cam > 0`으로 유효성을 자르는 회귀가 생기면 후방 커버리지가 0이 되어 여기서 걸린다.
    """
    spec = ROBOT_GRID_SPEC
    names = FINETUNE_CAMERA_NAMES
    cameras = [load_cameras(CALIB_PATH)[name] for name in names]
    ego_T_cams = load_ego_T_cams(CALIB_PATH)

    forward_m, lateral_m = cell_centers_m(spec)
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    points_ego = np.stack([fg.ravel(), lg.ravel(), np.zeros(fg.size), np.ones(fg.size)], axis=1)

    seen = np.zeros(fg.size, dtype=bool)
    for camera, name in zip(cameras, names):
        points_cam = (se3_inverse(ego_T_cams[name]) @ points_ego.T).T[:, :3]
        u, v, valid = camera.project(points_cam)
        seen |= valid & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

    seen = seen.reshape(spec.n_rows, spec.n_cols)
    rear_rows = forward_m < 0.0
    assert seen[rear_rows].mean() > 0.9, f"후방 커버리지 {seen[rear_rows].mean():.3f}"
