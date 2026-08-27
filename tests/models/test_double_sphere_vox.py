from pathlib import Path

import numpy as np
import pytest
import torch

from projects.bev_gt.grid import ROBOT_GRID_SPEC, OccupancyGridSpec, cell_centers_m
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam, LEGACY_HEIGHT_BINS

# 이 파일의 대조는 전부 **옛 `Y=1` 기하**에 대한 것이다 -- numpy 재구현과 값을 맞춰
# 보는 것이라 높이 축과 무관하고, 채택 기본값이 바뀌어도 그대로여야 한다.
LEGACY_HEIGHT_KWARGS = dict(height_bins=LEGACY_HEIGHT_BINS,
                            height_min_m=None, height_max_m=None)
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
    vox_util = build_double_sphere_vox_util(spec, cameras, **LEGACY_HEIGHT_KWARGS)

    B, S, C, H, W = 2, len(cameras), 8, 12, 16
    rgb_packed = torch.rand(B * S, C, H, W)
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.tile(np.linalg.inv(ref_T_cams), (B, 1, 1))).float()

    # 이 테스트는 **옛 Y=1 기하를 고정**한다 -- 채택 기본값이 바뀌어도 이 대조는
    # 그대로여야 한다(numpy 재구현과의 수치 대조라 높이 축과 무관하다).
    Z, Y, X = spec.n_rows, LEGACY_HEIGHT_BINS, spec.n_cols
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
    vox_util = build_double_sphere_vox_util(spec, cameras, **LEGACY_HEIGHT_KWARGS)

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

    multi = build_double_sphere_vox_util(spec, cameras, **LEGACY_HEIGHT_KWARGS)
    batched = multi.unproject_image_to_mem(
        torch.cat([rgb, rgb]), torch.cat([cam_T_ref, cam_T_ref]),
        torch.cat([cam_T_ref, cam_T_ref]), Z, Y, X,
    )
    # 배치를 두 번 반복했으므로 두 절반이 같아야 한다.
    torch.testing.assert_close(batched[: len(names)], batched[len(names):])

    for i, name in enumerate(names):
        single = build_double_sphere_vox_util(spec, [cameras[i]], **LEGACY_HEIGHT_KWARGS)
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


@requires_calib
def test_mirror_x_lifts_a_flipped_feature_map_into_the_mirrored_bev():
    """좌우 반전 증강의 기하 계약 -- **반전 입력이 정확히 반전된 BEV를 만든다.**

    성립하지 않으면 (반전 이미지, 반전 라벨) 쌍이 존재하지 않는 기하를 가르치게 된다.
    모델을 끼우지 않는 이유: conv encoder는 반전에 등변이 아니므로, 섞으면 "기하가 틀렸다"와
    "네트워크가 등변이 아니다"를 구분할 수 없다. 여기서 재는 것은 lifting 기하뿐이다.

    두 단계가 모두 필요하다는 것도 같은 테스트에서 확인한다(특징맵을 뒤집지 않으면 깨진다) --
    한쪽만 해도 통과하면 이 테스트는 아무것도 고정하지 못한다.
    """
    torch.manual_seed(0)
    spec = ROBOT_GRID_SPEC
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]
    plain = build_double_sphere_vox_util(spec, cameras, **LEGACY_HEIGHT_KWARGS)
    mirror = build_double_sphere_vox_util(spec, cameras, mirror_x=True, **LEGACY_HEIGHT_KWARGS)

    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    feat = torch.randn(len(cameras), 4, 36, 64)      # 512x288 입력 / stride 8

    volume = plain.unproject_image_to_mem(feat, camB_T_camA, camB_T_camA, Z, Y, X)
    mirrored = mirror.unproject_image_to_mem(
        torch.flip(feat, dims=[-1]), camB_T_camA, camB_T_camA, Z, Y, X
    )
    expected = torch.flip(volume, dims=[-1])         # memory X = 마지막 축 = 횡방향

    assert volume.abs().sum() > 0                     # 자명하게 0이면 아무것도 증명 못 한다
    # 오차 상한은 특징값 표준편차(약 0.55)보다 네 자릿수 작다. 반픽셀 어긋남이 있었다면
    # 이웃 값 차이(~1.4) 규모의 오차가 났을 것이므로, 이 상한이 곧 "어긋남 없음"이다.
    assert (mirrored - expected).abs().max() < 1e-3

    # 반증: 질의점만 반전하고 특징맵을 그대로 넣으면 데이터 자체 규모의 오차가 나야 한다.
    without_input_flip = mirror.unproject_image_to_mem(
        feat, camB_T_camA, camB_T_camA, Z, Y, X
    )
    assert (without_input_flip - expected).abs().max() > 1.0


@requires_calib
def test_flipping_memory_x_is_exactly_mirroring_ref_x():
    """격자 규약 -- 라벨을 마지막 축으로 뒤집는 것이 ego 횡방향 반전과 같아야 한다.

    ROI가 좌우 대칭(±half_width)이라 성립한다. 전후로는 비대칭(전방 4 m / 후방 2 m)이므로
    같은 논리가 성립하지 않고, 그래서 반전은 횡방향으로만 한다.
    """
    import utils.basic  # simple_bev submodule -- vox util import가 sys.path를 세팅한다

    spec = ROBOT_GRID_SPEC
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]
    vox_util = build_double_sphere_vox_util(spec, cameras, **LEGACY_HEIGHT_KWARGS)
    Z, Y, X = spec.n_rows, 1, spec.n_cols

    xyz_ref = vox_util.Mem2Ref(
        utils.basic.gridcloud3d(1, Z, Y, X, norm=False), Z, Y, X, assert_cube=False
    ).reshape(Z, Y, X, 3)
    x_ref = xyz_ref[..., 0]

    assert (x_ref + torch.flip(x_ref, dims=[-1])).abs().max() < 1e-4
    assert float(x_ref.max()) == pytest.approx(spec.half_width_m - spec.cell_m / 2, abs=1e-4)
