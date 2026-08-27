import numpy as np
import pytest
import torch

from projects.bev_gt.grid import ROBOT_GRID_SPEC, OccupancyGridSpec, cell_centers_m
from projects.datasets.simplebev_vox import (
    EGO_TO_REF_ROTATION,
    build_vox_util,
    ref_T_cam_from_ego_T_cam,
    vox_dims,
)

SPEC = OccupancyGridSpec(front_m=2.0, rear_m=1.0, half_width_m=1.5, cell_m=0.5)


def test_ego_to_ref_rotation_is_a_proper_rotation():
    R = EGO_TO_REF_ROTATION
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_grid_cell_centers_map_to_expected_mem_indices():
    """`cell_centers_m`의 (row 0=최전방, col 0=좌측) 관례가 Vox_util mem index와 그대로 맞는지."""
    Z, Y, X = vox_dims(SPEC)
    vox_util = build_vox_util(SPEC)

    forward_m, lateral_m = cell_centers_m(SPEC)
    forward_grid, lateral_grid = np.meshgrid(forward_m, lateral_m, indexing="ij")
    height = np.zeros_like(forward_grid)

    xyz_ego = np.stack([forward_grid.ravel(), lateral_grid.ravel(), height.ravel()], axis=1)
    xyz_ref = xyz_ego @ EGO_TO_REF_ROTATION.T  # row-vector form of ref = R @ ego

    xyz_ref_t = torch.from_numpy(xyz_ref).float().unsqueeze(0)
    xyz_mem = vox_util.Ref2Mem(xyz_ref_t, Z, Y, X).squeeze(0).numpy()

    mem_x = xyz_mem[:, 0].reshape(SPEC.n_rows, SPEC.n_cols)
    mem_z = xyz_mem[:, 2].reshape(SPEC.n_rows, SPEC.n_cols)

    np.testing.assert_allclose(mem_z[0, :], 0.0, atol=1e-4)  # row 0(최전방) -> Z-mem 첫 voxel
    np.testing.assert_allclose(mem_z[-1, :], SPEC.n_rows - 1.0, atol=1e-4)  # 최후방 -> 마지막
    np.testing.assert_allclose(mem_x[:, 0], 0.0, atol=1e-4)  # col 0(좌측) -> X-mem 첫 voxel
    np.testing.assert_allclose(mem_x[:, -1], SPEC.n_cols - 1.0, atol=1e-4)  # 우측 -> 마지막


def test_ref_T_cam_applies_rotation_and_preserves_translation_norm():
    ego_T_cam = np.eye(4)
    ego_T_cam[:3, 3] = [1.92, 0.0, 0.9]  # FV 카메라 근사 위치(ego 프레임)
    ref_T_cam = ref_T_cam_from_ego_T_cam(ego_T_cam)

    np.testing.assert_allclose(ref_T_cam[:3, :3], EGO_TO_REF_ROTATION)
    np.testing.assert_allclose(ref_T_cam[:3, 3], EGO_TO_REF_ROTATION @ [1.92, 0.0, 0.9])
    assert np.linalg.norm(ref_T_cam[:3, 3]) == pytest.approx(np.linalg.norm([1.92, 0.0, 0.9]))


# --- 높이 bin (Y>1) -----------------------------------------------------------------
# `docs/finetuning_guide.md` §9가 한계로 적어 둔 "Y=1이라 ego z=0 한 평면에서만 표본한다"를
# 여는 손잡이다. 라벨 파이프라인(`slab_label.py`)의 occupancy는 지상 0.87~1.67 m 슬래브의
# 기둥 질의이므로 표본 높이와 어긋나 있다.

def test_default_height_config_is_unchanged():
    """**기본값이 여태까지의 전 실험 설정과 같아야 한다** -- 이게 깨지면 옛 숫자가 전부 무효다."""
    from projects.datasets.simplebev_vox import vox_bounds, vox_dims
    assert vox_dims(ROBOT_GRID_SPEC) == (ROBOT_GRID_SPEC.n_rows, 1, ROBOT_GRID_SPEC.n_cols)
    assert vox_bounds(ROBOT_GRID_SPEC)[2:4] == (-0.25, 0.25)


def test_height_bin_centers_match_vox_util():
    """`height_bin_centers_m`가 `Vox_util`이 실제로 표본하는 높이와 일치해야 한다.

    표를 손으로 계산하지 않고 `Mem2Ref`로 직접 확인한다 -- 복셀 **중심**에서 표본한다는
    규약(`get_mem_T_ref`의 `-YMIN - vox_size_Y/2`)이 바뀌면 여기서 잡힌다.
    """
    import torch
    import utils.basic
    from projects.datasets.simplebev_vox import (
        build_vox_util, height_bin_centers_m, vox_bounds,
    )

    bins, low, high = 8, -0.125, 1.875
    vox = build_vox_util(ROBOT_GRID_SPEC, height_bins=bins, height_min_m=low, height_max_m=high)
    Z, Y, X = ROBOT_GRID_SPEC.n_rows, bins, ROBOT_GRID_SPEC.n_cols

    xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X, norm=False, device="cpu")
    xyz_ref = vox.Mem2Ref(xyz_mem, Z, Y, X, assert_cube=False)
    # ref.y = ego.z (EGO_TO_REF_ROTATION). mem 순서는 (Z, Y, X)라 Y가 가운데 축이다.
    heights = xyz_ref[0, :, 1].reshape(Z, Y, X)[0, :, 0].numpy()

    expected = height_bin_centers_m(
        vox_bounds(ROBOT_GRID_SPEC, height_min_m=low, height_max_m=high), bins)
    np.testing.assert_allclose(heights, expected, atol=1e-5)


def test_chosen_experiment_range_keeps_the_ground_plane():
    """실험이 고른 `[-0.125, 1.875]`/`Y=8`은 **bin 0이 정확히 `z=0`**이어야 한다.

    이것이 "정보가 순증한다"의 근거다 -- 기존 `Y=1`의 표본 평면이 그대로 살아 있으므로,
    결과가 나빠지면 원인을 '정보 부족'이 아니라 '용량/과적합'으로 좁힐 수 있다.
    """
    from projects.datasets.simplebev_vox import height_bin_centers_m, vox_bounds

    centers = height_bin_centers_m(
        vox_bounds(ROBOT_GRID_SPEC, height_min_m=-0.125, height_max_m=1.875), 8)
    np.testing.assert_allclose(centers, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], atol=1e-9)


def test_height_bins_round_trips_through_the_checkpoint_shape():
    """`Y`를 `bev_compressor` 형상에서 되읽는 경로가 실제로 맞는지 확인한다."""
    import torch.nn as nn
    from projects.models.simplebev_three_class import height_bins_from_state_dict

    for bins in (1, 4, 8):
        compressor = nn.Conv2d(128 * bins, 128, kernel_size=3, padding=1, bias=False)
        assert height_bins_from_state_dict({"bev_compressor.0.weight": compressor.weight}) == bins
