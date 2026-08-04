import numpy as np
import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec, cell_centers_m
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
