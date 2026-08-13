from pathlib import Path

import pytest
import torch

from projects.bev_gt.grid import SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC
from projects.datasets.synwoodscape_simplebev import (
    CAMERA_NAMES,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    SynWoodScapeSimpleBEVDataset,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OCCUPANCY_GT_ROOT = Path("dataset/synwoodscape_2head_roi_8_4_6_h08")
requires_data = pytest.mark.skipif(
    not (DATASET_ROOT / "calibration_data").exists() or not OCCUPANCY_GT_ROOT.exists(),
    reason="SynWoodScape dataset / occupancy GT not available locally",
)

SAMPLE_IDS = ["00000", "00001", "00250"]


@requires_data
def test_len_matches_sample_ids():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    assert len(ds) == len(SAMPLE_IDS)


@requires_data
def test_getitem_returns_expected_shapes_and_dtypes():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    item = ds[0]

    S = len(CAMERA_NAMES)
    assert item["rgb_camXs"].shape == (S, 3, RESIZE_HEIGHT, RESIZE_WIDTH)
    assert item["pix_T_cams"].shape == (S, 4, 4)
    assert item["cam0_T_camXs"].shape == (S, 4, 4)
    assert item["seg_bev_g"].shape == (1, SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_rows, SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_cols)
    assert item["vis_bev_g"].shape == item["seg_bev_g"].shape
    assert item["valid_bev_g"].shape == item["seg_bev_g"].shape

    for key in ("rgb_camXs", "pix_T_cams", "cam0_T_camXs", "seg_bev_g", "vis_bev_g", "valid_bev_g"):
        assert item[key].dtype == torch.float32


@requires_data
def test_rgb_is_in_unit_range():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    rgb = ds[0]["rgb_camXs"]
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


@requires_data
def test_seg_and_valid_bev_g_are_binary():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    item = ds[0]
    assert set(item["seg_bev_g"].unique().tolist()) <= {0.0, 1.0}
    assert set(item["vis_bev_g"].unique().tolist()) <= {0.0, 1.0}
    assert set(item["valid_bev_g"].unique().tolist()) <= {0.0, 1.0}


@requires_data
def test_pix_T_cams_is_valid_homogeneous_intrinsic():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    pix_T_cams = ds[0]["pix_T_cams"]
    for cam_idx in range(pix_T_cams.shape[0]):
        K = pix_T_cams[cam_idx]
        assert K[2, 2] == 1.0
        assert K[3, 3] == 1.0
        assert K[0, 0] > 0  # focal length
        assert K[1, 1] > 0


@requires_data
def test_cam0_T_camXs_rotation_blocks_are_orthonormal():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    cam0_T_camXs = ds[0]["cam0_T_camXs"]
    for cam_idx in range(cam0_T_camXs.shape[0]):
        R = cam0_T_camXs[cam_idx, :3, :3]
        identity_approx = R @ R.T
        torch.testing.assert_close(identity_approx, torch.eye(3), atol=1e-5, rtol=1e-5)


@requires_data
def test_different_samples_return_different_rgb_and_gt():
    ds = SynWoodScapeSimpleBEVDataset(SAMPLE_IDS)
    a, b = ds[0], ds[1]
    assert not torch.equal(a["rgb_camXs"], b["rgb_camXs"])
    # calibration tensors must stay identical across samples (same cameras)
    torch.testing.assert_close(a["pix_T_cams"], b["pix_T_cams"])
    torch.testing.assert_close(a["cam0_T_camXs"], b["cam0_T_camXs"])
