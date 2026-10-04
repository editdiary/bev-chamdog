"""`split_model`이 원래 `Segnet.forward`와 같은 출력을 내는지 -- 배포 측정의 전제다."""
from pathlib import Path

import numpy as np
import pytest
import torch

from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.deploy.split_model import split_model
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES, load_cameras, load_ego_T_cams
from projects.models.double_sphere_vox import build_double_sphere_vox_util
from projects.models.simplebev_three_class import ThreeClassSegnet
from projects.bev_gt.grid import OccupancyGridSpec

CALIB_PATH = Path("dataset/sj_datasets/common/calibration/calib.yaml")
pytestmark = [
    pytest.mark.skipif(not CALIB_PATH.exists(), reason="self-collected calibration not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Segnet keeps mean/std on CUDA"),
]


def _setup(height_bins):
    # 작은 격자(8의 배수)로 빠르게 -- 등가성은 격자 크기와 무관하다.
    spec = OccupancyGridSpec(front_m=1.6, rear_m=0.8, half_width_m=1.2, cell_m=0.1)
    cameras = [load_cameras(CALIB_PATH)[n] for n in FINETUNE_CAMERA_NAMES]
    vox_util = build_double_sphere_vox_util(spec, cameras, device="cuda", height_bins=height_bins,
                                            height_min_m=-0.25, height_max_m=1.75)
    torch.manual_seed(0)
    model = ThreeClassSegnet(spec.n_rows, vox_util.Y, spec.n_cols, vox_util, use_radar=False,
                             use_lidar=False, do_rgbcompress=True, encoder_type="res101",
                             rand_flip=False, num_classes=2).cuda().eval()
    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    cam0_T_camXs = torch.from_numpy(ref_T_cams).float().cuda()           # (S, 4, 4)
    pix_T_cams = torch.eye(4).repeat(len(cameras), 1, 1).cuda()
    return model, vox_util, pix_T_cams, cam0_T_camXs


@pytest.mark.parametrize("height_bins", [1, 4])
@torch.no_grad()
def test_split_matches_full_forward(height_bins):
    model, vox_util, pix_T_cams, cam0_T_camXs = _setup(height_bins)
    S, H, W = cam0_T_camXs.shape[0], 96, 160
    rgb = torch.rand(1, S, 3, H, W, device="cuda")

    _, _, logits, _, _ = model(rgb - 0.5, pix_T_cams[None], cam0_T_camXs[None], vox_util)
    want = torch.softmax(logits, dim=1)[:, 1]

    enc, bev = split_model(model, vox_util, pix_T_cams, cam0_T_camXs, (H, W))
    got = bev(enc(rgb[0]))

    assert got.shape == want.shape
    assert torch.equal(got, want), (got - want).abs().max()


@torch.no_grad()
def test_grid_is_independent_of_the_image():
    """`grid`·`valid`는 보정에서만 나온다 -- 상수로 박을 수 있다는 주장의 근거."""
    model, vox_util, pix_T_cams, cam0_T_camXs = _setup(4)
    _, bev_a = split_model(model, vox_util, pix_T_cams, cam0_T_camXs, (96, 160))
    _, bev_b = split_model(model, vox_util, pix_T_cams, cam0_T_camXs, (96, 160))
    assert torch.equal(bev_a.grid, bev_b.grid) and torch.equal(bev_a.valid, bev_b.valid)
    assert bev_a.valid.sum() > 0
