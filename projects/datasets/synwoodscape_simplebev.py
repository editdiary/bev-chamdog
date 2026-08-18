"""SynWoodScape -> two-head Simple-BEV input tensors.

`dataset/synwoodscape_2head_roi_8_4_6_h08/`의 finalized occupancy/visibility GT와
`dataset/synwoodscape/.../rgb_images/`, `calibration_data/`를 묶어
`(rgb_camXs, pix_T_cams, cam0_T_camXs, seg_bev_g, vis_bev_g, valid_bev_g)`를 반환한다.

Label contract:
- `seg_bev_g`: occupancy target, drivable=1/non-drivable=0
- `vis_bev_g`: visibility target, H=0.8 gather-column visibility with ego excluded
- `valid_bev_g`: 라벨이 존재하는 셀(= loss/metric의 바깥 경계). `vis_bev_g`와는 다른 개념이다.
  SynWoodScape는 ROI 전체가 라벨링돼 있으므로 전부 1이다. 관측 가능 영역으로 좁히는 건
  `vis_bev_g`의 역할이고, occupancy loss는 `vis * valid`로 마스킹된다. 둘을 같은 배열로
  주면 visibility loss가 positive 셀만 보게 되어 "전부 visible"이 전역 최적해가 된다.

4-cam(FV/MVL/MVR/RV)을 그대로 쓴다. Simple-BEV는 카메라별 전용 파라미터가 없어 pretrain
4-cam과 fine-tune 3-cam 전환이 구조적으로 막히지는 않는다.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from projects.bev_gt.grid import SYNWOODSCAPE_PRETRAIN_GRID_SPEC
from projects.datasets.photometric import apply_photometric, sample_photometric_params
from projects.datasets.simplebev_calib import ego_T_cam_from_camera, pinhole_pix_T_cam_from_camera
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.fisheye import load_camera

CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")
DEFAULT_DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
DEFAULT_OCCUPANCY_GT_ROOT = Path("dataset/synwoodscape_2head_roi_8_4_6_h08")
GRID_SPEC = SYNWOODSCAPE_PRETRAIN_GRID_SPEC

# 원본 종횡비(1280/966≈1.325)에 가까운 32의 배수 -- 왜곡을 최소화하면서 conv stride와도 맞는다.
RESIZE_WIDTH, RESIZE_HEIGHT = 512, 384


class SynWoodScapeSimpleBEVDataset(Dataset):
    def __init__(
        self,
        sample_ids,
        dataset_root=DEFAULT_DATASET_ROOT,
        occupancy_gt_root=DEFAULT_OCCUPANCY_GT_ROOT,
        camera_names=CAMERA_NAMES,
        resize_wh=(RESIZE_WIDTH, RESIZE_HEIGHT),
        augment=False,
    ):
        self.sample_ids = list(sample_ids)
        self.dataset_root = Path(dataset_root)
        self.occupancy_gt_root = Path(occupancy_gt_root)
        self.camera_names = tuple(camera_names)
        self.resize_wh = resize_wh
        self.augment = augment  # train split에서만 True -- val은 항상 원본이어야 비교가 된다

        cameras = {
            name: load_camera(self.dataset_root / "calibration_data" / f"{name}.json")
            for name in self.camera_names
        }
        out_w, out_h = self.resize_wh
        # 캘리브레이션은 샘플과 무관하게 고정이므로 샘플마다 다시 계산하지 않고 한 번만 만든다.
        if self.camera_names:
            self._pix_T_cams = torch.from_numpy(np.stack([
                pinhole_pix_T_cam_from_camera(cameras[name], out_w, out_h)
                for name in self.camera_names
            ])).float()
            self._cam0_T_camXs = torch.from_numpy(np.stack([
                ref_T_cam_from_ego_T_cam(ego_T_cam_from_camera(cameras[name]))
                for name in self.camera_names
            ])).float()
        else:
            self._pix_T_cams = torch.empty((0, 4, 4), dtype=torch.float32)
            self._cam0_T_camXs = torch.empty((0, 4, 4), dtype=torch.float32)

    def __len__(self):
        return len(self.sample_ids)

    def _load_rgb(self, sample_id: str, camera_name: str) -> np.ndarray:
        path = self.dataset_root / "rgb_images" / f"{sample_id}_{camera_name}.png"
        image = Image.open(path).convert("RGB").resize(self.resize_wh, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3), Segnet 내부 정규화 전 [0,1]

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        if self.camera_names:
            rgb_camXs = np.stack([self._load_rgb(sample_id, name) for name in self.camera_names])
            rgb_camXs = np.transpose(rgb_camXs, (0, 3, 1, 2))  # (S, H, W, 3) -> (S, 3, H, W)
        else:
            out_w, out_h = self.resize_wh
            rgb_camXs = np.empty((0, 3, out_h, out_w), dtype=np.float32)

        occupancy = np.load(self.occupancy_gt_root / f"{sample_id}_occupancy.npy")
        visible = np.load(self.occupancy_gt_root / f"{sample_id}_visible.npy")

        rgb_tensor = torch.from_numpy(rgb_camXs).float()
        if self.augment:
            # 광도만 바꾸므로 BEV GT는 손대지 않는다. 4개 카메라는 같은 파라미터를 공유한다.
            rgb_tensor = apply_photometric(rgb_tensor, sample_photometric_params())

        return {
            "sample_id": sample_id,
            "rgb_camXs": rgb_tensor,
            "pix_T_cams": self._pix_T_cams,
            "cam0_T_camXs": self._cam0_T_camXs,
            "seg_bev_g": torch.from_numpy(occupancy.astype(np.float32)).unsqueeze(0),
            "vis_bev_g": torch.from_numpy(visible.astype(np.float32)).unsqueeze(0),
            "valid_bev_g": torch.ones_like(torch.from_numpy(occupancy.astype(np.float32))).unsqueeze(0),
        }
