"""SynWoodScape -> Simple-BEV `Segnet.forward()` 입력 텐서 PyTorch `Dataset`.

`dataset/synwoodscape_occupancy_gt/`의 occupancy/visible GT(`tools/build_hybrid_occupancy.py`가
생성)와 `dataset/synwoodscape/.../rgb_images/`, `calibration_data/`를 묶어
`(rgb_camXs, pix_T_cams, cam0_T_camXs, seg_bev_g, valid_bev_g)`를 반환한다.

4-cam(FV/MVL/MVR/RV)을 그대로 쓴다 — 최종 자체 로봇 fine-tuning은 3-cam(후면 제외)이지만,
Simple-BEV는 카메라별 전용 파라미터가 없어(encoder는 이미지마다 공유, `feat_mem`은 카메라별
unproject 후 `reduce_masked_mean`으로 합쳐짐) pretrain 4-cam ↔ fine-tune 3-cam 전환이 구조적으로
문제 없다. 이미 4-cam raycast로 만들어둔 `visible.npy`를 그대로 재사용한다.

어안 투영은 이 단계에서 `simplebev_calib.pinhole_pix_T_cam_from_camera`의 임시 핀홀 근사다.
실제 `radial_poly` 투영으로의 교체(Simple-BEV lifting 자체를 어안으로 바꾸는 작업)는 ROADMAP
Phase 3.2에서 다룬다 — 이 데이터셋 클래스의 인터페이스(반환 텐서 shape/의미)는 그 교체와
무관하게 유지된다.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from projects.bev_gt.grid import SYNWOODSCAPE_PRETRAIN_GRID_SPEC
from projects.datasets.simplebev_calib import ego_T_cam_from_camera, pinhole_pix_T_cam_from_camera
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.fisheye import load_camera

CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")
DEFAULT_DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
DEFAULT_OCCUPANCY_GT_ROOT = Path("dataset/synwoodscape_occupancy_gt")
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
    ):
        self.sample_ids = list(sample_ids)
        self.dataset_root = Path(dataset_root)
        self.occupancy_gt_root = Path(occupancy_gt_root)
        self.camera_names = tuple(camera_names)
        self.resize_wh = resize_wh

        cameras = {
            name: load_camera(self.dataset_root / "calibration_data" / f"{name}.json")
            for name in self.camera_names
        }
        out_w, out_h = self.resize_wh
        # 캘리브레이션은 샘플과 무관하게 고정이므로 샘플마다 다시 계산하지 않고 한 번만 만든다.
        self._pix_T_cams = torch.from_numpy(np.stack([
            pinhole_pix_T_cam_from_camera(cameras[name], out_w, out_h)
            for name in self.camera_names
        ])).float()
        self._cam0_T_camXs = torch.from_numpy(np.stack([
            ref_T_cam_from_ego_T_cam(ego_T_cam_from_camera(cameras[name]))
            for name in self.camera_names
        ])).float()

    def __len__(self):
        return len(self.sample_ids)

    def _load_rgb(self, sample_id: str, camera_name: str) -> np.ndarray:
        path = self.dataset_root / "rgb_images" / f"{sample_id}_{camera_name}.png"
        image = Image.open(path).convert("RGB").resize(self.resize_wh, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3), Segnet 내부 정규화 전 [0,1]

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        rgb_camXs = np.stack([self._load_rgb(sample_id, name) for name in self.camera_names])
        rgb_camXs = np.transpose(rgb_camXs, (0, 3, 1, 2))  # (S, H, W, 3) -> (S, 3, H, W)

        occupancy = np.load(self.occupancy_gt_root / f"{sample_id}_occupancy.npy")
        visible = np.load(self.occupancy_gt_root / f"{sample_id}_visible.npy")

        return {
            "sample_id": sample_id,
            "rgb_camXs": torch.from_numpy(rgb_camXs).float(),
            "pix_T_cams": self._pix_T_cams,
            "cam0_T_camXs": self._cam0_T_camXs,
            "seg_bev_g": torch.from_numpy(occupancy.astype(np.float32)).unsqueeze(0),
            "valid_bev_g": torch.from_numpy(visible.astype(np.float32)).unsqueeze(0),
        }
