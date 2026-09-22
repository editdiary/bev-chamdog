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

from projects.bev_gt.grid import (
    SYNWOODSCAPE_NATIVE_GRID_SPEC,
    SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
)
from projects.common.free_space import decompose
from projects.common.soft_boundary import signed_distance_field
from projects.datasets.photometric import apply_photometric, sample_photometric_params
from projects.datasets.simplebev_calib import ego_T_cam_from_camera, pinhole_pix_T_cam_from_camera
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.fisheye import load_camera

CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")
DEFAULT_DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
DEFAULT_OCCUPANCY_GT_ROOT = Path("dataset/synwoodscape_2head_roi_8_4_6_h08")
GRID_SPEC = SYNWOODSCAPE_PRETRAIN_GRID_SPEC

# 소스 프로파일 -- **격자와 라벨 루트는 한 쌍으로만 의미가 있다.** 라벨이 그 ROI로 잘려
# 저장돼 있으므로 둘을 따로 고르게 두면 조용히 어긋난 학습이 돌아간다.
#
# `roi_8_4_6`  타깃 과제에 맞춰 좁게 자르고 **사람이 보정한** 라벨. 지금까지의 사전학습.
# `native`     소스가 실제로 주는 **30 m 전체**를 자율주행 관행 해상도(0.15 m/cell)로.
#              **수동 보정이 없다**(보정본은 8/4/±6에만 있다).
#
# 링 경계도 격자에 맞춰 같이 고른다 -- 8 m용 경계를 30 m 격자에 쓰면 바깥 대부분이 어느
# 링에도 안 들어간다.
NATIVE_OCCUPANCY_GT_ROOT = Path("dataset/synwoodscape_native_roi_15_15_15_h08")
SOURCE_PROFILES = {
    "roi_8_4_6": {
        "grid_spec": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
        "occupancy_gt_root": DEFAULT_OCCUPANCY_GT_ROOT,
        "ring_edges_m": (0.0, 2.0, 4.0, 6.0, 8.0),
        "manual_refinement": True,
    },
    "native": {
        "grid_spec": SYNWOODSCAPE_NATIVE_GRID_SPEC,
        "occupancy_gt_root": NATIVE_OCCUPANCY_GT_ROOT,
        "ring_edges_m": (0.0, 3.75, 7.5, 11.25, 15.0),
        "manual_refinement": False,
    },
}


def resolve_source_profile(name: str) -> dict:
    """프로파일 이름 -> `{grid_spec, occupancy_gt_root, ring_edges_m, manual_refinement}`."""
    if name not in SOURCE_PROFILES:
        raise ValueError(
            f"알 수 없는 source_profile {name!r}. 가능한 값: {sorted(SOURCE_PROFILES)}")
    return SOURCE_PROFILES[name]

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
        grid_spec=GRID_SPEC,
    ):
        self.sample_ids = list(sample_ids)
        self.dataset_root = Path(dataset_root)
        self.occupancy_gt_root = Path(occupancy_gt_root)
        self.camera_names = tuple(camera_names)
        self.resize_wh = resize_wh
        self.augment = augment  # train split에서만 True -- val은 항상 원본이어야 비교가 된다
        # 거리장을 미터로 만들려면 셀 크기가 필요하다. **라벨 루트와 짝이 맞아야 한다** --
        # `SOURCE_PROFILES`가 그 짝을 들고 있으니 따로 넘기지 말고 거기서 꺼내 쓴다.
        self.grid_spec = grid_spec

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
        valid = np.ones_like(occupancy, dtype=bool)

        # `d_bev_g` -- GT 경계까지의 부호 있는 수직 거리 [m]. `robot_simplebev.py`와 **같은
        # 함수·같은 규약**이다. soft-boundary loss가 영역을 나누는 데 쓴다.
        #
        # 여기서 계산하는 이유도 같다: 거리변환이 CPU numpy라 학습 루프에서 부르면 매 배치
        # GPU->CPU 왕복이 생긴다. `__getitem__`이면 `num_workers`가 병렬로 처리한다.
        # loss 종류와 무관하게 항상 넣는다 -- 배치 계약이 갈리면 두 loss의 런을 같은 코드로
        # 다룰 수 없다.
        #
        # 로봇 쪽과 다른 점 하나: SynWoodScape에는 `permanent_blind`(리그 자체가 영구히
        # 가리는 영역)와 `valid=0`(수집 아티팩트)이 없으므로 `keep`이 전부 True다.
        distance = signed_distance_field(
            decompose(occupancy, visible, valid)["free"], valid, self.grid_spec.cell_m
        )

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
            "valid_bev_g": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
            "d_bev_g": torch.from_numpy(distance).unsqueeze(0),
        }
