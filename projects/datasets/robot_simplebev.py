"""자체 수집 데이터셋(`dataset/sj_datasets/`) -> two-head Simple-BEV 입력 텐서.

SynWoodScape 쪽 대응물은 `projects/datasets/synwoodscape_simplebev.py`이고 반환 계약이 같다:
`(rgb_camXs, pix_T_cams, cam0_T_camXs, seg_bev_g, vis_bev_g, valid_bev_g)`.

--------------------------------------------------------------------------------
디렉터리 구조
--------------------------------------------------------------------------------
    dataset/sj_datasets/
      common/                       <- 모든 시퀀스가 공유
        calibration/calib.yaml
        self_mask.png               <- ego 테이블이 가리는 BEV 영역
        rear_self_box.png           <- 수집용 카트 손잡이 + 미는 사람 영역
      raws1/                        <- 시퀀스 하나
        rgb_images/sample_000000/cam_{front,left,right}.jpg
        occupancy_npy/sample_000000.npy
        visibility_npy/sample_000000.npy

시퀀스는 계속 추가되며, **train/val split은 프레임이 아니라 시퀀스 단위로 나눈다**
(`split_samples_by_sequence`). 한 시퀀스는 연속 주행을 거리 기반으로 샘플링한 것이라
프레임끼리 장면이 크게 겹쳐서, 프레임 단위로 섞으면 val이 train을 그대로 들여다본다.

--------------------------------------------------------------------------------
Label contract -- SynWoodScape와 다른 부분
--------------------------------------------------------------------------------
- `seg_bev_g`: occupancy. 라벨 그대로 drivable=1 / obstacle=0.
- `vis_bev_g`: **라벨의 raycast visibility에서 "카메라가 영구적으로 못 보는 영역"을 뺀 것.**
  라벨의 visibility는 "다른 장애물에 가렸는가"만 담고 있어서, 화각 밖(수평 장착 탓에 생긴
  반경 약 0.5 m 원반)과 ego 테이블에 가린 영역이 visible로 남아 있다. 둘 다 배포 때도
  그대로 존재하는 물리적 가림이므로 `vis=0`이 정확한 라벨이며, occupancy loss가 `vis*valid`로
  마스킹되므로 근거 없는 셀에서 예측을 요구하지 않게 된다.
- `valid_bev_g`: **수집 아티팩트(후방 카트 손잡이·미는 사람)만 0.**
  이건 배포 때 존재하지 않으므로 `vis=0`으로 두면 안 된다 -- 그러면 "후방은 항상 가려져 있다"는
  거짓을 visibility head가 학습한다(실제로는 left/right 카메라 주변부가 후방을 덮는다).
  `valid=0`이면 두 head 모두 gradient를 받지 않아 아무것도 배우지 않는다.

`valid`와 `vis`를 같은 배열로 주면 visibility loss가 positive 셀만 보게 되어 "전부 visible"이
전역 최적해가 된다 -- SynWoodScape 쪽에서 실제로 겪은 버그다.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from projects.bev_gt.camera_coverage import compute_camera_coverage_mask
from projects.bev_gt.grid import ROBOT_GRID_SPEC
from projects.datasets.photometric import apply_photometric, sample_photometric_params
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.double_sphere import (
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
)

DEFAULT_DATASET_ROOT = Path("dataset/sj_datasets")
DEFAULT_COMMON_ROOT = DEFAULT_DATASET_ROOT / "common"
GRID_SPEC = ROBOT_GRID_SPEC

# 원본 1280x720(16:9)의 종횡비를 유지하면서 인코더 stride 8로 나눠떨어지는 크기.
# SynWoodScape pretrain은 512x384였지만 그쪽 원본이 4:3이라서다 -- Segnet 인코더는
# 완전 합성곱이라 해상도가 달라도 가중치가 그대로 전이된다.
RESIZE_WIDTH, RESIZE_HEIGHT = 512, 288


def list_sequence_samples(sequence_root) -> list:
    """시퀀스 디렉터리 -> [(sequence_root, sample_id), ...] (sample_id 오름차순)."""
    sequence_root = Path(sequence_root)
    sample_ids = sorted(p.stem for p in (sequence_root / "occupancy_npy").glob("*.npy"))
    return [(sequence_root, sample_id) for sample_id in sample_ids]


def split_samples_by_sequence(sequence_roots, val_sequence_names) -> tuple:
    """시퀀스 단위 train/val split. `val_sequence_names`는 디렉터리 이름 집합.

    프레임 단위 split을 쓰지 않는 이유는 모듈 docstring 참고.
    """
    val_sequence_names = set(val_sequence_names)
    train, val = [], []
    for root in sequence_roots:
        root = Path(root)
        bucket = val if root.name in val_sequence_names else train
        bucket.extend(list_sequence_samples(root))
    return train, val


def _load_bev_png_mask(path, grid_spec) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    expected = (grid_spec.n_rows, grid_spec.n_cols)
    if mask.shape != expected:
        raise ValueError(f"{path}: mask shape {mask.shape} != grid {expected}")
    return mask


def build_bev_masks(common_root, grid_spec=GRID_SPEC, camera_names=FINETUNE_CAMERA_NAMES) -> tuple:
    """`(permanent_blind, invalid)` -- 각각 `vis=0`, `valid=0`으로 강제할 (n_rows, n_cols) bool.

    `permanent_blind`는 두 조각의 합집합이다:
    - 어떤 카메라 광선도 닿지 않는 셀 (캘리브레이션에서 계산)
    - `self_mask.png`: ego 테이블에 가린 셀 (이미지 픽셀을 지면에 투영해 만든 것이라
      안쪽 원반에는 구멍이 있어서, 위 계산과 반드시 합쳐야 한다)
    """
    common_root = Path(common_root)
    cameras = load_cameras(common_root / "calibration/calib.yaml")
    ego_T_cams = load_ego_T_cams(common_root / "calibration/calib.yaml")
    selected = {name: cameras[name] for name in camera_names}

    coverage = compute_camera_coverage_mask(grid_spec, selected, ego_T_cams)
    table = _load_bev_png_mask(common_root / "self_mask.png", grid_spec)
    invalid = _load_bev_png_mask(common_root / "rear_self_box.png", grid_spec)
    return (~coverage) | table, invalid


class RobotBEVDataset(Dataset):
    def __init__(
        self,
        samples,
        common_root=DEFAULT_COMMON_ROOT,
        camera_names=FINETUNE_CAMERA_NAMES,
        resize_wh=(RESIZE_WIDTH, RESIZE_HEIGHT),
        augment=False,
        grid_spec=GRID_SPEC,
    ):
        self.samples = [(Path(root), sample_id) for root, sample_id in samples]
        self.common_root = Path(common_root)
        self.camera_names = tuple(camera_names)
        self.resize_wh = resize_wh
        self.augment = augment  # train split에서만 True -- val은 항상 원본이어야 비교가 된다
        self.grid_spec = grid_spec

        calib_path = self.common_root / "calibration/calib.yaml"
        cameras = load_cameras(calib_path)
        ego_T_cams = load_ego_T_cams(calib_path)
        # `Segnet.forward`/`Vox_util`에 넘길 카메라 순서 -- rgb를 쌓는 순서와 반드시 같아야 한다.
        self.cameras = [cameras[name] for name in self.camera_names]

        permanent_blind, invalid = build_bev_masks(self.common_root, grid_spec, self.camera_names)
        self._permanent_blind = torch.from_numpy(permanent_blind)
        self._valid_bev_g = torch.from_numpy(~invalid).float().unsqueeze(0)

        # 캘리브레이션은 샘플과 무관하게 고정이므로 한 번만 만든다.
        self._cam0_T_camXs = torch.from_numpy(
            np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[name]) for name in self.camera_names])
        ).float()
        # `DoubleSphereVoxUtil`이 렌즈 파라미터를 직접 들고 있으므로 이 텐서는 쓰이지 않는다.
        # `Segnet.forward`가 인자로 요구하기 때문에 형상만 맞춰 넘긴다.
        self._pix_T_cams = torch.eye(4).repeat(len(self.camera_names), 1, 1)

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, sequence_root: Path, sample_id: str, camera_name: str) -> np.ndarray:
        path = sequence_root / "rgb_images" / sample_id / f"cam_{camera_name}.jpg"
        image = Image.open(path).convert("RGB").resize(self.resize_wh, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0  # Segnet 내부 정규화 전 [0,1]

    def __getitem__(self, index):
        sequence_root, sample_id = self.samples[index]

        rgb_camXs = np.stack(
            [self._load_rgb(sequence_root, sample_id, name) for name in self.camera_names]
        )
        rgb_camXs = np.transpose(rgb_camXs, (0, 3, 1, 2))  # (S, H, W, 3) -> (S, 3, H, W)
        rgb_tensor = torch.from_numpy(rgb_camXs).float()
        if self.augment:
            # 광도만 바꾸므로 BEV GT는 손대지 않는다. 카메라들은 같은 파라미터를 공유한다.
            rgb_tensor = apply_photometric(rgb_tensor, sample_photometric_params())

        occupancy = np.load(sequence_root / "occupancy_npy" / f"{sample_id}.npy")
        visible = np.load(sequence_root / "visibility_npy" / f"{sample_id}.npy")

        vis = torch.from_numpy(visible.astype(np.float32)).unsqueeze(0)
        vis = vis * (~self._permanent_blind).float().unsqueeze(0)

        return {
            "sample_id": f"{sequence_root.name}/{sample_id}",
            "rgb_camXs": rgb_tensor,
            "pix_T_cams": self._pix_T_cams,
            "cam0_T_camXs": self._cam0_T_camXs,
            "seg_bev_g": torch.from_numpy(occupancy.astype(np.float32)).unsqueeze(0),
            "vis_bev_g": vis,
            "valid_bev_g": self._valid_bev_g.clone(),
        }
