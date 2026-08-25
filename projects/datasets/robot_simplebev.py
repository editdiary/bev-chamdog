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
from projects.common.free_space import decompose
from projects.common.soft_boundary import signed_distance_field
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


def parse_sequence_names(value) -> list:
    """CLI로 들어온 시퀀스 목록 -> 이름 리스트.

    Fire는 `--train_sequences=raws1,raws3`처럼 콤마가 있는 값을 문자열이 아니라 튜플로
    파싱한다(`--train_sequences=raws1`은 문자열). 두 경우를 모두 받아준다.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [name for name in (str(item).strip() for item in items) if name]


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


def split_samples_by_random_frames(sequence_roots, val_fraction: float, split_seed: int) -> tuple:
    """**프로브 전용** 프레임 단위 무작위 split -- 모든 시퀀스의 프레임을 섞어 나눈다.

    ## 이건 성능 측정용이 아니다

    한 시퀀스는 연속 주행을 샘플링한 것이라 프레임끼리 장면이 크게 겹친다. 그래서 이 split은
    **val 프레임 바로 옆 프레임이 train에 들어간다** -- 누출이 설계상 존재한다. 여기서 나온
    숫자를 성능으로 보고하면 안 되고, `split_samples_by_sequence`가 기본인 이유가 그것이다.

    ## 그런데도 필요한 이유 (진단 문서 §28.4)

    답해야 할 질문이 **성능이 아니라 정보의 존재**다. `f1@10cm`이 train 프레임에서는 3~4 m
    에서도 0.938인데 시퀀스 holdout val에서는 0.355다(§28.2). 원인 후보가 둘이다.

    1. **모델이 train 191프레임을 외웠다** -- 그러면 "원거리 10 cm"는 계산이 아니라 회수이고,
       lifting 표본 밀도가 원거리 정밀도를 구속한다는 가설이 살아남는다.
    2. **정보가 정말 입력에 있다** -- 그러면 그 가설은 기각이고 병목은 장면 다양성이다.

    **못 본 프레임**(= 외울 수 없다)이면서 **같은 장면**(= 새 장면으로의 일반화를 요구하지
    않는다)인 프레임이 필요하고, 그것을 만드는 유일한 방법이 이 split이다. 여기서 원거리
    `f1@10cm`이 높게 나오면 (2)이고, 낮으면 아무것도 못 가른다 -- **한 방향으로만 결정적이다.**

    ## `split_seed`는 학습 시드와 다르다

    **시드마다 split이 바뀌면 시드 분산에 split 분산이 섞인다.** 그래서 split은 이 인자로
    따로 고정하고, 학습 시드 여러 개가 **같은 split**을 보게 한다.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction은 (0, 1)이어야 한다: {val_fraction}")
    samples = []
    for root in sequence_roots:
        samples.extend(list_sequence_samples(Path(root)))
    order = np.random.default_rng(int(split_seed)).permutation(len(samples))
    n_val = max(1, int(round(len(samples) * float(val_fraction))))
    val_idx = set(order[:n_val].tolist())
    # 원래 순서를 유지해 되돌린다 -- `split_*_samples.txt`가 사람이 읽을 수 있어야 한다.
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def split_samples_by_frame_blocks(sequence_roots, block_len: int, gap: int,
                                  split_seed: int) -> tuple:
    """**프로브 전용** -- 시퀀스마다 연속 블록 하나를 val로 떼고 **양쪽 `gap`프레임을 버린다.**

    `split_samples_by_random_frames`의 개선판이다(외부 검토 2026-08-25 제안). 무작위 split은
    **val 프레임 바로 옆 프레임이 train에 들어간다** -- 프레임 간격이 약 0.5~0.8 m이므로
    이웃은 거의 같은 이미지다. 그러면 "못 본 시점에서도 10 cm가 나온다"가 아니라
    "옆 프레임을 봤다"를 재게 된다.

    그래서 배치가 이렇게 된다 (한 시퀀스 안).

        train train train [버림 x gap] VAL x block_len [버림 x gap] train train train

    버려지는 프레임은 train도 val도 아니다. **val 프레임과 가장 가까운 train 프레임 사이가
    최소 `gap+1` 프레임(약 2 m)**이 되어, 이웃 누출이 크게 줄어든다.

    `block_len=5, gap=3`이면 7시퀀스에서 val 35 / 버림 42 / **train 190**이다 --
    시퀀스 holdout(train 192)과 train 크기가 거의 같아서 **데이터 양이 교란되지 않는다.**

    묻는 것은 여전히 성능이 아니다. **"같은 장면 분포 안에서, 학습하지 않은 시점에서도
    10 cm 정밀도를 낼 정보가 표현에 있는가"** 하나다.

    블록 시작 위치는 `split_seed`로 정하고 **양쪽 gap이 항상 잡히는 구간**에서만 고른다.
    `split_seed`는 학습 시드와 다른 인자다 -- 시드 여러 개가 같은 split을 봐야 한다.
    """
    block_len, gap = int(block_len), int(gap)
    if block_len < 1 or gap < 0:
        raise ValueError(f"block_len >= 1, gap >= 0이어야 한다: {block_len}, {gap}")
    rng = np.random.default_rng(int(split_seed))
    train, val = [], []
    for root in sequence_roots:
        samples = list_sequence_samples(Path(root))
        n = len(samples)
        last_start = n - block_len - gap
        if last_start < gap:
            raise ValueError(
                f"{Path(root).name}: 프레임 {n}개로는 block {block_len} + gap {gap}을 놓을 수 없다")
        start = int(rng.integers(gap, last_start + 1))
        for i, sample in enumerate(samples):
            if start <= i < start + block_len:
                val.append(sample)
            elif start - gap <= i < start + block_len + gap:
                continue  # 버린다 -- train도 val도 아니다
            else:
                train.append(sample)
    return train, val


def split_samples_within_sequences(sequence_roots, tail_fraction: float) -> tuple:
    """시퀀스 하나뿐일 때의 임시 split -- 각 시퀀스의 **뒤쪽 연속 구간**을 val로 뗀다.

    시퀀스가 둘 이상이면 `split_samples_by_sequence`를 쓰는 게 맞다. 이건 그때까지의
    임시방편이고, 숫자를 낙관적으로 만든다는 걸 알고 써야 한다:

    - sample_id 순서는 주행 순서이므로 뒤쪽 구간은 **공간적으로 연속된 한 덩어리**다.
      무작위 분할처럼 val 프레임 바로 옆 프레임이 train에 들어가는 일은 없다.
    - 다만 경계에 걸친 두 프레임(train 마지막 / val 첫)은 여전히 인접하고, 같은 온실
      통로를 같은 조명에서 찍은 것이라 도메인이 완전히 분리되지는 않는다.

    그래서 이 split의 val 숫자는 "학습이 망가지지 않았나"를 보는 sanity check이지
    일반화 성능이 아니다. 실험 A/B를 이 숫자로 판정하면 안 된다.
    """
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError(f"tail_fraction은 [0, 1)이어야 한다: {tail_fraction}")
    train, val = [], []
    for root in sequence_roots:
        samples = list_sequence_samples(root)
        n_val = int(round(len(samples) * tail_fraction))
        n_val = min(n_val, max(len(samples) - 1, 0))  # train이 비지 않도록
        if n_val:
            train.extend(samples[:-n_val])
            val.extend(samples[-n_val:])
        else:
            train.extend(samples)
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


def load_masked_labels(sequence_root, sample_id, permanent_blind, invalid) -> tuple:
    """`(occupancy, vis, valid)` -- 전부 (n_rows, n_cols) bool. 마스킹 규약의 단일 출처.

    `RobotBEVDataset.__getitem__`과 학습 스크립트의 클래스 비율 실측(`pos_weight`)이 같은
    정의를 봐야 한다. 예전에 SynWoodScape 쪽에서 `pos_weight`가 마스크를 무시하고 원본
    라벨을 세는 바람에 클래스 보정이 통째로 어긋난 적이 있다.
    """
    sequence_root = Path(sequence_root)
    occupancy = np.load(sequence_root / "occupancy_npy" / f"{sample_id}.npy").astype(bool)
    visible = np.load(sequence_root / "visibility_npy" / f"{sample_id}.npy").astype(bool)
    return occupancy, visible & ~permanent_blind, ~invalid


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

        self.permanent_blind, self.invalid = build_bev_masks(
            self.common_root, grid_spec, self.camera_names
        )

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

        occupancy, vis, valid = load_masked_labels(
            sequence_root, sample_id, self.permanent_blind, self.invalid
        )

        def as_tensor(mask):
            return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        # `d_bev_g` -- GT 경계까지의 부호 있는 수직 거리 [m]. **라벨만으로 결정되는 값이고
        # 하이퍼파라미터가 아니다** (`docs/soft_boundary_loss_design.md` §2.1). soft-boundary
        # loss가 영역을 나누는 데 쓴다.
        #
        # 여기서 계산하는 이유: 거리변환이 CPU numpy(scipy)라 학습 루프에서 부르면 매 배치
        # GPU->CPU 왕복이 생긴다. `__getitem__`에 두면 `num_workers`가 병렬로 처리하고
        # 120x120 두 번이라 비용이 사실상 0이다. loss 종류와 무관하게 항상 넣는다 --
        # 배치 계약이 갈리면 두 loss의 런을 같은 코드로 다룰 수 없다.
        #
        # `keep`에서 빠지는 셀(`permanent_blind`와 `valid=0`)은 거리 계산에서 free에 합쳐진다.
        # 합치지 않으면 그 정적 마스크의 테두리가 경계로 잡혀 대역이 오염된다. **감독에서의
        # 취급은 다르다** -- 원반은 hard `Ω_N`이다(같은 문서 §4.2).
        keep = valid.astype(bool) & ~self.permanent_blind
        distance = signed_distance_field(
            decompose(occupancy, vis, valid)["free"], keep, self.grid_spec.cell_m
        )

        return {
            "sample_id": f"{sequence_root.name}/{sample_id}",
            "rgb_camXs": rgb_tensor,
            "pix_T_cams": self._pix_T_cams,
            "cam0_T_camXs": self._cam0_T_camXs,
            "seg_bev_g": as_tensor(occupancy),
            "vis_bev_g": as_tensor(vis),
            "valid_bev_g": as_tensor(valid),
            "d_bev_g": torch.from_numpy(distance).unsqueeze(0),
        }
