from pathlib import Path

import numpy as np
import pytest
import torch

from projects.bev_gt.grid import ROBOT_GRID_SPEC
from projects.datasets.robot_simplebev import (
    GRID_SPEC,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    RobotBEVDataset,
    build_bev_masks,
    list_sequence_samples,
    split_samples_by_frame_blocks,
    split_samples_by_random_frames,
    split_samples_by_sequence,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES

DATASET_ROOT = Path("dataset/sj_datasets")
COMMON_ROOT = DATASET_ROOT / "common"
SEQUENCE_ROOT = DATASET_ROOT / "raws1"

requires_dataset = pytest.mark.skipif(
    not (SEQUENCE_ROOT / "occupancy_npy").exists() or not COMMON_ROOT.exists(),
    reason="self-collected dataset not available locally",
)


@requires_dataset
def test_grid_spec_matches_the_annotated_label_shape():
    occupancy = np.load(next((SEQUENCE_ROOT / "occupancy_npy").glob("*.npy")))
    assert occupancy.shape == (GRID_SPEC.n_rows, GRID_SPEC.n_cols)
    assert GRID_SPEC is ROBOT_GRID_SPEC


@requires_dataset
def test_list_and_split_sequences():
    samples = list_sequence_samples(SEQUENCE_ROOT)
    assert samples
    assert samples == sorted(samples, key=lambda s: s[1])

    train, val = split_samples_by_sequence([SEQUENCE_ROOT], val_sequence_names={"raws1"})
    assert not train and len(val) == len(samples)

    train, val = split_samples_by_sequence([SEQUENCE_ROOT], val_sequence_names=set())
    assert len(train) == len(samples) and not val


@requires_dataset
def test_permanent_blind_covers_both_the_fov_disc_and_the_table():
    """`self_mask.png`만 쓰면 안 되는 이유 -- 안쪽 원반이 구멍으로 남는다."""
    from PIL import Image

    permanent_blind, invalid = build_bev_masks(COMMON_ROOT)
    table = np.asarray(Image.open(COMMON_ROOT / "self_mask.png").convert("L")) > 127

    assert permanent_blind.shape == (GRID_SPEC.n_rows, GRID_SPEC.n_cols)
    assert invalid.shape == permanent_blind.shape
    # 테이블 마스크는 부분집합이어야 하고, 그것만으로는 한참 모자라야 한다.
    assert np.all(permanent_blind[table])
    assert permanent_blind.sum() > table.sum() * 1.5

    # 원점 주변은 어떤 카메라 광선도 닿지 않는다(수평 장착 + 렌즈 하향 화각 한계).
    forward_m = GRID_SPEC.front_m - (np.arange(GRID_SPEC.n_rows) + 0.5) * GRID_SPEC.cell_m
    lateral_m = GRID_SPEC.half_width_m - (np.arange(GRID_SPEC.n_cols) + 0.5) * GRID_SPEC.cell_m
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    near = np.hypot(fg, lg) <= 0.3
    assert permanent_blind[near].all()

    # 반대로 전방 2 m 부근은 잘 보여야 한다 -- 전부 가려졌다면 extrinsic이 틀린 것이다.
    ahead = (np.abs(fg - 2.0) < 0.2) & (np.abs(lg) < 0.2)
    assert not permanent_blind[ahead].any()


@requires_dataset
def test_getitem_tensor_contract():
    samples = list_sequence_samples(SEQUENCE_ROOT)[:2]
    dataset = RobotBEVDataset(samples, common_root=COMMON_ROOT)
    item = dataset[0]

    s = len(FINETUNE_CAMERA_NAMES)
    assert item["rgb_camXs"].shape == (s, 3, RESIZE_HEIGHT, RESIZE_WIDTH)
    assert item["pix_T_cams"].shape == (s, 4, 4)
    assert item["cam0_T_camXs"].shape == (s, 4, 4)
    for key in ("seg_bev_g", "vis_bev_g", "valid_bev_g"):
        assert item[key].shape == (1, GRID_SPEC.n_rows, GRID_SPEC.n_cols), key
        assert item[key].dtype == torch.float32, key
    assert item["sample_id"].startswith("raws1/")
    assert 0.0 <= item["rgb_camXs"].min() and item["rgb_camXs"].max() <= 1.0


@requires_dataset
def test_visibility_is_masked_but_valid_is_not_the_same_array():
    """`valid == vis`로 주면 visibility loss가 negative 셀을 못 봐서 head가 죽는다."""
    dataset = RobotBEVDataset(list_sequence_samples(SEQUENCE_ROOT)[:1], common_root=COMMON_ROOT)
    item = dataset[0]
    vis = item["vis_bev_g"][0].numpy().astype(bool)
    valid = item["valid_bev_g"][0].numpy().astype(bool)
    permanent_blind, invalid = build_bev_masks(COMMON_ROOT)

    assert not np.array_equal(vis, valid)
    # 영구 사각지대는 vis=0이지만 valid는 살아 있어야 한다(visibility head가 학습해야 하므로).
    assert not vis[permanent_blind].any()
    assert valid[permanent_blind & ~invalid].all()
    # 수집 아티팩트는 valid=0.
    assert not valid[invalid].any()
    # 두 head 모두 학습할 셀이 실제로 남아 있어야 한다.
    assert (vis & valid).sum() > 0
    assert (~vis & valid).sum() > 0


@requires_dataset
def test_masking_only_removes_visibility_never_adds_it():
    """마스킹은 라벨의 visible 영역을 좁히기만 해야 한다 -- 없던 visible을 만들면 버그다."""
    samples = list_sequence_samples(SEQUENCE_ROOT)[:3]
    dataset = RobotBEVDataset(samples, common_root=COMMON_ROOT)
    for index, (root, sample_id) in enumerate(samples):
        raw = np.load(root / "visibility_npy" / f"{sample_id}.npy").astype(bool)
        masked = dataset[index]["vis_bev_g"][0].numpy().astype(bool)
        assert np.all(masked <= raw)
        assert masked.sum() < raw.sum()


@requires_dataset
def test_augment_changes_images_but_never_labels():
    samples = list_sequence_samples(SEQUENCE_ROOT)[:1]
    plain = RobotBEVDataset(samples, common_root=COMMON_ROOT, augment=False)[0]
    augmented = RobotBEVDataset(samples, common_root=COMMON_ROOT, augment=True)[0]

    assert not torch.equal(plain["rgb_camXs"], augmented["rgb_camXs"])
    for key in ("seg_bev_g", "vis_bev_g", "valid_bev_g"):
        torch.testing.assert_close(plain[key], augmented[key])


@requires_dataset
def test_camera_order_is_consistent_between_images_and_extrinsics():
    """rgb 쌓는 순서와 `cam0_T_camXs`/`vox_util` 카메라 순서가 어긋나면 조용히 망가진다."""
    samples = list_sequence_samples(SEQUENCE_ROOT)[:1]
    dataset = RobotBEVDataset(samples, common_root=COMMON_ROOT)

    assert [cam.name for cam in dataset.cameras] == list(FINETUNE_CAMERA_NAMES)

    # 순서를 뒤집으면 extrinsic도 같이 뒤집혀야 한다.
    reversed_names = tuple(reversed(FINETUNE_CAMERA_NAMES))
    flipped = RobotBEVDataset(samples, common_root=COMMON_ROOT, camera_names=reversed_names)
    torch.testing.assert_close(
        dataset[0]["cam0_T_camXs"], flipped[0]["cam0_T_camXs"].flip(0)
    )
    torch.testing.assert_close(dataset[0]["rgb_camXs"], flipped[0]["rgb_camXs"].flip(0))


# --- 프로브 전용 split (진단 문서 §28.4) -------------------------------------------------

_ALL_SEQUENCES = ("raws1", "raws2", "raws3", "rawos1", "rawos2", "rawos3", "rawos4")


def _roots():
    return [DATASET_ROOT / name for name in _ALL_SEQUENCES]


def _index(sample):
    return int(sample[1].split("_")[1])


@requires_dataset
def test_random_frame_split_is_deterministic_disjoint_and_matches_the_holdout_size():
    """**train 크기가 시퀀스 holdout(192)과 같아야** 데이터 양이 교란되지 않는다."""
    train, val = split_samples_by_random_frames(_roots(), 0.28, split_seed=0)
    assert (len(train), len(val)) == (192, 75)
    assert not set(train) & set(val)
    again = split_samples_by_random_frames(_roots(), 0.28, split_seed=0)
    assert again == (train, val)
    assert split_samples_by_random_frames(_roots(), 0.28, split_seed=1)[1] != val


@requires_dataset
def test_random_frame_split_leaks_neighbours_which_is_why_the_block_split_exists():
    """무작위 split은 **val 바로 옆 프레임이 train에 있다** -- 이 프로브의 알려진 한계다."""
    train, val = split_samples_by_random_frames(_roots(), 0.28, split_seed=0)
    train_by_seq = {}
    for root, sample_id in train:
        train_by_seq.setdefault(root.name, set()).add(int(sample_id.split("_")[1]))
    adjacent = sum(1 for root, sample_id in val
                   if {int(sample_id.split("_")[1]) - 1, int(sample_id.split("_")[1]) + 1}
                   & train_by_seq.get(root.name, set()))
    assert adjacent > len(val) // 2


@requires_dataset
def test_block_split_keeps_every_val_frame_at_least_gap_plus_one_away_from_train():
    """이 split이 존재하는 이유가 이 성질 하나다. 깨지면 프로브가 답을 못 낸다."""
    block_len, gap = 5, 3
    train, val = split_samples_by_frame_blocks(_roots(), block_len, gap, split_seed=0)
    assert len(val) == block_len * len(_ALL_SEQUENCES)
    train_by_seq, val_by_seq = {}, {}
    for bucket, samples in ((train_by_seq, train), (val_by_seq, val)):
        for sample in samples:
            bucket.setdefault(sample[0].name, []).append(_index(sample))
    for name, val_idx in val_by_seq.items():
        assert sorted(val_idx) == list(range(min(val_idx), min(val_idx) + block_len))
        assert min(abs(v - t) for v in val_idx for t in train_by_seq[name]) >= gap + 1


@requires_dataset
def test_block_split_drops_frames_rather_than_reassigning_them():
    """버린 프레임은 train도 val도 아니다 -- train에 남기면 gap이 의미가 없다."""
    train, val = split_samples_by_frame_blocks(_roots(), 5, 3, split_seed=0)
    total = sum(len(list_sequence_samples(root)) for root in _roots())
    assert len(train) + len(val) == total - 2 * 3 * len(_ALL_SEQUENCES)
    assert (len(train), len(val)) == (190, 35)


@requires_dataset
def test_block_split_is_deterministic_and_split_seed_moves_the_block():
    train, val = split_samples_by_frame_blocks(_roots(), 5, 3, split_seed=0)
    assert (train, val) == split_samples_by_frame_blocks(_roots(), 5, 3, split_seed=0)
    assert split_samples_by_frame_blocks(_roots(), 5, 3, split_seed=7)[1] != val


@requires_dataset
def test_block_split_refuses_a_block_that_cannot_fit_with_two_sided_gaps():
    with pytest.raises(ValueError):
        split_samples_by_frame_blocks(_roots(), block_len=30, gap=10, split_seed=0)
