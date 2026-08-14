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
