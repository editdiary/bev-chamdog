from pathlib import Path

import numpy as np

from projects.bev_gt.grid import SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC
from projects.datasets.synwoodscape_simplebev import SynWoodScapeSimpleBEVDataset


def test_synwoodscape_dataset_loads_two_head_labels_from_training_package(tmp_path):
    dataset_root = tmp_path / "synwoodscape"
    rgb_root = dataset_root / "rgb_images"
    calib_root = dataset_root / "calibration_data"
    rgb_root.mkdir(parents=True)
    calib_root.mkdir(parents=True)

    occupancy_root = tmp_path / "labels"
    occupancy_root.mkdir()
    shape = (
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_rows,
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_cols,
    )
    visible = np.ones(shape, dtype=bool)
    visible[:10] = False
    np.save(occupancy_root / "00000_occupancy.npy", np.ones(shape, dtype=np.uint8))
    np.save(occupancy_root / "00000_visible.npy", visible)

    ds = SynWoodScapeSimpleBEVDataset(
        ["00000"],
        dataset_root=dataset_root,
        occupancy_gt_root=occupancy_root,
        camera_names=(),
    )
    item = ds[0]

    assert item["seg_bev_g"].shape == (1, 240, 240)
    assert item["vis_bev_g"].shape == item["seg_bev_g"].shape
    assert item["valid_bev_g"].shape == item["seg_bev_g"].shape
    assert item["vis_bev_g"][0, :10].max() == 0  # 실제 visibility label을 그대로 싣는다


def test_valid_mask_covers_whole_roi_and_is_not_the_visibility_mask(tmp_path):
    """`valid`(라벨이 존재하는 범위)와 `vis`(관측 가능 여부)는 다른 개념이다.

    둘을 같은 배열로 주면 visibility loss가 positive 셀만 보고 학습돼서 "전부 visible"이
    전역 최적해가 된다. SynWoodScape는 ROI 전체가 라벨링돼 있으므로 valid는 전부 1이다.
    """
    dataset_root = tmp_path / "synwoodscape"
    (dataset_root / "rgb_images").mkdir(parents=True)
    (dataset_root / "calibration_data").mkdir(parents=True)
    occupancy_root = tmp_path / "labels"
    occupancy_root.mkdir()
    shape = (
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_rows,
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_cols,
    )
    visible = np.ones(shape, dtype=bool)
    visible[:10] = False
    np.save(occupancy_root / "00000_occupancy.npy", np.ones(shape, dtype=np.uint8))
    np.save(occupancy_root / "00000_visible.npy", visible)

    ds = SynWoodScapeSimpleBEVDataset(
        ["00000"],
        dataset_root=dataset_root,
        occupancy_gt_root=occupancy_root,
        camera_names=(),
    )
    item = ds[0]

    assert item["valid_bev_g"].min() == 1
    assert item["valid_bev_g"].max() == 1
    assert not np.array_equal(item["valid_bev_g"].numpy(), item["vis_bev_g"].numpy())


def test_photometric_augmentation_never_touches_the_bev_labels(tmp_path):
    """광도 augmentation은 기하를 안 바꾸므로 BEV GT는 그대로여야 한다."""
    dataset_root = tmp_path / "synwoodscape"
    (dataset_root / "rgb_images").mkdir(parents=True)
    (dataset_root / "calibration_data").mkdir(parents=True)
    occupancy_root = tmp_path / "labels"
    occupancy_root.mkdir()
    shape = (
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_rows,
        SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC.n_cols,
    )
    occupancy = np.zeros(shape, dtype=np.uint8)
    occupancy[20:40] = 1
    visible = np.ones(shape, dtype=bool)
    visible[:10] = False
    np.save(occupancy_root / "00000_occupancy.npy", occupancy)
    np.save(occupancy_root / "00000_visible.npy", visible)

    def build(augment):
        return SynWoodScapeSimpleBEVDataset(
            ["00000"],
            dataset_root=dataset_root,
            occupancy_gt_root=occupancy_root,
            camera_names=(),
            augment=augment,
        )[0]

    plain, augmented = build(False), build(True)

    for key in ("seg_bev_g", "vis_bev_g", "valid_bev_g"):
        assert np.array_equal(plain[key].numpy(), augmented[key].numpy())
