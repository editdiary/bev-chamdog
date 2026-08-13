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
    np.save(occupancy_root / "00000_occupancy.npy", np.ones(shape, dtype=np.uint8))
    np.save(occupancy_root / "00000_visible.npy", np.ones(shape, dtype=bool))

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
