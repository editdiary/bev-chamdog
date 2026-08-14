from pathlib import Path

import numpy as np
import pytest

from projects.bev_gt.grid import ROBOT_GRID_SPEC
from projects.bev_gt.ipm import render_ipm
from projects.datasets.robot_simplebev import split_samples_within_sequences
from projects.geometry.double_sphere import (
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
)
from tools.visualize_robot_predictions import _sample_scores

COMMON_ROOT = Path("dataset/sj_datasets/common")
SEQUENCE_ROOT = Path("dataset/sj_datasets/raws1")
requires_dataset = pytest.mark.skipif(
    not (SEQUENCE_ROOT / "occupancy_npy").exists() or not COMMON_ROOT.exists(),
    reason="self-collected dataset not available locally",
)


def test_sample_scores_rank_worse_samples_higher():
    mask = np.ones((4, 4), bool)
    gt = np.ones((4, 4), np.uint8)
    gt[0] = 0  # 4 obstacle cells

    perfect = _sample_scores(gt.copy(), gt, mask)
    missed_all = _sample_scores(np.ones((4, 4), np.uint8), gt, mask)

    assert missed_all["missed_obstacle"] > perfect["missed_obstacle"]
    # obstacle_iou는 부호를 뒤집어 저장하므로 "클수록 나쁨"이어야 한다.
    assert missed_all["obstacle_iou"] > perfect["obstacle_iou"]
    assert perfect["_missed"] == 0 and missed_all["_missed"] == 4


def test_samples_without_gt_obstacles_sink_to_the_bottom():
    """GT에 obstacle이 없으면 IoU가 구조적으로 0이라, 최악 순위를 독차지하면 안 된다."""
    mask = np.ones((4, 4), bool)
    all_drivable = np.ones((4, 4), np.uint8)
    scores = _sample_scores(all_drivable.copy(), all_drivable, mask)

    assert scores["obstacle_iou"] == -np.inf
    assert scores["_obstacle_iou_text"].startswith("n/a")

    gt_with_obstacle = np.ones((4, 4), np.uint8)
    gt_with_obstacle[0] = 0
    worst_real = _sample_scores(np.ones((4, 4), np.uint8), gt_with_obstacle, mask)
    assert worst_real["obstacle_iou"] > scores["obstacle_iou"]


def test_false_alarm_is_measured_against_drivable_cells():
    mask = np.ones((4, 4), bool)
    gt = np.ones((4, 4), np.uint8)  # 전부 drivable
    pred = np.ones((4, 4), np.uint8)
    pred[0] = 0  # 4칸을 obstacle로 잘못 예측
    scores = _sample_scores(pred, gt, mask)
    assert scores["_false"] == 4
    assert scores["false_obstacle"] == pytest.approx(0.25)


@requires_dataset
def test_render_ipm_fills_the_camera_covered_area():
    from PIL import Image

    cameras = load_cameras(COMMON_ROOT / "calibration/calib.yaml")
    ego_T_cams = load_ego_T_cams(COMMON_ROOT / "calibration/calib.yaml")
    sample_id = sorted(p.stem for p in (SEQUENCE_ROOT / "occupancy_npy").glob("*.npy"))[0]
    images = {
        name: np.asarray(Image.open(
            SEQUENCE_ROOT / "rgb_images" / sample_id / f"cam_{name}.jpg").convert("RGB"))
        for name in FINETUNE_CAMERA_NAMES
    }

    ipm = render_ipm(images, cameras, ego_T_cams, ROBOT_GRID_SPEC)

    assert ipm.shape == (ROBOT_GRID_SPEC.n_rows, ROBOT_GRID_SPEC.n_cols, 3)
    filled = ipm.any(axis=2)
    # 화각 밖(원점 주변 원반)은 비고 나머지는 대부분 채워져야 한다.
    assert 0.85 < filled.mean() < 1.0
    # ego 원점은 그리드 중앙이 아니다 -- ROI가 전후 비대칭(전방 4 m / 후방 2 m)이라
    # x=0은 row (front_m / cell_m)에 온다.
    origin_row = int(ROBOT_GRID_SPEC.front_m / ROBOT_GRID_SPEC.cell_m)
    origin_col = int(ROBOT_GRID_SPEC.half_width_m / ROBOT_GRID_SPEC.cell_m)
    assert not filled[origin_row - 2:origin_row + 2, origin_col - 2:origin_col + 2].any()


@requires_dataset
def test_tail_split_is_contiguous_and_disjoint():
    train, val = split_samples_within_sequences([SEQUENCE_ROOT], tail_fraction=0.2)
    all_samples = sorted(p.stem for p in (SEQUENCE_ROOT / "occupancy_npy").glob("*.npy"))

    assert len(train) + len(val) == len(all_samples)
    assert not set(train) & set(val)
    # val은 시퀀스 끝의 연속 구간이어야 한다.
    assert [s for _, s in val] == all_samples[-len(val):]
    assert [s for _, s in train] == all_samples[:len(train)]


def test_tail_split_never_empties_training(tmp_path):
    (tmp_path / "occupancy_npy").mkdir(parents=True)
    for i in range(3):
        np.save(tmp_path / "occupancy_npy" / f"sample_{i:06d}.npy", np.zeros((2, 2), np.uint8))

    train, val = split_samples_within_sequences([tmp_path], tail_fraction=0.99)
    assert len(train) >= 1
    assert len(val) == len(train) + len(val) - len(train)

    with pytest.raises(ValueError):
        split_samples_within_sequences([tmp_path], tail_fraction=1.0)
