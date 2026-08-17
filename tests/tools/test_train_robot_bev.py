from pathlib import Path

import numpy as np
import pytest
import torch

from tools.train_robot_bev import (
    _baseline_iou_free,
    _write_epoch_metric_scalars,
    _write_free_space_scalars,
    compute_label_statistics,
)

DATASET_ROOT = Path("dataset/sj_datasets")
SEQUENCE_ROOT = DATASET_ROOT / "raws1"
requires_dataset = pytest.mark.skipif(
    not (SEQUENCE_ROOT / "occupancy_npy").exists(),
    reason="self-collected dataset not available locally",
)


def _write_sample(root, sample_id, occupancy, visible):
    (root / "occupancy_npy").mkdir(parents=True, exist_ok=True)
    (root / "visibility_npy").mkdir(parents=True, exist_ok=True)
    np.save(root / "occupancy_npy" / f"{sample_id}.npy", occupancy.astype(np.uint8))
    np.save(root / "visibility_npy" / f"{sample_id}.npy", visible.astype(np.uint8))


def test_label_statistics_count_only_masked_cells(tmp_path):
    """`pos_weight`는 loss가 실제로 보는 셀에서만 세어야 한다.

    마스크 밖 라벨을 같이 세면 클래스 보정이 통째로 어긋난다 -- 자체 데이터셋에서는
    마스킹 전후로 obstacle 비율이 몇 배 차이 난다.
    """
    occupancy = np.zeros((4, 4), bool)
    occupancy[0] = True                       # 4 drivable, 12 obstacle (마스킹 전)
    visible = np.ones((4, 4), bool)
    _write_sample(tmp_path, "sample_000000", occupancy, visible)

    permanent_blind = np.zeros((4, 4), bool)
    permanent_blind[2:] = True                # 아래 절반은 영구 사각 -> vis=0
    invalid = np.zeros((4, 4), bool)

    stats = compute_label_statistics([(tmp_path, "sample_000000")], permanent_blind, invalid)

    # 남는 셀은 위 두 행(8개): drivable 4, obstacle 4.
    assert stats["pos_weight"] == pytest.approx(1.0)
    assert stats["trivial_iou"] == pytest.approx(0.5)
    assert stats["supervised_fraction"] == pytest.approx(0.5)
    assert stats["obstacle_fraction"] == pytest.approx(0.5)

    # 마스킹을 무시했다면 pos_weight = 12/4 = 3.0이 나왔을 것이다.
    unmasked = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), invalid
    )
    assert unmasked["pos_weight"] == pytest.approx(3.0)


def test_invalid_mask_also_removes_cells(tmp_path):
    occupancy = np.ones((4, 4), bool)
    visible = np.ones((4, 4), bool)
    _write_sample(tmp_path, "sample_000000", occupancy, visible)

    invalid = np.zeros((4, 4), bool)
    invalid[:, :2] = True
    stats = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), invalid
    )
    assert stats["supervised_fraction"] == pytest.approx(0.5)


def test_label_statistics_handles_a_fully_masked_sample(tmp_path):
    """전부 마스킹된 샘플에서 0으로 나누지 않아야 한다."""
    _write_sample(tmp_path, "sample_000000", np.ones((4, 4), bool), np.zeros((4, 4), bool))
    stats = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    )
    assert stats["supervised_fraction"] == 0.0
    assert np.isfinite(stats["pos_weight"])


def test_constant_map_baseline_scores_masked_validation_free_space_on_cpu(tmp_path):
    """NaN baseline은 epoch 로그의 모델-vs-layout 비교를 무력화한다."""
    _write_sample(tmp_path, "sample_000000", np.array([[1, 0], [0, 0]], bool), np.ones((2, 2), bool))
    _write_sample(tmp_path, "sample_000001", np.ones((2, 2), bool), np.ones((2, 2), bool))

    score = _baseline_iou_free(
        [(tmp_path, "sample_000000"), (tmp_path, "sample_000001")],
        permanent_blind=np.zeros((2, 2), bool),
        invalid=np.zeros((2, 2), bool),
        constant_map=np.array([[1, 0], [0, 0]], bool),
        device="cpu",
    )

    assert score == pytest.approx(0.625)


class _ScalarWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_free_space_scalars_include_train_free_and_validation_range_and_rings():
    """free/range/ring metrics가 writer 경로에서 빠지면 TensorBoard에 관측값이 없다."""
    writer = _ScalarWriter()
    free = {"iou_free": 0.8, "fatal_rate": 0.1, "free_miss_rate": 0.2}
    range_metrics = {"abs_p50": 0.3, "abs_p90": 0.6, "over_mean": 0.2, "under_mean": 0.1}
    rings = {"0-1m": {"iou_free": 0.9}}

    _write_free_space_scalars(writer, "train", free, epoch=4)
    _write_free_space_scalars(writer, "val", free, epoch=4,
                              range_metrics=range_metrics, ring_metrics=rings)

    assert writer.scalars == [
        ("train/iou_free_epoch", 0.8, 4),
        ("train/fatal_rate_epoch", 0.1, 4),
        ("train/free_miss_rate_epoch", 0.2, 4),
        ("val/iou_free_epoch", 0.8, 4),
        ("val/fatal_rate_epoch", 0.1, 4),
        ("val/free_miss_rate_epoch", 0.2, 4),
        ("val/range_abs_p50_epoch", 0.3, 4),
        ("val/range_abs_p90_epoch", 0.6, 4),
        ("val/range_over_epoch", 0.2, 4),
        ("val/range_under_epoch", 0.1, 4),
        ("val/ring_0-1m_iou_free_epoch", 0.9, 4),
    ]


def test_epoch_metric_scalars_skip_non_finite_legacy_values():
    writer = _ScalarWriter()
    metrics = {
        "loss": 1.0, "loss_occ": float("nan"), "loss_vis": 0.5,
        "d_iou": float("nan"), "o_iou": float("nan"),
        "false_high": float("nan"), "false_low": float("nan"),
    }

    _write_epoch_metric_scalars(writer, "train", metrics, epoch=4)

    assert writer.scalars == [
        ("train/loss_epoch", 1.0, 4),
        ("train/loss_vis_epoch", 0.5, 4),
    ]


def test_occupancy_diagnostic_scalars_skip_non_finite_values():
    from projects.common.bev_occupancy_metrics import (
        summarize_occupancy_diagnostics,
        write_occupancy_diagnostics,
    )

    writer = _ScalarWriter()

    write_occupancy_diagnostics(writer, "train", summarize_occupancy_diagnostics([]), epoch=4)

    assert ("train/occupancy_obstacle_count_empty_epoch", 0, 4) in writer.scalars
    assert all(np.isfinite(value) for _, value, _ in writer.scalars)


def test_normalise_step_output_preserves_three_class_free_metrics():
    from tools.train_robot_bev import _normalise_step_output

    loss = torch.tensor(1.0)
    parts = {"loss_free": torch.tensor(0.1)}
    free = {"iou_free": 0.7}

    assert _normalise_step_output("three_class", (loss, parts, free)) == (loss, parts, free, None)


def test_normalise_step_output_preserves_two_head_legacy_diagnostics():
    from tools.train_robot_bev import _normalise_step_output

    loss = torch.tensor(1.0)
    parts = {"loss_occ": torch.tensor(0.4), "loss_vis": torch.tensor(0.6)}
    free = {"iou_free": 0.2}
    output = (
        loss, parts, torch.tensor(0.95), torch.tensor(0.80), 3,
        {"false_high": 0.1, "false_low": 0.2},
        {"obstacle_frac": 0.3},
        {"visible_coverage": 0.4},
        free,
    )

    normalised_loss, normalised_parts, normalised_free, legacy = _normalise_step_output("two_head", output)

    assert normalised_loss is loss
    assert normalised_parts is parts
    assert normalised_free is free
    assert legacy["d_iou"] == pytest.approx(0.95)
    assert legacy["o_iou"] == pytest.approx(0.80)
    assert legacy["o_count"] == 3
    assert legacy["vis"] == {"false_high": 0.1, "false_low": 0.2}
    assert legacy["occ"] == {"obstacle_frac": 0.3}
    assert legacy["deploy"] == {"visible_coverage": 0.4}


def test_main_rejects_unknown_head_before_touching_the_dataset(tmp_path):
    import tools.train_robot_bev as trainer

    with pytest.raises(ValueError, match="two_head.*three_class"):
        trainer.main(head="bogus", dataset_root=tmp_path, device="cpu")


@requires_dataset
def test_real_dataset_statistics_are_in_the_expected_range():
    """실제 라벨에서 마스킹 후 분포가 예상 범위인지 -- 마스크가 어긋나면 여기서 드러난다."""
    from projects.datasets.robot_simplebev import (
        DEFAULT_COMMON_ROOT,
        GRID_SPEC,
        build_bev_masks,
        list_sequence_samples,
    )
    from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES

    permanent_blind, invalid = build_bev_masks(
        DEFAULT_COMMON_ROOT, GRID_SPEC, FINETUNE_CAMERA_NAMES
    )
    stats = compute_label_statistics(
        list_sequence_samples(SEQUENCE_ROOT), permanent_blind, invalid
    )
    # occupancy loss가 덮는 셀은 그리드의 일부지만 0은 아니어야 한다.
    assert 0.10 < stats["supervised_fraction"] < 0.30
    # 마스킹된 영역 안에서는 obstacle이 소수 클래스다(통로가 대부분).
    assert 0.01 < stats["obstacle_fraction"] < 0.20
    assert stats["pos_weight"] < 1.0  # drivable이 다수 -> neg/pos < 1
