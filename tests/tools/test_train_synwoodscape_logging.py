import math

import pytest
import torch

from tools.train_synwoodscape import (
    compute_drivable_and_obstacle_iou,
    compute_occupancy_diagnostics,
    format_epoch_log,
    summarize_occupancy_diagnostics,
    weighted_mean,
)


def test_format_epoch_log_separates_epoch_train_and_val_with_metric_directions():
    text = format_epoch_log(
        epoch=12,
        num_epochs=60,
        epoch_time=138.2,
        train_loss=0.4821,
        train_occ_loss=0.1032,
        train_vis_loss=0.7578,
        train_d_iou=0.812,
        train_o_iou=0.436,
        train_v_false_high=0.021,
        train_v_false_low=0.184,
        val_loss=0.5014,
        val_occ_loss=0.1110,
        val_vis_loss=0.7808,
        val_d_iou=0.795,
        val_o_iou=0.453,
        val_v_false_high=0.025,
        val_v_false_low=0.197,
        val_score=0.624,
        best_val_score=0.610,
        is_new_best=True,
    )

    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0] == "epoch 012/60 | time  138.2s | val_iou_mean↑ 0.624 | best_val_iou_mean↑ 0.624 | checkpoint: new best"
    assert lines[1] == (
        "  train | loss_total↓ 0.4821 | loss_occ↓ 0.1032 | loss_vis↓ 0.7578 | "
        "iou_drivable↑ 0.812 | iou_obstacle↑ 0.436 | vis_false_high↓ 0.021 | vis_false_low↓ 0.184"
    )
    assert lines[2] == (
        "  val   | loss_total↓ 0.5014 | loss_occ↓ 0.1110 | loss_vis↓ 0.7808 | "
        "iou_drivable↑ 0.795 | iou_obstacle↑ 0.453 | vis_false_high↓ 0.025 | vis_false_low↓ 0.197"
    )


def test_compute_occupancy_diagnostics_reports_error_direction_and_obstacle_bins():
    logits = torch.tensor([[[[10.0, -10.0, -10.0], [10.0, 10.0, -10.0]]]])
    gt_drivable = torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
    valid = torch.ones_like(gt_drivable)

    metrics = compute_occupancy_diagnostics(logits, gt_drivable, valid)

    assert metrics["obstacle_frac"] == pytest.approx(0.5)
    assert metrics["false_obstacle"] == pytest.approx(1 / 3)
    assert metrics["missed_obstacle"] == pytest.approx(1 / 3)
    assert metrics["obstacle_iou_large"] == pytest.approx(0.5, abs=1e-4)
    assert metrics["obstacle_count_large"] == 1
    assert metrics["obstacle_count_empty"] == 0


# GT에 obstacle이 없는 샘플은 intersection이 항상 0이라 IoU가 0으로 고정된다 -- 완벽하게
# 맞혀도 0점이므로 obstacle IoU 평균에서 빼고, 대신 empty_false_alarm으로 따로 본다.
PERFECT_HALF_OBSTACLE_LOGITS = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
PERFECT_HALF_OBSTACLE_GT = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
ALL_DRIVABLE_LOGITS = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
ALL_DRIVABLE_GT = torch.ones(1, 1, 2, 2)


def test_obstacle_iou_ignores_samples_without_obstacle_in_gt():
    logits = torch.cat([PERFECT_HALF_OBSTACLE_LOGITS, ALL_DRIVABLE_LOGITS])
    gt_drivable = torch.cat([PERFECT_HALF_OBSTACLE_GT, ALL_DRIVABLE_GT])
    valid = torch.ones_like(gt_drivable)

    _, obstacle_iou, obstacle_count = compute_drivable_and_obstacle_iou(logits, gt_drivable, valid)

    assert obstacle_count == 1
    assert float(obstacle_iou) == pytest.approx(1.0, abs=1e-3)


def test_obstacle_iou_is_nan_when_no_sample_has_obstacle_in_gt():
    valid = torch.ones_like(ALL_DRIVABLE_GT)

    _, obstacle_iou, obstacle_count = compute_drivable_and_obstacle_iou(
        ALL_DRIVABLE_LOGITS, ALL_DRIVABLE_GT, valid
    )

    assert obstacle_count == 0
    assert math.isnan(float(obstacle_iou))


def test_occupancy_diagnostics_report_false_alarm_on_obstacle_free_samples():
    logits = torch.tensor([[[[10.0, 10.0], [10.0, -10.0]]]])  # 4칸 중 1칸을 obstacle로 오탐
    valid = torch.ones_like(ALL_DRIVABLE_GT)

    metrics = compute_occupancy_diagnostics(logits, ALL_DRIVABLE_GT, valid)

    assert metrics["obstacle_count_empty"] == 1
    assert metrics["empty_false_alarm"] == pytest.approx(0.25)
    assert metrics["empty_false_alarm_samples"] == 1
    # empty 샘플의 IoU는 구조적으로 0이라 의미가 없다 -- 값을 만들지 않는다.
    assert math.isnan(metrics["obstacle_iou_empty"])


def test_summarize_occupancy_diagnostics_aggregates_false_alarm_across_batches():
    valid = torch.ones_like(ALL_DRIVABLE_GT)
    clean = compute_occupancy_diagnostics(ALL_DRIVABLE_LOGITS, ALL_DRIVABLE_GT, valid)
    noisy = compute_occupancy_diagnostics(
        torch.tensor([[[[10.0, 10.0], [10.0, -10.0]]]]), ALL_DRIVABLE_GT, valid
    )

    summary = summarize_occupancy_diagnostics([clean, noisy])

    assert summary["obstacle_count_empty"] == 2
    assert summary["empty_false_alarm"] == pytest.approx(1 / 8)  # 오탐 1칸 / valid 8칸
    assert summary["empty_false_alarm_samples"] == 1


def test_epoch_log_shows_false_alarm_instead_of_iou_for_empty_bin():
    valid = torch.ones_like(ALL_DRIVABLE_GT)
    metrics = summarize_occupancy_diagnostics(
        [compute_occupancy_diagnostics(torch.tensor([[[[10.0, 10.0], [10.0, -10.0]]]]), ALL_DRIVABLE_GT, valid)]
    )

    text = format_epoch_log(
        epoch=1,
        num_epochs=60,
        epoch_time=1.0,
        train_loss=0.1,
        train_occ_loss=0.1,
        train_vis_loss=0.1,
        train_d_iou=0.9,
        train_o_iou=0.5,
        train_v_false_high=0.0,
        train_v_false_low=0.0,
        val_loss=0.1,
        val_occ_loss=0.1,
        val_vis_loss=0.1,
        val_d_iou=0.9,
        val_o_iou=0.5,
        val_v_false_high=0.0,
        val_v_false_low=0.0,
        val_occ_metrics=metrics,
        val_score=0.7,
        best_val_score=0.6,
        is_new_best=True,
    )

    assert "empty:fa 0.250/n1" in text.splitlines()[0]


def test_weighted_mean_skips_batches_without_valid_samples():
    assert weighted_mean([0.8, float("nan")], [4, 0]) == pytest.approx(0.8)
    assert math.isnan(weighted_mean([], []))
    assert weighted_mean([0.9, 0.5], [1, 3]) == pytest.approx(0.6)
