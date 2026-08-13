import pytest
import torch

from tools.train_synwoodscape import compute_occupancy_diagnostics, format_epoch_log


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
