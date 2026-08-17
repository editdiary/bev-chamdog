import io
import math
import re

import pytest
import torch

from projects.common import bev_occupancy_metrics
from projects.common.bev_occupancy_metrics import (
    _color_enabled,
    append_free_metrics,
    run_batch,
    summarize_free_metrics,
)
from tools.train_synwoodscape import (
    compute_deployment_occupancy_metrics,
    compute_drivable_and_obstacle_iou,
    compute_occupancy_diagnostics,
    format_epoch_log,
    summarize_deployment_metrics,
    summarize_occupancy_diagnostics,
    weighted_mean,
)


def test_checkpoint_score_uses_iou_free_when_legacy_iou_mean_prefers_another_model():
    """기존 평균은 free-space가 더 나쁜 모델을 best로 고르는 퇴행을 만든다."""
    better_free_space = bev_occupancy_metrics.select_checkpoint_score(
        d_iou=0.60, o_iou=0.20, free_metrics={"iou_free": 0.85}
    )
    better_legacy_mean = bev_occupancy_metrics.select_checkpoint_score(
        d_iou=0.95, o_iou=0.80, free_metrics={"iou_free": 0.70}
    )

    assert better_free_space == pytest.approx(0.85)
    assert better_free_space > better_legacy_mean


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
    # 직전 best 대비 증감을 같이 보여준다 -- 숫자 두 개를 눈으로 빼지 않아도 되도록.
    assert lines[0] == (
        "epoch 012/60 | time  138.2s | val_iou_free↑ 0.624 (+0.014) | "
        "best_val_iou_free↑ 0.624 | checkpoint: new best"
    )
    assert lines[1] == (
        "  train | loss_total↓ 0.4821 | loss_occ↓ 0.1032 | loss_vis↓ 0.7578 | "
        "iou_drivable↑ 0.812 | iou_obstacle↑ 0.436 | vis_false_high↓ 0.021 | vis_false_low↓ 0.184"
    )
    assert lines[2] == (
        "  val   | loss_total↓ 0.5014 | loss_occ↓ 0.1110 | loss_vis↓ 0.7808 | "
        "iou_drivable↑ 0.795 | iou_obstacle↑ 0.453 | vis_false_high↓ 0.025 | vis_false_low↓ 0.197"
    )


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _sample_epoch_log():
    return format_epoch_log(
        epoch=3, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_occ_loss=0.1, train_vis_loss=0.1,
        train_d_iou=0.9, train_o_iou=0.5, train_v_false_high=0.0, train_v_false_low=0.0,
        val_loss=0.1, val_occ_loss=0.1, val_vis_loss=0.1,
        val_d_iou=0.9, val_o_iou=0.5, val_v_false_high=0.0, val_v_false_low=0.0,
        val_score=0.7, best_val_score=0.6, is_new_best=True,
    )


def test_color_is_off_when_output_is_redirected(monkeypatch):
    """`python train.py > train.log`로 남긴 로그에 escape sequence가 섞이면 안 된다."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr("sys.stdout", io.StringIO())  # isatty() == False

    assert not _color_enabled()
    assert "\033[" not in _sample_epoch_log()


def test_color_only_adds_escape_codes_and_never_changes_the_text(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    plain = _sample_epoch_log()

    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    colored = _sample_epoch_log()

    assert "\033[" in colored
    assert _ANSI_RE.sub("", colored) == plain


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

    # bin 요약은 헤더에 붙이면 줄이 터미널 폭을 넘겨 val_iou_free가 묻히므로 별도 줄이다.
    assert text.splitlines()[1].startswith("  bins  | val_obst_iou_bins ")
    assert "empty:fa 0.250/n1" in text.splitlines()[1]


# 배포 시에는 GT visibility가 없다. 모델이 "보인다"고 예측한 영역에서만 occupancy를 믿게
# 되므로, 그 영역 기준 IoU가 실제로 쓰이는 성능이다.
DEPLOY_GT_DRIVABLE = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])  # 아래 줄이 obstacle
ALWAYS_DRIVABLE_OCC_LOGITS = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])  # obstacle을 통째로 놓침


def test_deployment_metrics_evaluate_occupancy_only_where_model_claims_visibility():
    vis_logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])  # 윗줄만 보인다고 예측
    valid = torch.ones_like(DEPLOY_GT_DRIVABLE)

    metrics = compute_deployment_occupancy_metrics(
        ALWAYS_DRIVABLE_OCC_LOGITS, vis_logits, DEPLOY_GT_DRIVABLE, valid
    )

    # 놓친 obstacle이 전부 "안 보인다"고 선언한 영역에 있어서 IoU에는 안 잡힌다.
    assert metrics["iou_drivable"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["obstacle_count"] == 0
    # 그래서 coverage를 같이 봐야 한다 -- 시야를 좁게 부르면 점수는 쉽게 올라간다.
    assert metrics["claimed_visible_cells"] == pytest.approx(2.0)
    assert metrics["valid_cells"] == pytest.approx(4.0)


def test_deployment_metrics_penalize_occupancy_error_inside_claimed_visibility():
    vis_logits = torch.full_like(DEPLOY_GT_DRIVABLE, 10.0)  # 전부 보인다고 예측
    valid = torch.ones_like(DEPLOY_GT_DRIVABLE)

    metrics = compute_deployment_occupancy_metrics(
        ALWAYS_DRIVABLE_OCC_LOGITS, vis_logits, DEPLOY_GT_DRIVABLE, valid
    )

    assert metrics["iou_drivable"] == pytest.approx(0.5, abs=1e-3)
    assert metrics["iou_obstacle"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["obstacle_count"] == 1


def test_summarize_deployment_metrics_reports_visible_coverage():
    valid = torch.ones_like(DEPLOY_GT_DRIVABLE)
    narrow = compute_deployment_occupancy_metrics(
        ALWAYS_DRIVABLE_OCC_LOGITS,
        torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]]),
        DEPLOY_GT_DRIVABLE,
        valid,
    )
    wide = compute_deployment_occupancy_metrics(
        ALWAYS_DRIVABLE_OCC_LOGITS, torch.full_like(DEPLOY_GT_DRIVABLE, 10.0), DEPLOY_GT_DRIVABLE, valid
    )

    summary = summarize_deployment_metrics([narrow, wide])

    assert summary["visible_coverage"] == pytest.approx(6 / 8)
    assert summary["iou_drivable"] == pytest.approx(0.75, abs=1e-3)
    # obstacle IoU는 obstacle이 실제로 평가된 샘플만으로 평균낸다.
    assert summary["iou_obstacle"] == pytest.approx(0.0, abs=1e-3)


def test_weighted_mean_skips_batches_without_valid_samples():
    assert weighted_mean([0.8, float("nan")], [4, 0]) == pytest.approx(0.8)
    assert math.isnan(weighted_mean([], []))
    assert weighted_mean([0.9, 0.5], [1, 3]) == pytest.approx(0.6)


def test_summarize_free_metrics_weights_iou_by_sample_count():
    dicts = [
        {"iou_free": 0.8, "iou_free_count": 3, "fatal_rate": 0.1, "fatal_denom": 100,
         "free_miss_rate": 0.2, "free_miss_denom": 10, "partition_defects": 0},
        {"iou_free": 0.4, "iou_free_count": 1, "fatal_rate": 0.5, "fatal_denom": 100,
         "free_miss_rate": 0.6, "free_miss_denom": 90, "partition_defects": 0},
    ]

    merged = summarize_free_metrics(dicts)

    assert merged["iou_free"] == pytest.approx((0.8 * 3 + 0.4 * 1) / 4)
    assert merged["fatal_rate"] == pytest.approx(0.3)
    assert merged["free_miss_rate"] == pytest.approx((0.2 * 10 + 0.6 * 90) / 100)


def test_summarize_free_metrics_of_an_empty_epoch_is_nan():
    merged = summarize_free_metrics([])

    assert math.isnan(merged["iou_free"])


def test_append_free_metrics_drops_batch_masks_before_epoch_accumulation():
    pred_free = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    gt_free = torch.zeros_like(pred_free)
    batches = []

    append_free_metrics(batches, {
        "iou_free": 0.5, "iou_free_count": 1,
        "fatal_rate": 0.25, "fatal_denom": 4,
        "free_miss_rate": 0.75, "free_miss_denom": 2,
        "partition_defects": 0,
        "pred_free": pred_free, "gt_free": gt_free,
    })

    assert batches == [{
        "iou_free": 0.5, "iou_free_count": 1,
        "fatal_rate": 0.25, "fatal_denom": 4,
        "free_miss_rate": 0.75, "free_miss_denom": 2,
        "partition_defects": 0,
    }]


def test_epoch_log_shows_iou_free_with_the_baseline_delta():
    """baseline 없이 free IoU만 표시하면 트리비얼 해와의 비교를 놓친다."""
    text = format_epoch_log(
        epoch=46, num_epochs=60, epoch_time=41.0,
        train_loss=0.03, train_occ_loss=0.003, train_vis_loss=0.06,
        train_d_iou=0.98, train_o_iou=0.78, train_v_false_high=0.001, train_v_false_low=0.02,
        val_loss=0.27, val_occ_loss=0.20, val_vis_loss=0.14,
        val_d_iou=0.89, val_o_iou=0.31, val_v_false_high=0.016, val_v_false_low=0.10,
        train_free_metrics={"iou_free": 0.97, "fatal_rate": 0.01,
                            "free_miss_rate": 0.02, "partition_defects": 0},
        val_free_metrics={"iou_free": 0.850, "fatal_rate": 0.0587,
                          "free_miss_rate": 0.0998, "partition_defects": 0},
        baseline_iou_free=0.673,
        val_score=0.850, best_val_score=0.840, is_new_best=True,
    )

    assert "iou_free" in text
    assert "0.850" in text
    assert "+0.177" in text
    assert "fatal" in text


def test_epoch_log_uses_free_iou_for_checkpoint_and_baseline_comparison():
    """checkpoint와 baseline 비교가 같은 free-space 목표를 가리켜야 한다."""
    text = format_epoch_log(
        epoch=1, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_occ_loss=0.1, train_vis_loss=0.1,
        train_d_iou=0.9, train_o_iou=0.3, train_v_false_high=0.1, train_v_false_low=0.1,
        val_loss=0.1, val_occ_loss=0.1, val_vis_loss=0.1,
        val_d_iou=0.9, val_o_iou=0.3, val_v_false_high=0.1, val_v_false_low=0.1,
        train_free_metrics={"iou_free": 0.9, "fatal_rate": 0.1,
                            "free_miss_rate": 0.1, "partition_defects": 0},
        val_free_metrics={"iou_free": 0.850, "fatal_rate": 0.1,
                          "free_miss_rate": 0.1, "partition_defects": 0},
        baseline_iou_free=0.673,
        val_score=0.600, best_val_score=0.500, is_new_best=True,
    )

    assert "val_iou_free" in text
    assert "iou_free↑ 0.850" in text
    assert "+0.177 vs baseline" in text


def test_epoch_log_flags_a_broken_partition_loudly():
    """free/occupied/unknown 분할 결함은 epoch 로그에서 눈에 띄어야 한다."""
    text = format_epoch_log(
        epoch=1, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_occ_loss=0.1, train_vis_loss=0.1,
        train_d_iou=0.1, train_o_iou=0.1, train_v_false_high=0.1, train_v_false_low=0.1,
        val_loss=0.1, val_occ_loss=0.1, val_vis_loss=0.1,
        val_d_iou=0.1, val_o_iou=0.1, val_v_false_high=0.1, val_v_false_low=0.1,
        train_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                            "free_miss_rate": 0.1, "partition_defects": 7},
        val_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                          "free_miss_rate": 0.1, "partition_defects": 0},
        val_score=0.1, best_val_score=0.0, is_new_best=True,
    )

    assert "partition" in text.lower()


class _CpuRunBatchModel:
    def __call__(self, rgb_camxs, pix_t_cams, cam0_t_camxs, vox_util):
        logits = torch.tensor([[[[3.0, -3.0], [3.0, 3.0]],
                               [[3.0, 3.0], [-3.0, 3.0]]]])
        return None, None, logits, None, None


def test_run_batch_returns_free_metrics_as_ninth_result_on_cpu():
    dummy = torch.zeros(1, 1, 1, 1, 1)
    batch = {
        "rgb_camXs": dummy,
        "pix_T_cams": dummy,
        "cam0_T_camXs": dummy,
        "seg_bev_g": torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]]),
        "vis_bev_g": torch.ones(1, 1, 2, 2),
        "valid_bev_g": torch.ones(1, 1, 2, 2),
    }

    result = run_batch(
        _CpuRunBatchModel(), batch, vox_util=None, pos_weight_tensor=torch.tensor(1.0),
        device="cpu", lambda_vis=0.5, vis_neg_weight=3.0,
    )

    assert len(result) == 9
    free_metrics = result[-1]
    assert {"iou_free", "iou_free_count", "fatal_rate", "fatal_denom",
            "free_miss_rate", "free_miss_denom", "partition_defects",
            "pred_free", "gt_free"} <= free_metrics.keys()
    assert free_metrics["pred_free"].dtype == torch.bool
    assert free_metrics["pred_free"].shape == (1, 1, 2, 2)
