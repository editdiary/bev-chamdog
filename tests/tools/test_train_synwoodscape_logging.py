"""pretrain/fine-tuning이 공유하는 보고 계층(`projects.common.bev_occupancy_metrics`) 테스트.

2-head가 제거되면서 여기 있던 occupancy 진단·deployment 지표·2-head `run_batch` 테스트는
같이 사라졌다. 남은 것은 두 학습 경로가 공유하는 것뿐이다: epoch 로그 형식, 체크포인트 선택
기준, free-space 지표의 epoch 집계, 그리고 3-class loss 항의 epoch 평균.
"""
import io
import math
import re

import pytest
import torch

from projects.common import bev_occupancy_metrics
from projects.common.bev_occupancy_metrics import (
    _color_enabled,
    append_free_metrics,
    mean_loss_parts,
    summarize_free_metrics,
)
from tools.train_synwoodscape import format_epoch_log, weighted_mean

_LOSS_PARTS = {"loss_unknown": 0.1032, "loss_free": 0.7578, "loss_occupied": 1.2044}


def test_checkpoint_score_is_iou_free():
    """체크포인트 선택은 `iou_free` 하나로 정해진다 -- 옛 `0.5*(d_iou+o_iou)` 평균이 아니다."""
    assert bev_occupancy_metrics.select_checkpoint_score(
        {"iou_free": 0.85}
    ) == pytest.approx(0.85)
    # 값이 그대로 흐르는지: 다른 키가 섞여 있어도 iou_free만 본다.
    assert bev_occupancy_metrics.select_checkpoint_score(
        {"iou_free": 0.70, "fatal_rate": 0.01}
    ) == pytest.approx(0.70)


def test_format_epoch_log_separates_epoch_train_and_val_with_metric_directions():
    text = format_epoch_log(
        epoch=12,
        num_epochs=60,
        epoch_time=138.2,
        train_loss=0.4821,
        train_loss_parts=_LOSS_PARTS,
        val_loss=0.5014,
        val_loss_parts={"loss_unknown": 0.1110, "loss_free": 0.7808,
                        "loss_occupied": 1.3001},
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
        "  train | loss_total↓ 0.4821 | loss_unknown↓ 0.1032 | "
        "loss_free↓ 0.7578 | loss_occupied↓ 1.2044"
    )
    assert lines[2] == (
        "  val   | loss_total↓ 0.5014 | loss_unknown↓ 0.1110 | "
        "loss_free↓ 0.7808 | loss_occupied↓ 1.3001"
    )


def test_epoch_log_shows_all_three_class_loss_terms():
    """세 항이 모두 보여야 한다. 옛 2-head 로그는 칸이 두 개뿐이어서 3-class로 바꾼 뒤
    `unknown` 항이 아예 표시되지 않았다 -- unknown이 셀의 85% 이상인 데이터라 이 항이
    안 보이면 loss가 어디서 오는지 읽을 수 없다."""
    text = format_epoch_log(
        epoch=1, num_epochs=1, epoch_time=1.0,
        train_loss=1.0, train_loss_parts=_LOSS_PARTS,
        val_loss=float("nan"), val_loss_parts={},
        val_score=0.5, best_val_score=0.0, is_new_best=True,
    )

    for name in ("loss_unknown", "loss_free", "loss_occupied"):
        assert name in text, f"{name}이 epoch 로그에 없다"


def test_epoch_log_renders_missing_loss_parts_as_dashes():
    """val을 돌리지 않은 epoch은 `loss_parts`가 빈 dict다. 그때 0.0000이 아니라 `-`가 나와야
    한다 -- 0으로 찍히면 "loss가 0이 됐다"로 오독된다."""
    text = format_epoch_log(
        epoch=1, num_epochs=1, epoch_time=1.0,
        train_loss=1.0, train_loss_parts=_LOSS_PARTS,
        val_loss=float("nan"), val_loss_parts={},
        val_score=float("nan"), best_val_score=0.0, is_new_best=False,
    )

    val_line = [line for line in text.splitlines() if line.startswith("  val")][0]
    assert "loss_unknown↓ -" in val_line
    assert "0.0000" not in val_line


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _sample_epoch_log():
    return format_epoch_log(
        epoch=3, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_loss_parts=_LOSS_PARTS,
        val_loss=0.1, val_loss_parts=_LOSS_PARTS,
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


def test_weighted_mean_skips_batches_without_valid_samples():
    assert weighted_mean([0.8, float("nan")], [4, 0]) == pytest.approx(0.8)
    assert math.isnan(weighted_mean([], []))
    assert weighted_mean([0.9, 0.5], [1, 3]) == pytest.approx(0.6)


def test_mean_loss_parts_averages_each_term_over_batches():
    batches = [
        {"loss_unknown": 1.0, "loss_free": 2.0, "loss_occupied": 3.0},
        {"loss_unknown": 3.0, "loss_free": 4.0, "loss_occupied": 9.0},
    ]

    merged = mean_loss_parts(batches)

    assert merged == pytest.approx(
        {"loss_unknown": 2.0, "loss_free": 3.0, "loss_occupied": 6.0}
    )
    # 마지막 배치만 쓰거나 항끼리 섞이면 위 값이 나오지 않는다.
    assert merged["loss_occupied"] != pytest.approx(9.0)


def test_mean_loss_parts_of_an_empty_epoch_is_empty():
    assert mean_loss_parts([]) == {}


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
        train_loss=0.03, train_loss_parts=_LOSS_PARTS,
        val_loss=0.27, val_loss_parts=_LOSS_PARTS,
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
        train_loss=0.1, train_loss_parts=_LOSS_PARTS,
        val_loss=0.1, val_loss_parts=_LOSS_PARTS,
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
        train_loss=0.1, train_loss_parts=_LOSS_PARTS,
        val_loss=0.1, val_loss_parts=_LOSS_PARTS,
        train_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                            "free_miss_rate": 0.1, "partition_defects": 7},
        val_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                          "free_miss_rate": 0.1, "partition_defects": 0},
        val_score=0.1, best_val_score=0.0, is_new_best=True,
    )

    assert "partition" in text.lower()
