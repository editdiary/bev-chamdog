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
    write_epoch_scalars,
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


def test_summarize_free_metrics_weights_every_metric_by_its_own_frame_count():
    """batch 값은 그 batch의 프레임 평균이다. epoch 값이 **정확한 프레임 macro**가 되려면 각
    지표를 **자기** count(평균에 들어간 프레임 수)로 가중해야 한다. 지표마다 count를 다르게
    줘서, 다른 지표의 count를 쓰거나 batch 수로 평균하면 깨지게 한다."""
    dicts = [
        {"iou_free": 0.8, "iou_free_count": 3, "iou_non_free": 0.9, "iou_non_free_count": 1,
         "precision": 0.1, "precision_count": 2, "recall": 0.2, "recall_count": 4,
         "partition_defects": 0},
        {"iou_free": 0.4, "iou_free_count": 1, "iou_non_free": 0.5, "iou_non_free_count": 3,
         "precision": 0.5, "precision_count": 2, "recall": 0.6, "recall_count": 1,
         "partition_defects": 0},
    ]

    merged = summarize_free_metrics(dicts)

    assert merged["iou_free"] == pytest.approx((0.8 * 3 + 0.4 * 1) / 4)
    assert merged["iou_non_free"] == pytest.approx((0.9 * 1 + 0.5 * 3) / 4)
    assert merged["precision"] == pytest.approx(0.3)
    assert merged["recall"] == pytest.approx((0.2 * 4 + 0.6 * 1) / 5)
    assert "fatal_rate" not in merged and "free_miss_rate" not in merged


def test_summarize_free_metrics_of_an_empty_epoch_is_nan():
    merged = summarize_free_metrics([])

    assert math.isnan(merged["iou_free"])


def test_append_free_metrics_drops_batch_masks_before_epoch_accumulation():
    pred_free = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    gt_free = torch.zeros_like(pred_free)
    batches = []

    scalars = {
        "iou_free": 0.5, "iou_free_count": 1,
        "iou_non_free": 0.9, "iou_non_free_count": 1,
        "precision": 0.25, "precision_count": 1,
        "recall": 0.75, "recall_count": 1,
        "partition_defects": 0,
    }

    append_free_metrics(batches, {
        **scalars,
        # 마스크는 4개 전부 버려져야 한다 -- epoch 내내 들고 있으면 240x240 bool이
        # 배치 수만큼 쌓인다.
        "pred_free": pred_free, "gt_free": gt_free,
        "pred_occupied": pred_free, "gt_occupied": gt_free,
    })

    assert batches == [scalars]


def test_epoch_log_shows_iou_free_with_the_baseline_delta():
    """baseline 없이 free IoU만 표시하면 트리비얼 해와의 비교를 놓친다."""
    text = format_epoch_log(
        epoch=46, num_epochs=60, epoch_time=41.0,
        train_loss=0.03, train_loss_parts=_LOSS_PARTS,
        val_loss=0.27, val_loss_parts=_LOSS_PARTS,
        train_free_metrics={"iou_free": 0.97, "iou_non_free": 0.9, "precision": 0.01,
                            "recall": 0.02, "partition_defects": 0},
        val_free_metrics={"iou_free": 0.850, "iou_non_free": 0.9, "precision": 0.0587,
                          "recall": 0.0998, "partition_defects": 0},
        baseline_iou_free=0.673,
        val_score=0.850, best_val_score=0.840, is_new_best=True,
    )

    assert "iou_free" in text
    assert "0.850" in text
    assert "+0.177" in text
    assert "P↑" in text and "R↑" in text and "iou_nf↑" in text
    assert "fatal" not in text and "free_miss" not in text


def test_epoch_log_uses_free_iou_for_checkpoint_and_baseline_comparison():
    """checkpoint와 baseline 비교가 같은 free-space 목표를 가리켜야 한다."""
    text = format_epoch_log(
        epoch=1, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_loss_parts=_LOSS_PARTS,
        val_loss=0.1, val_loss_parts=_LOSS_PARTS,
        train_free_metrics={"iou_free": 0.9, "iou_non_free": 0.9, "precision": 0.1,
                            "recall": 0.1, "partition_defects": 0},
        val_free_metrics={"iou_free": 0.850, "iou_non_free": 0.9, "precision": 0.1,
                          "recall": 0.1, "partition_defects": 0},
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
        train_free_metrics={"iou_free": 0.1, "iou_non_free": 0.9, "precision": 0.1,
                            "recall": 0.1, "partition_defects": 7},
        val_free_metrics={"iou_free": 0.1, "iou_non_free": 0.9, "precision": 0.1,
                          "recall": 0.1, "partition_defects": 0},
        val_score=0.1, best_val_score=0.0, is_new_best=True,
    )

    assert "partition" in text.lower()


class _ScalarWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def _tags(writer):
    return [tag for tag, _, _ in writer.scalars]


def test_epoch_scalars_write_each_class_loss_term():
    """세 클래스 loss 항이 각각 별도 tag로 나가야 한다 -- 총 loss만 보면 unknown이 셀의 85%를
    차지하는 데이터에서 occupied 항이 언제 죽었는지 알 수 없다."""
    writer = _ScalarWriter()

    write_epoch_scalars(writer, "train", {
        "loss": 1.0,
        "loss_parts": {"loss_unknown": 0.4, "loss_free": 0.2, "loss_occupied": 0.3},
    }, epoch=4)

    assert writer.scalars == [
        ("train/loss_total_epoch", 1.0, 4),
        ("train/loss_unknown_epoch", 0.4, 4),
        ("train/loss_free_epoch", 0.2, 4),
        ("train/loss_occupied_epoch", 0.3, 4),
    ]


def test_epoch_scalars_skip_non_finite_values():
    """val을 돌리지 않은 epoch은 loss가 NaN이다. TensorBoard에 NaN을 쓰면
    `NaN or Inf found` 경고가 나고 그래프가 끊긴다 (Task 16에서 실제로 겪었다)."""
    writer = _ScalarWriter()

    write_epoch_scalars(writer, "train", {
        "loss": float("nan"),
        "loss_parts": {"loss_unknown": float("nan"), "loss_free": 0.5},
    }, epoch=4)

    assert writer.scalars == [("train/loss_free_epoch", 0.5, 4)]


def test_epoch_scalars_write_every_metric_family_for_a_validation_epoch():
    """지표를 하나 추가하고 writer 배선을 잊으면 TensorBoard에 관측값이 없다 -- 학습을 33분
    돌린 뒤에야 알게 되는 종류의 누락이라 여기서 계약으로 고정한다."""
    writer = _ScalarWriter()

    write_epoch_scalars(writer, "val", {
        "loss": 1.0,
        "loss_parts": {},
        "free": {"iou_free": 0.8, "iou_non_free": 0.95, "precision": 0.9, "recall": 0.85},
        # 옛 키가 dict에 남아 있어도 tag로 나가면 안 된다(2026-10-02에 뺐다).
        "range": {"mae": 0.25, "bias": -0.05, "missed_obstacle_rate": 0.04},
        "rings": {"0.0-1.5m": {"iou_free": 0.9, "iou_non_free": 0.97,
                               "precision": 0.93, "recall": 0.91}},
        "boundary": {"20cm": {"bf": 0.5, "precision": 0.6, "recall": 0.4, "n_frames": 3}},
    }, epoch=4)

    # 2026-10-02 지표 집합을 **정확히** 고정한다(`projects/common/metric_spec.py`).
    assert _tags(writer) == [
        "val/loss_total_epoch",
        "val/iou_free_epoch", "val/iou_non_free_epoch", "val/precision_epoch",
        "val/recall_epoch",
        "val/ring_0.0-1.5m_iou_free_epoch", "val/ring_0.0-1.5m_iou_non_free_epoch",
        "val/ring_0.0-1.5m_precision_epoch", "val/ring_0.0-1.5m_recall_epoch",
        "val/bf_20cm_epoch", "val/bf_20cm_precision_epoch", "val/bf_20cm_recall_epoch",
    ]


def test_a_full_validation_epoch_writes_exactly_the_tags_the_readers_expect():
    """**쓰는 쪽과 읽는 쪽의 계약.** 실제 링·눈금으로 val epoch을 쓰면 나오는 tag 집합이
    `metric_spec.EXPORTED_METRICS`와 정확히 같아야 한다. 한쪽만 바뀌면 패키지 생성기가 빈칸을
    읽는다 -- 그것을 9시간 캠페인이 끝난 뒤가 아니라 여기서 잡는다."""
    from projects.common.metric_spec import (
        BF_TOLERANCES_M, EXPORTED_METRICS, FREE_METRICS, ring_names, tolerance_key)

    free = {name: 0.5 for name in FREE_METRICS}
    writer = _ScalarWriter()
    write_epoch_scalars(writer, "val", {
        "loss": 1.0, "loss_parts": {}, "free": free,
        "rings": {ring: dict(free) for ring in ring_names()},
        "boundary": {tolerance_key(t): {"bf": 0.5, "precision": 0.5, "recall": 0.5,
                                        "n_frames": 1} for t in BF_TOLERANCES_M},
    }, epoch=4)

    written = set(_tags(writer)) - {"val/loss_total_epoch"}
    assert written == {tag for tag, _, _ in EXPORTED_METRICS}


def test_epoch_scalars_of_a_train_epoch_skip_the_val_only_families():
    """train dict에는 range/ring/tolerance 키가 아예 없다 -- 광선·거리변환은 val에서만 돈다.
    `.get()`이 아니라 `[]`로 읽는 순간 학습 첫 epoch에서 KeyError로 죽는다."""
    writer = _ScalarWriter()

    write_epoch_scalars(writer, "train", {
        "loss": 1.0, "loss_parts": {},
        "free": {"iou_free": 0.8, "iou_non_free": 0.95, "precision": 0.9, "recall": 0.85},
    }, epoch=4)

    assert _tags(writer) == [
        "train/loss_total_epoch",
        "train/iou_free_epoch", "train/iou_non_free_epoch", "train/precision_epoch",
        "train/recall_epoch",
    ]
