"""캠페인 v3 판정 규칙(짝지은 양측 t-검정, 95 % 신뢰구간)을 고정한다."""
import math

import pytest
from scipy import stats

from projects.common.paired_stats import paired_test, verdict


def test_matches_scipy_paired_t_test():
    diffs = [0.010, 0.012, 0.008, 0.011, 0.009]
    out = paired_test(diffs)
    ref = stats.ttest_1samp(diffs, 0.0)
    assert out["t"] == pytest.approx(ref.statistic)
    assert out["p"] == pytest.approx(ref.pvalue)
    assert out["df"] == 4 and out["n"] == 5
    assert out["ci_low"] > 0 and out["significant"] is True


def test_the_95_percent_rule_is_stricter_than_the_old_two_se_rule():
    """|Δ̄| = 2.5·SE: 옛 규칙(2·SE)은 통과, 95 %(2.776·SE)는 통과하지 못한다."""
    base = [-2.0, -1.0, 0.0, 1.0, 2.0]                 # 평균 0, sd = sqrt(2.5)
    se0 = math.sqrt(2.5) / math.sqrt(5)
    diffs = [b + 2.5 * se0 for b in base]
    out = paired_test(diffs)
    assert out["mean"] / out["se"] == pytest.approx(2.5)
    assert out["significant"] is False
    assert out["ci_low"] < 0 < out["ci_high"]
    diffs = [b + 3.0 * se0 for b in base]
    assert paired_test(diffs)["significant"] is True


def test_degenerate_inputs_are_never_significant():
    assert paired_test([])["significant"] is False
    assert paired_test([0.01])["significant"] is False
    assert paired_test([0.01] * 5)["significant"] is False      # sd = 0
    assert paired_test([0.01, None, float("nan"), 0.02, 0.015])["n"] == 3


def test_verdict_uses_direction_only_when_significant():
    sig = paired_test([0.010, 0.012, 0.008, 0.011, 0.009])
    assert verdict(sig, True) == "arm better"
    assert verdict(sig, False) == "control better"
    assert "no significant" in verdict(paired_test([0.01, -0.01, 0.02, -0.02, 0.0]), True)


def test_paper_table_marks_only_significant_cells():
    from projects.common.paired_stats import paper_metric_table
    seeds = (0, 1, 2, 3, 4)
    control = {s: {"iou_free": 0.800 + 0.001 * s, "recall": 0.90 + 0.001 * s} for s in seeds}
    better = {s: {"iou_free": 0.810 + 0.001 * s + 0.0002 * (s % 2), "recall": 0.90 + 0.001 * ((s * 3) % 5)}
              for s in seeds}
    header, rows = paper_metric_table({"ctrl": control, "arm": better}, "ctrl",
                                      ["iou_free", "recall"], seeds)
    assert header == ["condition", "iou_free", "recall"]
    assert rows[0][0] == "ctrl" and not rows[0][1].endswith("*")
    assert rows[1][1].endswith("*")            # +0.010, 5/5 같은 방향
    assert not rows[1][2].endswith("*")        # 같은 값들을 섞은 것뿐 -- 평균 차이 0
    assert rows[0][1] == "0.802 ± 0.002"
