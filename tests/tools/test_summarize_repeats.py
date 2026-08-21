import math

import pytest

from tools.summarize_repeats import (
    SELECTION_TAG,
    aggregate,
    group_runs,
    pick_epochs,
    _mean_sd,
)


def _run(name, selection, seed=0, extra=None, config=None):
    series = {SELECTION_TAG: selection}
    series.update(extra or {})
    return {"name": name, "config": {"seed": seed, **(config or {})}, "series": series}


def test_pick_epochs_separates_the_declared_epoch_from_the_selected_one():
    """고정 epoch은 **선택이 개입하지 않는** 값이고 best는 max라 위로 편향된다.
    둘을 같은 함수가 내놓지 않으면 표에서 섞여 편향의 크기를 볼 수 없다."""
    selection = {10: 0.70, 20: 0.85, 30: 0.80}

    picks = pick_epochs({SELECTION_TAG: selection}, fixed_epoch=30)

    assert picks["fixed"] == 30
    assert picks["best"] == 20      # max는 ep20이고 고정 epoch과 다르다
    assert picks["last"] == 30
    assert selection[picks["fixed"]] == 0.80


def test_pick_epochs_falls_back_to_the_last_epoch_and_reports_it():
    """`fixed_epoch`를 안 주면 도달한 마지막 epoch을 쓴다. `last`를 함께 내놓아야
    중간에 죽은 런을 조용히 짧은 값으로 집계하지 않는다."""
    picks = pick_epochs({SELECTION_TAG: {5: 0.5, 12: 0.6}})

    assert picks["fixed"] == 12 and picks["last"] == 12


def test_pick_epochs_marks_a_missing_fixed_epoch_instead_of_guessing():
    """요청한 epoch에 값이 없으면 **가장 가까운 값으로 대체하지 않는다** -- 그러면 죽은 런이
    표에서 정상으로 보이고 평균이 조용히 다른 것을 재게 된다."""
    picks = pick_epochs({SELECTION_TAG: {10: 0.5, 20: 0.6}}, fixed_epoch=40)

    assert picks["fixed"] is None
    assert picks["best"] == 20


def test_mean_sd_returns_nan_for_a_single_run_rather_than_zero():
    """n=1에서 sd를 0으로 내놓으면 "분산이 없다"로 읽힌다 -- 이 도구의 존재 이유가
    분산을 재는 것이므로 그 오독이 가장 비싸다."""
    mean, sd, n = _mean_sd([0.8])

    assert mean == pytest.approx(0.8) and n == 1 and math.isnan(sd)


def test_mean_sd_skips_missing_values_without_counting_them():
    mean, sd, n = _mean_sd([0.8, None, float("nan"), 0.6])

    assert n == 2 and mean == pytest.approx(0.7)
    assert sd == pytest.approx(0.1414, abs=1e-3)


def test_aggregate_reports_the_sample_sd_across_runs_for_both_epoch_choices():
    """세 시드의 σ가 이 도구의 산출물이다. fixed와 best가 **다른 σ**를 갖는 것이 정상이고
    (best는 max라 분포가 다르다) 표가 그 둘을 구별해야 한다."""
    runs = [
        _run("s0", {10: 0.70, 20: 0.80}, seed=0),
        _run("s1", {10: 0.72, 20: 0.78}, seed=1),
        _run("s2", {10: 0.74, 20: 0.76}, seed=2),
    ]

    summary = aggregate(runs, fixed_epoch=10)

    fixed_mean, fixed_sd, fixed_n = summary["iou_free"]["fixed"]
    best_mean, _, best_n = summary["iou_free"]["best"]
    assert fixed_n == 3 and best_n == 3
    assert fixed_mean == pytest.approx(0.72)
    assert fixed_sd == pytest.approx(0.02)
    # best는 각 런의 max이므로 전부 ep20이고 평균이 고정 epoch보다 높다 -- 그 편향이 보여야 한다.
    assert best_mean == pytest.approx(0.78)
    assert summary["_epochs"]["best"] == [20, 20, 20]


def test_aggregate_keeps_a_metric_that_only_some_runs_logged():
    """지표 집합을 줄인 뒤(2026-08-21) 옛 런과 새 런을 한 표에 놓으면 일부 tag가 없다.
    그때 전체가 n/a가 되면 안 되고, 있는 런만으로 집계하되 n이 줄어야 한다."""
    runs = [
        _run("s0", {10: 0.7}, extra={"val/range_mae_epoch": {10: 0.25}}),
        _run("s1", {10: 0.7}),
    ]

    summary = aggregate(runs, fixed_epoch=10)

    assert summary["range_mae"]["fixed"][2] == 1
    assert summary["iou_free"]["fixed"][2] == 2


def test_group_runs_splits_by_a_config_key_for_loso_folds():
    """LOSO는 fold마다 `val_sequences`가 다르다 -- 그 키로 묶으면 fold별 표가 공짜로 나온다."""
    runs = [
        _run("a", {10: 0.8}, config={"val_sequences": "raws1"}),
        _run("b", {10: 0.7}, config={"val_sequences": "raws1"}),
        _run("c", {10: 0.6}, config={"val_sequences": "rawos3"}),
    ]

    grouped = group_runs(runs, group_by="val_sequences")

    assert sorted(grouped) == ["rawos3", "raws1"]
    assert len(grouped["raws1"]) == 2 and len(grouped["rawos3"]) == 1


def test_group_runs_without_a_key_keeps_every_run_in_one_group():
    runs = [_run("a", {10: 0.8}), _run("b", {10: 0.7})]

    assert list(group_runs(runs)) == ["all"]
    assert len(group_runs(runs)["all"]) == 2
