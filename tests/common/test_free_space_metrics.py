import math

import pytest
import torch

from projects.common.free_space_metrics import (
    fatal_rate,
    free_miss_rate,
    iou_free,
    weighted_mean,
)


def _mask(values):
    return torch.tensor(values, dtype=torch.bool).unsqueeze(0).unsqueeze(0)


def test_iou_free_matches_hand_computation():
    gt = _mask([[True, True], [False, False]])     # free 2칸
    pred = _mask([[True, False], [True, False]])   # free 2칸, 교집합 1칸
    valid = torch.ones_like(gt)

    value, count = iou_free(pred, gt, valid)

    assert value == pytest.approx(1 / 3)           # 교집합 1 / 합집합 3
    assert count == 1


def test_perfect_prediction_scores_one_and_zero_errors():
    gt = _mask([[True, False], [True, False]])
    valid = torch.ones_like(gt)

    assert iou_free(gt, gt, valid)[0] == pytest.approx(1.0)
    assert fatal_rate(gt, gt, valid)[0] == pytest.approx(0.0)
    assert free_miss_rate(gt, gt, valid)[0] == pytest.approx(0.0)


def test_all_free_prediction_is_penalised_by_iou_free():
    """'전부 free'는 iou_drivable을 이겼던 트리비얼 해다. iou_free는 그걸 벌해야 한다."""
    gt = _mask([[True, False, False, False]])      # 4칸 중 1칸만 free
    pred = torch.ones_like(gt)
    valid = torch.ones_like(gt)

    assert iou_free(pred, gt, valid)[0] == pytest.approx(0.25)
    assert fatal_rate(pred, gt, valid)[0] == pytest.approx(0.75)


def test_metrics_ignore_invalid_cells():
    gt = _mask([[True, False]])
    pred = _mask([[True, True]])
    valid = _mask([[True, False]])                 # 두 번째 칸은 집계에서 빠진다

    assert iou_free(pred, gt, valid)[0] == pytest.approx(1.0)
    assert fatal_rate(pred, gt, valid)[0] == pytest.approx(0.0)


def test_empty_denominator_yields_nan_and_zero_weight():
    """GT에도 예측에도 free가 없는 샘플은 IoU가 0/0이다. 0점으로 세면 지표가 왜곡된다."""
    gt = _mask([[False, False]])
    pred = _mask([[False, False]])
    valid = torch.ones_like(gt)

    value, count = iou_free(pred, gt, valid)
    assert math.isnan(value)
    assert count == 0

    rate, denom = fatal_rate(pred, gt, valid)
    assert math.isnan(rate)
    assert denom == 0


def test_weighted_mean_skips_zero_weight_entries():
    assert weighted_mean([1.0, float("nan")], [2, 0]) == pytest.approx(1.0)
    assert math.isnan(weighted_mean([1.0], [0]))


def test_free_metrics_from_masks_is_the_shared_aggregator():
    """2-head와 3-class가 같은 집계기를 써야 A/B가 공정하다."""
    from projects.common.free_space import decompose
    from projects.common.free_space_metrics import free_metrics_from_masks

    occ = torch.tensor([[[[1.0, 0.0]]]])
    vis = torch.tensor([[[[1.0, 1.0]]]])
    valid = torch.ones_like(occ)
    gt_parts = decompose(occ, vis, valid)

    result = free_metrics_from_masks(gt_parts["free"], gt_parts, valid)

    assert result["iou_free"] == pytest.approx(1.0)
    assert result["partition_defects"] == 0
    assert torch.equal(result["gt_free"], gt_parts["free"])


def test_free_metrics_from_masks_wires_every_key_correctly():
    """`free_metrics_from_masks`는 2-head/3-class A/B 비교가 흐르는 유일한 통로다.

    앞선 스모크 테스트는 완벽한 예측만 넣어서 `fatal_denom`/`free_miss_denom`이
    통째로 빠지거나 두 지표가 뒤바뀌어도 통과했을 것이다. 여기서는 불완전한 예측을 써서
    `fatal_rate != free_miss_rate`, `fatal_denom != free_miss_denom`이 되도록 만들어
    뒤바뀜을 실제로 탐지할 수 있게 한다.

    격자 8칸, valid=1 vis=1 (unknown 없음) 이므로 free = occ, occupied = ~occ.
        gt_free   = [T, T, T, T, F, F, F, F]   (4칸)
        pred_free = [T, T, F, F, T, F, F, F]   (3칸)

    손계산:
        intersection (pred & gt)      = {0, 1}                -> 2
        union        (pred | gt)      = {0, 1, 2, 3, 4}        -> 5
        iou_free = 2 / 5 = 0.4, count = 1 (union > 0인 샘플 1개)

        fatal:  pred & ~gt              = {4}                  -> 1
                denom = |pred|          = {0, 1, 4}             -> 3
                fatal_rate = 1 / 3

        miss:   ~pred & gt               = {2, 3}               -> 2
                denom = |gt|            = {0, 1, 2, 3}          -> 4
                free_miss_rate = 2 / 4 = 0.5

    fatal_rate(1/3) != free_miss_rate(0.5), fatal_denom(3) != free_miss_denom(4)이므로
    두 지표/분모가 뒤바뀌면 이 테스트가 반드시 깨진다.
    """
    from projects.common.free_space import decompose
    from projects.common.free_space_metrics import free_metrics_from_masks

    occ = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]])
    vis = torch.ones_like(occ)
    valid = torch.ones_like(occ)
    gt_parts = decompose(occ, vis, valid)

    pred_free = torch.tensor(
        [[[[True, True, False, False], [True, False, False, False]]]]
    )

    result = free_metrics_from_masks(pred_free, gt_parts, valid)

    assert result["iou_free"] == pytest.approx(0.4)
    assert result["iou_free_count"] == 1
    assert result["fatal_rate"] == pytest.approx(1 / 3)
    assert result["fatal_denom"] == 3
    assert result["free_miss_rate"] == pytest.approx(0.5)
    assert result["free_miss_denom"] == 4
    assert result["partition_defects"] == 0
    assert torch.equal(result["pred_free"], pred_free)
    assert torch.equal(result["gt_free"], gt_parts["free"])
