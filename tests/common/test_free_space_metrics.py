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


import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.free_space_metrics import range_error, summarize_range_error
from projects.common.polar import build_ray_index

RANGE_SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def _forward_corridor(free_cells_ahead):
    """전방 한 줄만 `free_cells_ahead` 칸 열린 (1, 1, H, W) bool 텐서."""
    grid = np.zeros((RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), bool)
    grid[19 - free_cells_ahead:19, 20] = True
    return torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)


def test_range_error_is_zero_when_prediction_matches():
    gt = _forward_corridor(10)
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(gt, gt, valid, rays)

    assert result["n_paired_rays"] >= 1
    assert result["abs_p50"] == pytest.approx(0.0)
    assert result["over_mean"] == pytest.approx(0.0)


def test_over_prediction_is_reported_separately_from_under_prediction():
    """과대(위험)와 과소(보수적)는 비용이 다르므로 한 숫자로 뭉개면 안 된다."""
    gt = _forward_corridor(6)
    pred = _forward_corridor(10)                 # 0.2 m 더 멀리 열려 있다고 예측
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, valid, rays)

    assert result["over_mean"] == pytest.approx(0.2, abs=RANGE_SPEC.cell_m)
    assert result["under_mean"] == pytest.approx(0.0)


def test_censored_rays_are_counted_but_excluded_from_the_regression():
    gt = torch.ones((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(gt, gt, valid, rays)

    assert result["censored_gt"] == 4
    assert result["n_paired_rays"] == 0
    assert math.isnan(result["abs_p50"])


def test_summarize_range_error_weights_by_paired_ray_count():
    """`abs_p50`/`abs_p90`은 delta 전체 통계이므로 `n_paired_rays`로 가중해야 한다.

    (`over_count`/`under_count` 키는 이 라운드에서 인터페이스에 추가된 필수 필드라
    dict를 유효하게 만들기 위해 채워 넣었을 뿐, 이 테스트의 검증 대상은 여전히
    `n_paired_rays` 가중이다 -- `over_mean`/`under_mean` 가중 검증은 아래
    `test_summarize_range_error_weights_over_and_under_by_their_own_counts`가 맡는다.)
    """
    a = {"abs_p50": 0.1, "abs_p90": 0.2, "over_mean": 0.0, "under_mean": 0.1,
         "n_paired_rays": 30, "over_count": 0, "under_count": 30,
         "censored_gt": 1, "censored_pred": 2, "no_free_gt": 3}
    b = {"abs_p50": 0.3, "abs_p90": 0.4, "over_mean": 0.0, "under_mean": 0.3,
         "n_paired_rays": 10, "over_count": 0, "under_count": 10,
         "censored_gt": 0, "censored_pred": 0, "no_free_gt": 1}

    merged = summarize_range_error([a, b])

    assert merged["abs_p50"] == pytest.approx((0.1 * 30 + 0.3 * 10) / 40)
    assert merged["censored_gt"] == 1
    assert merged["n_paired_rays"] == 40


def test_summarize_range_error_weights_over_and_under_by_their_own_counts():
    """`over_mean`은 over 사건들만의 평균이므로 `n_paired_rays`가 아니라 `over_count`로
    가중해야 한다. 두 배치의 `n_paired_rays`는 같게(100, 100) 두고 `over_count`는
    크게 다르게(1 vs 99) 만들어서, `n_paired_rays` 가중과 `over_count` 가중이 뚜렷이
    다른 값을 내도록 설계했다 (batch 예시: PR 코멘트의 25배 과대 사례).

    손계산:
        올바름 (over_count 가중):  (1 * 1.00 + 99 * 0.01) / (1 + 99)
                                  = (1.00 + 0.99) / 100 = 0.0199
        틀림   (n_paired_rays 가중): (1.00 * 100 + 0.01 * 100) / (100 + 100)
                                  = (100 + 1) / 200 = 0.505   <- 25배 이상 과대
    """
    a = {"abs_p50": 0.5, "abs_p90": 0.9, "over_mean": 1.00, "under_mean": 0.0,
         "n_paired_rays": 100, "over_count": 1, "under_count": 0,
         "censored_gt": 0, "censored_pred": 0, "no_free_gt": 0}
    b = {"abs_p50": 0.5, "abs_p90": 0.9, "over_mean": 0.01, "under_mean": 0.0,
         "n_paired_rays": 100, "over_count": 99, "under_count": 0,
         "censored_gt": 0, "censored_pred": 0, "no_free_gt": 0}

    merged = summarize_range_error([a, b])

    correct = (1 * 1.00 + 99 * 0.01) / 100
    wrong_if_paired_weighted = (1.00 * 100 + 0.01 * 100) / 200
    assert correct == pytest.approx(0.0199)
    assert wrong_if_paired_weighted == pytest.approx(0.505)
    assert merged["over_mean"] == pytest.approx(correct)
    assert merged["over_count"] == 100
    assert merged["n_paired_rays"] == 200


def test_range_error_reports_over_and_under_counts():
    """`range_error`가 `over_count`/`under_count`를 실제로 채워 돌려주는지 확인한다."""
    gt = _forward_corridor(6)
    pred = _forward_corridor(10)                 # 전방 광선 1개만 0.2 m 과대 예측
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, valid, rays)

    assert result["n_paired_rays"] == 1
    assert result["over_count"] == 1
    assert result["under_count"] == 0
