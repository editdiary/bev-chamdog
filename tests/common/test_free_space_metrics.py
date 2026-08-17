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


from projects.common.free_space_metrics import build_ring_masks, metrics_per_ring


def test_ring_masks_partition_the_grid_by_distance_from_the_origin():
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))

    assert [name for name, _ in rings] == ["0.0-0.5m", "0.5-1.0m"]
    inner, outer = rings[0][1], rings[1][1]
    assert not (inner & outer).any()                      # 겹치지 않는다
    assert inner[19, 20] and not outer[19, 20]            # 원점 바로 앞은 안쪽 링


def test_metrics_per_ring_isolates_far_field_failure():
    """근거리의 쉬운 성능이 원거리 실패를 가리는 것을 막는 것이 M4의 목적이다."""
    gt = torch.ones((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))
    outer = torch.from_numpy(rings[1][1])
    pred = gt.clone()
    pred[0, 0][outer] = False                             # 바깥 링만 전부 틀린다
    valid = torch.ones_like(gt)

    per_ring = metrics_per_ring(pred, gt, valid, rings)

    assert per_ring["0.0-0.5m"]["iou_free"] == pytest.approx(1.0)
    assert per_ring["0.5-1.0m"]["iou_free"] == pytest.approx(0.0)


def test_metrics_per_ring_isolates_far_field_fatal_rate():
    """`fatal_rate`는 모듈 docstring이 말하는 "직접적인 위험량"이다. 위 테스트는 `iou_free`
    만 링별로 확인하므로, `metrics_per_ring`이 `fatal_rate` 호출에도 실제로 링 마스크를
    곱하는지(전체 `valid`가 아니라 `ring_valid`를 넘기는지)는 검출하지 못한다. 안쪽 링은
    pred==gt(위험 없음), 바깥 링은 pred가 전부 잘못 free라고 주장(전부 위험)하도록 만들어
    두 값이 0.0/1.0으로 뚜렷이 갈리게 한다."""
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))
    inner_mask, _ = rings[0][1], rings[1][1]

    gt = torch.zeros((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    gt[0, 0][torch.from_numpy(inner_mask)] = True     # 안쪽 링만 실제 free, 바깥 링은 occupied
    pred = torch.ones_like(gt)                        # 예측은 어디서나 free라고 주장
    valid = torch.ones_like(gt)

    per_ring = metrics_per_ring(pred, gt, valid, rings)

    assert per_ring["0.0-0.5m"]["fatal_rate"] == pytest.approx(0.0)   # 안쪽: pred==gt, 위험 없음
    assert per_ring["0.5-1.0m"]["fatal_rate"] == pytest.approx(1.0)   # 바깥: 전부 오탐(occupied인데 free라 함)


def test_metrics_per_ring_reports_matching_count_and_denom_fields():
    """`iou_free_count`/`fatal_denom`이 실제로 각자의 지표와 짝지어 나오는지 확인한다
    (Task 2 리뷰가 잡아낸 것과 같은 부류의 배선 결함 -- 값이 그럴듯해서 아무도 눈치채지
    못한다). 안쪽 링은 iou 평균에 배치 1개가 들어가 count=1이고, fatal 분모는 `|pred|`인
    링 전체 칸 수라서 두 값이 (1 vs 수백) 크게 다르다 -- 뒤바뀌면 반드시 걸린다."""
    gt = torch.ones((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))
    inner_mask, outer_mask = rings[0][1], rings[1][1]
    outer = torch.from_numpy(outer_mask)
    pred = gt.clone()
    pred[0, 0][outer] = False
    valid = torch.ones_like(gt)

    per_ring = metrics_per_ring(pred, gt, valid, rings)

    assert per_ring["0.0-0.5m"]["iou_free_count"] == 1
    assert per_ring["0.0-0.5m"]["fatal_denom"] == int(inner_mask.sum())


ASYMMETRIC_SPEC = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=0.5)


def test_ring_masks_assign_front_m_to_rows_and_half_width_m_to_cols():
    """RANGE_SPEC은 `front_m == half_width_m == 1.0`이라 `origin_row`/`origin_col`이 어느
    필드에서 오는지 바꿔도 결과가 같다 -- row/col 배정 버그를 검출할 수 없는 fixture다.
    `front_m != half_width_m`인 ASYMMETRIC_SPEC으로 직접 검증한다. `build_ring_masks`가
    존재하는 이유인 전후 비대칭이 바로 이 값이며, 실제 `ROBOT_GRID_SPEC`(front_m=4.0,
    rear_m=2.0, half_width_m=3.0)도 이런 비대칭 케이스다."""
    rings = build_ring_masks(ASYMMETRIC_SPEC, edges_m=(0.0, 0.5, 3.0))
    inner_mask = rings[0][1]

    # 올바른 배정: origin_row = front_m/cell_m - 0.5 = 5.5, origin_col = half_width_m/cell_m
    # - 0.5 = 1.5. 셀 (5, 1)은 원점에서 대각선으로 0.5칸(0.5*sqrt(2)*0.5m ≈ 0.354m)
    # 떨어져 있어 0.0-0.5m 링 안이다.
    assert inner_mask[5, 1]
    # row/col을 뒤바꾸면 origin이 (1.5, 5.5)가 되어 이 셀까지 거리가 ≈2.85m로 늘어나
    # 0.0-0.5m 링 밖으로 밀려난다 -- 이 assert 하나로 뒤바뀜을 잡는다.
