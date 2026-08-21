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

    result = free_metrics_from_masks(gt_parts, gt_parts, valid)

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
    # 나머지 칸은 occupied/unknown으로 채워 예측도 완전한 분할이 되게 한다.
    #     pred_occupied = {5, 6, 7},  pred_unknown = {2, 3}
    pred_occupied = torch.tensor(
        [[[[False, False, False, False], [False, True, True, True]]]]
    )
    pred_unknown = torch.tensor(
        [[[[False, False, True, True], [False, False, False, False]]]]
    )
    pred_parts = {"free": pred_free, "occupied": pred_occupied, "unknown": pred_unknown}

    result = free_metrics_from_masks(pred_parts, gt_parts, valid)

    assert result["iou_free"] == pytest.approx(0.4)
    assert result["iou_free_count"] == 1
    assert result["fatal_rate"] == pytest.approx(1 / 3)
    assert result["fatal_denom"] == 3
    assert result["free_miss_rate"] == pytest.approx(0.5)
    assert result["free_miss_denom"] == 4
    assert result["partition_defects"] == 0
    assert torch.equal(result["pred_free"], pred_free)
    assert torch.equal(result["gt_free"], gt_parts["free"])
    assert torch.equal(result["pred_occupied"], pred_occupied)
    assert torch.equal(result["gt_occupied"], gt_parts["occupied"])


def test_iou_free_counts_free_predicted_inside_gt_unknown_as_an_error():
    """**task 정의를 못박는 테스트다.** (D) 정식화는 "보이면서 빈 곳"만 drivable이고
    보이지 않는 곳은 전부 non-drivable이다. 따라서 GT가 `unknown`인 셀을 free라고 예측하면
    그건 오차이고 `iou_free`의 분모에 들어가야 한다.

    이 계약을 잃으면 `iou_free_known`(2026-08-21 제거)처럼 `unknown`을 채점에서 빼는 변형이
    다시 들어오고, 그 순간 라벨이 non-drivable이라고 선언한 셀의 78.5 %가 무료가 된다.

    격자 4칸, valid=1:
        vis      = [T, T, F, F]        -> gt_unknown = {2, 3}
        occ      = [T, F, T, T]        -> gt_free = {0}, gt_occupied = {1}
        pred_free = [T, F, T, F]       -> 2번은 unknown 안의 오예측

        inter = {0}                 -> 1
        union = {0, 2}              -> 2        -> 0.5
    2번을 분모에서 빼면 1.0이 나오므로, 값이 0.5여야 계약이 지켜진 것이다.
    """
    from projects.common.free_space import decompose
    from projects.common.free_space_metrics import free_metrics_from_masks

    occ = torch.tensor([[[[1.0, 0.0, 1.0, 1.0]]]])
    vis = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
    valid = torch.ones_like(occ)
    gt_parts = decompose(occ, vis, valid)

    pred_free = torch.tensor([[[[True, False, True, False]]]])
    pred_parts = {
        "free": pred_free,
        "occupied": torch.tensor([[[[False, True, False, False]]]]),
        "unknown": torch.tensor([[[[False, False, False, True]]]]),
    }

    result = free_metrics_from_masks(pred_parts, gt_parts, valid)

    assert result["iou_free"] == pytest.approx(0.5)
    # unknown을 채점에서 빼는 변형이 dict에 다시 생기면 여기서 걸린다.
    assert "iou_free_known" not in result
    # 두께 1셀 표면의 면적 IoU도 함께 뺐다 -- 경계 정밀도는 `occupied_metrics`의 f1@τ가 잰다.
    assert "iou_occupied" not in result and "iou_unknown" not in result


import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.free_space_metrics import (
    _delta_rates,
    _delta_stats,
    range_error,
    summarize_range_error,
    summarize_range_error_by_gt_range,
)
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


def _range_batch(deltas, *, missed_obstacle=0):
    """`range_error`가 돌려주는 것과 **같은 키를 전부** 갖는 batch 결과 하나를 만든다.

    batch별 통계(`abs_p50` 등)까지 채우는 것이 중요하다 -- 이게 없으면 옛 구현(batch별 값을
    가중평균)으로 되돌렸을 때 테스트가 값이 틀려서가 아니라 `KeyError`로 죽어서, 실제로
    무엇을 검증하는지 알 수 없게 된다.

    `missed_obstacle`은 짝지어지지 않은(=회귀 통계에 없는) 광선이므로 `ok_gt`에는 더해지되
    `deltas`에는 들어가지 않는다 -- 실제 `range_error`가 세는 방식과 같다.
    """
    delta = np.asarray(deltas, dtype=float)
    counts = {
        "n_paired_rays": delta.size,
        "over_count": int((delta > 0).sum()),
        "under_count": int((delta < 0).sum()),
        "censored_gt": 0, "censored_pred": 0, "no_free_gt": 0,
        # 짝지어진 광선은 정의상 GT가 RAY_OK이고, 놓친 광선도 GT는 OK다.
        "ok_gt": delta.size + missed_obstacle, "missed_obstacle": missed_obstacle,
    }
    return {
        "deltas": delta,
        # GT 거리를 짝지어 보관하는 것도 계약이다 -- 거리별 층화가 이 배열을 쓴다.
        "gt_ranges": np.full(delta.shape, 1.0),
        **_delta_stats(delta),
        **counts,
        **_delta_rates(counts),
    }


def test_summarize_range_error_pools_the_samples_before_taking_percentiles():
    """백분위수는 평균낼 수 없다 -- 표본을 다 모은 뒤 한 번만 계산해야 한다.

    손계산: 전체 |dr| = [0.1, 0.1, 0.9] -> 중위수 0.10 (참값).
    배치별 중위수를 `n_paired_rays`로 가중평균하면
        batch A [0.1, 0.9] -> 0.50,  batch B [0.1] -> 0.10
        (0.50*2 + 0.10*1) / 3 = 0.3667   <- 참값의 3.7배

    같은 표본을 어떻게 자르느냐만 달라져도 값이 바뀌던 것이 이 지표의 batch-size 의존이었다.
    """
    merged = summarize_range_error([_range_batch([0.1, 0.9]), _range_batch([0.1])])

    assert merged["abs_p50"] == pytest.approx(0.1)
    wrong_if_batch_averaged = (0.5 * 2 + 0.1 * 1) / 3
    assert wrong_if_batch_averaged == pytest.approx(0.3667, abs=1e-4)
    assert merged["abs_p50"] != pytest.approx(wrong_if_batch_averaged)
    assert merged["n_paired_rays"] == 3


def test_summarize_range_error_does_not_depend_on_how_the_samples_are_batched():
    """같은 표본이면 batch로 어떻게 잘라도 결과가 같아야 한다 -- 실측 회귀 가드.

    같은 체크포인트를 bs4/bs8로 재채점했을 때 `abs_p50` 0.177 vs 0.172,
    `abs_p90` 0.701 vs 0.718로 갈렸던 것이 이 성질이 없어서였다.
    """
    sample = [0.05, 0.4, 0.02, 0.9, 0.3, 0.15, 0.7, 0.01, 0.25, 0.6, 0.11]
    one = summarize_range_error([_range_batch(sample)])
    split_by_2 = summarize_range_error([_range_batch(sample[i:i + 2])
                                        for i in range(0, len(sample), 2)])
    split_by_5 = summarize_range_error([_range_batch(sample[i:i + 5])
                                        for i in range(0, len(sample), 5)])

    for key in ("abs_p50", "abs_p90", "over_mean", "under_mean", "n_paired_rays"):
        assert one[key] == pytest.approx(split_by_2[key]), key
        assert one[key] == pytest.approx(split_by_5[key]), key


def test_summarize_range_error_averages_over_events_over_their_own_subset():
    """`over_mean`은 over 사건들만의 평균이다. 표본을 모아서 재면 자동으로 그렇게 된다.

    예전에는 batch별 `over_mean`을 합치느라 `over_count` 가중이라는 별도 규칙이 필요했고,
    `n_paired_rays`로 가중하면 25배까지 틀어졌다. 표본을 모으면 그 규칙 자체가 사라진다.

    손계산: over 사건 = 1건 x 1.00 m + 99건 x 0.01 m -> (1.00 + 0.99) / 100 = 0.0199
    """
    a = _range_batch([1.00] + [-0.5] * 99)      # over 1건, under 99건
    b = _range_batch([0.01] * 99 + [-0.5])      # over 99건, under 1건

    merged = summarize_range_error([a, b])

    assert merged["over_mean"] == pytest.approx((1.00 + 99 * 0.01) / 100)
    assert merged["over_count"] == 100
    assert merged["under_count"] == 100
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


def _open_to_grid_edge():
    """전방이 격자 끝까지 열려 있는 예측 -- 전방 광선이 `RAY_CENSORED`가 된다.

    `_forward_corridor(19)`가 정확히 그 경계다(실측 확인: status 0 -> 2로 바뀐다).
    """
    return _forward_corridor(19)


def test_mae_and_bias_pin_the_sign_convention_of_the_range_error():
    """`bias > 0` = 장애물을 실제보다 **멀다**고 예측 = free의 과대추정 = 위험한 쪽.

    다른 문헌은 같은 사건을 "장애물의 과소추정"이라 부르므로, 부호 규약을 코드 밖에서
    옮겨 읽으면 뒤집힌다. 그래서 두 방향을 각각 고정한다.

    전방 광선만 `RAY_OK`인 격자를 쓴다(나머지 세 방향은 free가 없어 `RAY_NO_FREE`):
        gt corridor(6)  -> r_gt   = 0.35 m
        pred corridor(10) -> r_pred = 0.55 m     -> dr = +0.20  (멀다고 예측)
    """
    valid = torch.ones_like(_forward_corridor(6))
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    too_far = range_error(_forward_corridor(10), _forward_corridor(6), valid, rays)
    assert too_far["mae"] == pytest.approx(0.2, abs=RANGE_SPEC.cell_m)
    assert too_far["bias"] == pytest.approx(0.2, abs=RANGE_SPEC.cell_m)

    too_near = range_error(_forward_corridor(6), _forward_corridor(10), valid, rays)
    assert too_near["mae"] == pytest.approx(0.2, abs=RANGE_SPEC.cell_m)
    # mae는 같고 bias만 부호가 뒤집힌다 -- 절댓값만 보면 두 실패가 구별되지 않는다.
    assert too_near["bias"] == pytest.approx(-0.2, abs=RANGE_SPEC.cell_m)


def test_missed_obstacle_rate_counts_the_rays_the_regression_silently_drops():
    """거리 통계가 정직하지 않을 수 있는 지점. GT에는 장애물이 있는데 예측이 격자 끝까지
    free라고 보면(=완전히 놓쳤다) 그 광선은 `RAY_CENSORED`가 되어 **회귀 통계에서 빠진다.**

    즉 이 배치에서 모델은 최대로 틀렸는데 `mae`는 nan이다. `missed_obstacle_rate`가 1.0으로
    그 사실을 드러내야 한다. 이 숫자가 없으면 "mae가 좋아졌다"를 혼자 읽게 된다.
    """
    valid = torch.ones_like(_forward_corridor(6))
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(_open_to_grid_edge(), _forward_corridor(6), valid, rays)

    assert result["ok_gt"] == 1
    assert result["missed_obstacle"] == 1
    assert result["missed_obstacle_rate"] == pytest.approx(1.0)
    assert result["n_paired_rays"] == 0
    assert math.isnan(result["mae"])


def test_a_perfect_mae_can_coexist_with_missed_obstacles():
    """앞 테스트를 강화한다: 한 샘플은 완벽히 맞히고 다른 샘플은 통째로 놓치면
    `mae`는 0.000인데 절반의 광선을 놓친 상태다. 두 숫자를 항상 같이 읽어야 하는 이유다.
    """
    gt = torch.cat([_forward_corridor(6), _forward_corridor(6)])
    pred = torch.cat([_forward_corridor(6), _open_to_grid_edge()])
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, torch.ones_like(gt), rays)

    assert result["mae"] == pytest.approx(0.0)
    assert result["missed_obstacle_rate"] == pytest.approx(0.5)


def test_range_error_stratified_by_gt_range_uses_the_gt_distance_not_the_prediction():
    """거리별 층화는 **GT 거리**로 묶는다 -- 예측 거리로 묶으면 구간 정의가 모델에 따라
    움직여 run 간 비교가 무의미해진다.

    샘플 두 장, 전방 광선만 `RAY_OK`:
        A: gt corridor(4)  r_gt 0.25 m, pred corridor(6)  -> dr = +0.10
        B: gt corridor(16) r_gt 0.85 m, pred corridor(16) -> dr =  0.00
    경계 (0, 0.5, 1.0)이면 A는 근거리 구간, B는 원거리 구간에 들어간다. 예측 거리로 묶으면
    A의 r_pred가 0.35라 여전히 근거리 구간이므로, 이 테스트만으로는 GT/예측 구분이 안 된다 --
    그래서 예측을 원거리로 크게 밀어 구간이 바뀌는 경우를 아래에서 따로 본다.
    """
    gt = torch.cat([_forward_corridor(4), _forward_corridor(16)])
    pred = torch.cat([_forward_corridor(6), _forward_corridor(16)])
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, torch.ones_like(gt), rays)
    bins = summarize_range_error_by_gt_range([result], (0.0, 0.5, 1.0))

    assert bins["0.0-0.5m"]["mae"] == pytest.approx(0.1, abs=1e-9)
    assert bins["0.0-0.5m"]["n_paired_rays"] == 1
    assert bins["0.5-1.0m"]["mae"] == pytest.approx(0.0)
    assert bins["0.5-1.0m"]["n_paired_rays"] == 1


def test_stratification_bin_membership_follows_the_gt_even_when_the_prediction_is_far_off():
    """근거리 장애물을 원거리로 예측한 광선은 **근거리 구간**에 들어가야 한다.

        gt corridor(4)   r_gt   = 0.25 m   -> 0.0-0.5m 구간
        pred corridor(16) r_pred = 0.85 m  -> 예측으로 묶으면 0.5-1.0m 구간

    예측 거리로 묶는 구현이면 근거리 구간이 비고 원거리 구간에 0.6 m 오차가 찍힌다.
    """
    gt = _forward_corridor(4)
    pred = _forward_corridor(16)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    bins = summarize_range_error_by_gt_range(
        [range_error(pred, gt, torch.ones_like(gt), rays)], (0.0, 0.5, 1.0)
    )

    assert bins["0.0-0.5m"]["n_paired_rays"] == 1
    assert bins["0.0-0.5m"]["mae"] == pytest.approx(0.6, abs=RANGE_SPEC.cell_m)
    assert bins["0.5-1.0m"]["n_paired_rays"] == 0
    assert math.isnan(bins["0.5-1.0m"]["mae"])


def test_predicting_everything_blocked_is_not_counted_as_a_missed_obstacle():
    """예측에 free가 전혀 없는 광선(`RAY_NO_FREE`)은 "놓쳤다"가 아니다.

    그건 "전부 막혔다고 봤다"는 과잉 보수라서 비용의 종류가 정반대다 -- planner를 세울 뿐
    위험하게 만들지 않는다. 두 실패를 한 숫자에 합치면 `missed_obstacle_rate`가 "위험한
    실패율"이라는 의미를 잃는다.

    GT는 전방 광선이 `RAY_OK`이고, 예측은 free가 한 칸도 없어 같은 광선이 `RAY_NO_FREE`다.
    """
    gt = _forward_corridor(6)
    pred = torch.zeros_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, torch.ones_like(gt), rays)

    assert result["ok_gt"] == 1
    assert result["missed_obstacle"] == 0
    assert result["missed_obstacle_rate"] == pytest.approx(0.0)
    assert result["n_paired_rays"] == 0


def test_summarize_recomputes_the_missed_rate_from_the_summed_counts():
    """비율은 batch별로 평균낼 수 없다 -- 합산된 카운트에서 다시 계산해야 batch-size 불변이다.

        batch A: ok_gt 1, missed 1   -> 비율 1.0
        batch B: ok_gt 9, missed 0   -> 비율 0.0
        카운트 합산: 1 / 10 = 0.1  (참값)
        batch별 단순 평균: (1.0 + 0.0) / 2 = 0.5   <- 참값의 5배
    """
    merged = summarize_range_error([
        _range_batch([], missed_obstacle=1),
        _range_batch([0.0] * 9),
    ])

    assert merged["ok_gt"] == 10
    assert merged["missed_obstacle"] == 1
    assert merged["missed_obstacle_rate"] == pytest.approx(0.1)
    assert merged["missed_obstacle_rate"] != pytest.approx(0.5)
