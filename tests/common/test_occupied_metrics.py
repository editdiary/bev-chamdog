"""occupied tolerance F1 테스트.

이 지표가 존재하는 이유 자체가 "면적 IoU는 두께 1셀 표면에 쓰면 안 된다"이므로, 첫 테스트가
바로 그 상황(예측이 몇 셀 밀렸을 때 IoU는 0인데 F1@τ는 1.0)을 고정한다.
"""
import math

import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.free_space_metrics import iou_masked
from projects.common.occupied_metrics import (
    derive_occupied,
    summarize_tolerance_f1,
    tolerance_counts,
    tolerance_key,
)
from projects.common.polar import build_ray_index

CELL_M = 0.05  # 실제 격자와 같은 해상도 -- τ를 셀 수로 환산해 손계산할 수 있게 맞춘다


def _grid(cells, size=7):
    """`cells`에 있는 (row, col)만 True인 `(1, 1, size, size)` 마스크."""
    mask = torch.zeros(1, 1, size, size, dtype=torch.bool)
    for row, col in cells:
        mask[0, 0, row, col] = True
    return mask


def test_tolerance_key_is_stable_and_readable():
    """dict 키와 TensorBoard tag가 이 함수 하나를 통과한다 -- float을 그대로 쓰면 `0.1` /
    `0.10000000000000001` 같은 표기가 섞여 run 간 tag가 갈린다."""
    assert tolerance_key(0.10) == "10cm"
    assert tolerance_key(0.20) == "20cm"
    assert tolerance_key(0.40) == "40cm"


_DERIVE_SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def test_derive_occupied_returns_the_boundary_per_sample_in_a_batch():
    """batch의 각 샘플을 독립으로 처리하고 `(B, 1, H, W)` bool 계약을 지킨다.

    샘플 0은 원점을 포함하는 사각형 free, 샘플 1은 free가 전혀 없다. 하나로 뭉쳐 처리하면
    (예: 배치 전체를 OR) 샘플 1에도 표면이 생겨 이 테스트가 깨진다. 유도 규칙 자체의 기하는
    `tests/common/test_polar.py`가 고정한다.
    """
    size = _DERIVE_SPEC.n_rows
    free = torch.zeros(2, 1, size, size, dtype=torch.bool)
    free[0, 0, 10:30, 10:30] = True
    valid = torch.ones_like(free)

    derived = derive_occupied(free, valid, build_ray_index(_DERIVE_SPEC, n_theta=720))

    assert derived.shape == free.shape and derived.dtype == torch.bool
    assert not derived[1].any()                      # free가 없으면 표면도 없다
    assert derived[0].any()
    assert not (derived[0] & free[0]).any()          # free 셀은 표면이 아니다
    border = torch.zeros_like(free[0])
    border[0, 9:31, 9:31] = True
    assert not (derived[0] & ~(border & ~free[0])).any()


def test_derive_occupied_respects_the_valid_mask():
    """`valid=0`은 수집 아티팩트라 free로 취급할 수 없다 -- 그 경계에서 표면이 생겨야 한다."""
    size = _DERIVE_SPEC.n_rows
    free = torch.zeros(1, 1, size, size, dtype=torch.bool)
    free[0, 0, 10:30, 10:30] = True
    valid = torch.ones_like(free)
    valid[0, 0, :20] = False                          # 전방 절반을 무효로 만든다

    rays = build_ray_index(_DERIVE_SPEC, n_theta=720)
    derived = derive_occupied(free, valid, rays)

    # 무효 영역은 free가 아니므로 그 경계(row 19)에서 광선이 멈춘다 -- 원래 사각형의
    # 앞쪽 테두리(row 9)까지 뚫고 나가면 `valid`를 무시한 것이다.
    effective_free = free & valid
    assert not (derived & effective_free).any()
    assert derived[0, 0, 19].any()
    assert not derived[0, 0, :19].any()


def test_a_shifted_wall_scores_zero_iou_but_full_f1_within_tolerance():
    """이 지표가 존재하는 이유. 격자 폭을 가로지르는 벽이 3셀(0.15 m) **통째로** 밀린 경우:

        IoU        = 0        (교집합이 없다)
        f1@10cm    = 0        (0.15 m > 0.10 m -- 허용오차 안에 없다)
        f1@20cm    = 1.0      (0.15 m <= 0.20 m -- 허용오차 안이다)

    벽이 폭 전체를 덮어야 모든 셀의 거리가 정확히 0.15 m로 같아진다. 일부 구간만 덮으면
    끝 셀들이 GT 끝 셀에 더 가까워져 거리가 셀마다 달라지고, 그러면 τ의 임계 동작을
    깨끗하게 고정할 수 없다.

    τ가 실제로 임계값으로 작동하는지까지 같이 본다. 거리를 셀 단위로 쓰고 `cell_m`을
    곱하는 것을 잊으면 0.15 m가 3.0으로 읽혀 두 τ 모두 실패하고 이 테스트가 깨진다.
    """
    gt = _grid([(3, col) for col in range(7)])
    pred = _grid([(6, col) for col in range(7)])
    valid = torch.ones_like(gt)

    assert iou_masked(pred, gt, valid)[0] == pytest.approx(0.0)

    counts = tolerance_counts(pred, gt, valid, CELL_M, tolerances=(0.10, 0.20))
    result = summarize_tolerance_f1([counts])

    assert result["10cm"]["f1"] == pytest.approx(0.0)
    assert result["20cm"]["f1"] == pytest.approx(1.0)


def test_precision_and_recall_are_computed_from_opposite_distance_fields():
    """precision과 recall이 서로 **반대 방향**의 거리장을 봐야 한다.

    두 거리장을 뒤바꾸는 변이를 실제로 넣어 보면, hit 카운트가 우연히 같은 배치에서는
    분모만 다르므로 값이 그대로 나온다. 그래서 **hit 카운트가 서로 다른** 배치를 쓴다.

    GT = {(3,3), (0,0)},  예측 = {(3,4), (3,5), (3,6)},  cell = 0.05 m, τ = 0.10 m:
        예측 -> 가장 가까운 GT:  0.05 hit / 0.10 hit / 0.15 miss   -> hit_pred = 2, n_pred = 3
        GT   -> 가장 가까운 예측: (3,3) 0.05 hit
                                 (0,0) hypot(3,4)*0.05 = 0.25 miss -> hit_gt = 1, n_gt = 2
        precision = 2/3,  recall = 1/2,  f1 = 4/7

    hit_pred(2) != hit_gt(1)이므로 거리장을 뒤바꾸면 precision 1/3, recall 1.0이 되어 깨진다.
    """
    gt = _grid([(3, 3), (0, 0)])
    pred = _grid([(3, 4), (3, 5), (3, 6)])
    valid = torch.ones_like(gt)

    result = summarize_tolerance_f1(
        [tolerance_counts(pred, gt, valid, CELL_M, tolerances=(0.10,))]
    )["10cm"]

    assert result["hit_pred"] == 2 and result["n_pred"] == 3
    assert result["hit_gt"] == 1 and result["n_gt"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(4 / 7)


def test_invalid_cells_are_excluded_from_both_sets():
    """`valid=0`은 수집 아티팩트라 지표에서 빠져야 한다. 예측이 그 안에서만 틀리면 만점이다."""
    gt = _grid([(3, 3)])
    pred = _grid([(3, 3), (0, 0)])
    valid = torch.ones_like(gt)
    valid[0, 0, 0, 0] = False  # 예측이 틀린 유일한 칸을 무효로 만든다

    result = summarize_tolerance_f1(
        [tolerance_counts(pred, gt, valid, CELL_M, tolerances=(0.10,))]
    )["10cm"]

    assert result["precision"] == pytest.approx(1.0)
    assert result["n_pred"] == 1


def _four_samples():
    """occupied 셀 수가 샘플마다 다른 4장. batch를 어떻게 자르든 결과가 같아야 한다.

    셀 수를 일부러 크게 다르게 둔다(1/1/4/4). 같으면 "배치별 F1 평균"과 "카운트 합산"이
    우연히 같은 값이 되어 batch-size 의존을 탐지할 수 없다.
    """
    return [
        (_grid([(3, 3)]), _grid([(3, 3)])),                       # 완전 정답 1칸
        (_grid([(1, 1)]), _grid([(6, 6)])),                       # 완전 오답 1칸
        (_grid([(3, i) for i in range(1, 5)]),
         _grid([(3, i) for i in range(1, 5)])),                   # 완전 정답 4칸
        (_grid([(5, i) for i in range(1, 5)]),
         _grid([(0, i) for i in range(1, 5)])),                   # 완전 오답 4칸
    ]


def _f1_over_batches(batches, tolerances=(0.10,)):
    dicts = []
    for group in batches:
        preds = torch.cat([pred for pred, _ in group])
        gts = torch.cat([gt for _, gt in group])
        valid = torch.ones_like(gts)
        dicts.append(tolerance_counts(preds, gts, valid, CELL_M, tolerances=tolerances))
    return summarize_tolerance_f1(dicts)["10cm"]


def test_f1_does_not_depend_on_how_the_samples_are_batched():
    """카운트를 합산하지 않고 배치별 F1을 평균하면 batch 크기에 따라 값이 달라진다.

    같은 함정을 range 백분위수에서 이미 한 번 밟았다(`summarize_range_error` 주석). 여기서는
    프레임당 occupied 셀 수가 크게 달라 그 편차가 더 크다.
    """
    samples = _four_samples()
    one_batch = _f1_over_batches([samples])
    two_batches = _f1_over_batches([samples[:2], samples[2:]])
    four_batches = _f1_over_batches([[sample] for sample in samples])

    assert one_batch["f1"] == pytest.approx(two_batches["f1"])
    assert one_batch["f1"] == pytest.approx(four_batches["f1"])
    # 자명한 값(0이나 1)이면 위 일치가 아무것도 증명하지 않는다.
    assert 0.0 < one_batch["f1"] < 1.0


def test_batch_averaged_f1_would_give_a_different_number():
    """위 불변성이 우연이 아님을 보인다 -- 옛 방식(배치별 F1의 단순 평균)은 값이 다르다.

    절단을 비대칭으로 둔다:
        전체 합산:  hit 5 / n 10  -> precision = recall = 0.5  -> f1 0.5
        배치 A = 샘플1(정답 1칸)               -> 1/1  -> f1 1.000
        배치 B = 샘플2,3,4(오답1+정답4+오답4)  -> 4/9  -> f1 0.444
        단순 평균 = (1.000 + 0.444) / 2 = 0.722   <- 참값의 1.44배

    참값 0.5가 두 배치 F1 사이에 있지도 않은 것이 요점이다 -- 배치별 평균은 셀이 적은
    샘플에 과도한 가중을 준다.
    """
    samples = _four_samples()
    pooled = _f1_over_batches([samples])
    assert pooled["f1"] == pytest.approx(0.5)

    first = _f1_over_batches([[samples[0]]])
    rest = _f1_over_batches([samples[1:]])
    assert first["f1"] == pytest.approx(1.0)
    assert rest["f1"] == pytest.approx(4 / 9)

    naive_average = (first["f1"] + rest["f1"]) / 2
    assert naive_average == pytest.approx((1.0 + 4 / 9) / 2)
    assert naive_average != pytest.approx(pooled["f1"])


def test_predicting_no_obstacle_at_all_scores_zero_not_nan():
    """"장애물을 하나도 예측하지 않았다"는 퇴행 해다. nan으로 빠져나가면 로그에서 `-`로
    보이고, 그 epoch이 best로 뽑히는 사고로 이어진다."""
    gt = _grid([(3, 3)])
    pred = torch.zeros_like(gt)
    valid = torch.ones_like(gt)

    result = summarize_tolerance_f1(
        [tolerance_counts(pred, gt, valid, CELL_M, tolerances=(0.10,))]
    )["10cm"]

    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_nothing_to_measure_is_nan():
    """GT에도 예측에도 occupied가 없으면 정말로 잴 것이 없다 -- 그때만 nan이다."""
    empty = torch.zeros(1, 1, 7, 7, dtype=torch.bool)
    valid = torch.ones_like(empty)

    result = summarize_tolerance_f1(
        [tolerance_counts(empty, empty, valid, CELL_M, tolerances=(0.10,))]
    )["10cm"]

    assert math.isnan(result["f1"])
    assert math.isnan(result["precision"])


def test_an_epoch_without_validation_summarises_to_an_empty_dict():
    """`empty_epoch_metrics`가 이 계약에 의존한다 -- 로그 포매터가 빈 dict를 받아 `-`를 찍는다."""
    assert summarize_tolerance_f1([]) == {}
