"""free-space 지표 -- M1 `iou_free`, M2 `fatal_rate`, M2b `free_miss_rate`.

왜 `iou_drivable`/`iou_obstacle`을 대신하는가: 자체 로봇 데이터에서 평가 마스크
(`vis & valid`) 안 drivable 비율이 93.5%라 "전부 drivable"이 `iou_drivable` 0.935를
받아 학습된 모델(0.891)을 이겼다. `iou_obstacle`은 두께 1셀 표면에 IoU를 적용한 것이라
한 칸 정렬 오차가 점수를 반토막 낸다. 두 head의 결합 결과인 `free`를 재면 트리비얼 해가
이기지 못한다 (모델 0.850 vs constant map 0.673 vs 전부 free ~0.17).

전부 `(B, 1, H, W)` bool 텐서만 받고 `(값, 가중치)`를 돌려준다 -- 가중치는 batch 간
가중평균용이며, 분모가 0인 batch를 0점으로 세지 않기 위해 필요하다.
"""
import numpy as np
import torch

from projects.common.free_space import partition_defect_count
from projects.common.polar import RAY_CENSORED, RAY_NO_FREE, RAY_OK, first_free_range


def weighted_mean(values, weights) -> float:
    """weight가 0인 항목은 값이 NaN일 수 있으므로(= 셀 수 있는 샘플이 없던 batch) 건너뛴다."""
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total


def iou_free(free_pred, free_gt, valid):
    """M1. per-sample IoU의 평균과, 평균에 실제로 들어간 샘플 수.

    GT에도 예측에도 free가 없는 샘플은 union이 0이라 IoU가 정의되지 않는다. 0점으로 세면
    "free가 없는 장면을 완벽히 맞혔는데 0점"이 되어 지표가 왜곡되므로 평균에서 뺀다.
    """
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    dims = list(range(1, pred.ndim))
    intersection = (pred & gt).sum(dim=dims).float()
    union = (pred | gt).sum(dim=dims).float()
    has_union = union > 0
    count = int(has_union.sum().item())
    if count == 0:
        return float("nan"), 0
    return float((intersection[has_union] / union[has_union]).mean().item()), count


def _rate(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum().item())
    if denominator == 0:
        return float("nan"), 0
    return int(numerator_mask.sum().item()) / denominator, denominator


def fatal_rate(free_pred, free_gt, valid):
    """M2. "갈 수 있다고 믿은 곳 중 틀린 비율" = 1 - precision(free).

    분모를 `|free_pred|`로 잡는 이유: `|~free_gt|`로 잡으면 분모가 격자의 83%라 값이 항상
    작게 나와 변별력이 없다 (같은 체크포인트에서 0.0113 vs 0.0587). planner 관점에서도
    "내가 신뢰한 영역의 오류율"이 직접적인 위험량이다.
    """
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(pred & ~gt, pred)


def free_miss_rate(free_pred, free_gt, valid):
    """M2b. 보수성(정보 낭비). `fatal_rate`와 비용이 다르므로 절대 합치지 않는다."""
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(~pred & gt, gt)


def free_metrics_from_masks(pred_free, gt_parts, valid) -> dict:
    """M1·M2·M2b + 분할 불변식을 한 번에. **2-head와 3-class가 공유한다.**

    두 정식화는 `pred_free`를 만드는 방식만 다르고(두 sigmoid의 AND vs. softmax argmax)
    그 이후 집계는 완전히 같다. 집계를 각자 복사해 두면 한쪽만 고쳐지는 순간 A/B가
    무의미해지므로 여기 한 벌만 둔다.

    `pred_free`/`gt_free`를 함께 실어 보내는 이유: M3·M4는 광선 루프가 CPU numpy라 학습
    step마다 돌리면 병목이 된다(이 리그는 이미 데이터 로딩이 병목이다). val 경로가
    forward를 다시 하지 않고 이 마스크를 받아 따로 계산한다.
    """
    gt_free = gt_parts["free"]
    iou, iou_count = iou_free(pred_free, gt_free, valid)
    fatal, fatal_denom = fatal_rate(pred_free, gt_free, valid)
    miss, miss_denom = free_miss_rate(pred_free, gt_free, valid)
    return {
        "iou_free": iou, "iou_free_count": iou_count,
        "fatal_rate": fatal, "fatal_denom": fatal_denom,
        "free_miss_rate": miss, "free_miss_denom": miss_denom,
        # 배선이 틀리면 조용히 이상한 숫자가 나오는 대신 여기서 0이 아니게 된다.
        "partition_defects": partition_defect_count(gt_parts, valid.bool()),
        "pred_free": pred_free, "gt_free": gt_free,
    }


_RANGE_STAT_KEYS = ("abs_p50", "abs_p90", "over_mean", "under_mean")
_RANGE_COUNTS = ("n_paired_rays", "over_count", "under_count",
                 "censored_gt", "censored_pred", "no_free_gt")


def _delta_stats(delta: np.ndarray) -> dict:
    """`dr = r_pred - r_gt` 표본 하나에서 range 통계 네 개를 낸다.

    batch별로도, epoch 전체를 모은 뒤에도 **같은 함수**를 쓴다 -- 그래야 "배치별로 낸 값"과
    "전체로 낸 값"이 정의상 같은 것이 되고, 집계가 통계 종류마다 다른 규칙을 갖지 않는다.
    """
    if delta.size == 0:
        return dict.fromkeys(_RANGE_STAT_KEYS, float("nan"))
    over, under = delta[delta > 0], -delta[delta < 0]
    return {
        "abs_p50": float(np.percentile(np.abs(delta), 50)),
        "abs_p90": float(np.percentile(np.abs(delta), 90)),
        # 빈 부분집합은 0.0으로 둔다: over/under는 강한 부등호로 나눈 진짜 분할이라 원소가
        # 하나라도 있으면 평균이 0일 수 없다. 따라서 0.0은 "over 사건이 없었다"만 뜻한다.
        "over_mean": float(over.mean()) if over.size else 0.0,
        "under_mean": float(under.mean()) if under.size else 0.0,
    }


def range_error(free_pred, free_gt, valid, rays) -> dict:
    """M3. 방위각별 "첫 장애물까지의 거리" 오차 `dr = r_pred - r_gt`.

    GT와 예측이 **둘 다** `RAY_OK`인 광선만 회귀 통계에 넣는다. censored(격자 끝까지 free)를
    r_max로 대체해 섞으면 통계가 그 상수에 눌린다.

    numpy로 계산하므로 CPU로 내린다 -- 학습 루프의 매 step이 아니라 val에서만 부르는 것을
    전제로 한다(광선 루프가 batch당 수 ms 든다).
    """
    pred_np = (free_pred.bool() & valid.bool()).cpu().numpy()
    gt_np = (free_gt.bool() & valid.bool()).cpu().numpy()

    deltas, counts = [], dict.fromkeys(_RANGE_COUNTS, 0)
    for i in range(gt_np.shape[0]):
        r_gt, s_gt = first_free_range(gt_np[i, 0], rays)
        r_pred, s_pred = first_free_range(pred_np[i, 0], rays)
        counts["censored_gt"] += int((s_gt == RAY_CENSORED).sum())
        counts["censored_pred"] += int((s_pred == RAY_CENSORED).sum())
        counts["no_free_gt"] += int((s_gt == RAY_NO_FREE).sum())
        paired = (s_gt == RAY_OK) & (s_pred == RAY_OK)
        deltas.append(r_pred[paired] - r_gt[paired])

    delta = np.concatenate(deltas) if deltas else np.empty(0)
    counts["n_paired_rays"] = int(delta.size)
    counts["over_count"] = int((delta > 0).sum())
    counts["under_count"] = int((delta < 0).sum())
    # `deltas`를 그대로 실어 보낸다 -- `summarize_range_error`가 epoch 전체를 모아 한 번에
    # 백분위수를 내야 하기 때문이다. batch당 수천 개 float이라 비용은 무시할 수준이다.
    return {"deltas": delta, **_delta_stats(delta), **counts}


def summarize_range_error(dicts) -> dict:
    """batch 단위 결과를 합친다. **delta 표본을 모두 모아 한 번에 통계를 낸다.**

    백분위수는 평균낼 수 없다 -- 배치별 중위수의 가중평균은 전체 분포의 중위수가 아니다.
    예: 전체 |dr|이 [0.1, 0.1, 0.9]이면 참값은 0.10인데, [0.1, 0.9]와 [0.1]로 잘라
    가중평균하면 (0.5*2 + 0.1*1)/3 = 0.37이 나온다. 자르는 방식만 바뀌어도 숫자가 달라진다.

    실제로 같은 체크포인트를 bs4/bs8로 재채점했을 때 `abs_p50`이 0.177 vs 0.172,
    `abs_p90`이 0.701 vs 0.718로 갈렸고, 스위트에서 이 두 지표만 batch-size 불변이 아니었다
    (`docs/free_space_metric_migration.md` §9.4). 표본을 모아 한 번에 계산하면 그 의존이
    정의상 사라진다.

    `over_mean`/`under_mean`도 같은 표본에서 직접 낸다. 예전에는 부분집합 평균이라
    `over_count`로 가중해야 하는 별도 규칙이 필요했는데(`n_paired_rays`로 가중하면 25배까지
    틀어졌다), 모아서 계산하면 그 규칙 자체가 필요 없어진다.

    카운트는 **합**으로 둔다 -- batch당 평균으로 나누면 batch 크기가 바뀔 때 숫자가 따라
    움직여 run 간 비교가 안 된다.
    """
    if not dicts:
        return {**dict.fromkeys(_RANGE_STAT_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_COUNTS, 0)}
    delta = np.concatenate([d["deltas"] for d in dicts])
    result = _delta_stats(delta)
    for key in _RANGE_COUNTS:
        result[key] = sum(d[key] for d in dicts)
    return result


DEFAULT_RING_EDGES_M = (0.0, 1.5, 3.0, 4.0)


def build_ring_masks(grid_spec, edges_m=DEFAULT_RING_EDGES_M):
    """원점으로부터의 거리 링 마스크. 격자가 전후 비대칭이라 바깥 링은 격자에서 잘린다 --
    그래도 "근거리는 맞는데 원거리에서 무너진다"를 보는 목적에는 충분하다."""
    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    rows, cols = np.mgrid[0:grid_spec.n_rows, 0:grid_spec.n_cols]
    distance_m = np.hypot(rows - origin_row, cols - origin_col) * grid_spec.cell_m
    return [
        (f"{lo}-{hi}m", (distance_m >= lo) & (distance_m < hi))
        for lo, hi in zip(edges_m[:-1], edges_m[1:])
    ]


def metrics_per_ring(free_pred, free_gt, valid, ring_masks) -> dict:
    """링마다 M1·M2를 다시 잰다. `valid`에 링 마스크를 곱해 같은 함수를 재사용한다."""
    result = {}
    for name, mask in ring_masks:
        ring = torch.from_numpy(mask).to(valid.device).view(1, 1, *mask.shape)
        ring_valid = valid.bool() & ring
        iou, iou_count = iou_free(free_pred, free_gt, ring_valid)
        rate, denom = fatal_rate(free_pred, free_gt, ring_valid)
        result[name] = {
            "iou_free": iou, "iou_free_count": iou_count,
            "fatal_rate": rate, "fatal_denom": denom,
        }
    return result
