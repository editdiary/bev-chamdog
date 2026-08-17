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


_RANGE_ABS_KEYS = ("abs_p50", "abs_p90")          # 전체 delta 통계 -> n_paired_rays로 가중
# 부분집합 통계 -> 그 부분집합의 크기로 가중해야 한다. 아래 summarize_range_error 참고.
_RANGE_PARTITION_KEYS = {"over_mean": "over_count", "under_mean": "under_count"}
_RANGE_COUNTS = ("n_paired_rays", "over_count", "under_count",
                 "censored_gt", "censored_pred", "no_free_gt")


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
    over, under = delta[delta > 0], -delta[delta < 0]
    counts["over_count"] = int(over.size)
    counts["under_count"] = int(under.size)
    if delta.size == 0:
        return {**dict.fromkeys(_RANGE_ABS_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_PARTITION_KEYS, float("nan")), **counts}
    return {
        "abs_p50": float(np.percentile(np.abs(delta), 50)),
        "abs_p90": float(np.percentile(np.abs(delta), 90)),
        # 빈 부분집합은 0.0이 아니라 0.0 그대로 둔다: over/under는 강한 부등호로 나눈
        # 진짜 분할이라 원소가 하나라도 있으면 평균이 0일 수 없다. 따라서 batch 안에서
        # 0.0은 "over 사건이 없었다"만 뜻하고 "상쇄돼 0"과 헷갈리지 않는다.
        "over_mean": float(over.mean()) if over.size else 0.0,
        "under_mean": float(under.mean()) if under.size else 0.0,
        **counts,
    }


def summarize_range_error(dicts) -> dict:
    """batch 단위 결과를 합친다.

    **가중치가 통계마다 다르다.** `abs_p50`/`abs_p90`은 delta 전체에 대한 통계라
    `n_paired_rays`로 가중하는 것이 맞지만, `over_mean`은 **over 사건들만**의 평균이므로
    `over_count`로 가중해야 한다. `n_paired_rays`로 가중하면 다음처럼 크게 틀어진다:

        batch A: paired 100, over 1건 x 1.00 m -> over_mean 1.00
        batch B: paired 100, over 99건 x 0.01 m -> over_mean 0.01
        올바른 값 (1*1.00 + 99*0.01)/100 = 0.02
        n_paired_rays 가중 (1.00*100 + 0.01*100)/200 = 0.505   <- 25배 과대

    over-prediction은 "통로가 실제보다 길다"는 안전 실패라 이 숫자가 그대로 배포 판단과
    논문에 들어간다. 카운트는 **합**으로 둔다 -- batch당 평균으로 나누면 batch 크기가
    바뀔 때 숫자가 따라 움직여 run 간 비교가 안 된다.
    """
    if not dicts:
        return {**dict.fromkeys(_RANGE_ABS_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_PARTITION_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_COUNTS, 0)}
    paired = [d["n_paired_rays"] for d in dicts]
    result = {key: weighted_mean([d[key] for d in dicts], paired) for key in _RANGE_ABS_KEYS}
    for key, count_key in _RANGE_PARTITION_KEYS.items():
        result[key] = weighted_mean([d[key] for d in dicts], [d[count_key] for d in dicts])
    for key in _RANGE_COUNTS:
        result[key] = sum(d[key] for d in dicts)
    return result
