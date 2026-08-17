"""free-space 지표 -- M1 `iou_free`, M2 `fatal_rate`, M2b `free_miss_rate`.

왜 `iou_drivable`/`iou_obstacle`을 대신하는가: 자체 로봇 데이터에서 평가 마스크
(`vis & valid`) 안 drivable 비율이 93.5%라 "전부 drivable"이 `iou_drivable` 0.935를
받아 학습된 모델(0.891)을 이겼다. `iou_obstacle`은 두께 1셀 표면에 IoU를 적용한 것이라
한 칸 정렬 오차가 점수를 반토막 낸다. 두 head의 결합 결과인 `free`를 재면 트리비얼 해가
이기지 못한다 (모델 0.850 vs constant map 0.673 vs 전부 free ~0.17).

전부 `(B, 1, H, W)` bool 텐서만 받고 `(값, 가중치)`를 돌려준다 -- 가중치는 batch 간
가중평균용이며, 분모가 0인 batch를 0점으로 세지 않기 위해 필요하다.
"""
import torch

from projects.common.free_space import partition_defect_count


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
