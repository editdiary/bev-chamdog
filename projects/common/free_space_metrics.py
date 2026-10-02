"""free-space 지표 -- `iou_free`, `iou_non_free`, `precision`, `recall` (전부 프레임 macro).

**2026-10-02 지표 개편.** 이름·눈금·집계 규칙의 정본은 `projects/common/metric_spec.py`다.
`fatal_rate`(= 1 − precision)와 `free_miss_rate`(= 1 − recall)는 **셀을 한 통에 부은 micro**
였고 `iou_free`만 프레임 macro였다. 논문 표 안에서 집계가 섞이지 않도록 전부 macro로 바꾸고
이름을 precision/recall로 적는다. 옛 두 함수와 M3(range) 함수는 옛 런 재채점용 legacy로
남긴다 -- 학습 로그와 논문 패키지에는 더 이상 나가지 않는다.

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
from projects.common.metric_spec import RING_EDGES_M
from projects.common.polar import RAY_CENSORED, RAY_NO_FREE, RAY_OK, first_free_range


def weighted_mean(values, weights) -> float:
    """weight가 0인 항목은 값이 NaN일 수 있으므로(= 셀 수 있는 샘플이 없던 batch) 건너뛴다."""
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total


def iou_masked(pred_mask, gt_mask, valid):
    """per-sample IoU의 평균과, 평균에 실제로 들어간 샘플 수.

    GT에도 예측에도 그 클래스가 없는 샘플은 union이 0이라 IoU가 정의되지 않는다. 0점으로 세면
    "그 클래스가 없는 장면을 완벽히 맞혔는데 0점"이 되어 지표가 왜곡되므로 평균에서 뺀다.
    """
    valid_b = valid.bool()
    pred, gt = pred_mask.bool() & valid_b, gt_mask.bool() & valid_b
    dims = list(range(1, pred.ndim))
    intersection = (pred & gt).sum(dim=dims).float()
    union = (pred | gt).sum(dim=dims).float()
    has_union = union > 0
    count = int(has_union.sum().item())
    if count == 0:
        return float("nan"), 0
    return float((intersection[has_union] / union[has_union]).mean().item()), count


def iou_free(free_pred, free_gt, valid):
    """M1. 주 지표. `iou_masked`의 free 전용 이름이다.

    이름을 남겨 두는 이유: 체크포인트 선택과 baseline 대조가 전부 이 이름을 통과하므로,
    일반화된 함수로 바꾸면서 호출부가 다른 클래스를 실수로 넣는 일이 없도록 고정한다.
    """
    return iou_masked(free_pred, free_gt, valid)


def iou_non_free(free_pred, free_gt, valid):
    """non-free 클래스 IoU. `valid` 안에서 free가 아닌 셀 전부(occupied ∪ unknown)다.

    **변별력이 낮다는 것을 알고 읽는다.** non-free가 `valid`의 약 83 %라서 "전부 non-free"라는
    자명해가 이미 ≈ 0.83을 받는다. 논문 본문 지표가 아니라 보조 열이다(사용자 결정 2026-10-02).
    """
    valid_b = valid.bool()
    return iou_masked(~free_pred.bool() & valid_b, ~free_gt.bool() & valid_b, valid_b)


def _frame_confusion(free_pred, free_gt, valid):
    """프레임마다 `(TP, FP, FN)` -- 셀 수, `(B,)` float."""
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    dims = list(range(1, pred.ndim))
    return ((pred & gt).sum(dim=dims).float(), (pred & ~gt).sum(dim=dims).float(),
            (~pred & gt).sum(dim=dims).float())


def _macro_ratio(numerator, denominator, defined):
    """프레임별 비율의 평균과 평균에 들어간 프레임 수.

    분모가 0인 정의된 프레임은 0점이다 -- 예: GT free가 있는데 예측 free가 하나도 없으면
    precision 0. 그 프레임을 빼면 "아무것도 주장하지 않는" 퇴행 해가 평균에서 빠져나간다.
    """
    count = int(defined.sum().item())
    if count == 0:
        return float("nan"), 0
    ratio = torch.where(denominator > 0, numerator / denominator.clamp(min=1.0),
                        torch.zeros_like(numerator))
    return float(ratio[defined].mean().item()), count


def free_precision(free_pred, free_gt, valid):
    """free Precision = TP / (TP + FP), 프레임 macro. 옛 `fatal_rate`(micro)의 보수다.

    **혼자 읽으면 속는다.** macro에서는 free를 몇 셀만 예측하고 맞힌 프레임이 1.0을 받아 큰
    프레임과 같은 표를 행사한다. 반드시 `free_recall`과 같이 읽는다.
    """
    tp, fp, fn = _frame_confusion(free_pred, free_gt, valid)
    return _macro_ratio(tp, tp + fp, (tp + fp + fn) > 0)


def free_recall(free_pred, free_gt, valid):
    """free Recall = TP / (TP + FN), 프레임 macro. 옛 `free_miss_rate`(micro)의 보수다."""
    tp, fp, fn = _frame_confusion(free_pred, free_gt, valid)
    return _macro_ratio(tp, tp + fn, (tp + fp + fn) > 0)


def _rate(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum().item())
    if denominator == 0:
        return float("nan"), 0
    return int(numerator_mask.sum().item()) / denominator, denominator


def fatal_rate(free_pred, free_gt, valid):
    """**legacy(micro)** -- 2026-10-02 이후 학습 로그에 쓰지 않는다. 옛 런 재채점용.

    M2. "갈 수 있다고 믿은 곳 중 틀린 비율" = 1 - precision(free).

    분모를 `|free_pred|`로 잡는 이유: `|~free_gt|`로 잡으면 분모가 격자의 83%라 값이 항상
    작게 나와 변별력이 없다 (같은 체크포인트에서 0.0113 vs 0.0587). planner 관점에서도
    "내가 신뢰한 영역의 오류율"이 직접적인 위험량이다.
    """
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(pred & ~gt, pred)


def free_miss_rate(free_pred, free_gt, valid):
    """**legacy(micro)** -- 옛 런 재채점용. M2b. 보수성(정보 낭비) = 1 - recall(free)."""
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(~pred & gt, gt)


def free_metrics_from_masks(pred_parts, gt_parts, valid) -> dict:
    """셀 단위 지표 전부 + 분할 불변식을 한 번에. 학습 루프와 재채점이 공유한다.

    집계를 호출부마다 복사해 두면 한쪽만 고쳐지는 순간 두 학습 스크립트의 숫자를 나란히
    읽을 수 없으므로 여기 한 벌만 둔다.

    **여기서 재는 것은 넷이다: `iou_free`, `iou_non_free`, `precision`, `recall`** -- 전부
    프레임 macro이고 `*_count`는 평균에 들어간 프레임 수다(batch 간 가중평균용).
    2026-10-02에 `fatal_rate`/`free_miss_rate`(micro)를 precision/recall(macro)로 바꿨다.

    `iou_non_free`는 아래 2026-08-21의 삭제 사유에 걸리지 않는다: 유도된 1셀 표면도 아니고
    task 정의를 바꾸지도 않는다 -- binary head가 실제로 내는 둘째 클래스의 IoU다.

    2026-08-21에 지운 세 항목과 그 이유(되살리려는 유혹을 막기 위해 남긴다):

    - `iou_occupied`/`iou_unknown`: binary 정식화에서 `occupied`는 head가 없고 예측 free의
      경계에서 **유도**되며 `unknown`은 그 나머지다. 즉 둘 다 `free`의 결정론적 함수라 독립
      정보가 없다. 게다가 두께 1셀 표면에 면적 IoU를 씌운 값은 한 칸 밀리면 반토막 나서
      val 0.071까지 떨어졌다 -- 품질 신호로 읽을 수 없는 숫자다. 경계 정밀도는
      `occupied_metrics`의 tolerance F1이 재고, 그쪽이 이 역할을 완전히 대체한다.
    - `iou_free_known`: `valid`를 GT가 관측한 셀(free ∪ occupied)로 좁힌 `iou_free`였다.
      **task 정의와 충돌한다.** 이 프로젝트의 (D) 정식화는 "보이면서 빈 곳"만 drivable이고
      **보이지 않는 곳은 전부 non-drivable**이다. `unknown`을 채점에서 빼면 라벨이
      non-drivable이라고 선언한 셀의 78.5 %를 빼는 것이므로 다른 task를 재게 된다.
      실측 근거는 `docs/finetune_overfitting_diagnosis.md` §22.4/§23에 남아 있다.

    마스크를 함께 실어 보내는 이유: BF@τ는 광선 루프와 거리변환이 CPU numpy라 학습 step마다
    돌리면 병목이 된다. val 경로가 forward를 다시 하지 않고 이 마스크를 받아 따로 계산한다.
    """
    pred_free, gt_free = pred_parts["free"], gt_parts["free"]
    pred_occupied, gt_occupied = pred_parts["occupied"], gt_parts["occupied"]
    return {
        **free_scores(pred_free, gt_free, valid),
        # 배선이 틀리면 조용히 이상한 숫자가 나오는 대신 여기서 0이 아니게 된다.
        "partition_defects": partition_defect_count(gt_parts, valid.bool()),
        "pred_free": pred_free, "gt_free": gt_free,
        "pred_occupied": pred_occupied, "gt_occupied": gt_occupied,
    }


def free_scores(free_pred, free_gt, valid) -> dict:
    """`{지표: 값, 지표_count: 프레임 수}` -- `metric_spec.FREE_METRICS` 넷. 링도 같은 함수를 탄다."""
    out = {}
    for name, fn in (("iou_free", iou_free), ("iou_non_free", iou_non_free),
                     ("precision", free_precision), ("recall", free_recall)):
        out[name], out[f"{name}_count"] = fn(free_pred, free_gt, valid)
    return out


# --- 이하 M3(range)는 **legacy**다 ---------------------------------------------------------
# 2026-10-02 사용자 결정으로 학습 로그·논문 패키지에서 뺐다. 옛 런 재채점
# (`tools/rescore_checkpoints.py`)과 옛 연구 도구가 쓰므로 계산 함수는 남긴다.

_RANGE_STAT_KEYS = ("mae", "abs_p50", "abs_p90", "bias", "over_mean", "under_mean")
_RANGE_COUNTS = ("n_paired_rays", "over_count", "under_count",
                 "censored_gt", "censored_pred", "no_free_gt",
                 "ok_gt", "missed_obstacle")
_RANGE_RATE_KEYS = ("missed_obstacle_rate",)


def _delta_stats(delta: np.ndarray) -> dict:
    """`dr = r_pred - r_gt` 표본 하나에서 range 통계를 낸다. 단위는 전부 meter다.

    batch별로도, epoch 전체를 모은 뒤에도 **같은 함수**를 쓴다 -- 그래야 "배치별로 낸 값"과
    "전체로 낸 값"이 정의상 같은 것이 되고, 집계가 통계 종류마다 다른 규칙을 갖지 않는다.

    **부호 규약: `dr > 0`은 장애물을 실제보다 멀다고 예측한 것 = free의 과대추정 = 위험한
    쪽이다.** `over_*`가 그 쪽이다. 다른 문헌은 같은 사건을 "장애물의 과소추정"이라 부르므로
    `over`/`under`라는 단어만 옮겨 읽으면 부호가 뒤집힌다.

    `mae`와 `abs_p50`/`abs_p90`을 함께 두는 이유: 실측 분포가 꼬리가 두꺼워(p50 0.175 m 대
    p90 0.700 m) 평균은 소수 광선에 끌려다니고, 백분위수만 보면 "평균 몇 cm 틀리나"에
    답할 수 없다. `bias`는 `over`/`under`가 상쇄된 순수 편향이라 보수적/낙관적 성향을 한
    숫자로 보여준다.
    """
    if delta.size == 0:
        return dict.fromkeys(_RANGE_STAT_KEYS, float("nan"))
    over, under = delta[delta > 0], -delta[delta < 0]
    return {
        "mae": float(np.abs(delta).mean()),
        "abs_p50": float(np.percentile(np.abs(delta), 50)),
        "abs_p90": float(np.percentile(np.abs(delta), 90)),
        "bias": float(delta.mean()),
        # 빈 부분집합은 0.0으로 둔다: over/under는 강한 부등호로 나눈 진짜 분할이라 원소가
        # 하나라도 있으면 평균이 0일 수 없다. 따라서 0.0은 "over 사건이 없었다"만 뜻한다.
        "over_mean": float(over.mean()) if over.size else 0.0,
        "under_mean": float(under.mean()) if under.size else 0.0,
    }


def _delta_rates(counts: dict) -> dict:
    """카운트에서 유도되는 비율. **합산된 카운트에서 다시 계산해야** batch-size 불변이다.

    `missed_obstacle_rate` = (GT에는 장애물이 있는데 예측은 격자 끝까지 free인 광선) /
    (GT에 장애물이 있는 광선). 이것이 없으면 위의 거리 통계가 정직하지 않다 -- 회귀 통계는
    GT와 예측이 **둘 다** `RAY_OK`인 광선만 쓰므로, 장애물을 완전히 놓친 광선은 표본에서
    빠진다. 즉 많이 놓치면 `mae`가 오히려 **좋아진다.** 두 숫자는 항상 같이 읽는다.
    """
    ok = counts["ok_gt"]
    return {"missed_obstacle_rate":
            counts["missed_obstacle"] / ok if ok else float("nan")}


def range_error(free_pred, free_gt, valid, rays) -> dict:
    """M3. 방위각별 "첫 장애물까지의 거리" 오차 `dr = r_pred - r_gt`.

    GT와 예측이 **둘 다** `RAY_OK`인 광선만 회귀 통계에 넣는다. censored(격자 끝까지 free)를
    r_max로 대체해 섞으면 통계가 그 상수에 눌린다.

    numpy로 계산하므로 CPU로 내린다 -- 학습 루프의 매 step이 아니라 val에서만 부르는 것을
    전제로 한다(광선 루프가 batch당 수 ms 든다).
    """
    pred_np = (free_pred.bool() & valid.bool()).cpu().numpy()
    gt_np = (free_gt.bool() & valid.bool()).cpu().numpy()

    deltas, gt_ranges, counts = [], [], dict.fromkeys(_RANGE_COUNTS, 0)
    for i in range(gt_np.shape[0]):
        r_gt, s_gt = first_free_range(gt_np[i, 0], rays)
        r_pred, s_pred = first_free_range(pred_np[i, 0], rays)
        ok_gt = s_gt == RAY_OK
        counts["censored_gt"] += int((s_gt == RAY_CENSORED).sum())
        counts["censored_pred"] += int((s_pred == RAY_CENSORED).sum())
        counts["no_free_gt"] += int((s_gt == RAY_NO_FREE).sum())
        counts["ok_gt"] += int(ok_gt.sum())
        # 놓친 장애물: GT에는 있는데 예측은 격자 끝까지 free. `RAY_NO_FREE`는 세지 않는다 --
        # 그건 "전부 막혔다고 봤다"는 과잉 보수라서 비용의 종류가 정반대다.
        counts["missed_obstacle"] += int((ok_gt & (s_pred == RAY_CENSORED)).sum())
        paired = ok_gt & (s_pred == RAY_OK)
        deltas.append(r_pred[paired] - r_gt[paired])
        gt_ranges.append(r_gt[paired])

    delta = np.concatenate(deltas) if deltas else np.empty(0)
    gt_range = np.concatenate(gt_ranges) if gt_ranges else np.empty(0)
    counts["n_paired_rays"] = int(delta.size)
    counts["over_count"] = int((delta > 0).sum())
    counts["under_count"] = int((delta < 0).sum())
    # `deltas`를 그대로 실어 보낸다 -- `summarize_range_error`가 epoch 전체를 모아 한 번에
    # 백분위수를 내야 하기 때문이다. batch당 수천 개 float이라 비용은 무시할 수준이다.
    # `gt_range`는 같은 표본을 GT 거리로 층화하기 위해 짝지어 보관한다.
    return {"deltas": delta, "gt_ranges": gt_range,
            **_delta_stats(delta), **counts, **_delta_rates(counts)}


def summarize_range_error(dicts) -> dict:
    """batch 단위 결과를 합친다. **delta 표본을 모두 모아 한 번에 통계를 낸다.**

    백분위수는 평균낼 수 없다 -- 배치별 중위수의 가중평균은 전체 분포의 중위수가 아니다.
    예: 전체 |dr|이 [0.1, 0.1, 0.9]이면 참값은 0.10인데, [0.1, 0.9]와 [0.1]로 잘라
    가중평균하면 (0.5*2 + 0.1*1)/3 = 0.37이 나온다. 자르는 방식만 바뀌어도 숫자가 달라진다.

    실제로 같은 체크포인트를 bs4/bs8로 재채점했을 때 `abs_p50`이 0.177 vs 0.172,
    `abs_p90`이 0.701 vs 0.718로 갈렸고, 스위트에서 이 두 지표만 batch-size 불변이 아니었다
    (`docs/archive/free_space_metric_migration.md` §9.4). 표본을 모아 한 번에 계산하면 그 의존이
    정의상 사라진다.

    `over_mean`/`under_mean`도 같은 표본에서 직접 낸다. 예전에는 부분집합 평균이라
    `over_count`로 가중해야 하는 별도 규칙이 필요했는데(`n_paired_rays`로 가중하면 25배까지
    틀어졌다), 모아서 계산하면 그 규칙 자체가 필요 없어진다.

    카운트는 **합**으로 둔다 -- batch당 평균으로 나누면 batch 크기가 바뀔 때 숫자가 따라
    움직여 run 간 비교가 안 된다.
    """
    if not dicts:
        return {**dict.fromkeys(_RANGE_STAT_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_RATE_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_COUNTS, 0)}
    delta = np.concatenate([d["deltas"] for d in dicts])
    result = _delta_stats(delta)
    for key in _RANGE_COUNTS:
        result[key] = sum(d[key] for d in dicts)
    result.update(_delta_rates(result))
    return result


def summarize_range_error_by_gt_range(dicts, edges_m) -> dict:
    """같은 표본을 **GT 거리**로 층화한다 -- "이 거리의 장애물을 얼마나 정확히 찾나".

    M4(`metrics_per_ring`)와 다른 것을 잰다: M4는 셀을 링 마스크로 나누는데, 광선 하나는
    여러 링을 지나므로 거리 오차를 셀 기준으로 나눌 수 없다. 그래서 광선을 그 광선의
    **GT 거리**로 묶는다.

    예측 거리로 묶으면 안 된다 -- 구간의 정의 자체가 모델에 따라 움직여 run 간 비교가
    무의미해진다. GT 거리로 묶으면 구간은 데이터가 고정한다.

    왜 필요한가: 전체 평균 하나로는 "근거리 5 cm / 원거리 1.5 m"인 모델과 "전 구간 40 cm"인
    모델이 구별되지 않는다. BEV 변환이 거리에 따라 기하를 얼마나 잘 복원하는지가 여기서
    드러난다.
    """
    if not dicts:
        return {}
    delta = np.concatenate([d["deltas"] for d in dicts])
    gt_range = np.concatenate([d["gt_ranges"] for d in dicts])
    result = {}
    for lo, hi in zip(edges_m[:-1], edges_m[1:]):
        selected = (gt_range >= lo) & (gt_range < hi)
        result[f"{lo}-{hi}m"] = {**_delta_stats(delta[selected]),
                                 "n_paired_rays": int(selected.sum())}
    return result


DEFAULT_RING_EDGES_M = RING_EDGES_M


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
    """링마다 `free_scores` 넷을 다시 잰다. `valid`에 링 마스크를 곱해 같은 함수를 재사용한다.

    링 밖 셀은 그 링의 채점에서 빠지므로, 링 안에 예측·GT free가 둘 다 없는 프레임은 그 링의
    평균에서 빠진다(`*_count`가 줄어든다).
    """
    result = {}
    for name, mask in ring_masks:
        ring = torch.from_numpy(mask).to(valid.device).view(1, 1, *mask.shape)
        result[name] = free_scores(free_pred, free_gt, valid.bool() & ring)
    return result
