"""경계 지표 -- **Boundary F-measure `BF@τ`** (2026-10-02까지 이름은 `f1@τ`/`occupied_f1_*`).

**무엇을 재나 -- 전체 셀이 아니라 경계 셀만이다.** 아래 `O_pred`/`O_gt`는 ego 원점 광선이
처음 멈춘 **두께 1셀 표면**(격자의 약 1.1 %)이다. 예측 쪽은 `derive_occupied`가 예측 free에서
유도하고, GT 쪽은 라벨의 `occupied = ~occ & vis`(같은 ego raycast)다. 그래서 이름을 BF로
바꿨다(사용자 지적, 2026-10-02). **DAVIS 계열 BF와 다른 점:** 그쪽은 분할 경계 전체(contour)를
쓰지만 여기 경계는 **ego에서 보이는 free 공간의 첫 표면**이다 -- 물체 뒤편 윤곽은 경계
집합에 없다. 논문에 이 정의를 한 줄로 적는다.

**집계는 프레임 macro다**(`boundary_f_scores` -> `summarize_boundary_f`). 프레임마다 P·R·F를
내고 프레임 평균한다. 예전 `tolerance_counts`/`summarize_tolerance_f1`(micro, τ 0.10/0.20/0.40)은
옛 런 재채점용 legacy로 남는다.

**왜 면적 IoU가 아닌가.** `occupied`는 광선이 멈춘 두께 1셀 표면이라 예측이 한 칸 옆으로
밀리기만 해도 교집합이 0이 되어 IoU가 무너진다. 실측으로 `iou_obstacle` 0.312가 이미지를
보지 않는 frontier 규칙 0.473에 졌고(`docs/archive/free_space_metric_migration.md` §1) 그래서 이
프로젝트는 occupied 면적 IoU를 주 지표에서 내렸다. 그런데 그 자리를 비워 두면 **"장애물을
얼마나 잘 찾나"를 직접 재는 지표가 하나도 없다** -- M1/M2/M2b는 전부 free 기준이고, M3는
방위각마다 첫 경계 하나만 본다.

**해결 방향: 라벨을 두껍게 만들지 않고 지표에 허용오차를 준다.** GT를 dilation해서 IoU를
안정화하는 것은 지표 문제를 라벨 정의로 감추는 것이라 평가의 의미가 흐려진다. GT 두께는
센서·맵 구축의 물리적 정의가 결정하게 두고, "몇 cm까지 맞은 것으로 볼지"를 지표 쪽에서
명시적인 파라미터 τ로 노출한다.

    d(p, O_gt) = min_{g in O_gt} ||p - g||_2      (예측 occupied 셀 -> 가장 가까운 GT occupied)
    precision_τ = |{p in O_pred : d(p, O_gt) <= τ}| / |O_pred|
    recall_τ    = |{g in O_gt   : d(g, O_pred) <= τ}| / |O_gt|
    f1_τ        = 2 P R / (P + R)

여러 τ를 동시에 보고한다. 하나만 고르면 그 값이 곧 숨은 하이퍼파라미터가 되고, 여러 개를
나란히 두면 "얼마나 밀렸나"의 분포 자체가 읽힌다.

**M3와 짝을 이룬다.** M3의 광선은 free가 끊기는 곳에서 멈추므로 예측 `unknown`도 장애물처럼
광선을 세운다 -- 애매한 영역을 전부 unknown으로 칠하면 거리 오차가 좋아지는 구멍이 있다.
그 해는 여기서 occupied recall이 떨어져 벌을 받는다. 반대로 이 지표는 첫 표면 뒤쪽까지
전부 세므로 "planner가 실제로 쓰는 자유거리"를 재지 못한다. 두 지표는 서로의 사각을 덮는다.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from projects.common.metric_spec import BF_TOLERANCES_M, tolerance_key  # noqa: F401
from projects.common.npsafe import bool_not

from projects.common.polar import first_free_range, frontier_cells

# 2 / 4 / 6셀(0.05 m 격자). 셀 하나(0.05 m)는 라벨 자체의 이산화 오차와 구별되지 않아 넣지
# 않았다. 정본은 `metric_spec.BF_TOLERANCES_M`이다.
DEFAULT_TOLERANCES_M = BF_TOLERANCES_M
# 2026-10-02 이전 눈금. legacy `tolerance_counts`의 기본값으로만 남는다 -- 옛 도구가 옛 런을
# 다시 읽을 때 숫자가 바뀌지 않게 하려는 것이다.
LEGACY_TOLERANCES_M = (0.10, 0.20, 0.40)

_COUNT_KEYS = ("hit_pred", "n_pred", "hit_gt", "n_gt")
_BF_SUM_KEYS = ("bf_sum", "precision_sum", "recall_sum", "n_frames")


def _distance_field_m(mask: np.ndarray, cell_m: float):
    """`mask`의 가장 가까운 True까지의 거리 [m]. `mask`가 비면 `None`.

    비었을 때 거리는 정의되지 않는다(scipy는 배경이 없는 입력에 큰 상수를 채워 넣는데 그것을
    거리로 읽으면 조용히 틀린다). 호출부가 `None`을 보고 그 방향의 hit을 0으로 센다.
    """
    if not mask.any():
        return None
    # **`~mask`를 직접 넘기지 않는다** -- 호출부의 `occ_pred`/`occ_gt`가 제자리에서
    # 뒤집힐 수 있다(`projects/common/npsafe.py`). 그러면 f1@τ가 조용히 반대가 된다.
    return distance_transform_edt(bool_not(mask), sampling=cell_m)


def derive_occupied(free_pred, valid, rays) -> torch.Tensor:
    """**free 예측만으로 occupied를 유도한다** -- 광선이 멈춘 셀 (`polar.frontier_cells`).

    라벨에서 `occupied = ~occ & vis`이고 `vis`가 ego 원점 raycast이므로 GT occupied는
    free 영역의 ego 기준 경계다. 그래서 occupied를 예측 클래스에서 빼도(§13.3 (D))
    `iou_occupied`/`f1@τ`를 계속 보고할 수 있다. GT free를 넣은 상한 실측은
    `tools/measure_derived_occupied.py`에 있다.

    3-class head의 occupied 채널과 **같은 지표 함수로** 채점되도록 같은 계약
    `(B, 1, H, W)` bool 텐서를 돌려준다. numpy 광선 루프라 CPU로 내리고 val에서만 부른다.

    정적 마스크(`permanent_blind`/`invalid`)로 경계가 생긴 셀을 빼면 실측에서 `f1@10cm`이
    0.003 오르는데(GT 상한 기준), 그 이득이 마스크를 이 함수까지 끌고 들어올 값은 아니라고
    보고 넣지 않았다.
    """
    free_np = (free_pred.bool() & valid.bool()).cpu().numpy()
    derived = np.stack([
        frontier_cells(*first_free_range(free_np[i, 0], rays), rays, free_np[i, 0].shape)
        for i in range(free_np.shape[0])
    ])[:, None]
    return torch.from_numpy(derived).to(free_pred.device)


def boundary_f_scores(occ_pred, occ_gt, valid, cell_m, tolerances=BF_TOLERANCES_M) -> dict:
    """τ마다 **프레임별** BF·precision·recall의 합과 프레임 수. 비율은 `summarize_boundary_f`가 낸다.

    합과 수를 돌려주는 이유: batch 간에 그대로 더하면 정확한 프레임 macro가 된다 -- batch를
    어떻게 자르든 값이 같다(옛 micro가 피하려던 batch-size 의존은 프레임 단위로 나누는 순간
    사라진다).

    프레임 규칙(`metric_spec` 머리말과 같다):
      - 예측·GT 경계가 둘 다 비면 그 프레임은 빠진다(잴 것이 없다).
      - 한쪽만 비면 그쪽 비율을 0으로 본다 -- 예측 경계가 없으면 precision 0, F 0.
      - F = 2PR/(P+R), P+R = 0이면 0.

    **macro F는 macro P·R의 조화평균이 아니다** -- 프레임마다 F를 내고 평균한다(DAVIS와 같다).
    """
    pred_np = (occ_pred.bool() & valid.bool()).cpu().numpy()
    gt_np = (occ_gt.bool() & valid.bool()).cpu().numpy()

    sums = {tolerance_key(t): dict.fromkeys(_BF_SUM_KEYS, 0.0) for t in tolerances}
    for i in range(gt_np.shape[0]):
        pred, gt = pred_np[i, 0], gt_np[i, 0]
        n_pred, n_gt = int(pred.sum()), int(gt.sum())
        if n_pred == 0 and n_gt == 0:
            continue
        to_gt, to_pred = _distance_field_m(gt, cell_m), _distance_field_m(pred, cell_m)
        for tolerance in tolerances:
            hit_pred = int((pred & (to_gt <= tolerance)).sum()) if to_gt is not None else 0
            hit_gt = int((gt & (to_pred <= tolerance)).sum()) if to_pred is not None else 0
            precision = hit_pred / n_pred if n_pred else 0.0
            recall = hit_gt / n_gt if n_gt else 0.0
            total = precision + recall
            bucket = sums[tolerance_key(tolerance)]
            bucket["bf_sum"] += (2 * precision * recall / total) if total else 0.0
            bucket["precision_sum"] += precision
            bucket["recall_sum"] += recall
            bucket["n_frames"] += 1
    return sums


def summarize_boundary_f(dicts) -> dict:
    """batch별 합을 더해 프레임 macro로 나눈다 -- `{"10cm": {"bf", "precision", "recall", "n_frames"}}`."""
    if not dicts:
        return {}
    result = {}
    for key in dicts[0]:
        summed = {name: sum(d[key][name] for d in dicts) for name in _BF_SUM_KEYS}
        n = int(summed["n_frames"])
        if n == 0:
            result[key] = {"bf": float("nan"), "precision": float("nan"),
                           "recall": float("nan"), "n_frames": 0}
        else:
            result[key] = {"bf": summed["bf_sum"] / n, "precision": summed["precision_sum"] / n,
                           "recall": summed["recall_sum"] / n, "n_frames": n}
    return result


def tolerance_counts(occ_pred, occ_gt, valid, cell_m,
                     tolerances=LEGACY_TOLERANCES_M) -> dict:
    """**legacy(micro)** -- 옛 런 재채점용. 새 학습 로그는 `boundary_f_scores`를 쓴다.

    τ마다 precision/recall의 분자·분모를 센다. **비율이 아니라 카운트를 돌려준다.**

    batch마다 F1을 내고 평균하면 batch 크기에 따라 값이 달라진다 -- 프레임당 occupied 셀 수가
    장면에 따라 수십 배 차이 나기 때문이다. 카운트를 모아 마지막에 한 번 나누면
    (`summarize_tolerance_f1`) 그 의존이 정의상 사라진다. M3의 백분위수에서 같은 함정을 이미
    한 번 밟았다(`summarize_range_error` 주석).

    거리변환이 CPU numpy라 샘플마다 2회 돈다 -- M3와 마찬가지로 val에서만 부르는 것을
    전제한다.
    """
    pred_np = (occ_pred.bool() & valid.bool()).cpu().numpy()
    gt_np = (occ_gt.bool() & valid.bool()).cpu().numpy()

    counts = {tolerance_key(t): dict.fromkeys(_COUNT_KEYS, 0) for t in tolerances}
    for i in range(gt_np.shape[0]):
        pred, gt = pred_np[i, 0], gt_np[i, 0]
        # 거리장은 샘플마다 두 번만 만들고 모든 τ가 재사용한다 -- τ는 임계값일 뿐이다.
        to_gt, to_pred = _distance_field_m(gt, cell_m), _distance_field_m(pred, cell_m)
        n_pred, n_gt = int(pred.sum()), int(gt.sum())
        for tolerance in tolerances:
            bucket = counts[tolerance_key(tolerance)]
            bucket["n_pred"] += n_pred
            bucket["n_gt"] += n_gt
            if to_gt is not None:
                bucket["hit_pred"] += int((pred & (to_gt <= tolerance)).sum())
            if to_pred is not None:
                bucket["hit_gt"] += int((gt & (to_pred <= tolerance)).sum())
    return counts


def summarize_tolerance_f1(dicts) -> dict:
    """**legacy.** batch별 카운트를 합친 뒤 비율을 계산한다(micro-average).

    퇴행 해를 nan으로 빠져나가게 두지 않는다: 예측 occupied가 하나도 없으면 precision을 0으로
    본다("장애물을 아예 예측하지 않았다"는 실패이지 측정 불가가 아니다). 양쪽이 다 비었을
    때만 nan이다 -- 그때는 정말로 잴 것이 없다.
    """
    if not dicts:
        return {}
    result = {}
    for key in dicts[0]:
        summed = {name: sum(d[key][name] for d in dicts) for name in _COUNT_KEYS}
        if summed["n_pred"] == 0 and summed["n_gt"] == 0:
            precision = recall = f1 = float("nan")
        else:
            precision = summed["hit_pred"] / summed["n_pred"] if summed["n_pred"] else 0.0
            recall = summed["hit_gt"] / summed["n_gt"] if summed["n_gt"] else 0.0
            total = precision + recall
            f1 = (2 * precision * recall / total) if total else 0.0
        result[key] = {"f1": f1, "precision": precision, "recall": recall, **summed}
    return result
