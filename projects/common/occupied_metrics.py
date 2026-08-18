"""occupied 전용 지표 -- 거리 허용오차 F1 (`f1@τ`).

**왜 면적 IoU가 아닌가.** `occupied`는 광선이 멈춘 두께 1셀 표면이라 예측이 한 칸 옆으로
밀리기만 해도 교집합이 0이 되어 IoU가 무너진다. 실측으로 `iou_obstacle` 0.312가 이미지를
보지 않는 frontier 규칙 0.473에 졌고(`docs/free_space_metric_migration.md` §1) 그래서 이
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
from scipy.ndimage import distance_transform_edt

# 격자 셀이 0.05 m이므로 2 / 4 / 8 셀에 해당한다. 셀 하나(0.05 m)는 라벨 자체의 이산화
# 오차와 구별되지 않아 넣지 않았다.
DEFAULT_TOLERANCES_M = (0.10, 0.20, 0.40)

_COUNT_KEYS = ("hit_pred", "n_pred", "hit_gt", "n_gt")


def tolerance_key(tolerance_m: float) -> str:
    """`0.2 -> "20cm"`. 로그 tag와 dict 키에 float을 그대로 쓰면 표기가 갈린다."""
    return f"{round(tolerance_m * 100)}cm"


def _distance_field_m(mask: np.ndarray, cell_m: float):
    """`mask`의 가장 가까운 True까지의 거리 [m]. `mask`가 비면 `None`.

    비었을 때 거리는 정의되지 않는다(scipy는 배경이 없는 입력에 큰 상수를 채워 넣는데 그것을
    거리로 읽으면 조용히 틀린다). 호출부가 `None`을 보고 그 방향의 hit을 0으로 센다.
    """
    if not mask.any():
        return None
    return distance_transform_edt(~mask, sampling=cell_m)


def tolerance_counts(occ_pred, occ_gt, valid, cell_m,
                     tolerances=DEFAULT_TOLERANCES_M) -> dict:
    """τ마다 precision/recall의 분자·분모를 센다. **비율이 아니라 카운트를 돌려준다.**

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
    """batch별 카운트를 합친 뒤 비율을 계산한다(micro-average).

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
