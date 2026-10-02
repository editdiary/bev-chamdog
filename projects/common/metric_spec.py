"""논문 지표의 **이름·tag·눈금 정본**. 학습 로그를 쓰는 쪽과 읽는 쪽이 모두 여기를 본다.

**왜 따로 두나.** 2026-10-02까지 지표 tag 표가 학습 로거 한 벌, 보고 도구 여섯 벌, 패키지
생성기 네 벌로 복사돼 있었다. 지표를 하나 바꾸면 그중 일부만 고쳐지고, 나머지는 **없는 tag를
조용히 빈칸으로** 읽는다. 여기 한 벌만 두고 나머지는 import한다. torch에 의존하지 않으므로
TensorBoard만 읽는 도구도 가볍게 가져다 쓴다.

## 2026-10-02 지표 집합 (사용자 확정)

    iou_free       free 클래스 IoU                        (주 지표, 체크포인트 선택 기준)
    iou_non_free   non-free(= valid 안의 나머지) 클래스 IoU (보조: 자명해가 이미 ~0.83을 받는다)
    precision      free 기준 Precision = TP / (TP + FP)
    recall         free 기준 Recall    = TP / (TP + FN)
    bf_{τ}         Boundary F-measure, 허용오차 τ ∈ {0.10, 0.20, 0.30} m
    ring_*         거리 링마다 위 네 개(iou_free / iou_non_free / precision / recall)

**전부 macro다** -- 프레임마다 값을 낸 뒤 프레임 평균한다. 이전에는 `iou_free`만 macro였고
`fatal_rate`(= 1 − precision) · `free_miss_rate`(= 1 − recall) · `f1@τ`는 전체 셀을 한 통에
부은 micro였다. 한 논문 표 안에서 집계 방식이 섞이지 않게 통일했다.

**프레임을 평균에서 빼는 것은 예측과 GT가 둘 다 빈 경우뿐이다.** 한쪽만 비면 정의되지 않는
비율을 0으로 본다 -- "장애물을 아예 예측하지 않았다"는 측정 불가가 아니라 실패다.
IoU(합집합 0인 프레임만 뺀다)와 같은 규칙이다.

**뺀 것.** `fatal_rate`/`free_miss_rate`(이름을 precision/recall로 바꿨다), `range_*`
(방위각 자유거리 지표 전부 -- 사용자가 논문에서 뺐다), `f1@40cm`(눈금을 0.30 m로 바꿨다).
계산 함수는 옛 런 재채점을 위해 `free_space_metrics`/`occupied_metrics`에 legacy로 남는다.
"""

# BF 허용오차 [m]. 셀 0.05 m 기준 2 / 4 / 6셀. **최댓값을 soft-boundary 반폭 δ = 0.30 m와
# 맞췄다**(사용자 결정 2026-10-02). 옛 눈금은 (0.10, 0.20, 0.40)이었다.
BF_TOLERANCES_M = (0.10, 0.20, 0.30)

# 거리 링 경계 [m]. 로봇 격자(전방 4 / 후방 2 / 좌우 3 m) 기준이라 3-4 m 링은 전방만 남은
# 부분 고리다.
RING_EDGES_M = (0.0, 1.5, 3.0, 4.0)

# 프레임 단위 free 지표. 순서가 곧 표의 열 순서다 -- `iou_free`가 첫 열이어야 한다.
FREE_METRICS = ("iou_free", "iou_non_free", "precision", "recall")

SELECTION_TAG = "val/iou_free_epoch"

# --- 손실 로그 (2026-10-02, 사용자 요청: 어떤 실험이든 같은 손실 항을 같은 이름으로) ----------
# 정의는 `projects/common/binary_metrics.decompose_loss`. 세 영역은 soft-boundary의 Ω_F / Ω_N / Ω_B
# (δ = 0.30 m 고정)이다. `loss_*`는 그 런이 최적화한 손실의 영역 평균이고, `ce_*`는 모든 런에
# 같은 눈금(가중치 없는 hard CE)이다. `val/loss_epoch`은 그 런의 총 목적함수다.
DECOMPOSITION_DELTA_M = 0.30
LOSS_REGIONS = ("free", "not_free", "boundary")
COMMON_LOSS_PARTS = (
    "loss_free", "loss_not_free", "loss_boundary", "loss_boundary_floor",
    "ce_free", "ce_not_free", "ce_boundary", "ce_all",
    "frac_free", "frac_not_free", "frac_boundary",
)


def tolerance_key(tolerance_m: float) -> str:
    """`0.2 -> "20cm"`. 로그 tag와 dict 키에 float을 그대로 쓰면 표기가 갈린다."""
    return f"{round(tolerance_m * 100)}cm"


def bf_name(tolerance_m: float) -> str:
    """`0.1 -> "bf_10cm"`."""
    return f"bf_{tolerance_key(tolerance_m)}"


def ring_name(lo: float, hi: float) -> str:
    """`(0.0, 1.5) -> "0.0-1.5m"`. `free_space_metrics.build_ring_masks`와 같은 표기다."""
    return f"{lo}-{hi}m"


def ring_names(edges_m=RING_EDGES_M):
    return tuple(ring_name(lo, hi) for lo, hi in zip(edges_m[:-1], edges_m[1:]))


def _tag(split: str, name: str) -> str:
    return f"{split}/{name}_epoch"


BF_METRICS = tuple(bf_name(t) for t in BF_TOLERANCES_M)

#: 논문 본문 지표. `(tag, 이름, 클수록 좋은가)`. 전부 클수록 좋다.
PAPER_METRICS = tuple((_tag("val", name), name, True) for name in FREE_METRICS + BF_METRICS)

#: BF의 precision/recall. 헤드라인은 `bf_*` 하나이고 이것은 해석용 보조 열이다.
BF_DETAIL_METRICS = tuple(
    (_tag("val", f"{bf}_{part}"), f"{bf}_{part}", True)
    for bf in BF_METRICS for part in ("precision", "recall"))

#: 링별 지표. 이름은 `ring_0.0-1.5m_iou_free` 꼴이다.
RING_METRICS = tuple(
    (_tag("val", f"ring_{ring}_{metric}"), f"ring_{ring}_{metric}", True)
    for ring in ring_names() for metric in FREE_METRICS)

#: CSV로 내보내는 전부. 패키지의 시드별 원자료 표가 이 순서를 쓴다.
EXPORTED_METRICS = PAPER_METRICS + BF_DETAIL_METRICS + RING_METRICS


def epoch_tag(split: str, name: str) -> str:
    """학습 로거가 쓰는 tag. 읽는 쪽과 같은 함수를 지나게 해 표기가 갈리지 않게 한다."""
    return _tag(split, name)


def require_tags(series: dict, run_name: str, metrics=PAPER_METRICS) -> None:
    """런에 논문 지표 tag가 전부 있는지. **없으면 빈칸으로 넘어가지 않고 멈춘다.**

    2026-10-02 이전 런은 `fatal_rate`·`occupied_f1_*`·`range_*` tag로 기록돼 있어서 새 tag를
    읽으면 전부 비어 나온다. 그 결과가 빈 CSV 열로 조용히 패키지에 들어가는 것을 막는다.
    """
    missing = [tag for tag, _, _ in metrics if tag not in series]
    if missing:
        raise SystemExit(
            f"{run_name}에 지표 tag {len(missing)}개가 없다(예: {missing[:3]}). "
            "2026-10-02 이전 지표 정의로 기록된 런이면 새 패키지 원천으로 쓸 수 없다 "
            "(projects/common/metric_spec.py)")
