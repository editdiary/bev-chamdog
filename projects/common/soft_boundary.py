"""경계 불확실성을 명시적으로 모델링하는 loss -- `L_region + λ_B · L_soft-boundary`.

설계 정본은 `docs/soft_boundary_loss_design.md`이고 근거 실측은
`docs/finetune_overfitting_diagnosis.md` §26이다. 한 줄 요약: val loss 증가분의 90 %가 GT
경계 ±20 cm 안에서 나오고 경계에서 먼 영역은 오히려 좋아진다. 경계는 라벨이 신뢰할 수 없는
자리인데 hard CE가 거기서도 `p→1`을 요구하는 것이 원인이다.

**셀을 세 영역으로 나눈다.** 부호 있는 거리 `d_i`(아래)를 기준으로

- `Ω_F` (`d > δ`) -- 확실한 drivable. hard BCE(목표 1)
- `Ω_N` (`d < −δ`) -- 확실한 non-drivable. hard BCE(목표 0)
- `Ω_B` (`|d| ≤ δ`) -- uncertain boundary. **soft** BCE(목표는 `d`의 함수)

세 항을 각자 자기 집합에서 평균 낸 뒤 `½·L_F + ½·L_N + λ_B·L_B`로 합친다. 앞의 두 개를
per-set 평균으로 두는 것이 역빈도 클래스 가중치를 대체한다 -- 셀 개수와 무관하게 두 클래스가
정확히 절반씩 기여하고, **배치 구성과 데이터셋 통계에 의존하지 않는다**(설계 문서 §3.1).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# 대역 폭 [m]. 0.15 m = 3셀. 하이퍼파라미터이므로 **라벨에서 추정할 수 없다** -- `P(free | d)`가
# 완벽한 계단이라는 것을 실측했다(설계 문서 §7). 스윕으로 정한다.
DEFAULT_DELTA_M = 0.15
DEFAULT_LAMBDA_B = 0.5

TARGET_LINEAR = "linear"
TARGET_GAUSSIAN = "gaussian"

# gaussian target의 **모양 매개변수** `alpha = sigma / delta`.
#
# **왜 sigma를 미터로 주지 않고 alpha로 주는가.** 정규화된 target에 `u = d/delta`를 넣으면
#
#     y = (Phi(u/alpha) - Phi(-1/alpha)) / (Phi(1/alpha) - Phi(-1/alpha))
#
# 이 되어 **`delta`가 식에서 사라진다.** 즉 모양은 `alpha`만이 정하고 `delta`는 대역 폭만
# 정한다. 두 손잡이가 직교한다 -- sigma를 절대값으로 주면 `delta`를 바꿀 때 모양이 조용히
# 같이 바뀌고, 그러면 "폭을 넓힌 효과"와 "모양을 둔하게 한 효과"가 섞인다.
#
# 두 극단이 알려진 loss다: `alpha -> 0`은 hard step, **`alpha -> inf`는 정확히 선형 target**
# (`Phi(x) ~ 0.5 + phi(0)x`로 전개하면 `1/2(1 + d/delta)`가 나온다). 즉 선형은 별도 형태가
# 아니라 이 계열의 한쪽 끝이다.
#
# **아래로는 격자 양자화가 한계를 만든다.** 5 cm 격자에서 셀 중심의 최소 `|d|`가 1셀이므로
# `alpha <~ 0.2`면 셀이 실제로 놓이는 자리에서 target이 이미 1.0에 붙어 soft target이
# 이름만 남는다(delta=0.15, alpha=0.15에서 최근접 셀 target이 0.987이다).
DEFAULT_SIGMA_ALPHA = 0.5

# 대역 target을 0.5 쪽으로 섞는 계수. `y' = kappa*y + (1-kappa)/2`. **1.0이 기본이고 그때
# 동작은 전과 완전히 같다.**
#
# **왜 필요한가 -- `delta`와 `alpha`가 둘 다 막혀 있기 때문이다** (2026-08-27 실측).
#
# `L_B`의 val KL은 어떤 `alpha`에서도 수렴하지 않는다. epoch 1~3에 최저를 찍고 끝까지
# 오르기만 하며, 가장 나은 `alpha=1.0`에서도 +48 %다(`runs/alpha_y4`). 원인은 target이
# 입력이 담고 있는 것보다 날카롭다는 것이다 -- BCE의 최소화자는 `E[y|image]`인데, 실측
# 특징맵 해상도가 1~2 m에서 10.5 cm, 3~4 m에서 20~24 cm이므로 Bayes 최적 예측기조차 그만큼
# 뭉개져 있다. 그런데 target은 그보다 좁은 폭을 요구한다.
#
# 그 폭을 넓히는 손잡이가 둘인데 **둘 다 이 데이터에서 막힌다.**
#
# - `alpha`는 **포화한다.** 함의하는 경계 오차의 표준편차 상한이 `delta/sqrt(3)` = 8.7 cm
#   (선형 극한)이고, `alpha=1.0`에서 이미 8.1 cm로 상한의 93 %다. 무한대로 키워도 0.6 cm를
#   더 못 넓힌다.
# - `delta`는 **기하를 먹는다.** 이 로봇은 좁은 통로를 다니고 자유 셀의 절반이 벽에서
#   32 cm 이내다(val 75프레임 실측). `delta=0.45`면 자유공간의 69 %가 hard `Omega_F`
#   자격을 잃고 `delta=0.60`이면 프레임의 28 %가 hard free 셀을 하나도 못 가진다.
#
# **`kappa`는 대역 폭을 안 건드리면서 상한 없이 평평하게 만든다.** `Omega_F`도 통로도
# 그대로다. 뜻도 다르다 -- `delta`가 "경계 위치가 ±delta 안에 있다"인 반면 `kappa`는
# "경계 위치와 무관하게 (1-kappa)의 확률로 이 라벨은 정보가 없다"는 **평평한 라벨 잡음**이다.
# 이 데이터셋 라벨이 LiDAR/SLAM 위에 사람 손 보정이 얹힌 것이므로 그런 성분을 가정하는 것
# 자체는 자연스럽지만, **보정 절차가 기록돼 있지 않아 `kappa`를 데이터에서 추정할 수 없다.**
# `delta`와 마찬가지로 사전지식에서 오는 값이고 스윕으로 고른다.
#
# 설계 문서가 label smoothing을 "과신을 직접 겨냥한 손잡이"라고 부르면서
# (`segmentation_loss.py`) CE 경로에만 배선해 둔 것을 soft-boundary 쪽으로 가져온 것이다.
# 실측 근거: val loss가 오르는 이유가 "더 많이 틀려서"가 아니라 "같은 만큼 틀리되 확신이
# 커져서"이고 정답 확률의 기하평균이 train 0.995 대 val 0.620이다(설계 문서 §16.2).
DEFAULT_KAPPA = 1.0

# 세 영역 **전부**의 target을 안쪽으로 당기는 균일 label smoothing. `Omega_F`는 `1-eps`,
# `Omega_N`은 `eps`, 대역은 `eps + (1-2eps)*y`. **0.0이 기본이고 그때 동작은 전과 같다.**
#
# **왜 `kappa`로 부족한가** (2026-08-27 실측). `kappa`는 대역 target을 실제로 평평하게 만들고
# 모델의 확신도 실제로 줄였다(대역의 확신 예측 50.7 % -> 21 %). 그런데 `L_B`의 val KL 상승은
# 안 줄었고 `kappa <= 0.4`에서는 **더 나빠졌다**(U자). 이유가 둘이다.
#
# 1. target을 0.5로 보내면 **확신 있는 예측에 대한 KL이 오히려 커진다** --
#    `kappa -> 0` 극한에서 `KL = -log2 - 0.5*log(p(1-p))`이고 `p`가 0/1에 가까울수록 발산한다.
#    평평한 target은 "너도 반드시 불확실해야 한다"는 요구이기 때문이다.
# 2. **모델이 target을 따라오지 못한다.** target 평탄도가 사실상 같은 두 config를 비교하면
#    (`kappa=0.4`의 target 엔트로피 0.652 대 `delta=0.45`의 0.658) 모델과 target의 격차가
#    `kappa` 쪽은 -0.142인데 `delta` 쪽은 -0.065다. 그리고 `delta` 쪽만 수렴한다.
#
# 차이는 **대역 밖**에 있었다. `Omega_F`의 예측 엔트로피가 `kappa`에서는 0.079 -> 0.109로
# 거의 안 움직이는데 `delta=0.45`에서는 0.260으로 3.3배가 된다. **모델의 확신은 대역 밖
# 88 %가 정하고, 대역은 셀의 11.6 %인 얇은 띠라 양옆 hard 영역의 확신을 물려받는다.**
# 즉 `delta=0.45`가 수렴한 것은 target을 평평하게 해서가 아니라 hard 감독의 비중 자체를
# 바꿔(`Omega_F` 15.3 % -> 6.3 %) 모델을 **전역적으로** 덜 확신하게 만들었기 때문이다.
#
# `eps`는 그 전역 효과를 **기하를 안 건드리고** 얻으려는 것이다 -- `delta`는 0.15로 두므로
# 좁은 통로에서 hard `Omega_F`를 잃지 않는다.
#
# **뜻이 양쪽에서 다르다는 유보.** `Omega_F` 16만 셀은 전부 사람이 "보이고 drivable"이라
# 라벨한 것이라 `eps`가 곧 어노테이션 오류율이다. 그런데 `Omega_N`은 **93.3 %가 vis=0**
# (벽 뒤, raycast가 정한 것)이고 사람이 라벨한 obstacle은 0.0 %다 -- 거기서 `eps`는
# "벽 뒤가 eps 확률로 free다"라는 **거짓이고 `fatal` 방향**의 주장이 된다. 그래도 대칭으로
# 두는 이유는 검증하려는 가설이 "확신은 대역 밖 88 %가 만든다"인데 그중 76.6 %가
# `Omega_N`이어서, 비대칭으로 가면 가설을 제대로 시험하지 못하기 때문이다. **`fatal_rate`와
# `missed_obstacle`로 감시하고, 나빠지면 `Omega_N`용 eps를 따로 둔다.**
DEFAULT_EPS = 0.0


def resolve_sigma(delta, sigma=None, alpha=None) -> float:
    """`sigma`[m]와 `alpha`(=sigma/delta) 중 하나를 받아 `sigma`[m]로 돌려준다.

    **둘을 동시에 받으면 실패한다.** 조용히 하나를 이기게 두면 로그의 config와 실제로 쓰인
    모양이 갈리고, 그건 스윕 표를 통째로 무의미하게 만든다.
    """
    if sigma is not None and alpha is not None:
        raise ValueError(f"sigma와 alpha 중 하나만 줘야 한다: sigma={sigma}, alpha={alpha}")
    if alpha is not None:
        if float(alpha) <= 0:
            raise ValueError(f"alpha는 양수여야 한다: {alpha}")
        return float(alpha) * float(delta)
    if sigma is None or float(sigma) <= 0:
        raise ValueError(f"gaussian target에는 양수 sigma 또는 alpha가 필요하다: {sigma}")
    return float(sigma)


def signed_distance_field(free: np.ndarray, keep: np.ndarray, cell_m: float) -> np.ndarray:
    """`d_i` -- GT 경계까지의 **부호 있는 수직 거리** [m]. `d > 0`이면 drivable 쪽.

    `dist(x, Γ) = min_{q∈Γ} ‖x − q‖₂`에 부호를 붙인 것이고, `Γ`는 free 영역의 경계다.
    **방위각 광선 위의 반경 거리가 아니라 경계면까지의 최단 거리다.** 두 정의를 실측 비교했을
    때 Jaccard가 0.305였고 반경 정의는 경계에서 15 cm 안인 셀의 11 %를 놓쳤다(설계 문서 §2.2) --
    경계가 광선을 스치듯 지나가는 통로 측벽에서 반경 거리는 크지만 수직 거리는 작기 때문이다.
    soft label의 근거가 "경계면이 공간상 몇 cm 밀릴 수 있다"이므로 재야 할 것은 수직 거리다.

    이것은 **`f1@τ`가 쓰는 것과 같은 거리 연산자**다(`occupied_metrics._distance_field_m`도
    `distance_transform_edt(sampling=cell_m)`). 다른 것은 두 가지뿐이다: `f1@τ`는 크기만
    쓰고 부호를 안 쓰며, 대상 집합이 free 영역 경계가 아니라 `occupied` 표면이다.
    **올리려는 지표와 loss가 같은 거리를 재는 것**이 수직 정의를 고른 첫 번째 이유다.

    `keep`이 False인 셀(`permanent_blind`)을 free에 합쳐서 거리를 잰다 -- ego 아래 원반은
    프레임마다 같은 정적 경계라 합치지 않으면 그 테두리가 경계로 잡혀 대역이 오염된다.
    합치는 것은 거리 계산에서만이고, 그 셀들의 **감독은 `region_masks`가 따로 정한다.**

    거리는 셀 **중심** 사이로 재므로 경계에 붙은 셀도 `|d| = cell_m`이고 0이 되지 않는다.
    실제 경계면은 두 중심 사이(약 `cell_m/2`)에 있으므로 크기가 반 셀만큼 크게 나오는데,
    **양쪽에 대칭으로 걸리므로 target의 `y = 0.5` 교차점은 경계면에 그대로 남는다.**
    """
    free_ext = free | ~keep
    inside = distance_transform_edt(free_ext, sampling=cell_m)
    outside = distance_transform_edt(~free_ext, sampling=cell_m)
    return np.where(free_ext, inside, -outside).astype(np.float32)


def region_masks(d, valid, permanent_blind, delta: float = DEFAULT_DELTA_M) -> dict:
    """`(Ω_F, Ω_N, Ω_B)` bool 마스크. 세 집합은 겹치지 않고 `valid`를 정확히 덮는다.

    **`permanent_blind`는 대역 분할에서 빼되 `Ω_N`에 넣는다.** 두 가지를 동시에 만족해야
    하기 때문이다.

    - 분할에서 빼야 하는 이유: 원반은 ego 근처라 `r`이 작아 `d`가 큰 양수로 나오고, 그대로
      두면 라벨이 `unknown`(= not_free)인 셀이 `Ω_F`로 들어가 **free로 학습된다.**
    - 그래도 감독해야 하는 이유: **지표는 `valid` 전체에서 채점되고 원반은 `valid`다.**
      감독을 아예 빼면 모델이 원반에 free를 찍어도 loss가 벌하지 않는데 `fatal_rate`는
      나빠진다. 정적이라 배우기 쉬운 셀이므로 hard 목표 0으로 두는 것이 맞다.

    라벨이 `d`의 부호와 어긋나는 셀은 실측 0.00~0.01 %이므로(설계 문서 §4.1) `Ω_F`/`Ω_N`의
    목표는 상수 1/0으로 두고 라벨을 다시 읽지 않는다.
    """
    valid_b = valid.bool()
    blind_b = permanent_blind.bool()
    partitioned = valid_b & ~blind_b
    omega_f = partitioned & (d > delta)
    omega_n = (partitioned & (d < -delta)) | (valid_b & blind_b)
    return {"omega_f": omega_f, "omega_n": omega_n,
            "omega_b": partitioned & ~omega_f & ~omega_n}


def soft_target(d, delta: float = DEFAULT_DELTA_M, kind: str = TARGET_LINEAR,
                sigma=None, alpha=None, kappa: float = DEFAULT_KAPPA):
    """`Ω_B`의 목표 확률 `y = P(이 셀이 실제로 drivable)`. `[0, 1]`.

    두 형태는 **같은 질문에 다른 사전분포로 답한 것**이고 둘 다 정확한 사후확률이다.
    경계가 `R_true = R_gt + ε`로 흔들린다고 하면 `P(drivable) = P(ε > −d)`이므로

    - `linear` -- `ε ~ Uniform(−δ, δ)` → `y = ½(1 + d/δ)`. **손잡이가 `δ` 하나**이고
      `d = ±δ`에서 정확히 1/0이 되어 hard 영역과 이어진다. 먼저 이것으로 검증한다.
    - `gaussian` -- `ε ~ N(0, σ²)` → `y ∝ Φ(d/σ)`. `σ`가 "경계를 몇 cm 모르는가"라는
      물리량이 되어 해석·측정이 가능하지만 손잡이가 하나 늘어난다.

    **`gaussian`은 대역 안에서 정규화한다.** 생 `Φ(d/σ)`는 유한 구간에서 0/1에 도달하지 않아
    `d = ±δ`에서 목표가 튀고(`σ = δ`면 0.841 대 옆 셀 1.0), 불연속 위치가 임의로 고른 `δ`가
    된다. 정규화한 형태는 "오차가 ±δ 안에 있다는 것은 안다"는 **절단 정규 사전분포에서의
    정확한 사후확률**이므로 물리적 해석을 잃지 않는다.

    **`kappa`는 대역 target을 0.5 쪽으로 섞는다** -- `y' = κ·y + (1−κ)/2`. 자세한 근거는
    `DEFAULT_KAPPA`의 주석에 있다. `κ = 1.0`이면 동작이 전과 완전히 같다.

    `Ω_B` 밖의 값도 계산되지만 의미가 없다 -- 호출부가 마스크로 걸러야 한다.
    """
    if kind == TARGET_LINEAR:
        y = (0.5 * (1.0 + d / delta)).clamp(0.0, 1.0)
    elif kind == TARGET_GAUSSIAN:
        sigma = resolve_sigma(delta, sigma, alpha)
        edge = torch.special.ndtr(torch.tensor(delta / sigma, dtype=d.dtype, device=d.device))
        lo = 1.0 - edge                              # Φ(−δ/σ) = 1 − Φ(δ/σ)
        y = ((torch.special.ndtr(d / sigma) - lo) / (edge - lo)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"soft target 종류는 {TARGET_LINEAR} 또는 {TARGET_GAUSSIAN}여야 한다: {kind}")
    return apply_kappa(y, kappa)


def _binary_entropy(eps: float) -> float:
    """`H(ε) = −ε ln ε − (1−ε) ln(1−ε)`. `Ω_F`/`Ω_N`에서 줄일 수 없는 상수다."""
    import math
    if eps <= 0.0:
        return 0.0
    return -(eps * math.log(eps) + (1.0 - eps) * math.log(1.0 - eps))


def apply_kappa(y, kappa: float = DEFAULT_KAPPA):
    """`y' = κ·y + (1−κ)/2`. **0.5 쪽으로의 수축이고 대역 폭은 건드리지 않는다.**

    `κ = 1`은 항등이므로 기존 런과 bit 단위로 같다.

    **대칭성이 보존된다** -- `y(d) + y(−d) = 1`이면 `y'(d) + y'(−d) = κ·1 + (1−κ) = 1`이다.
    그래서 `L_range`의 `arc`에 대한 중립성(제자리 대역은 arc에 0을 기여)이 그대로 남는다
    (`tests/tools/test_diagnose_range_dead_zone.py`가 그 성질을 고정한다).
    """
    kappa = float(kappa)
    if not 0.0 < kappa <= 1.0:
        raise ValueError(f"kappa는 (0, 1] 범위여야 한다: {kappa}")
    if kappa == 1.0:
        return y
    return kappa * y + (1.0 - kappa) * 0.5


def _masked_mean(values, selector):
    """`selector` 안의 평균. 집합이 비면 0을 돌려준다 -- nan이 총합을 오염시키면 안 된다."""
    picked = values * selector
    return picked.sum() / (selector.sum() + 1e-6)


def target_entropy(y):
    """`H(y) = −[y log y + (1−y) log(1−y)]` -- soft BCE가 내려갈 수 있는 **하한**.

    target이 모델과 무관한 상수이므로 `BCE(z, y)`의 최소값은 0이 아니라 `p = y`에서의 `H(y)`다.
    선형 target이면 `y`가 `[0, 1]`에 균일해 평균 하한이 0.5 nats이고, `λ_B = 0.5`면 전체
    loss에 상수 0.25가 얹힌다. **이것을 모르면 하한에 붙어 평평해지는 것을 "수렴 실패"로
    오독한다** -- 그래서 `L_B` 대신 `KL = L_B − H̄`를 로그해야 한다(설계 문서 §5.4).

    **`xlogy`를 쓴다.** `clamp` 후 `y·log y`로 쓰면 `y = 1`에서 nan이 나온다 -- float32에서는
    `1 − 1e-12`가 정확히 1.0으로 반올림되어 `(1−y)·log(1−y)`가 `0 · −inf`가 된다. 그 nan은
    `Ω_B` 밖의 셀에서 생기는데 **마스크로 곱해도 사라지지 않고** 총합을 오염시킨다.
    `xlogy(0, 0)`은 정의상 0이므로 그 경로가 아예 없어진다.
    """
    return -(torch.xlogy(y, y) + torch.xlogy(1.0 - y, 1.0 - y))


def compute_soft_boundary_loss(logits, d, valid, permanent_blind, delta=DEFAULT_DELTA_M,
                               lambda_b=DEFAULT_LAMBDA_B, kind=TARGET_LINEAR, sigma=None,
                               alpha=None, range_term=None, lambda_r=0.0,
                               kappa=DEFAULT_KAPPA, eps=DEFAULT_EPS):
    """`(총 loss, 항별 dict)`. `logits`는 `(B, 2, H, W)`이고 채널 1이 `free`다.

    `masked_weighted_ce`와 같은 계약을 지킨다 -- 학습 루프·로깅이 두 loss를 바꿔 끼울 수
    있어야 하기 때문이다. 다른 것은 항의 이름과 `kl_boundary`가 추가된다는 점뿐이다.

    `share_*`는 각 항이 총 loss에 기여하는 몫이고 정의상 합이 1이다. `frac_*`는 셀 비율이다.
    **둘을 같이 내놓는 이유:** `λ_B = 0.5`는 작아 보이지만 `Ω_B`가 셀의 10 % 남짓이라
    셀당 가중치는 `Ω_N`의 7배쯤 된다. 두 숫자를 나란히 두지 않으면 그 사실이 안 보인다.

    `range_term`은 `range_loss.compute_range_loss`가 돌려준 `(loss, parts)`이고 `None`이면
    항이 아예 없다(대조군 런이 새 경로를 타지 않아야 한다). **총 loss와 `share_*`를 한
    자리에서만 계산하기 위해** 밖에서 더하지 않고 여기로 받는다 -- 두 곳에서 더하면 몫의
    합이 1이 아니게 되고, 그 표가 이 프로젝트에서 항의 영향력을 읽는 주된 도구다.
    """
    log_probs = F.log_softmax(logits, dim=1)
    log_free = log_probs[:, 1:2]
    log_not_free = log_probs[:, 0:1]

    regions = region_masks(d, valid, permanent_blind, delta)
    omega_f = regions["omega_f"].float()
    omega_n = regions["omega_n"].float()
    omega_b = regions["omega_b"].float()

    # `Ω_F`/`Ω_N`의 목표는 `1−ε`/`ε`이다. `ε = 0`이면 상수 1/0으로 전과 같다 -- 라벨과 `d`의
    # 부호가 어긋나는 비율이 0.01 %라 라벨을 다시 읽지 않는다.
    eps = float(eps)
    if not 0.0 <= eps < 0.5:
        raise ValueError(f"eps는 [0, 0.5) 범위여야 한다: {eps}")
    if eps == 0.0:
        loss_free = _masked_mean(-log_free, omega_f)
        loss_not_free = _masked_mean(-log_not_free, omega_n)
    else:
        loss_free = _masked_mean(-((1.0 - eps) * log_free + eps * log_not_free), omega_f)
        loss_not_free = _masked_mean(-(eps * log_free + (1.0 - eps) * log_not_free), omega_n)

    # **`ε`은 대역 target도 같은 범위로 당겨야 한다.** 안 그러면 정규화된 target이
    # `y(±δ) = 1/0`을 정확히 주므로 대역 **끝**이 바로 옆 hard 영역(`1−ε`/`ε`)보다 더
    # 확신에 찬 거꾸로 된 불연속이 생긴다. 당기면 `y' = ε + (1−2ε)y`인데 이것은
    # `0.5 + (1−2ε)(y − 0.5)`와 같다 -- 즉 **대역에서 `ε`은 정확히 `κ = 1−2ε`이다.**
    # 그래서 두 손잡이를 곱해서 함께 적용한다.
    y = soft_target(d, delta, kind, sigma, alpha, kappa * (1.0 - 2.0 * eps))
    per_cell_b = -(y * log_free + (1.0 - y) * log_not_free)
    loss_boundary = _masked_mean(per_cell_b, omega_b)
    entropy = _masked_mean(target_entropy(y), omega_b)

    contributions = [0.5 * loss_free, 0.5 * loss_not_free, lambda_b * loss_boundary]
    names = ["free", "not_free", "boundary"]
    n_valid = valid.float().sum() + 1e-6
    parts = {
        "loss_free": loss_free,
        "loss_not_free": loss_not_free,
        "loss_boundary": loss_boundary,
        # 경계 항의 **진짜 진행도**. `loss_boundary`는 `entropy`가 하한이라 0으로 안 간다.
        "kl_boundary": loss_boundary - entropy,
        "entropy_boundary": entropy,
        # **`ε`이 `Ω_F`/`Ω_N`에도 상수 하한 `H(ε)`을 만든다.** ε=0.10이면 0.325다 --
        # 빼지 않고 읽으면 `loss_free`가 0.07 -> 0.40으로 뛰는 것이 성능 붕괴처럼 보인다.
        "kl_free": loss_free - _binary_entropy(eps),
        "kl_not_free": loss_not_free - _binary_entropy(eps),
    }
    range_contribution = 0.0
    if range_term is not None:
        range_loss, range_parts = range_term
        range_contribution = lambda_r * range_loss
        contributions.append(range_contribution)
        names.append("range")
        parts.update(range_parts)
    total = sum(contributions)
    for name, value in zip(names, contributions):
        parts[f"share_{name}"] = value / (total + 1e-6)
    # **`share_boundary`만 보면 경계 항의 영향력을 과대평가한다.** 그 항에는 target 엔트로피가
    # 상수로 들어 있고 상수는 gradient가 0이다. 실측 스모크에서 `share_boundary`가 0.74였는데
    # 그중 절반 이상이 그 하한이었다. 그래서 **축소 가능한 부분만으로 다시 센 몫**을 같이
    # 낸다 -- 이쪽이 학습을 실제로 지배하는 비율이다.
    reducible = (contributions[0] + contributions[1]
                 + lambda_b * parts["kl_boundary"] + range_contribution)
    parts["share_boundary_kl"] = lambda_b * parts["kl_boundary"] / (reducible + 1e-6)
    for name, mask in (("free", omega_f), ("not_free", omega_n), ("boundary", omega_b)):
        parts[f"frac_{name}"] = mask.sum() / n_valid
    return total, parts
