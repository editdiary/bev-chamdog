"""`L_range` -- 방위각별 자유거리에 걸리는 **보조** loss. 설계는 `docs/soft_boundary_loss_design.md` §13.

**왜 이 항이 필요한가.** 현행 loss(`soft_boundary.compute_soft_boundary_loss`)는 셀마다
독립인 BCE의 합이라 **광선 방향으로 셀을 묶는 항이 하나도 없다.** 그래서 "이 방향으로
free가 몇 m까지 이어진다고 예측했는가"를 어떤 항도 직접 묻지 않는다. 결과로 per-cell BCE가
거의 벌하지 않는 오차 모드가 남는다 -- 벽 뒤에 free 섬 하나를 찍으면 셀 몇 개분 BCE만 물지만
그 방향의 자유거리는 통째로 망가진다.

**왜 dead zone인가.** 단순 MAE로 걸면 main loss에서 완화한 경계 불확실성을 보조항이 다시
강요한다. `soft_boundary`가 경계 셀의 **확신**만 낮추고 **위치**는 라벨에 못박아 둔 것이
train/val KL 격차 28~42배의 원인이므로(설계 문서 §9.5의 원인 (3)), 거리에도 같은 못박기를
넣으면 병을 키운다. `e^eff = max(0, e − δ_R)`로 두면 δ_R 안에서 loss가 **평평**해져
gradient가 정확히 0이 되고, **모델이 그 자유도에 용량을 쓸 수 없다.**

이것은 §12.2의 허용오차 F1과 **같은 메커니즘이고 좌표계만 다르다** -- 허용오차 F1은 2D 셀
공간에서 어느 방향으로 밀려도 용서하고, 이 항은 광선당 스칼라 하나에서 **반경 방향** 변위만
용서한다. 그리고 §5.2의 유도(`R_true = R_gt + ε`)가 애초에 반경 변위이므로, 불확실성 가정이
원래 살던 좌표계로 돌아온 형태다. §2.1이 `d`를 수직 거리로 고른 것과 모순되지 않는다 --
그건 "어느 셀을 부드럽게 하나"의 결정이었고 이건 "허용 폭을 어느 좌표에서 재나"의 결정이다.

**전부 torch로, 배치의 라벨에서 직접 계산한다.** `arc_gt`를 `__getitem__`에서 미리 계산해
배치에 실을 수도 있었지만 그러면 좌우 반전 증강에서 조용히 틀린다 -- `(n_theta,)` 배열의
좌우 반전은 공간 축 flip이 아니라 `θ → 2π − θ`, 즉 **순열**(`roll(flip(x), 1)`)이다.
여기서는 이미 반전된 라벨 텐서에서 매번 다시 뽑으므로 **θ 축이 저장되지 않고 그 함정이
아예 생기지 않는다.** 비용은 고정 인덱스 gather 한 번(프레임당 720×200)이라 사실상 0이다.
"""
import numpy as np
import torch
import torch.nn.functional as F

# 허용 반폭 [m]. "GT와 이만큼 안이면 같은 경계로 본다." **하이퍼파라미터이고 라벨에서
# 추정할 수 없다** -- 설계 문서 §7이 `P(free | d)`가 완벽한 계단임을 실측했고, 같은 이유로
# 반경 방향 편차도 라벨 자신에서는 잴 수 없다. 0.20 m = 4셀이고 주 지표 `f1@10cm`의 허용
# 오차(2셀)의 2배다.
DEFAULT_DELTA_R_M = 0.20

# Huber의 전환점 [m]. `e^eff <= beta`는 이차, 그 위는 선형이다. **미터 단위로 둔다** --
# 정규화된 스케일에서 주면 실효 오차가 항상 beta보다 훨씬 작아져 Huber가 순수 L2로 퇴화하고
# "outlier에 덜 끌려간다"는 도입 이유가 사라진다. 0.10 m는 dead zone 통과 후의 전형적인
# 잔차 규모다(`range_mae` 실측 0.235 m − δ_R 0.20 m).
DEFAULT_HUBER_BETA_M = 0.10

# 보조항 가중치. **0.0이 기본이고 그때 이 항은 계산되지 않는다** -- 대조군 런이 새 코드
# 경로를 타지 않아야 한다.
#
# **이 값은 추측하지 말고 gradient 비로 캘리브레이션한다**(`tools/measure_range_gradient.py`).
# `L_range`는 미터 단위이고 BCE는 nats라 두 항의 절대값에 공통 스케일이 없다. 그래서
# `lambda_r`을 숫자로 고르면 그 뜻이 정해지지 않는다. 대신 `‖∂(λ_R·L_range)/∂logits‖ /
# ‖∂L_BCE/∂logits‖ ≈ 0.1`이 되도록 맞추면 "gradient의 10 %"라는 뜻 있는 값이 된다.
DEFAULT_LAMBDA_R = 0.0


class RayGather:
    """고정 격자의 방위각 광선을 따라 `(B, 1, H, W)` 장을 `(B, n_theta, n_steps)`로 뽑는다.

    `polar.build_ray_index`가 만든 인덱스를 torch로 한 번만 옮겨 둔다 -- 격자가 고정이므로
    매 배치 다시 만들 이유가 없다. **`radii_m`의 간격이 `step_m`**이고 `step_cells=0.5`가
    기본이므로 셀 하나가 광선 위에서 약 2회 표본된다. 예측과 목표에 **같은 연산자**를 쓰는
    한 그 중복은 양쪽에서 상쇄되므로 보정하지 않는다.
    """

    def __init__(self, rays, grid_shape, device=None):
        self.n_rows, self.n_cols = grid_shape
        self.step_m = float(rays.radii_m[1] - rays.radii_m[0])
        self.rows = torch.from_numpy(np.ascontiguousarray(rays.rows)).long()
        self.cols = torch.from_numpy(np.ascontiguousarray(rays.cols)).long()
        self.inside = torch.from_numpy(np.ascontiguousarray(rays.inside))
        self._step_index = torch.arange(self.rows.shape[1])
        if device is not None:
            self.to(device)

    def to(self, device):
        self.rows = self.rows.to(device)
        self.cols = self.cols.to(device)
        self.inside = self.inside.to(device)
        self._step_index = self._step_index.to(device)
        return self

    @property
    def step_index(self):
        return self._step_index

    def __call__(self, field):
        """`(B, 1, H, W)` -> `(B, n_theta, n_steps)`. 격자 밖 표본은 호출부가 `inside`로 지운다."""
        if field.shape[-2:] != (self.n_rows, self.n_cols):
            raise ValueError(f"격자 크기가 광선 인덱스와 다르다: {tuple(field.shape[-2:])} "
                             f"!= {(self.n_rows, self.n_cols)}")
        return field[:, 0][:, self.rows, self.cols]


def ray_is_ok(sampled_free, inside):
    """`polar.RAY_OK`와 **같은 판정**을 torch로. `(B, n_theta)` bool.

    광선 위에 free가 있고(`RAY_NO_FREE` 아님) 그 뒤 격자 안에서 non-free를 만나야
    (`RAY_CENSORED` 아님) 자유거리가 정의된다. 세 상태의 실측 비율은 `RAY_OK` 35 % /
    `RAY_CENSORED` 12 % / `RAY_NO_FREE` 53 %(val split 40프레임)다.

    **`RAY_OK`만 loss에 쓴다.** 나머지 두 상태를 넣으면 안 되는 이유가 각각 다르다.

    - `RAY_NO_FREE`(53 %)는 목표가 0이고 그 셀들은 이미 hard `Ω_N`으로 잘 학습된다. 넣으면
      광선 평균의 절반이 상수 0인 항이 되어 `L_range` 값이 읽히지 않고 `λ_R`의 실효 크기가
      2배 희석된다.
    - `RAY_CENSORED`(12 %)는 격자 끝까지 free라 **하한만 아는** 광선이다. 양방향 loss를 걸면
      모르는 값을 목표로 삼는 것이 된다.

    이것은 M3(`free_space_metrics.range_error`)가 "GT와 예측이 둘 다 `RAY_OK`인 광선만
    회귀 통계에 넣는다"고 정한 규약과 같은 판단이다.
    """
    has_free = sampled_free.any(dim=-1)
    # `argmax`는 True가 없으면 0을 주는데, 그 경우는 `has_free`가 False라 아래에서 걸러진다.
    first = sampled_free.to(torch.uint8).argmax(dim=-1, keepdim=True)
    steps = torch.arange(sampled_free.shape[-1], device=sampled_free.device)
    blocked = ((steps >= first) & inside & ~sampled_free).any(dim=-1)
    return has_free & blocked


def compute_range_loss(prob_free, free_gt, valid, gather, delta_r=DEFAULT_DELTA_R_M,
                       beta=DEFAULT_HUBER_BETA_M):
    """`(loss, 항별 dict)`. `prob_free`는 `(B, 1, H, W)`의 `p(free)`다.

    **목표는 `polar.first_free_range`의 `R_gt`가 아니라 `arc = Δr·Σ(free)`다.** 둘은 같은
    양이 아니다 -- `R_gt`는 ego 원점부터의 반지름이고 `first_free_range`의 계약은 ego 아래
    `permanent_blind` 원반을 **건너뛰고 잰다**는 것이므로, 실측에서 `R_gt − arc`가 평균
    **0.758 m**(val split 40프레임)다. 제안된 `δ_R = 0.20 m`의 3.8배인 계통 편차이므로 그대로
    쓰면 dead zone이 무력화되고, loss가 그 상수를 줄이려고 **원반 안을 free로 예측하라고
    요구한다** -- 원반은 hard `Ω_N`(목표 0)으로 감독되는 자리라 두 항이 정면으로 싸운다.

    예측과 목표에 **같은 연산자**를 쓰면 그 편차가 정의상 사라진다. 부수 효과로 720광선·0.5셀
    표본화와 라벨 raycast의 이산화 불일치(실측 0.035 m)도 같이 상쇄된다. `arc`와 `R_gt`는
    거의 정적인 per-θ 상수만큼 차이 나므로 **`δ_R`의 물리적 뜻("이 방향 자유거리를 몇 cm까지
    용서하나")은 그대로 보존된다.**

    합은 광선 전체에 걸린다 -- 첫 장애물 앞까지가 아니다. GT free는 ego 원점 raycast의
    결과라 광선 위에서 연속 구간 하나이므로(`free = occ & vis`) 두 정의가 GT에서는 같지만,
    **예측에서는 다르고 그 차이가 이 항이 잡으려는 오차 모드다** -- 벽 뒤의 free 섬이 `arc`를
    늘려 벌을 받는다.

    `valid = 0` 셀은 양쪽에서 뺀다. GT free는 이미 0이지만(`decompose`가 `valid`를 요구한다)
    예측 확률은 그 셀에서 감독되지 않아 아무 값이나 들어 있다.
    """
    inside = gather.inside
    valid_ray = gather(valid.to(prob_free.dtype)) * inside
    sampled_gt = gather(free_gt.to(torch.bool)) & inside

    arc_gt = sampled_gt.to(prob_free.dtype).sum(dim=-1) * gather.step_m
    arc_hat = (gather(prob_free) * valid_ray).sum(dim=-1) * gather.step_m

    ok = ray_is_ok(sampled_gt, inside)
    ok_f = ok.to(prob_free.dtype)
    n_ok = ok_f.sum() + 1e-6

    signed = arc_hat - arc_gt
    error = signed.abs()
    # **dead zone.** 여기서 loss가 평평해지고 gradient가 정확히 0이 된다 -- 그것이 이 항의
    # 요점이다(모듈 docstring).
    effective = (error - delta_r).clamp_min(0.0)
    # `F.smooth_l1_loss(e, 0, beta=β)`가 정확히 `ρ_β(e) = e²/(2β) | e−β/2`다.
    per_ray = F.smooth_l1_loss(effective, torch.zeros_like(effective),
                              reduction="none", beta=beta)

    loss = (per_ray * ok_f).sum() / n_ok
    parts = {
        "loss_range": loss,
        # dead zone **전**의 순수 거리 오차 [m]. 항이 실제로 일하는지 읽는 값이고,
        # `range_mae`(M3, argmax 첫 hit)와는 다른 양이므로 이름을 따로 둔다.
        "range_arc_mae": (error * ok_f).sum() / n_ok,
        # 부호 있는 평균. **`fatal` 방향의 감시용이다** -- dead zone이 `|e|`에 대칭이라
        # "벽 20 cm 안쪽까지 free" 예측이 무벌점이 되는데 로봇에게는 그쪽이 치명 방향이다
        # (설계 문서 §4.5의 제약). 양수면 자유공간을 과대예측하고 있다는 뜻이다.
        "range_arc_bias": (signed * ok_f).sum() / n_ok,
        "frac_rays_used": ok_f.mean(),
    }
    return loss, parts
