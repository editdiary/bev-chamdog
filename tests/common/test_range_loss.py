"""`L_range`의 계약 테스트.

고정하려는 것은 **설계 문서 §13이 주장하는 성질들**이다.

- `ray_is_ok`가 `polar.first_free_range`의 `RAY_OK`와 **정확히 같은 광선**을 고른다
  (torch로 다시 구현했으므로 원본과 갈리면 조용히 틀린다)
- 목표가 `R_gt`가 아니라 `arc`이고, 그래서 `permanent_blind` 원반의 계통 편차가 없다
- **dead zone 안에서 gradient가 정확히 0이다** -- 이 항의 존재 이유다
- 벽 뒤의 free 섬이 벌을 받는다 (per-cell BCE가 거의 안 보는 오차 모드)
- `valid = 0` 셀이 양쪽에서 빠진다
"""
import numpy as np
import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.polar import RAY_OK, build_ray_index, first_free_range
from projects.common.soft_boundary import compute_soft_boundary_loss, signed_distance_field
from projects.common.range_loss import (
    DEFAULT_HUBER_BETA_M,
    RayGather,
    compute_range_loss,
    ray_is_ok,
)

# 작은 격자로 광선 수를 줄여 테스트를 빠르게 유지한다. 규약(θ=0이 전방, row 감소)은
# 격자 크기와 무관하다.
SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)
N_THETA = 64


@pytest.fixture(scope="module")
def rays():
    return build_ray_index(SPEC, n_theta=N_THETA)


@pytest.fixture(scope="module")
def gather(rays):
    return RayGather(rays, (SPEC.n_rows, SPEC.n_cols))


def _disc(radius_m):
    """원점 중심 반지름 `radius_m`의 원반 마스크 -- free 영역으로 쓴다."""
    forward = SPEC.front_m - (np.arange(SPEC.n_rows) + 0.5) * SPEC.cell_m
    lateral = SPEC.half_width_m - (np.arange(SPEC.n_cols) + 0.5) * SPEC.cell_m
    r = np.hypot(forward[:, None], lateral[None, :])
    return r < radius_m


def _as_batch(mask):
    return torch.from_numpy(np.asarray(mask)).view(1, 1, *np.shape(mask))


def test_ray_is_ok_matches_first_free_range(rays, gather):
    """torch 판정이 numpy 원본과 광선 하나까지 일치해야 한다."""
    for free in (_disc(0.6), _disc(0.95), np.zeros((SPEC.n_rows, SPEC.n_cols), bool)):
        _, status = first_free_range(free, rays)
        sampled = gather(_as_batch(free)) & gather.inside
        got = ray_is_ok(sampled, gather.inside)[0].numpy()
        assert (got == (status == RAY_OK)).all()


def test_arc_target_has_no_blind_disc_offset(gather):
    """목표를 `arc`로 두면 원반만큼의 계통 편차가 생기지 않는다.

    실측에서 `R_gt − arc`가 0.758 m였다(`compute_range_loss` docstring). GT를 그대로
    예측으로 넣으면 오차는 0이어야 하고, 그것이 두 양을 같은 연산자로 재고 있다는 증거다.
    """
    free = _disc(0.6)
    valid = np.ones_like(free)
    prob = _as_batch(free).float()
    _, parts = compute_range_loss(prob, _as_batch(free), _as_batch(valid), gather)
    assert parts["range_arc_mae"].item() == pytest.approx(0.0, abs=1e-6)
    assert parts["loss_range"].item() == pytest.approx(0.0, abs=1e-9)
    assert parts["frac_rays_used"].item() > 0.5


def test_gradient_is_exactly_zero_inside_dead_zone(gather):
    """**이 항의 존재 이유.** δ_R 안의 변위에는 gradient가 없다.

    경계를 한 셀(0.05 m) 안쪽으로 민 예측을 준다. `δ_R = 0.20`이면 그 오차는 dead zone
    안이므로 gradient가 정확히 0이고, `δ_R = 0.0`이면 0이 아니어야 한다.
    """
    free = _disc(0.6)
    shrunk = _disc(0.55)
    valid = _as_batch(np.ones_like(free))

    for delta_r, expect_zero in ((0.20, True), (0.0, False)):
        prob = _as_batch(shrunk).float().requires_grad_(True)
        loss, parts = compute_range_loss(prob, _as_batch(free), valid, gather,
                                         delta_r=delta_r)
        loss.backward()
        grad_norm = prob.grad.abs().sum().item()
        assert parts["range_arc_mae"].item() > 0.01, "예측이 실제로 밀려 있어야 한다"
        if expect_zero:
            assert loss.item() == 0.0
            assert grad_norm == 0.0
        else:
            assert loss.item() > 0.0
            assert grad_norm > 0.0


def test_free_island_behind_the_wall_is_penalized(gather):
    """per-cell BCE가 거의 안 보는 오차 모드를 이 항이 잡는다.

    광선 위 free 셀 수가 늘어나므로 `arc`가 길어지고, dead zone을 넘으면 벌을 받는다.
    """
    free = _disc(0.4)
    with_island = free.copy()
    with_island[0:6, 0:6] = True          # 벽 뒤 모서리에 free 섬
    valid = _as_batch(np.ones_like(free))

    _, clean = compute_range_loss(_as_batch(free).float(), _as_batch(free), valid,
                                  gather, delta_r=0.0)
    _, dirty = compute_range_loss(_as_batch(with_island).float(), _as_batch(free), valid,
                                  gather, delta_r=0.0)
    assert clean["range_arc_mae"].item() == pytest.approx(0.0, abs=1e-6)
    assert dirty["range_arc_mae"].item() > 0.0
    # 섬은 free를 **더** 찍은 것이므로 과대예측 쪽으로 치우쳐야 한다.
    assert dirty["range_arc_bias"].item() > 0.0


def test_invalid_cells_are_excluded_from_both_sides(gather):
    """`valid = 0` 셀의 예측 확률은 감독되지 않으므로 `arc`에 들어가면 안 된다."""
    free = _disc(0.6)
    valid = np.ones_like(free)
    valid[0:8, :] = False                 # 격자 앞쪽 띠를 무효로

    # 무효 영역에 free를 잔뜩 찍은 예측. 제외되므로 GT와 같은 `arc`가 나와야 한다.
    prob = free.astype(np.float32).copy()
    prob[0:8, :] = 1.0
    gt_free = free & valid                # `decompose`가 `valid`를 요구하는 것과 같게

    _, parts = compute_range_loss(_as_batch(prob), _as_batch(gt_free), _as_batch(valid),
                                  gather)
    assert parts["range_arc_mae"].item() == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_folds_the_range_term_into_total_and_shares(gather):
    """`share_*`의 합이 1로 유지되고, `range_term=None`이면 항이 아예 없어야 한다.

    몫의 합이 1이라는 성질이 이 프로젝트에서 항의 영향력을 읽는 주된 도구다(§5.6). 총 loss를
    두 곳에서 더하면 그 표가 조용히 무의미해지므로 여기서 고정한다.
    """
    free = _disc(0.6)
    valid = _as_batch(np.ones_like(free))
    blind = _as_batch(np.zeros_like(free))
    d = _as_batch(signed_distance_field(free, np.ones_like(free), SPEC.cell_m))
    logits = torch.zeros(1, 2, SPEC.n_rows, SPEC.n_cols)
    logits[:, 1] = 1.0

    range_term = compute_range_loss(_as_batch(_disc(0.3)).float(), _as_batch(free),
                                    valid, gather, delta_r=0.0)

    without, parts_without = compute_soft_boundary_loss(logits, d, valid, blind)
    with_range, parts_with = compute_soft_boundary_loss(
        logits, d, valid, blind, range_term=range_term, lambda_r=0.3
    )

    assert "share_range" not in parts_without and "loss_range" not in parts_without
    assert with_range.item() == pytest.approx(
        without.item() + 0.3 * range_term[0].item(), rel=1e-6
    )
    shares = [parts_with[f"share_{name}"].item()
              for name in ("free", "not_free", "boundary", "range")]
    assert sum(shares) == pytest.approx(1.0, abs=1e-4)
    assert parts_with["share_range"].item() > 0.0


def test_huber_is_quadratic_then_linear(gather):
    """`ρ_β`가 β에서 이차 -> 선형으로 바뀐다. 큰 outlier에 덜 끌려가는 근거.

    **광선마다 오차가 달라 `mean(ρ(e))`를 `ρ(mean(e))`와 비교할 수 없다**(Jensen). 그래서
    오차 분포에 의존하지 않는 두 성질로 고정한다.

    - 이차 영역(`모든 e < β`)에서는 `ρ ∝ 1/β`이므로 β를 2배로 하면 loss가 정확히 절반이다
    - 선형 영역(`모든 e > β`)에서는 `ρ = e − β/2`가 선형이라 평균과 교환되므로
      `loss = mae − β/2`가 **정확히** 성립한다
    """
    free = _disc(0.6)
    valid = _as_batch(np.ones_like(free))

    def loss_for(radius_m, beta):
        _, parts = compute_range_loss(_as_batch(_disc(radius_m)).float(), _as_batch(free),
                                      valid, gather, delta_r=0.0, beta=beta)
        return parts["loss_range"].item(), parts["range_arc_mae"].item()

    beta = DEFAULT_HUBER_BETA_M
    small_loss, small_e = loss_for(0.57, beta)
    big_loss, big_e = loss_for(0.30, beta)
    assert small_e < beta < big_e

    # 이차 영역: β를 2배로 하면 loss가 절반.
    small_loss_wide, _ = loss_for(0.57, 2 * beta)
    assert small_loss == pytest.approx(2 * small_loss_wide, rel=1e-4)
    # 선형 영역: 평균과 교환되므로 등식이 정확하다.
    assert big_loss == pytest.approx(big_e - beta / 2, rel=1e-4)
