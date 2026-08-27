"""`tools/diagnose_range_dead_zone.py`의 전제를 고정한다.

이 도구가 재는 값이 뜻을 가지려면 **부호로 나눈 뒤 다시 합친 것이 loss가 보는 값과 같아야
한다.** 즉 `e_over − e_under == arc_hat − arc_gt`여야 하고, 이것이 깨지면 [A]의 상쇄율도
[C2]의 층별 분해도 `compute_range_loss`와 다른 양을 재게 된다 -- 그런데 숫자는 여전히
그럴듯하게 나오므로 조용히 틀린다. 그래서 테스트로 못박는다.

마스킹도 같이 고정한다. `arc_hat`은 `valid`로 지우고 `arc_gt`는 `inside`로만 지우는데
(`compute_range_loss`의 규약), 잔차를 만들 때 한쪽 마스크를 빠뜨리면 `valid = 0` 셀에서
가짜 불일치가 생긴다.
"""
import numpy as np
import torch

from projects.common.polar import build_ray_index
from projects.common.range_loss import RayGather, compute_range_loss
from projects.datasets.robot_simplebev import GRID_SPEC


def _fixture(seed=0, batch=2):
    torch.manual_seed(seed)
    gather = RayGather(build_ray_index(GRID_SPEC), (GRID_SPEC.n_rows, GRID_SPEC.n_cols))
    shape = (batch, 1, GRID_SPEC.n_rows, GRID_SPEC.n_cols)
    prob = torch.rand(shape)
    free_gt = (torch.rand(shape) > 0.5).float()
    # `valid = 0`을 일부러 섞는다 -- 마스킹이 어긋나면 여기서 갈린다.
    valid = (torch.rand(shape) > 0.1).float()
    return gather, prob, free_gt, valid


def _decomposition(gather, prob, free_gt, valid):
    """도구의 `collect`가 하는 것과 **같은 계산**. 여기서 갈라지면 테스트가 뜻을 잃는다."""
    inside = gather.inside
    valid_ray = gather(valid.to(prob.dtype)) * inside
    sampled_gt = gather(free_gt.to(torch.bool)) & inside
    resid = gather(prob) * valid_ray - sampled_gt.to(prob.dtype)
    over = resid.clamp_min(0.0).sum(-1) * gather.step_m
    under = (-resid).clamp_min(0.0).sum(-1) * gather.step_m
    return over, under


def test_signed_split_reconstructs_what_the_loss_sees():
    gather, prob, free_gt, valid = _fixture()
    inside = gather.inside
    sampled_gt = gather(free_gt.to(torch.bool)) & inside
    arc_gt = sampled_gt.to(prob.dtype).sum(-1) * gather.step_m
    arc_hat = (gather(prob) * gather(valid) * inside).sum(-1) * gather.step_m

    over, under = _decomposition(gather, prob, free_gt, valid)
    torch.testing.assert_close(over - under, arc_hat - arc_gt, rtol=0, atol=1e-5)


def test_arc_bias_matches_compute_range_loss():
    """`RAY_OK` 광선만 평균하면 `compute_range_loss`의 `range_arc_bias`와 같아야 한다."""
    from projects.common.range_loss import ray_is_ok

    gather, prob, free_gt, valid = _fixture(seed=1)
    over, under = _decomposition(gather, prob, free_gt, valid)
    ok = ray_is_ok(gather(free_gt.to(torch.bool)) & gather.inside, gather.inside)
    mine = ((over - under) * ok).sum() / (ok.sum() + 1e-6)

    _, parts = compute_range_loss(prob, free_gt, valid, gather)
    torch.testing.assert_close(mine, parts["range_arc_bias"], rtol=0, atol=1e-5)


def test_absolute_disagreement_bounds_the_signed_error():
    """`e_abs >= |e_signed|`. 상쇄율이 음수가 될 수 없다는 것과 같은 말이다."""
    gather, prob, free_gt, valid = _fixture(seed=2)
    over, under = _decomposition(gather, prob, free_gt, valid)
    assert bool((over + under >= (over - under).abs() - 1e-6).all())


def test_soft_target_contributes_zero_to_the_arc():
    """**대역이 제자리에 있으면 soft target은 `arc`를 흔들지 않는다.**

    `y(d) + y(−d) = 1`이므로 경계 기준 대칭인 표본 쌍에서 `(y(d) − 1) + (y(−d) − 0) = 0`이다.
    이것이 성립하지 않으면 `δ_R`의 여유가 soft 대역의 번짐을 흡수하는 데 쓰이고 있다는 뜻이
    되어 [C2]의 '설명 상한 0.00'이 거짓이 된다. `D_range_s0`의 train 실측이
    `mean s_band = −0.0011`로 이것을 뒷받침한다.
    """
    from projects.common.soft_boundary import TARGET_GAUSSIAN, soft_target

    delta = 0.15
    # **중점 표본이다.** `linspace(-δ, δ, 2n+1)`을 쓰면 `d = 0`인 점이 짝 없이 남고, 그 점의
    # 부호가 부동소수점으로 정해져 잔차가 ±y(0) = ±0.5만큼 튄다. 실제 광선 표본도 셀 중심을
    # 지나므로 경계에 정확히 걸리지 않는다(`signed_distance_field`가 중심 사이로 재고,
    # 그래서 경계에 붙은 셀도 `|d| = cell_m`이다).
    n = 200
    d = (torch.arange(2 * n, dtype=torch.float32) + 0.5) / n * delta - delta
    for kind, kwargs in (("linear", {}), (TARGET_GAUSSIAN, {"alpha": 0.5})):
        y = soft_target(d, delta=delta, kind=kind, **kwargs)
        # `y(d) + y(−d) == 1`
        torch.testing.assert_close(y + y.flip(0), torch.ones_like(y), rtol=0, atol=1e-5)
        # 그 결과 hard GT(`d > 0`)와의 잔차 합이 0이다.
        residual = (y - (d > 0).to(y.dtype)).sum()
        assert abs(float(residual)) < 1e-3, f"{kind}: {residual}"


def test_band_stratification_partitions_the_total():
    """층 3개의 합이 총량과 같아야 한다 -- 층 경계가 겹치거나 비면 [C]의 몫이 거짓이 된다."""
    edges = ((0.0, 0.15), (0.15, 0.5), (0.5, np.inf))
    d = torch.rand(1000) * 2.0
    masks = [(d >= lo) & (d < hi) for lo, hi in edges]
    stacked = torch.stack(masks).sum(0)
    assert bool((stacked == 1).all())
