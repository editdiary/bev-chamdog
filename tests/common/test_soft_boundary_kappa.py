"""대역 target 수축 계수 `kappa`의 계약을 고정한다 (2026-08-27).

`y' = κ·y + (1−κ)/2`. **`δ`(대역 폭)·`α`(모양)와 직교하는 셋째 손잡이**이고, 도입 이유는
앞의 둘이 둘 다 막혀 있다는 실측이다 -- `α`는 함의 폭이 `δ/√3`에 포화하고(α=1.0에서 이미
상한의 93 %), `δ`는 넓히면 좁은 통로에서 hard `Ω_F`를 먹는다(δ=0.45면 자유공간의 69 %가
자격을 잃는다). 근거 전체는 `soft_boundary.DEFAULT_KAPPA`의 주석에 있다.

여기서 못박는 것은 넷이다.

1. **`κ = 1.0`은 항등이다.** 기존 런이 새 경로를 타서 조용히 달라지면 안 된다.
2. **대역 폭을 안 건드린다.** `region_masks`가 `κ`를 아예 안 본다 -- 이것이 `δ`와 다른 점의
   전부이고, 사용자가 `δ` 확대를 거부한 이유(통로 기하)가 여기 걸려 있다.
3. **대칭성이 보존된다.** `y(d) + y(−d) = 1`이 `κ` 후에도 성립해야 `L_range`의 arc 중립성이
   남는다(`tests/tools/test_diagnose_range_dead_zone.py`가 그 성질을 쓴다).
4. **평평해지는 방향이 맞다.** target 엔트로피가 `κ`에 대해 단조 증가해야 한다 -- 그게
   "확신을 낮춘다"는 이 손잡이의 정의다.
"""
import math

import pytest
import torch

from projects.common.soft_boundary import (
    TARGET_GAUSSIAN,
    TARGET_LINEAR,
    apply_kappa,
    compute_soft_boundary_loss,
    region_masks,
    soft_target,
    target_entropy,
)

_DELTA = 0.15
_KINDS = ((TARGET_LINEAR, {}), (TARGET_GAUSSIAN, {"alpha": 0.5}))


def _band_d():
    """대역 안 표본. **중점 표본이라 `d = 0`이 짝 없이 남지 않는다.**"""
    n = 200
    return (torch.arange(2 * n, dtype=torch.float32) + 0.5) / n * _DELTA - _DELTA


@pytest.mark.parametrize("kind,kwargs", _KINDS)
def test_kappa_one_is_the_identity(kind, kwargs):
    d = _band_d()
    torch.testing.assert_close(soft_target(d, _DELTA, kind, kappa=1.0, **kwargs),
                               soft_target(d, _DELTA, kind, **kwargs), rtol=0, atol=0)


@pytest.mark.parametrize("kind,kwargs", _KINDS)
def test_kappa_shrinks_toward_one_half(kind, kwargs):
    d = _band_d()
    base = soft_target(d, _DELTA, kind, **kwargs)
    for kappa in (0.8, 0.5, 0.2):
        y = soft_target(d, _DELTA, kind, kappa=kappa, **kwargs)
        torch.testing.assert_close(y, kappa * base + (1 - kappa) * 0.5, rtol=0, atol=1e-6)
        # 0.5로부터의 거리가 정확히 `kappa`배가 된다.
        torch.testing.assert_close((y - 0.5).abs(), kappa * (base - 0.5).abs(),
                                   rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind,kwargs", _KINDS)
def test_kappa_preserves_the_symmetry_that_keeps_the_arc_neutral(kind, kwargs):
    """`y(d) + y(−d) = 1`이 `κ` 후에도 성립한다: `κ·1 + (1−κ) = 1`."""
    d = _band_d()
    for kappa in (1.0, 0.6, 0.3):
        y = soft_target(d, _DELTA, kind, kappa=kappa, **kwargs)
        torch.testing.assert_close(y + y.flip(0), torch.ones_like(y), rtol=0, atol=1e-5)
        # 그 결과 hard GT와의 잔차 합이 0 -- `L_range`의 `arc`가 안 흔들린다.
        assert abs(float((y - (d > 0).to(y.dtype)).sum())) < 1e-3


@pytest.mark.parametrize("kind,kwargs", _KINDS)
def test_entropy_rises_monotonically_as_kappa_falls(kind, kwargs):
    """엔트로피 = 줄일 수 없는 하한. `κ`를 낮추면 커져야 한다 -- 그게 '확신을 낮춘다'는 뜻이다."""
    d = _band_d()
    entropies = [float(target_entropy(soft_target(d, _DELTA, kind, kappa=k, **kwargs)).mean())
                 for k in (1.0, 0.8, 0.6, 0.4, 0.2)]
    assert entropies == sorted(entropies), entropies
    # `κ → 0`이면 대역 전체가 0.5, 즉 완전한 무지(ln 2)에 수렴한다.
    flat = float(target_entropy(soft_target(d, _DELTA, kind, kappa=1e-4, **kwargs)).mean())
    assert flat == pytest.approx(math.log(2), abs=1e-6)


def test_kappa_does_not_change_which_cells_are_supervised():
    """**`δ`와 갈리는 지점.** `region_masks`는 `κ`를 인자로 받지도 않는다."""
    torch.manual_seed(0)
    d = (torch.rand(1, 1, 40, 40) - 0.5) * 2.0
    valid = torch.ones_like(d)
    blind = torch.zeros_like(d)
    base = region_masks(d, valid, blind, delta=_DELTA)
    # 대역 폭을 넓히면 집합이 바뀐다 -- 대조군.
    wider = region_masks(d, valid, blind, delta=0.45)
    assert int(wider["omega_b"].sum()) > int(base["omega_b"].sum())
    assert int(wider["omega_f"].sum()) < int(base["omega_f"].sum())


def test_loss_is_bit_identical_at_kappa_one():
    """총 loss 경로 전체에서 `κ = 1.0`이 항등인지 -- 회귀 방지의 본체."""
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 32, 32)
    d = (torch.rand(2, 1, 32, 32) - 0.5) * 2.0
    valid = (torch.rand(2, 1, 32, 32) > 0.1).float()
    blind = (torch.rand(2, 1, 32, 32) > 0.9).float()
    kw = dict(delta=_DELTA, lambda_b=0.5, kind=TARGET_GAUSSIAN, alpha=0.5)
    ref, ref_parts = compute_soft_boundary_loss(logits, d, valid, blind, **kw)
    got, got_parts = compute_soft_boundary_loss(logits, d, valid, blind, kappa=1.0, **kw)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for key in ref_parts:
        torch.testing.assert_close(got_parts[key], ref_parts[key], rtol=0, atol=0)


def test_kappa_out_of_range_fails_loudly():
    """0이나 음수는 확률이 아니고, 1보다 크면 0.5에서 **멀어지는** 반대 동작이 된다."""
    y = torch.tensor([0.3, 0.7])
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="kappa"):
            apply_kappa(y, bad)


# --- 균일 label smoothing `eps` -------------------------------------------------
#
# `kappa`가 대역만 건드려서 실패한 자리를 여는 손잡이다. 여기서 못박는 것은 셋이다.
#
# 1. **`eps = 0`이 항등이다** -- 기존 런이 새 경로를 타면 안 된다.
# 2. **`eps`가 대역에서 정확히 `kappa = 1-2eps`로 작동한다** -- 이것이 대역 끝의 거꾸로 된
#    불연속(대역 끝 1.0 대 바로 옆 `1-eps`)을 막는 유일한 형태다.
# 3. **`Omega_F`/`Omega_N`에 상수 하한 `H(eps)`이 생긴다** -- 빼지 않고 읽으면 성능 붕괴로
#    오독된다(eps=0.10에서 0.325).


def test_eps_zero_is_the_identity():
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 32, 32)
    d = (torch.rand(2, 1, 32, 32) - 0.5) * 2.0
    valid = (torch.rand(2, 1, 32, 32) > 0.1).float()
    blind = (torch.rand(2, 1, 32, 32) > 0.9).float()
    kw = dict(delta=_DELTA, lambda_b=0.5, kind=TARGET_GAUSSIAN, alpha=0.5)
    ref, ref_parts = compute_soft_boundary_loss(logits, d, valid, blind, **kw)
    got, got_parts = compute_soft_boundary_loss(logits, d, valid, blind, eps=0.0, **kw)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for key in ref_parts:
        torch.testing.assert_close(got_parts[key], ref_parts[key], rtol=0, atol=0)


def test_eps_meets_the_band_edge_without_a_jump():
    """**대역 끝이 바로 옆 hard 영역과 정확히 이어져야 한다.**

    정규화된 target은 `y(±δ) = 1/0`을 정확히 주므로, `eps`를 hard 영역에만 걸면 대역 끝이
    `Ω_F`보다 더 확신에 찬 거꾸로 된 불연속이 생긴다. `y' = ε + (1−2ε)y`가 그것을 없앤다.
    """
    from projects.common.soft_boundary import apply_kappa as _ak
    for eps in (0.02, 0.10, 0.25):
        edge = torch.tensor([_DELTA, -_DELTA])
        y = _ak(soft_target(edge, _DELTA, TARGET_GAUSSIAN, alpha=0.5), 1 - 2 * eps)
        torch.testing.assert_close(y, torch.tensor([1 - eps, eps]), rtol=0, atol=1e-6)


def test_eps_puts_a_constant_floor_on_the_hard_terms():
    """`H(ε)`만큼은 어떤 예측으로도 못 줄인다. `kl_free`/`kl_not_free`가 그것을 뺀 값이다."""
    from projects.common.soft_boundary import _binary_entropy
    eps = 0.1
    # 완벽한 예측(logit이 목표를 정확히 재현)에서도 loss가 H(eps)로 남는다.
    p = torch.full((1, 1, 8, 8), 1 - eps)
    logits = torch.cat([(1 - p).log(), p.log()], dim=1)
    d = torch.full((1, 1, 8, 8), 1.0)                      # 전부 Ω_F
    valid, blind = torch.ones_like(d), torch.zeros_like(d)
    _, parts = compute_soft_boundary_loss(logits, d, valid, blind, delta=_DELTA,
                                          lambda_b=0.5, kind=TARGET_GAUSSIAN, alpha=0.5, eps=eps)
    assert float(parts["loss_free"]) == pytest.approx(_binary_entropy(eps), abs=1e-5)
    assert float(parts["kl_free"]) == pytest.approx(0.0, abs=1e-5)
    assert _binary_entropy(eps) == pytest.approx(0.325083, abs=1e-5)


def test_eps_out_of_range_fails_loudly():
    logits = torch.randn(1, 2, 8, 8)
    d = torch.zeros(1, 1, 8, 8)
    v = torch.ones_like(d)
    for bad in (-0.1, 0.5, 0.9):
        with pytest.raises(ValueError, match="eps"):
            compute_soft_boundary_loss(logits, d, v, torch.zeros_like(d), eps=bad)
