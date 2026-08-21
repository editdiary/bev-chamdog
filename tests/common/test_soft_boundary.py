"""soft-boundary loss의 계약 테스트.

고정하려는 것은 **설계 문서가 주장하는 성질들**이다: 세 영역이 `valid`를 정확히 덮는지,
`permanent_blind`가 `Ω_F`로 새지 않는지, 선형 target이 hard 영역과 이어지는지, soft BCE의
하한이 target 엔트로피인지, 그리고 최적점이 `p = y`라 argmax 경계가 `d = 0`에 놓이는지.
"""
import numpy as np
import pytest
import torch

from projects.common.soft_boundary import (
    TARGET_GAUSSIAN,
    TARGET_LINEAR,
    compute_soft_boundary_loss,
    region_masks,
    signed_distance_field,
    soft_target,
    target_entropy,
)

CELL_M = 0.05


def _free_half(n=12):
    """왼쪽 절반이 free인 격자. 경계가 col 5|6 사이에 수직으로 선다."""
    free = np.zeros((n, n), bool)
    free[:, :6] = True
    return free


def test_signed_distance_sign_and_magnitude():
    free = _free_half()
    keep = np.ones_like(free)
    d = signed_distance_field(free, keep, CELL_M)
    # 경계에 붙은 셀은 셀 중심 사이 거리이므로 정확히 한 셀이고 0이 아니다.
    assert d[0, 5] == pytest.approx(CELL_M)
    assert d[0, 6] == pytest.approx(-CELL_M)
    assert (d[:, :6] > 0).all() and (d[:, 6:] < 0).all()
    # 안쪽으로 들어갈수록 커진다.
    assert d[0, 0] > d[0, 4]


def test_permanent_blind_is_merged_for_distance_but_not_for_supervision():
    """원반이 경계를 만들지 않아야 하고, 그러면서도 `Ω_N`으로 감독돼야 한다."""
    free = _free_half()
    keep = np.ones_like(free)
    keep[5:7, 2:4] = False          # free 영역 안의 정적 사각 -- free에 합쳐진다
    d = signed_distance_field(free, keep, CELL_M)
    # 합치지 않았다면 원반 테두리가 경계가 되어 이웃 셀의 `d`가 한 셀로 떨어진다.
    assert d[5, 1] > CELL_M

    blind = torch.from_numpy(~keep).view(1, 1, *free.shape)
    valid = torch.ones_like(blind)
    regions = region_masks(torch.from_numpy(d).view(1, 1, *free.shape), valid, blind, delta=0.15)
    assert not regions["omega_f"][blind].any(), "원반이 Ω_F로 새면 free로 학습된다"
    assert regions["omega_n"][blind].all(), "원반은 hard not_free로 감독돼야 한다"


def test_regions_partition_valid_exactly():
    free = _free_half()
    keep = np.ones_like(free)
    keep[0, 0] = False
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    valid[0, 0, 11, 11] = False     # `valid=0`은 어느 영역에도 들어가지 않는다
    blind = torch.from_numpy(~keep).view(1, 1, *free.shape)

    regions = region_masks(d, valid, blind, delta=0.15)
    union = regions["omega_f"] | regions["omega_n"] | regions["omega_b"]
    assert torch.equal(union, valid)
    pairs = [("omega_f", "omega_n"), ("omega_f", "omega_b"), ("omega_n", "omega_b")]
    for a, b in pairs:
        assert not (regions[a] & regions[b]).any(), f"{a}와 {b}가 겹친다"


def test_linear_target_meets_hard_region_at_the_band_edge():
    """`d = ±δ`에서 정확히 1/0이어야 옆의 hard 셀과 목표가 튀지 않는다."""
    delta = 0.15
    d = torch.tensor([-delta, -delta / 2, 0.0, delta / 2, delta])
    y = soft_target(d, delta, TARGET_LINEAR)
    assert y[0].item() == pytest.approx(0.0)
    assert y[2].item() == pytest.approx(0.5)
    assert y[-1].item() == pytest.approx(1.0)
    assert torch.all(y[1:] > y[:-1]), "단조여야 argmax 경계가 d=0에 놓인다"


def test_gaussian_target_is_normalized_over_the_band():
    """정규화하지 않은 `Φ`는 대역 끝에서 1에 못 미친다 -- 그 결함이 없어야 한다."""
    delta, sigma = 0.15, 0.15
    d = torch.tensor([-delta, 0.0, delta])
    y = soft_target(d, delta, TARGET_GAUSSIAN, sigma)
    assert y[0].item() == pytest.approx(0.0, abs=1e-6)
    assert y[1].item() == pytest.approx(0.5, abs=1e-6)
    assert y[2].item() == pytest.approx(1.0, abs=1e-6)
    # 생 Φ라면 대역 끝이 Φ(1) = 0.841이었다 -- 정규화가 실제로 걸렸는지 확인한다.
    assert torch.special.ndtr(torch.tensor(1.0)).item() == pytest.approx(0.841, abs=1e-3)


def test_gaussian_requires_sigma():
    with pytest.raises(ValueError):
        soft_target(torch.zeros(3), 0.15, TARGET_GAUSSIAN)
    with pytest.raises(ValueError):
        soft_target(torch.zeros(3), 0.15, "step")


def test_soft_bce_floor_is_target_entropy():
    """target이 상수이므로 최소값은 0이 아니라 `H(y)`다. 이것이 KL 로깅의 근거다."""
    y = torch.tensor([0.5, 0.25, 0.75])
    # `p = y`가 되도록 logit을 직접 만든다 (채널 1 = free).
    logit_free = torch.log(y)
    logit_not_free = torch.log(1.0 - y)
    logits = torch.stack([logit_not_free, logit_free], dim=0).view(1, 2, 1, 3)
    log_probs = torch.log_softmax(logits, dim=1)
    bce = -(y * log_probs[0, 1, 0] + (1 - y) * log_probs[0, 0, 0])
    assert torch.allclose(bce, target_entropy(y), atol=1e-6)
    assert target_entropy(torch.tensor([0.5])).item() == pytest.approx(np.log(2.0))
    # 확정 target의 엔트로피는 0이므로 하한이 없다 -- hard 항과의 대조.
    assert target_entropy(torch.tensor([1.0])).item() == pytest.approx(0.0, abs=1e-6)


def _perfect_logits(free, d, delta, kind=TARGET_LINEAR, sigma=None):
    """`Ω_F`는 free, `Ω_N`은 not_free, `Ω_B`는 `p = y`로 맞춘 이상적인 logits."""
    y = soft_target(d, delta, kind, sigma).clone()
    y[d > delta] = 1.0 - 1e-6
    y[d < -delta] = 1e-6
    y = y.clamp(1e-6, 1 - 1e-6)
    return torch.cat([torch.log(1 - y), torch.log(y)], dim=1)


def test_kl_goes_to_zero_at_the_optimum_while_loss_does_not():
    """`loss_boundary`는 하한에 붙어 남지만 `kl_boundary`는 0으로 가야 한다."""
    free = _free_half()
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    blind = torch.zeros_like(valid)

    logits = _perfect_logits(free, d, 0.15)
    total, parts = compute_soft_boundary_loss(logits, d, valid, blind, delta=0.15)
    assert parts["kl_boundary"].item() == pytest.approx(0.0, abs=1e-4)
    assert parts["entropy_boundary"].item() > 0.1, "선형 target이면 하한이 실제로 존재한다"
    assert parts["loss_boundary"].item() == pytest.approx(
        parts["entropy_boundary"].item(), abs=1e-4)
    assert total.item() > 0.0, "총 loss는 최적점에서도 0이 아니다"


def test_argmax_boundary_sits_at_d_zero_at_the_optimum():
    """soft target이 결정 위치를 옮기지 않는다는 설계 주장(§5.5)을 고정한다."""
    free = _free_half()
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    logits = _perfect_logits(free, d, 0.15)
    pred_free = logits.argmax(dim=1, keepdim=True) == 1
    assert torch.equal(pred_free, d > 0)


def test_shares_sum_to_one_and_fracs_cover_valid():
    free = _free_half()
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    blind = torch.zeros_like(valid)
    logits = torch.zeros(1, 2, *free.shape)      # 균일 예측

    _, parts = compute_soft_boundary_loss(logits, d, valid, blind, delta=0.15)
    shares = sum(parts[f"share_{n}"].item() for n in ("free", "not_free", "boundary"))
    fracs = sum(parts[f"frac_{n}"].item() for n in ("free", "not_free", "boundary"))
    assert shares == pytest.approx(1.0, abs=1e-4)
    assert fracs == pytest.approx(1.0, abs=1e-4)


def test_empty_region_contributes_zero_not_nan():
    """전부 free인 프레임에서 `Ω_N`이 비어도 총 loss가 nan이 되면 안 된다."""
    free = np.ones((8, 8), bool)
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    blind = torch.zeros_like(valid)
    total, parts = compute_soft_boundary_loss(torch.zeros(1, 2, 8, 8), d, valid, blind)
    assert torch.isfinite(total)
    assert parts["frac_not_free"].item() == pytest.approx(0.0)


def test_lambda_b_scales_only_the_boundary_term():
    free = _free_half()
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    blind = torch.zeros_like(valid)
    logits = torch.zeros(1, 2, *free.shape)

    base, base_parts = compute_soft_boundary_loss(logits, d, valid, blind, lambda_b=0.5)
    zero, zero_parts = compute_soft_boundary_loss(logits, d, valid, blind, lambda_b=0.0)
    assert base_parts["loss_free"].item() == pytest.approx(zero_parts["loss_free"].item())
    assert zero_parts["share_boundary"].item() == pytest.approx(0.0)
    expected = zero.item() + 0.5 * base_parts["loss_boundary"].item()
    assert base.item() == pytest.approx(expected, abs=1e-5)


def test_share_boundary_kl_excludes_the_constant_floor():
    """`share_boundary`는 상수 하한을 포함해 경계 항의 영향력을 과대평가한다.

    **완전 최적점에서 재면 안 된다** -- 세 항이 다 0으로 가서 두 몫이 0/0이 되고 비교가
    무의미해진다. 세 항이 모두 살아 있는 균일 예측(`p = 0.5`)에서 재야 차이가 드러난다.
    """
    free = _free_half()
    keep = np.ones_like(free)
    d = torch.from_numpy(signed_distance_field(free, keep, CELL_M)).view(1, 1, *free.shape)
    valid = torch.ones(1, 1, *free.shape, dtype=torch.bool)
    blind = torch.zeros_like(valid)

    # 균일 예측에서는 세 항의 BCE가 모두 log2다(soft target이어도 그렇다) -- 그래서
    # `share_boundary`는 정확히 1/3이 되고, 그중 하한을 뺀 몫은 그보다 훨씬 작아야 한다.
    _, parts = compute_soft_boundary_loss(
        torch.zeros(1, 2, *free.shape), d, valid, blind, delta=0.15)
    assert parts["share_boundary"].item() == pytest.approx(1.0 / 3.0, abs=1e-3)
    # 임의의 비율 임계값을 두지 않는다 -- 하한의 크기는 대역 안 `y` 분포에 달려 있고 그것은
    # 격자·`δ`에 따라 달라진다. 고정할 계약은 "하한을 빼면 몫이 줄어든다"와 그 하한이
    # 실제로 존재한다는 것뿐이다.
    assert parts["share_boundary_kl"].item() < parts["share_boundary"].item()
    assert parts["entropy_boundary"].item() > 0.1


def test_alpha_makes_the_shape_independent_of_delta():
    """**alpha 매개화의 존재 이유.** 같은 alpha면 delta가 달라도 `u = d/delta`에서 같은 target.

    이 성질이 없으면 delta를 넓힐 때 모양이 조용히 같이 둔해져서, 스윕에서 "폭을 넓힌 효과"와
    "모양을 둔하게 한 효과"가 분리되지 않는다.
    """
    alpha, u = 0.4, torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0])
    ys = [soft_target(u * delta, delta, TARGET_GAUSSIAN, alpha=alpha)
          for delta in (0.10, 0.15, 0.30)]
    for other in ys[1:]:
        assert torch.allclose(ys[0], other, atol=1e-6)
    # sigma를 절대값으로 고정하면 반대로 delta에 따라 모양이 **달라진다** (대조).
    fixed = [soft_target(u * delta, delta, TARGET_GAUSSIAN, sigma=0.05)
             for delta in (0.10, 0.30)]
    assert not torch.allclose(fixed[0], fixed[1], atol=1e-3)


def test_large_alpha_converges_to_the_linear_target():
    """선형은 별도 형태가 아니라 `alpha -> inf` 극한이다."""
    delta = 0.15
    d = torch.tensor([-0.12, -0.05, 0.0, 0.05, 0.12])
    linear = soft_target(d, delta, TARGET_LINEAR)
    assert torch.allclose(soft_target(d, delta, TARGET_GAUSSIAN, alpha=50.0),
                          linear, atol=1e-3)
    # 작은 alpha는 계단으로 간다 -- 반대쪽 극한.
    steep = soft_target(d, delta, TARGET_GAUSSIAN, alpha=0.05)
    assert steep[0].item() == pytest.approx(0.0, abs=1e-6)
    assert steep[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert (steep - linear).abs().max().item() > 0.2, "계단과 선형은 크게 달라야 한다"


def test_sigma_and_alpha_are_mutually_exclusive():
    """둘을 다 받으면 config에 적힌 모양과 실제 모양이 갈릴 수 있으므로 즉시 실패한다."""
    with pytest.raises(ValueError):
        soft_target(torch.zeros(3), 0.15, TARGET_GAUSSIAN, sigma=0.05, alpha=0.5)
    with pytest.raises(ValueError):
        soft_target(torch.zeros(3), 0.15, TARGET_GAUSSIAN, alpha=-0.1)
