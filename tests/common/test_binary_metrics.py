"""(D) binary 정식화의 loss·지표 테스트.

3-class와 **같은 지표 dict**를 돌려주는 것이 이 정식화의 계약이다(`occupied`는 예측하지 않고
예측 free의 경계에서 유도한다). 그래서 마지막 두 테스트가 그 유도가 실제로 일어나는지를
고정한다 -- 유도가 빠지면 `f1@τ`가 조용히 0이 되어 3-class 런과 비교할 수 없다.
"""
import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.binary_metrics import (
    CLASS_ORDER,
    COMMON_LOSS_PARTS,
    DECOMPOSITION_DELTA_M,
    LOSS_PART_NAMES,
    NOT_FREE,
    class_weights_from_labels,
    compute_binary_loss,
    compute_free_metrics,
    run_batch,
    run_batch_soft_boundary,
    to_class_index,
)
from projects.common.free_space import FREE, decompose
from projects.common.polar import build_ray_index


SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def _logits_favouring(class_id, shape=(1, 2, 2, 2)):
    logits = torch.full(shape, -5.0)
    logits[:, class_id] = 5.0
    return logits


def _square_scene():
    """원점을 포함하는 free 사각형 + 그 테두리가 occupied, 나머지가 unknown인 라벨.

    `occupied`가 free의 ego 기준 경계라는 라벨의 성질을 그대로 갖는 최소 장면이다 --
    유도가 되돌려야 하는 것이 정확히 이 테두리다.
    """
    size = SPEC.n_rows
    occ = torch.zeros(1, 1, size, size, dtype=torch.bool)
    vis = torch.zeros_like(occ)
    occ[0, 0, 10:30, 10:30] = True          # drivable
    vis[0, 0, 9:31, 9:31] = True            # 관측 = free 사각형 + 테두리 한 겹
    valid = torch.ones_like(occ)
    return occ, vis, valid


def test_loss_is_near_zero_when_the_prediction_is_confidently_right():
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)

    loss, parts = compute_binary_loss(
        _logits_favouring(FREE), target, valid, torch.ones(2)
    )

    assert float(loss) < 0.01
    # `compute_binary_loss`는 손실 함수 자체의 항(클래스별)을 돌려준다. 학습 로그로 나가는 것은
    # `run_batch`가 만드는 공통 분해다(`COMMON_LOSS_PARTS`).
    assert set(parts) == {"loss_free", "loss_not_free", "share_free", "share_not_free"}


def test_invalid_cells_contribute_no_gradient():
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    logits = _logits_favouring(NOT_FREE).requires_grad_()

    loss, _ = compute_binary_loss(logits, target, valid, torch.ones(2))
    loss.backward()

    assert float(loss) == pytest.approx(0.0)
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_free_keeps_index_one_so_the_channel_convention_does_not_flip():
    """3-class와 같은 `FREE=1`을 쓴다 -- 시각화·재채점이 "1번 채널이 free"를 전제한다."""
    assert CLASS_ORDER == (NOT_FREE, FREE) == (0, 1)
    assert LOSS_PART_NAMES == ("free", "not_free", "boundary")

    occ = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.bool)
    vis = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.bool)
    valid = torch.ones_like(occ)

    index = to_class_index(decompose(occ, vis, valid))

    assert index.tolist() == [[[[FREE, NOT_FREE], [NOT_FREE, FREE]]]]


def test_occupied_and_unknown_are_counted_together_as_not_free():
    """가중치가 세는 정의가 loss가 보는 정의와 같아야 한다.

    셀 4개 중 free 2 / occupied 1 / unknown 1이면 binary에서는 2 대 2라 가중치가 1:1이다.
    `occupied`만 not-free로 세면 3 대 1이 되어 값이 달라지므로 이 테스트가 그 실수를 잡는다.
    """
    occ = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    vis = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
    valid = torch.ones_like(occ)

    weights = class_weights_from_labels([(occ, vis, valid)])

    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_class_weights_are_inverse_frequency_and_cap_never_binds_at_two_classes():
    """최빈 클래스를 1로 정규화하므로 가중치는 곧 **셀 수의 비**다.

    8칸 중 free 2 / not_free 6 -> free 가중치 3.0. 로봇 split의 실제 값(free 3.94)이 상한
    20에 한참 못 미치는 것과 같은 구조다 -- 3-class에서 상한이 loss 균형을 결정하던 상황이
    정식화 자체로 사라진다.
    """
    occ = torch.tensor([[[[1, 1, 0, 0, 0, 0, 0, 0]]]], dtype=torch.bool)
    vis = torch.ones_like(occ)
    valid = torch.ones_like(occ)

    weights = class_weights_from_labels([(occ, vis, valid)])

    assert weights[NOT_FREE] == pytest.approx(1.0)
    assert weights[FREE] == pytest.approx(3.0)
    assert float(weights.max()) < 20.0


def test_metrics_derive_occupied_from_the_predicted_free_boundary():
    """이 정식화의 핵심 계약 -- `occupied`를 예측하지 않고 유도해 3-class와 같은 표를 만든다."""
    occ, vis, valid = _square_scene()
    gt = decompose(occ, vis, valid)
    logits = torch.full((1, 2, SPEC.n_rows, SPEC.n_cols), -5.0)
    logits[:, NOT_FREE] = 5.0
    logits[:, FREE][gt["free"][:, 0]] = 5.0      # free를 정확히 맞힌 예측
    logits[:, NOT_FREE][gt["free"][:, 0]] = -5.0

    metrics = compute_free_metrics(
        logits, occ, vis, valid, build_ray_index(SPEC, n_theta=720)
    )

    assert metrics["iou_free"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["partition_defects"] == 0
    # 유도 결과가 지표 dict에 실려 나가야 val 루프의 `BF@τ`가 계산된다.
    assert metrics["pred_occupied"].any()


class _BinaryStubModel:
    def __init__(self, logits):
        self.logits = logits
        self.seen_rgb = None

    def __call__(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util):
        self.seen_rgb = rgb_camXs
        batch = rgb_camXs.shape[0]
        return None, None, self.logits.expand(batch, -1, -1, -1).contiguous(), None, None


def _batch(occ, vis, valid, d):
    return {
        "rgb_camXs": torch.ones(1, 1, 3, 2, 2),
        "pix_T_cams": torch.eye(4).view(1, 1, 4, 4),
        "cam0_T_camXs": torch.eye(4).view(1, 1, 4, 4),
        "seg_bev_g": occ.float(), "vis_bev_g": vis.float(), "valid_bev_g": valid,
        "d_bev_g": d,
    }


def _scene_with_distance():
    """사각형 장면과 **라벨과 부호가 맞는** 거리장. free 셀은 양수, 나머지는 음수이고, 크기는
    무작위라 세 영역(경계에서 먼 free / 먼 non-free / 대역)이 모두 생긴다."""
    occ, vis, valid = _square_scene()
    free = decompose(occ, vis, valid)["free"]
    torch.manual_seed(0)
    mag = torch.rand(1, 1, SPEC.n_rows, SPEC.n_cols) * 0.8
    d = torch.where(free, mag, -mag)
    return occ, vis, valid, d


def _random_logits():
    torch.manual_seed(1)
    return torch.randn(1, 2, SPEC.n_rows, SPEC.n_cols)


def _run_both(logits, occ, vis, valid, d):
    rays = build_ray_index(SPEC, n_theta=720)
    blind = torch.zeros(1, 1, SPEC.n_rows, SPEC.n_cols)
    ce = run_batch(_BinaryStubModel(logits), _batch(occ, vis, valid, d), vox_util=object(),
                   class_weights=torch.tensor([1.0, 3.0]), device="cpu", rays=rays,
                   permanent_blind=blind)
    soft = run_batch_soft_boundary(_BinaryStubModel(logits), _batch(occ, vis, valid, d),
                                   object(), "cpu", rays, blind,
                                   delta=DECOMPOSITION_DELTA_M, lambda_b=0.5,
                                   target="gaussian", sigma=0.10)
    return ce, soft


def test_both_losses_log_exactly_the_same_loss_tags():
    """**사용자 요청(2026-10-02): 어떤 실험이든 손실 로그가 같아야 한다.** 가중 BCE와 soft-BCE가
    정확히 같은 항 집합을 내야 "A는 경계 손실이 오르고 B는 수렴한다"를 두 곡선으로 말할 수 있다."""
    occ, vis, valid, d = _scene_with_distance()
    (_, ce_parts, _), (_, soft_parts, _) = _run_both(_random_logits(), occ, vis, valid, d)
    public = lambda parts: tuple(k for k in parts if not k.startswith("_"))   # noqa: E731
    assert public(ce_parts) == public(soft_parts) == COMMON_LOSS_PARTS
    # 학습 루프는 **모든** 항(내부 값 포함)에 `.item()`을 부른다 -- 하나라도 float이면 첫 배치에서 죽는다.
    for parts in (ce_parts, soft_parts):
        assert all(torch.is_tensor(v) for v in parts.values()), \
            [k for k, v in parts.items() if not torch.is_tensor(v)]


def test_each_loss_decomposes_back_into_its_own_total():
    """분해가 그 런의 손실을 정확히 쪼갠 것인지. 가중 BCE는 셀 평균이므로 `Σ frac·항`,
    soft-BCE는 `½·F + ½·N + λ_B·B`가 총 손실과 같아야 한다."""
    occ, vis, valid, d = _scene_with_distance()
    (ce_loss, ce, _), (soft_loss, soft, _) = _run_both(_random_logits(), occ, vis, valid, d)

    assert sum(float(ce[f"frac_{r}"]) for r in ("free", "not_free", "boundary")) \
        == pytest.approx(1.0, abs=1e-5)
    recombined = sum(float(ce[f"frac_{r}"]) * float(ce[f"loss_{r}"])
                     for r in ("free", "not_free", "boundary"))
    assert recombined == pytest.approx(float(ce_loss), rel=1e-4)
    assert 0.5 * float(soft["loss_free"]) + 0.5 * float(soft["loss_not_free"]) \
        + 0.5 * float(soft["loss_boundary"]) == pytest.approx(float(soft_loss), rel=1e-4)


def test_the_common_ruler_is_identical_across_losses_and_the_floor_is_explicit():
    """같은 예측이면 손실과 무관하게 `ce_*`·`frac_*`가 같아야 한다(공통 눈금). soft-BCE의
    경계 밖 두 항은 hard CE와 같고, 경계 항은 0이 아닌 하한(엔트로피)을 가진다. 가중 BCE는 하한이 0."""
    occ, vis, valid, d = _scene_with_distance()
    (_, ce, _), (_, soft, _) = _run_both(_random_logits(), occ, vis, valid, d)
    for key in ("ce_free", "ce_not_free", "ce_boundary", "ce_all",
                "frac_free", "frac_not_free", "frac_boundary"):
        assert float(ce[key]) == pytest.approx(float(soft[key]), rel=1e-6), key
    assert float(soft["loss_free"]) == pytest.approx(float(soft["ce_free"]), rel=1e-5)
    assert float(soft["loss_not_free"]) == pytest.approx(float(soft["ce_not_free"]), rel=1e-5)
    assert float(soft["loss_boundary_floor"]) > 0.1
    assert float(ce["loss_boundary_floor"]) == 0.0
    # hard CE 전체는 세 영역의 셀 수 가중평균이다.
    assert sum(float(ce[f"frac_{r}"]) * float(ce[f"ce_{r}"])
               for r in ("free", "not_free", "boundary")) == pytest.approx(float(ce["ce_all"]), rel=1e-4)


def test_run_batch_refuses_a_batch_without_a_distance_field():
    """두 데이터셋 모두 `d_bev_g`를 만든다. 없으면 분해가 빠진 채 조용히 다른 로그가 나오는
    대신 즉시 멈춘다."""
    occ, vis, valid = _square_scene()
    batch = _batch(occ, vis, valid, None)
    del batch["d_bev_g"]
    with pytest.raises(KeyError):
        run_batch(_BinaryStubModel(torch.zeros(1, 2, SPEC.n_rows, SPEC.n_cols)), batch,
                  vox_util=object(), class_weights=torch.ones(2), device="cpu",
                  rays=build_ray_index(SPEC, n_theta=720))


def test_run_batch_returns_loss_parts_and_free_metrics():
    occ, vis, valid = _square_scene()
    gt = decompose(occ, vis, valid)
    logits = torch.full((1, 2, SPEC.n_rows, SPEC.n_cols), -5.0)
    logits[:, NOT_FREE] = 5.0
    logits[:, FREE][gt["free"][:, 0]] = 5.0
    logits[:, NOT_FREE][gt["free"][:, 0]] = -5.0
    model = _BinaryStubModel(logits)
    batch = _batch(occ, vis, valid, torch.zeros_like(occ.float()))

    loss, parts, free_metrics = run_batch(
        model, batch, vox_util=object(), class_weights=torch.ones(2), device="cpu",
        rays=build_ray_index(SPEC, n_theta=720),
    )

    assert float(loss) < 0.01
    assert tuple(k for k in parts if not k.startswith("_")) == COMMON_LOSS_PARTS
    assert free_metrics["iou_free"] == pytest.approx(1.0)
    assert torch.equal(model.seen_rgb, torch.full_like(batch["rgb_camXs"], 0.5))


def test_label_smoothing_caps_the_loss_a_confident_cell_can_produce():
    """val loss가 오르는 이유가 "확신을 갖고 틀리는 것"이므로(§17) 그 상한을 직접 건다.

    같은 예측에 대해 (a) 완전히 맞은 셀은 loss가 0이 아니게 되고(smoothing이 다른 클래스에도
    질량을 주므로), (b) 확신을 갖고 틀린 셀의 loss는 **줄어든다**. 둘 다 확인해야 한다 --
    (a)만 보면 상수를 더한 것과 구별되지 않는다.
    """
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    right = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    wrong = torch.full((1, 1, 2, 2), NOT_FREE, dtype=torch.long)
    logits = _logits_favouring(FREE)

    plain_right, _ = compute_binary_loss(logits, right, valid, torch.ones(2))
    smooth_right, _ = compute_binary_loss(logits, right, valid, torch.ones(2),
                                          label_smoothing=0.1)
    plain_wrong, _ = compute_binary_loss(logits, wrong, valid, torch.ones(2))
    smooth_wrong, _ = compute_binary_loss(logits, wrong, valid, torch.ones(2),
                                          label_smoothing=0.1)

    assert float(smooth_right) > float(plain_right)
    assert float(smooth_wrong) < float(plain_wrong)
    # 기본값은 smoothing 없음 -- 옛 런과 loss 값이 비교 가능해야 한다.
    assert float(compute_binary_loss(logits, right, valid, torch.ones(2))[0]) == pytest.approx(
        float(plain_right)
    )


def test_epoch_aggregation_weights_region_means_by_their_cell_counts():
    """배치마다 영역 셀 수가 다르면 같은 무게로 평균한 값은 "그 영역 모든 셀의 평균"이 아니다.
    두 배치로 나눠 모은 값이 한 배치로 계산한 값과 같아야 하고, 가중 BCE의 `Σ frac·loss_r`이
    epoch 단위로도 셀 평균 손실과 같아야 한다."""
    from projects.common.bev_occupancy_metrics import mean_loss_parts
    occ, vis, valid, d = _scene_with_distance()
    logits = _random_logits()
    rays = build_ray_index(SPEC, n_theta=720)
    weights = torch.tensor([1.0, 3.0])
    # 장면 둘: 원래 장면과, 왼쪽 절반을 invalid로 만든 장면 -- 영역 셀 수가 크게 다르다.
    valid2 = valid.clone()
    valid2[..., : SPEC.n_cols // 2] = False
    batches = [_batch(occ, vis, valid, d), _batch(occ, vis, valid2, d)]
    per_batch = [run_batch(_BinaryStubModel(logits), b, vox_util=object(), class_weights=weights,
                           device="cpu", rays=rays)[1] for b in batches]
    merged = mean_loss_parts([{k: float(v) for k, v in p.items()} for p in per_batch])

    joint = {k: torch.cat([b[k] for b in batches]) if torch.is_tensor(b[k]) else b[k]
             for b in batches for k in b}
    joint.update({k: torch.cat([b[k] for b in batches])
                  for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g", "d_bev_g")})
    joint_loss, whole, _ = run_batch(_BinaryStubModel(logits), joint, vox_util=object(),
                                     class_weights=weights, device="cpu", rays=rays)
    for key in COMMON_LOSS_PARTS:
        assert merged[key] == pytest.approx(float(whole[key]), rel=1e-4), key
    assert "_n_valid" not in merged
    recombined = sum(merged[f"frac_{r}"] * merged[f"loss_{r}"] for r in ("free", "not_free", "boundary"))
    assert recombined == pytest.approx(float(joint_loss), rel=1e-4)


def test_epoch_loss_is_the_objective_on_the_whole_set_for_both_losses():
    """epoch 총 손실 = 그 epoch의 모든 셀을 한 집합으로 본 목적함수. 영역 크기가 다른 두 배치로
    나눠 모아도 한 배치로 계산한 손실과 같아야 하고, 분해(`Σ frac·loss_r` / `½F + ½N + λ_B·B`)가
    epoch 단위에서 정확히 맞아야 한다. 배치 평균이었을 때는 soft-BCE에서 6~7 % 어긋났다."""
    from projects.common.bev_occupancy_metrics import epoch_loss, mean_loss_parts
    occ, vis, valid, d = _scene_with_distance()
    logits = _random_logits()
    rays = build_ray_index(SPEC, n_theta=720)
    blind = torch.zeros(1, 1, SPEC.n_rows, SPEC.n_cols)
    valid2 = valid.clone()
    valid2[..., : SPEC.n_cols // 2] = False
    batches = [_batch(occ, vis, valid, d), _batch(occ, vis, valid2, d)]
    joint = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}

    def soft(batch):
        return run_batch_soft_boundary(_BinaryStubModel(logits), batch, object(), "cpu", rays,
                                       blind, delta=DECOMPOSITION_DELTA_M, lambda_b=0.5,
                                       target="gaussian", sigma=0.10)

    def ce(batch):
        return run_batch(_BinaryStubModel(logits), batch, vox_util=object(),
                         class_weights=torch.tensor([1.0, 3.0]), device="cpu", rays=rays,
                         permanent_blind=blind)

    for step in (ce, soft):
        outs = [step(b) for b in batches]
        merged = mean_loss_parts([{k: float(v) for k, v in o[1].items()} for o in outs])
        value = epoch_loss([(float(o[0]), 1) for o in outs], merged)
        whole_loss, whole_parts, _ = step(joint)
        assert value == pytest.approx(float(whole_loss), rel=1e-4), step.__name__
        for key in COMMON_LOSS_PARTS:
            assert merged[key] == pytest.approx(float(whole_parts[key]), rel=1e-4), key
