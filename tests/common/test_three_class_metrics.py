import pytest
import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.common.three_class_metrics import (
    CLASS_ORDER,
    MAX_CLASS_WEIGHT,
    class_weights_from_labels,
    compute_free_metrics,
    compute_three_class_loss,
    run_batch,
)


def _logits_favouring(class_id, shape=(1, 3, 2, 2)):
    logits = torch.full(shape, -5.0)
    logits[:, class_id] = 5.0
    return logits


def test_loss_is_near_zero_when_the_prediction_is_confidently_right():
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)

    loss, parts = compute_three_class_loss(
        _logits_favouring(FREE), target, valid, torch.ones(3)
    )

    assert float(loss) < 0.01
    assert set(parts) == {"loss_free", "loss_occupied", "loss_unknown",
                          "share_free", "share_occupied", "share_unknown"}


def test_invalid_cells_contribute_no_gradient():
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    logits = _logits_favouring(OCCUPIED).requires_grad_()

    loss, _ = compute_three_class_loss(logits, target, valid, torch.ones(3))
    loss.backward()

    assert float(loss) == pytest.approx(0.0)
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_class_weights_scale_the_occupied_term():
    target = torch.full((1, 1, 2, 2), OCCUPIED, dtype=torch.long)
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    logits = _logits_favouring(FREE)

    light, _ = compute_three_class_loss(logits, target, valid, torch.ones(3))
    heavy, _ = compute_three_class_loss(
        logits, target, valid, torch.tensor([1.0, 1.0, 10.0])
    )

    assert float(heavy) > float(light) * 5


def test_class_weights_use_unknown_free_occupied_order():
    assert CLASS_ORDER == (UNKNOWN, FREE, OCCUPIED)

    occ = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    vis = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
    valid = torch.ones_like(occ)

    weights = class_weights_from_labels([(occ, vis, valid)])

    assert weights.tolist() == pytest.approx([2.0, 1.0, 2.0])


def test_class_weights_ignore_invalid_cells():
    occ = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)
    vis = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.bool)
    valid = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)

    weights = class_weights_from_labels([(occ, vis, valid)])

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_class_weights_are_clipped():
    occ = torch.ones((1, 1, 1, 102), dtype=torch.bool)
    vis = torch.ones_like(occ)
    valid = torch.ones_like(occ)
    occ[..., 100] = 0
    vis[..., 101] = 0

    weights = class_weights_from_labels([(occ, vis, valid)])

    assert weights.tolist() == pytest.approx([20.0, 1.0, 20.0])


def test_class_weights_accumulate_across_samples():
    """여러 샘플이 하나의 분포로 합산돼야 한다 -- 마지막 샘플만 세면 split 전체의 불균형을
    반영하지 못한다. free만 있는 샘플과 occupied만 있는 샘플을 합치면 균형이 된다."""
    free_only = (
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
    )
    occupied_only = (
        torch.zeros((1, 1, 1, 4), dtype=torch.bool),
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
    )

    weights = class_weights_from_labels([free_only, occupied_only])

    # unknown 0개 -> clamp(min=1)에 걸려 8/1 = 8, free/occupied는 각각 8/4 = 2.
    assert weights.tolist() == pytest.approx([4.0, 1.0, 1.0])
    # 한 샘플만 셌다면 free 또는 occupied 한쪽이 0이 되어 20.0 캡에 걸린다.
    assert max(weights.tolist()) < 20.0


def test_compute_free_metrics_uses_the_shared_free_space_aggregator():
    seg_g = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    vis_g = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    valid_g = torch.ones_like(seg_g)
    logits = _logits_favouring(FREE)
    logits[:, :, 0, 1] = -5.0
    logits[:, OCCUPIED, 0, 1] = 5.0
    logits[:, :, 1, 0] = -5.0
    logits[:, UNKNOWN, 1, 0] = 5.0

    metrics = compute_free_metrics(logits, seg_g, vis_g, valid_g)

    assert metrics["iou_free"] == pytest.approx(1.0)
    assert metrics["fatal_rate"] == pytest.approx(0.0)
    assert metrics["partition_defects"] == 0
    assert torch.equal(metrics["pred_free"], metrics["gt_free"])


class _ThreeClassStubModel:
    def __init__(self, logits):
        self.logits = logits
        self.seen_rgb = None

    def __call__(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util):
        self.seen_rgb = rgb_camXs
        batch = rgb_camXs.shape[0]
        logits = self.logits.expand(batch, -1, -1, -1).contiguous()
        return None, None, logits, None, None


def test_run_batch_returns_loss_parts_and_free_metrics():
    logits = _logits_favouring(FREE, shape=(1, 3, 1, 2))
    logits[:, :, 0, 1] = -5.0
    logits[:, UNKNOWN, 0, 1] = 5.0
    model = _ThreeClassStubModel(logits)
    batch = {
        "rgb_camXs": torch.ones(1, 1, 3, 2, 2),
        "pix_T_cams": torch.eye(4).view(1, 1, 4, 4),
        "cam0_T_camXs": torch.eye(4).view(1, 1, 4, 4),
        "seg_bev_g": torch.tensor([[[[1.0, 0.0]]]]),
        "vis_bev_g": torch.tensor([[[[1.0, 0.0]]]]),
        "valid_bev_g": torch.ones(1, 1, 1, 2, dtype=torch.bool),
    }

    loss, parts, free_metrics = run_batch(
        model, batch, vox_util=object(), class_weights=torch.ones(3), device="cpu"
    )

    assert float(loss) < 0.01
    assert set(parts) == {"loss_free", "loss_occupied", "loss_unknown",
                          "share_free", "share_occupied", "share_unknown"}
    assert free_metrics["iou_free"] == pytest.approx(1.0)
    assert torch.equal(model.seen_rgb, torch.full_like(batch["rgb_camXs"], 0.5))


def test_loss_shares_sum_to_one_and_expose_which_class_dominates():
    """`share_*`는 각 클래스가 **총 loss에 실제로 기여하는 몫**이다.

    클래스별 평균(`loss_*`)만으로는 병리가 안 보인다 -- 실측에서 val `loss_occupied` 145가
    "occupied가 셀의 1.1 %인데 총 loss의 67 %"라는 뜻이라는 것을 셀 비율을 손으로 곱해야
    알 수 있었다(`docs/finetune_overfitting_diagnosis.md` §12).

    셀 4개 중 occupied 1개만 틀리게 만들고 그 클래스에 가중치 10을 준다. 나머지는 정답이라
    loss가 거의 0이므로, occupied의 몫이 1에 가까워야 한다.
    """
    logits = torch.zeros(1, 3, 1, 4)
    class_index = torch.tensor([[[[UNKNOWN, FREE, FREE, OCCUPIED]]]])
    for i, correct in enumerate((UNKNOWN, FREE, FREE)):
        logits[0, correct, 0, i] = 20.0
    logits[0, FREE, 0, 3] = 20.0  # occupied 셀만 자신 있게 틀린다
    valid = torch.ones(1, 1, 1, 4)
    weights = torch.tensor([1.0, 1.0, 10.0])

    _, parts = compute_three_class_loss(logits, class_index, valid, weights)

    shares = [float(parts[f"share_{name}"]) for name in ("unknown", "free", "occupied")]
    assert sum(shares) == pytest.approx(1.0, abs=1e-4)
    assert shares[2] > 0.99, shares
    # 몫과 평균은 다른 것을 잰다 -- 평균이 큰 클래스가 셀이 적으면 몫은 작을 수 있다.
    assert float(parts["loss_occupied"]) > float(parts["loss_free"])


def test_the_weight_cap_is_configurable_and_one_means_no_weighting():
    """상한을 스윕하려면 인자로 받아야 한다. `1`은 "가중치를 쓰지 않는다"와 같아야 한다 --
    최소값이 1로 정규화되므로 전부 1로 눌린다.
    """
    # 8칸 중 free 4 / unknown 3 / occupied 1 -> 역빈도 원값이 occupied 4.0으로 상한보다 크다.
    # (free 2 / unknown 1 / occupied 1이면 원값이 정확히 2.0이 되어 상한 2와 구별되지 않는다.)
    occ = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]]])
    vis = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]])
    valid = torch.ones_like(occ)

    uncapped = class_weights_from_labels([(occ, vis, valid)], max_class_weight=1e9)
    capped = class_weights_from_labels([(occ, vis, valid)], max_class_weight=2.0)
    unweighted = class_weights_from_labels([(occ, vis, valid)], max_class_weight=1.0)

    assert uncapped[OCCUPIED] > 2.0
    assert capped[OCCUPIED] == pytest.approx(2.0)
    assert unweighted.tolist() == [1.0, 1.0, 1.0]
    # 기본값은 모듈 상수를 따른다 -- 호출부가 상한을 안 넘겨도 동작이 바뀌지 않아야 한다.
    assert torch.equal(
        class_weights_from_labels([(occ, vis, valid)]),
        class_weights_from_labels([(occ, vis, valid)], max_class_weight=MAX_CLASS_WEIGHT),
    )
