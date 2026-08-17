import pytest
import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.common.three_class_metrics import (
    CLASS_ORDER,
    compute_free_metrics,
    compute_three_class_loss,
    default_class_weights,
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
    assert set(parts) == {"loss_free", "loss_occupied", "loss_unknown"}


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


def test_default_class_weights_use_unknown_free_occupied_order():
    assert CLASS_ORDER == (UNKNOWN, FREE, OCCUPIED)

    def load_labels(_sequence_root, _sample_id, _permanent_blind, _invalid):
        occ = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
        vis = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
        valid = torch.ones_like(occ)
        return occ, vis, valid

    weights = default_class_weights(
        [(object(), "sample")],
        permanent_blind=torch.zeros(1, dtype=torch.bool),
        invalid=torch.zeros(1, dtype=torch.bool),
        load_labels=load_labels,
    )

    assert weights.tolist() == pytest.approx([2.0, 1.0, 2.0])


def test_default_class_weights_ignore_invalid_cells():
    def load_labels(_sequence_root, _sample_id, _permanent_blind, _invalid):
        occ = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)
        vis = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.bool)
        valid = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
        return occ, vis, valid

    weights = default_class_weights(
        [(object(), "sample")],
        permanent_blind=torch.zeros(1, dtype=torch.bool),
        invalid=torch.zeros(1, dtype=torch.bool),
        load_labels=load_labels,
    )

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_default_class_weights_are_clipped():
    def load_labels(_sequence_root, _sample_id, _permanent_blind, _invalid):
        occ = torch.ones((1, 1, 1, 102), dtype=torch.bool)
        vis = torch.ones_like(occ)
        valid = torch.ones_like(occ)
        occ[..., 100] = 0
        vis[..., 101] = 0
        return occ, vis, valid

    weights = default_class_weights(
        [(object(), "sample")],
        permanent_blind=torch.zeros(1, dtype=torch.bool),
        invalid=torch.zeros(1, dtype=torch.bool),
        load_labels=load_labels,
    )

    assert weights.tolist() == pytest.approx([20.0, 1.0, 20.0])


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
    assert set(parts) == {"loss_free", "loss_occupied", "loss_unknown"}
    assert free_metrics["iou_free"] == pytest.approx(1.0)
    assert torch.equal(model.seen_rgb, torch.full_like(batch["rgb_camXs"], 0.5))
