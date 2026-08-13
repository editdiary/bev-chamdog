import torch

from projects.models.simplebev_two_head import (
    TwoHeadDecoder,
    compute_two_head_loss,
    visibility_error_rates,
)


def test_two_head_decoder_returns_separate_occupancy_and_visibility_logits():
    decoder = TwoHeadDecoder(in_channels=8)
    x = torch.randn(2, 8, 16, 16)

    out = decoder(x)

    assert out["occupancy"].shape == (2, 1, 16, 16)
    assert out["visibility"].shape == (2, 1, 16, 16)
    assert out["occupancy"] is not out["visibility"]


def test_two_head_loss_masks_occupancy_with_visibility_and_trains_visibility_everywhere_valid():
    occ_logit = torch.zeros(1, 1, 1, 3, requires_grad=True)
    vis_logit = torch.zeros(1, 1, 1, 3, requires_grad=True)
    occ_gt = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    vis_gt = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    valid_gt = torch.ones_like(occ_gt)

    total, parts = compute_two_head_loss(
        occ_logit,
        vis_logit,
        occ_gt,
        vis_gt,
        valid_gt,
        lambda_vis=0.5,
        vis_neg_weight=3.0,
    )
    total.backward()

    assert parts["loss_occ"] > 0
    assert parts["loss_vis"] > 0
    assert occ_logit.grad[0, 0, 0, 1].item() == 0.0
    assert vis_logit.grad[0, 0, 0, 1].item() > 0.0


def test_visibility_error_rates_are_asymmetric():
    vis_prob = torch.tensor([[[[0.8, 0.7, 0.2, 0.1]]]])
    vis_gt = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    valid_gt = torch.ones_like(vis_gt)

    metrics = visibility_error_rates(vis_prob, vis_gt, valid_gt)

    assert metrics["false_high"] == 0.5
    assert metrics["false_low"] == 0.5
