"""Two-head Simple-BEV wrapper for occupancy and visibility pretraining.

The upstream Simple-BEV submodule stays untouched. This module reuses its
encoder, lifting, BEV compressor, and decoder trunk, then replaces the final
segmentation projection with two independent binary heads.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

from nets.segnet import Decoder, Segnet  # noqa: E402


class TwoHeadSegmentationHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, padding=0),
        )
        self.visibility_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return torch.cat([self.occupancy_head(x), self.visibility_head(x)], dim=1)


class TwoHeadDecoder(Decoder):
    """Simple-BEV decoder trunk with independent occupancy and visibility heads."""

    def __init__(self, in_channels: int, predict_future_flow: bool = False):
        super().__init__(in_channels=in_channels, n_classes=1, predict_future_flow=predict_future_flow)
        self.segmentation_head = TwoHeadSegmentationHead(in_channels)

    def forward(self, x, bev_flip_indices=None):
        out = super().forward(x, bev_flip_indices=bev_flip_indices)
        occ_logit, vis_logit = split_two_head_logits(out["segmentation"])
        out["occupancy"] = occ_logit
        out["visibility"] = vis_logit
        return out


class TwoHeadSegnet(Segnet):
    """`Segnet` variant whose segmentation output has two logits: occ, visibility."""

    def __init__(self, *args, **kwargs):
        latent_dim = kwargs.get("latent_dim", 128)
        super().__init__(*args, **kwargs)
        self.decoder = TwoHeadDecoder(in_channels=latent_dim, predict_future_flow=False)


def split_two_head_logits(two_head_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if two_head_logits.ndim != 4 or two_head_logits.shape[1] != 2:
        raise ValueError(f"expected logits with shape (B, 2, H, W), got {tuple(two_head_logits.shape)}")
    return two_head_logits[:, 0:1], two_head_logits[:, 1:2]


def compute_two_head_loss(
    occ_logit: torch.Tensor,
    vis_logit: torch.Tensor,
    occ_gt: torch.Tensor,
    vis_gt: torch.Tensor,
    valid_gt: torch.Tensor,
    *,
    lambda_vis: float = 0.5,
    vis_neg_weight: float = 3.0,
    occ_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = valid_gt.float()

    w_occ = vis_gt.float() * valid
    occ_target = occ_gt.clamp(0, 1).float()
    occ_loss = F.binary_cross_entropy_with_logits(
        occ_logit,
        occ_target,
        pos_weight=occ_pos_weight,
        reduction="none",
    )
    occ_loss = (occ_loss * w_occ).sum() / (w_occ.sum() + 1e-6)

    vis_target = vis_gt.float()
    asymmetric_weight = torch.where(
        vis_target > 0.5,
        torch.ones_like(vis_target),
        torch.full_like(vis_target, float(vis_neg_weight)),
    )
    w_vis = asymmetric_weight * valid
    vis_loss = F.binary_cross_entropy_with_logits(vis_logit, vis_target, reduction="none")
    vis_loss = (vis_loss * w_vis).sum() / (w_vis.sum() + 1e-6)

    total = occ_loss + float(lambda_vis) * vis_loss
    return total, {"loss_occ": occ_loss, "loss_vis": vis_loss}


def visibility_error_rates(
    vis_prob: torch.Tensor,
    vis_gt: torch.Tensor,
    valid_gt: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    valid = valid_gt.bool()
    pred_visible = vis_prob > threshold
    gt_visible = vis_gt > threshold

    false_high_den = ((~gt_visible) & valid).sum().clamp(min=1)
    false_low_den = (gt_visible & valid).sum().clamp(min=1)
    false_high = (pred_visible & (~gt_visible) & valid).sum().float() / false_high_den
    false_low = ((~pred_visible) & gt_visible & valid).sum().float() / false_low_den
    return {"false_high": float(false_high.item()), "false_low": float(false_low.item())}
