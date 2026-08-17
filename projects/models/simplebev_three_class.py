"""Three-class Simple-BEV wrapper for free/occupied/unknown logits.

The upstream Simple-BEV submodule stays untouched. This module reuses its
encoder, lifting, BEV compressor, and decoder trunk, then replaces the final
segmentation projection with one 3-channel softmax-ready head.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

from nets.segnet import Decoder, Segnet  # noqa: E402

NUM_CLASSES = 3


def build_three_class_head(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, NUM_CLASSES, kernel_size=1, padding=0),
    )


class ThreeClassDecoder(Decoder):
    """Simple-BEV decoder trunk with a 3-class segmentation projection."""

    def __init__(self, in_channels: int, predict_future_flow: bool = False):
        super().__init__(in_channels=in_channels, n_classes=1, predict_future_flow=predict_future_flow)
        self.segmentation_head = build_three_class_head(in_channels)

    def forward(self, x, bev_flip_indices=None):
        out = super().forward(x, bev_flip_indices=bev_flip_indices)
        out["three_class"] = out["segmentation"]
        return out


class ThreeClassSegnet(Segnet):
    """`Segnet` variant whose segmentation output has free/occupied/unknown logits."""

    def __init__(self, *args, **kwargs):
        latent_dim = kwargs.get("latent_dim", 128)
        super().__init__(*args, **kwargs)
        self.decoder = ThreeClassDecoder(in_channels=latent_dim, predict_future_flow=False)


def load_trunk_weights(model, checkpoint_path, device) -> dict:
    """Load shape-compatible trunk weights from a two-head checkpoint.

    Shape-compatible keys are copied explicitly so skipped projection weights
    are visible in the returned report instead of being hidden by
    `strict=False`.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()

    transfer, skipped = {}, []
    for name, tensor in source.items():
        if name in target and target[name].shape == tensor.shape:
            transfer[name] = tensor
        else:
            skipped.append(name)

    target.update(transfer)
    model.load_state_dict(target)
    model.to(device)
    return {"loaded": len(transfer), "skipped": skipped}
