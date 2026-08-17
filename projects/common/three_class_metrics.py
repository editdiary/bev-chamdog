"""Loss, metrics, and batch step for the three-class BEV formulation.

This is the sibling of `bev_occupancy_metrics.py`. Both paths return the same
free-space metric dictionary so logging, checkpoint selection, and A/B
comparisons can stay on one metric contract.
"""
import torch
import torch.nn.functional as F

from projects.common.free_space import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    decompose,
    decompose_from_class_index,
    to_class_index,
)
from projects.common.free_space_metrics import free_metrics_from_masks

CLASS_ORDER = (UNKNOWN, FREE, OCCUPIED)
_PART_BY_CLASS = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied"}
MAX_CLASS_WEIGHT = 20.0


def compute_three_class_loss(logits, class_index, valid, class_weights):
    """Masked weighted cross-entropy over valid BEV cells only."""
    valid_f = valid.float()
    per_cell = F.cross_entropy(
        logits,
        class_index.squeeze(1),
        weight=class_weights.to(logits.device),
        reduction="none",
    ).unsqueeze(1)
    total = (per_cell * valid_f).sum() / (valid_f.sum() + 1e-6)

    parts = {}
    valid_b = valid.bool()
    for class_id in CLASS_ORDER:
        selector = ((class_index == class_id) & valid_b).float()
        parts[f"loss_{_PART_BY_CLASS[class_id]}"] = (
            (per_cell * selector).sum() / (selector.sum() + 1e-6)
        )
    return total, parts


def class_weights_from_labels(label_triples) -> torch.Tensor:
    """Inverse-frequency class weights from an iterable of ``(occ, vis, valid)`` triples.

    데이터셋을 모른다 -- 호출자가 자기 방식으로 라벨을 읽어 세 마스크만 넘긴다. 로봇 쪽은
    `permanent_blind`/`rear_self_box`가 이미 반영된 마스크를 넘기고, SynWoodScape 쪽은
    `valid`가 전부 1인 마스크를 넘긴다. 이렇게 두는 이유는 두 데이터셋의 "관측 불가" 개념이
    서로 다르고(리그 고정 마스크 대 없음), 가중치 계산은 그 차이를 알 필요가 없기 때문이다.
    """
    counts = torch.zeros(3, dtype=torch.float64)
    for occ, vis, valid in label_triples:
        occ = torch.as_tensor(occ).bool()
        vis = torch.as_tensor(vis).bool()
        valid = torch.as_tensor(valid).bool()

        counts[UNKNOWN] += float(((~vis) & valid).sum())
        counts[FREE] += float((occ & vis & valid).sum())
        counts[OCCUPIED] += float(((~occ) & vis & valid).sum())

    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.min().clamp(min=1e-12)
    return weights.clamp(max=MAX_CLASS_WEIGHT).float()


def compute_free_metrics(logits, seg_g, vis_g, valid_g) -> dict:
    """Convert 3-class logits to free-space metrics with the shared aggregator."""
    gt = decompose(seg_g, vis_g, valid_g)
    pred = decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid_g)
    return free_metrics_from_masks(pred["free"], gt, valid_g)


def run_batch(model, batch, vox_util, class_weights, device):
    """Run one three-class train/eval batch."""
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    _, _, logits, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
    class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
    loss, loss_parts = compute_three_class_loss(logits, class_index, valid_bev_g, class_weights)
    return loss, loss_parts, compute_free_metrics(logits, seg_bev_g, vis_bev_g, valid_bev_g)
