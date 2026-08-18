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

# 역빈도 가중치의 상한. **근거가 없는 임의값이고, 실제로 작동 중이다** -- 로봇 train split에서
# `occupied`를 67.93 -> 20으로 자른다(4샘플 overfit split에서는 ~112 -> 20). 즉 loss 균형을
# 실질적으로 결정하는 하이퍼파라미터인데 한 번도 실측 검증되지 않았고, 하필 `fatal_rate`
# (장애물을 free로 오인 -- planner 안전상 가장 비싼 오류)와 직결된다.
#
# 본학습으로 기준 숫자를 확보한 뒤 이 값을 스윕하거나 다른 완화 방식(median-frequency,
# √/log 역빈도) 또는 다른 loss(Focal, Lovasz-Softmax, Dice)로 바꾸는 것을 검토한다.
# 배경과 실측값은 `docs/BEV_loss_and_metrics_design.md` §1.7이 정본이다.
MAX_CLASS_WEIGHT = 20.0


def compute_three_class_loss(logits, class_index, valid, class_weights):
    """Masked weighted cross-entropy over valid BEV cells only.

    `loss_*`는 그 클래스 셀의 **평균**이고 `share_*`는 그 클래스가 **총 loss에 실제로 기여하는
    몫**이다. 둘을 같이 내놓는 이유: 평균만 보면 병리가 안 보인다. 2026-08-18 실측에서 val
    `loss_occupied`가 145였는데, 그것만으로는 occupied가 셀의 1.1 %인데 **총 loss의 67 %**를
    차지한다는 사실이 드러나지 않았다 -- 셀 비율을 손으로 곱해 봐야 알 수 있었다.
    `share_*`는 정의상 합이 1이므로 어느 클래스가 학습을 지배하는지 바로 읽힌다.
    """
    valid_f = valid.float()
    per_cell = F.cross_entropy(
        logits,
        class_index.squeeze(1),
        weight=class_weights.to(logits.device),
        reduction="none",
    ).unsqueeze(1)
    weighted_sum = (per_cell * valid_f).sum()
    total = weighted_sum / (valid_f.sum() + 1e-6)

    parts = {}
    valid_b = valid.bool()
    for class_id in CLASS_ORDER:
        name = _PART_BY_CLASS[class_id]
        selector = ((class_index == class_id) & valid_b).float()
        class_sum = (per_cell * selector).sum()
        parts[f"loss_{name}"] = class_sum / (selector.sum() + 1e-6)
        parts[f"share_{name}"] = class_sum / (weighted_sum + 1e-6)
    return total, parts


def class_weights_from_labels(label_triples, max_class_weight=None) -> torch.Tensor:
    """Inverse-frequency class weights from an iterable of ``(occ, vis, valid)`` triples.

    데이터셋을 모른다 -- 호출자가 자기 방식으로 라벨을 읽어 세 마스크만 넘긴다. 로봇 쪽은
    `permanent_blind`/`rear_self_box`가 이미 반영된 마스크를 넘기고, SynWoodScape 쪽은
    `valid`가 전부 1인 마스크를 넘긴다. 이렇게 두는 이유는 두 데이터셋의 "관측 불가" 개념이
    서로 다르고(리그 고정 마스크 대 없음), 가중치 계산은 그 차이를 알 필요가 없기 때문이다.
    """
    counts = torch.zeros(3, dtype=torch.float64)
    for occ, vis, valid in label_triples:
        # 분해는 `free_space.decompose` 하나만 쓴다 -- 여기서 조합을 다시 쓰면 loss가 보는
        # 클래스 정의와 가중치가 세는 정의가 갈라질 수 있다. 예전에 `pos_weight`가 마스크를
        # 무시하고 원본 라벨을 세는 바람에 클래스 보정이 통째로 어긋난 적이 있다.
        parts = decompose(occ, vis, valid)
        for class_id in CLASS_ORDER:
            counts[class_id] += float(parts[_PART_BY_CLASS[class_id]].sum())

    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.min().clamp(min=1e-12)
    cap = MAX_CLASS_WEIGHT if max_class_weight is None else float(max_class_weight)
    # `cap=1`은 "가중치를 아예 쓰지 않는다"가 된다 -- 최소값이 1로 정규화돼 있으므로 전부 1로
    # 눌린다. 한 번도 돌려 본 적이 없는 기준선이고, 스윕의 한쪽 끝이다.
    return weights.clamp(max=cap).float()


def compute_free_metrics(logits, seg_g, vis_g, valid_g) -> dict:
    """Convert 3-class logits to free-space metrics with the shared aggregator."""
    gt = decompose(seg_g, vis_g, valid_g)
    pred = decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid_g)
    return free_metrics_from_masks(pred, gt, valid_g)


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
