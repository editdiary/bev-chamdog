"""Loss, metrics, and batch step for the three-class BEV formulation.

This is the sibling of `bev_occupancy_metrics.py`. Both paths return the same
free-space metric dictionary so logging, checkpoint selection, and A/B
comparisons can stay on one metric contract.
"""
import torch

from projects.common.free_space import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    decompose,
    decompose_from_class_index,
    to_class_index,
)
from projects.common.free_space_metrics import free_metrics_from_masks
from projects.common.segmentation_loss import (
    DEFAULT_MAX_CLASS_WEIGHT,
    inverse_frequency_weights,
    masked_weighted_ce,
)

CLASS_ORDER = (UNKNOWN, FREE, OCCUPIED)
_PART_BY_CLASS = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied"}
LOSS_PART_NAMES = ("unknown", "free", "occupied")

# 역빈도 가중치의 기본 상한. 로봇 train split에서 `occupied`를 67.4 -> 20으로 자른다.
#
# **2026-08-18 실측 결론: 20은 해롭고, 그렇다고 상한을 고르는 문제도 아니다.**
# `occupied`는 셀의 1.1 %인데 val loss의 67 %를 만들고, 가중치를 제거한 raw CE가 7.25 nats
# (정답 클래스 확률 0.07 %)까지 간다 -- 확신을 갖고 틀린다. 반대로 상한 1(가중치 없음)은
# 초기에 occupied 예측이 붕괴한다. 근본 원인은 CE가 **면적** loss인데 `occupied`는 두께 1셀
# **표면**이라 셀 단위 정확도가 본질적으로 달성 불가능하다는 것이다.
#
# 다음 단계는 상한 스윕이 아니라 정식화를 바꾸는 것이다 -- (D) binary로 갔다(§15).
# **정본: `docs/finetune_overfitting_diagnosis.md` §12(실측)·§13(설계)·§15(판정).**
# 호출부는 `max_class_weight=`로 이 값을 덮어쓸 수 있다(`1`이면 가중치 없음).
MAX_CLASS_WEIGHT = DEFAULT_MAX_CLASS_WEIGHT


def compute_three_class_loss(logits, class_index, valid, class_weights):
    """Masked weighted cross-entropy over valid BEV cells only.

    항의 의미(`loss_*` 대 `share_*`)는 `segmentation_loss.masked_weighted_ce`가 정본이다.
    binary 경로와 **같은 함수**를 타야 두 정식화의 loss 곡선을 나란히 읽을 수 있다.
    """
    return masked_weighted_ce(logits, class_index, valid, class_weights, _PART_BY_CLASS)


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

    return inverse_frequency_weights(
        counts, MAX_CLASS_WEIGHT if max_class_weight is None else max_class_weight
    )


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
