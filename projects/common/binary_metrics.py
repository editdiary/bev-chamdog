"""(D) binary 정식화 -- `free` / `not-free` 한 장만 예측한다.

**왜 클래스를 둘로 줄이나.** 3-class에서 `occupied`는 셀의 1.1 %인데 val loss의 67 %를
만들었고, 어떤 클래스 가중치도 그것을 고치지 못했다 -- CE는 면적 loss인데 GT `occupied`는
두께 1셀 표면이라 셀 단위 정확도가 본질적으로 달성 불가능하기 때문이다
(`docs/finetune_overfitting_diagnosis.md` §12-§13).

**무엇을 잃나 -- 실측으로 거의 없다(§15).** 라벨에서 `occupied = ~occ & vis`이고 `vis`가 ego
원점 raycast이므로 GT `occupied`는 free 영역의 ego 기준 경계다. GT free를 광선에 되돌리면
`f1@20cm` 0.984로 복원된다. 그래서 예측에서도 `occupied`를 **유도**해 보고하고
(`occupied_metrics.derive_occupied`), 지표 집합은 3-class와 완전히 같게 유지한다.

`unknown`은 사라지지 않는다 -- `not_free` 안에 들어간다. 즉 모델은 여전히 "어디까지 보이는가"를
배워야 하고, 바뀐 것은 **"왜 못 가는가"를 loss가 더 이상 묻지 않는다**는 것뿐이다.
"""
import torch

from projects.common.free_space import FREE, decompose
from projects.common.free_space_metrics import free_metrics_from_masks
from projects.common.occupied_metrics import derive_occupied
from projects.common.segmentation_loss import (
    DEFAULT_MAX_CLASS_WEIGHT,
    inverse_frequency_weights,
    masked_weighted_ce,
)

# `FREE`는 3-class와 같은 인덱스 1을 쓴다 -- 두 정식화의 logits를 같은 시각화·재채점 코드가
# 다룰 때 "1번 채널이 free"라는 규약이 갈리지 않도록 한다.
NOT_FREE = 0
CLASS_ORDER = (NOT_FREE, FREE)
_PART_BY_CLASS = {NOT_FREE: "not_free", FREE: "free"}
LOSS_PART_NAMES = ("not_free", "free")

# 3-class와 같은 기본 상한을 쓰지만 **실제로는 걸리지 않는다** -- 로봇 train split에서
# 순수 역빈도가 free 3.94 / not_free 1.00이다. 상한이 loss 균형을 결정하던 3-class의
# 상황(occupied 67.9 -> 20)이 정식화 자체로 사라진 것이 (D)의 요점 중 하나다.
MAX_CLASS_WEIGHT = DEFAULT_MAX_CLASS_WEIGHT


def to_class_index(parts: dict) -> torch.Tensor:
    """분해 -> `(B, 1, H, W)` long. `free`가 1, 나머지 전부가 0이다.

    `valid` 밖은 `NOT_FREE`로 떨어지지만 loss에서 마스킹되므로 값은 의미가 없다
    (3-class의 `free_space.to_class_index`가 `UNKNOWN`으로 떨어뜨리는 것과 같은 이유).
    """
    return parts["free"].long()


def compute_binary_loss(logits, class_index, valid, class_weights):
    """`valid` 셀에 한정한 가중 2-class cross-entropy.

    구현은 3-class와 **같은 함수**(`segmentation_loss.masked_weighted_ce`)다. 정식화만 바꾸고
    loss의 정규화·집계는 그대로 두어야 두 런의 loss 곡선을 나란히 읽을 수 있다.
    """
    return masked_weighted_ce(logits, class_index, valid, class_weights, _PART_BY_CLASS)


def class_weights_from_labels(label_triples, max_class_weight=None) -> torch.Tensor:
    """`(occ, vis, valid)` 묶음에서 역빈도 가중치 `[not_free, free]`.

    `not_free`는 `occupied ∪ unknown`이다. 분해는 3-class와 같은
    `free_space.decompose` 하나만 쓴다 -- 여기서 마스크를 다시 조합하면 loss가 보는 클래스
    정의와 가중치가 세는 정의가 갈라진다(예전에 `pos_weight`가 마스크를 무시해 클래스 보정이
    통째로 어긋난 적이 있다).
    """
    counts = torch.zeros(2, dtype=torch.float64)
    for occ, vis, valid in label_triples:
        parts = decompose(occ, vis, valid)
        free_count = float(parts["free"].sum())
        counts[FREE] += free_count
        counts[NOT_FREE] += float(parts["occupied"].sum()) + float(parts["unknown"].sum())
    return inverse_frequency_weights(
        counts, MAX_CLASS_WEIGHT if max_class_weight is None else max_class_weight
    )


def compute_free_metrics(logits, seg_g, vis_g, valid_g, rays) -> dict:
    """binary logits -> 3-class와 **같은 지표 dict**.

    `occupied`는 예측하지 않고 예측 free의 경계에서 유도하며, `unknown`은 나머지다. 이렇게
    분할을 복원해 두면 `free_metrics_from_masks`·`f1@τ`·M3·ring이 전부 그대로 돌아가고,
    3-class 런과 지표를 한 표에 놓을 수 있다. 유도 비용은 bs8에서 25 ms이므로 train에서도
    매 배치 계산한다 -- val에서만 계산하면 train/val 곡선이 다른 것을 재게 된다.
    """
    gt = decompose(seg_g, vis_g, valid_g)
    valid_b = valid_g.bool()
    pred_free = (logits.argmax(dim=1, keepdim=True) == FREE) & valid_b
    pred_occupied = derive_occupied(pred_free, valid_g, rays) & valid_b
    pred = {
        "free": pred_free,
        "occupied": pred_occupied,
        "unknown": valid_b & ~pred_free & ~pred_occupied,
    }
    return free_metrics_from_masks(pred, gt, valid_g)


def run_batch(model, batch, vox_util, class_weights, device, rays):
    """Run one binary train/eval batch."""
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    _, _, logits, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
    class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
    loss, loss_parts = compute_binary_loss(logits, class_index, valid_bev_g, class_weights)
    return loss, loss_parts, compute_free_metrics(
        logits, seg_bev_g, vis_bev_g, valid_bev_g, rays
    )
