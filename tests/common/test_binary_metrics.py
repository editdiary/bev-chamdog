"""(D) binary 정식화의 loss·지표 테스트.

3-class와 **같은 지표 dict**를 돌려주는 것이 이 정식화의 계약이다(`occupied`는 예측하지 않고
예측 free의 경계에서 유도한다). 그래서 마지막 두 테스트가 그 유도가 실제로 일어나는지를
고정한다 -- 유도가 빠지면 `iou_occupied`/`f1@τ`가 조용히 0이 되어 3-class 런과 비교할 수 없다.
"""
import pytest
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.binary_metrics import (
    CLASS_ORDER,
    LOSS_PART_NAMES,
    NOT_FREE,
    class_weights_from_labels,
    compute_binary_loss,
    compute_free_metrics,
    run_batch,
    to_class_index,
)
from projects.common.free_space import FREE, decompose
from projects.common.polar import build_ray_index

_PART_KEYS = {"loss_free", "loss_not_free", "share_free", "share_not_free"}

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
    assert set(parts) == _PART_KEYS


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
    assert LOSS_PART_NAMES == ("not_free", "free")

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
    assert metrics["fatal_rate"] == pytest.approx(0.0)
    # GT occupied(테두리)를 유도로 되돌린다. 두께 1셀이라 면적 IoU도 여기서는 높아야 한다.
    assert metrics["iou_occupied"] > 0.9
    assert metrics["partition_defects"] == 0
    # 유도 결과가 지표 dict에 실려 나가야 val 루프의 `f1@τ`가 계산된다.
    assert metrics["pred_occupied"].any()


class _BinaryStubModel:
    def __init__(self, logits):
        self.logits = logits
        self.seen_rgb = None

    def __call__(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util):
        self.seen_rgb = rgb_camXs
        batch = rgb_camXs.shape[0]
        return None, None, self.logits.expand(batch, -1, -1, -1).contiguous(), None, None


def test_run_batch_returns_loss_parts_and_free_metrics():
    occ, vis, valid = _square_scene()
    gt = decompose(occ, vis, valid)
    logits = torch.full((1, 2, SPEC.n_rows, SPEC.n_cols), -5.0)
    logits[:, NOT_FREE] = 5.0
    logits[:, FREE][gt["free"][:, 0]] = 5.0
    logits[:, NOT_FREE][gt["free"][:, 0]] = -5.0
    model = _BinaryStubModel(logits)
    batch = {
        "rgb_camXs": torch.ones(1, 1, 3, 2, 2),
        "pix_T_cams": torch.eye(4).view(1, 1, 4, 4),
        "cam0_T_camXs": torch.eye(4).view(1, 1, 4, 4),
        "seg_bev_g": occ.float(),
        "vis_bev_g": vis.float(),
        "valid_bev_g": valid,
    }

    loss, parts, free_metrics = run_batch(
        model, batch, vox_util=object(), class_weights=torch.ones(2), device="cpu",
        rays=build_ray_index(SPEC, n_theta=720),
    )

    assert float(loss) < 0.01
    assert set(parts) == _PART_KEYS
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
