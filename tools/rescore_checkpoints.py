"""기존 체크포인트를 새 free-space 지표로 재채점한다 (스펙 Phase 1).

재학습하지 않고 지표만 바꿔 다시 재는 것이 요점이다. 새 지표가 옛 지표를 포함·확장하는지,
그리고 트리비얼 baseline과의 순서가 맞는지를 여기서 확인한 뒤에야 학습 스크립트를 건드린다.

실행:
    CUDA_VISIBLE_DEVICES=0 python tools/rescore_checkpoints.py \\
        --checkpoint=runs/robot_bev/ckpt/<run>/model_best-000000046.pth \\
        --train_sequences=raws1,raws2,raws3,rawos1,rawos4 --val_sequences=rawos3
"""
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.baselines import all_free_map, as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    build_ring_masks,
    fatal_rate,
    free_miss_rate,
    iou_free,
    metrics_per_ring,
    range_error,
    summarize_range_error,
    weighted_mean,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.bev_occupancy_metrics import compute_drivable_and_obstacle_iou  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
    parse_sequence_names,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_two_head import TwoHeadSegnet, split_two_head_logits  # noqa: E402

_COLUMNS = (
    ("name", "체크포인트"), ("split", "val"),
    ("iou_free", "iou_free↑"), ("baseline_iou_free", "baseline iou_free"),
    ("all_free_iou_free", "all-free iou_free"),
    ("fatal_rate", "fatal↓"), ("baseline_fatal_rate", "baseline fatal"),
    ("iou_drivable", "(참고) iou_drivable"), ("iou_obstacle", "(참고) iou_obstacle"),
)


def format_markdown_table(rows) -> str:
    """baseline을 모델 값 바로 옆에 붙인다 -- 숫자를 혼자 읽게 두는 것이 이번 결함의 원인이었다."""
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    rule = "|" + "---|" * len(_COLUMNS)
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _ in _COLUMNS:
            value = row.get(key, float("nan"))
            cells.append(value if isinstance(value, str) else f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_checkpoint_state_dict(model, checkpoint_path, device) -> None:
    """체크포인트를 `strict=False`로 얹되, 키가 하나라도 안 맞으면 조용히 넘어가지 않는다.

    `tools/train_robot_bev.py`의 `load_initial_weights`와 같은 계약이다. 이 계약이 없으면
    `--checkpoint` 경로가 틀렸거나(예: 다른 run의 체크포인트) `encoder_type`이 학습 때와
    달라 아키텍처가 어긋나도 `strict=False`가 안 맞는 키를 조용히 버리고 나머지만 얹어
    실행이 끝까지 간다 -- 그 결과는 일부가 무작위 초기화인 채로 나온 그럴듯한 숫자라
    Phase 1 게이트가 통과했다고 착각하게 만든다. 여기서 즉시 실패해야 그 실수를 잡는다.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(state.get("model_state_dict", state), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"체크포인트가 모델과 맞지 않는다: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )


def _collect_free_masks(samples, permanent_blind, invalid):
    masks = []
    for sequence_root, sample_id in samples:
        occ, vis, valid = load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
        masks.append(occ & vis & valid)
    return masks


def score_split(model, loader, vox_util, rays, ring_masks, device, constant_map) -> dict:
    ious, iou_counts, fatals, fatal_denoms, misses, miss_denoms = [], [], [], [], [], []
    base_ious, base_counts, base_fatals, base_denoms = [], [], [], []
    allfree_ious, allfree_counts = [], []
    d_ious, o_ious, o_counts, range_dicts, ring_dicts = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb_camXs"].to(device) - 0.5
            _, _, two_head, _, _ = model(
                rgb, batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device), vox_util
            )
            occ_logits, vis_logits = split_two_head_logits(two_head)
            seg_g = batch["seg_bev_g"].to(device)
            vis_g = batch["vis_bev_g"].to(device)
            valid = batch["valid_bev_g"].to(device)

            gt = decompose(seg_g, vis_g, valid)["free"]
            pred = decompose(torch.sigmoid(occ_logits), torch.sigmoid(vis_logits), valid)["free"]
            batch_size = gt.shape[0]
            base = as_batch(constant_map, batch_size, device)
            allfree = as_batch(all_free_map(constant_map.shape), batch_size, device)

            for values, counts, fn, arg in (
                (ious, iou_counts, iou_free, pred),
                (base_ious, base_counts, iou_free, base),
                (allfree_ious, allfree_counts, iou_free, allfree),
            ):
                value, count = fn(arg, gt, valid)
                values.append(value)
                counts.append(count)
            for values, denoms, fn, arg in (
                (fatals, fatal_denoms, fatal_rate, pred),
                (base_fatals, base_denoms, fatal_rate, base),
                (misses, miss_denoms, free_miss_rate, pred),
            ):
                value, denom = fn(arg, gt, valid)
                values.append(value)
                denoms.append(denom)

            d_iou, o_iou, o_count = compute_drivable_and_obstacle_iou(
                occ_logits, seg_g, vis_g * valid
            )
            d_ious.append(float(d_iou.item()))
            o_ious.append(float(o_iou.item()))
            o_counts.append(o_count)
            range_dicts.append(range_error(pred, gt, valid, rays))
            ring_dicts.append(metrics_per_ring(pred, gt, valid, ring_masks))

    rings = {}
    for name, _ in ring_masks:
        rings[name] = {
            "iou_free": weighted_mean([d[name]["iou_free"] for d in ring_dicts],
                                      [d[name]["iou_free_count"] for d in ring_dicts]),
            "fatal_rate": weighted_mean([d[name]["fatal_rate"] for d in ring_dicts],
                                        [d[name]["fatal_denom"] for d in ring_dicts]),
        }
    return {
        "iou_free": weighted_mean(ious, iou_counts),
        "baseline_iou_free": weighted_mean(base_ious, base_counts),
        "all_free_iou_free": weighted_mean(allfree_ious, allfree_counts),
        "fatal_rate": weighted_mean(fatals, fatal_denoms),
        "baseline_fatal_rate": weighted_mean(base_fatals, base_denoms),
        "free_miss_rate": weighted_mean(misses, miss_denoms),
        "iou_drivable": float(np.mean(d_ious)) if d_ious else float("nan"),
        "iou_obstacle": weighted_mean(o_ious, o_counts),
        "range": summarize_range_error(range_dicts),
        "rings": rings,
    }


def main(
    checkpoint,
    train_sequences="raws1,raws2,raws3,rawos1,rawos4",
    val_sequences="rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=4,
    num_workers=4,
    n_theta=None,
    device="cuda",
):
    dataset_root = Path(dataset_root)
    train_samples = [
        s for name in parse_sequence_names(train_sequences)
        for s in list_sequence_samples(dataset_root / name)
    ]
    val_samples = [
        s for name in parse_sequence_names(val_sequences)
        for s in list_sequence_samples(dataset_root / name)
    ]
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    constant_map = constant_free_map(
        _collect_free_masks(train_samples, permanent_blind, invalid)
    )

    val_ds = RobotBEVDataset(val_samples, common_root=common_root)
    loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    vox_util = build_double_sphere_vox_util(GRID_SPEC, val_ds.cameras, device=device)
    model = TwoHeadSegnet(
        Z, Y, X, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    load_checkpoint_state_dict(model, checkpoint, device)
    model.eval()

    # n_theta는 명시적으로 지정하지 않는 한 `build_ray_index`의 기본값(720, Task 6)을
    # 그대로 따른다 -- 여기서 옛 기본값(360)을 하드코딩해 조용히 되돌리면 안 된다.
    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta=n_theta)
    scores = score_split(
        model, loader, vox_util,
        rays,
        build_ring_masks(GRID_SPEC), device, constant_map,
    )
    row = {"name": Path(checkpoint).parent.name, "split": val_sequences, **scores}
    print(format_markdown_table([row]))
    print()
    # M2b(free_miss_rate)는 표(_COLUMNS)에는 안 넣었지만 score_split이 이미 계산해 두고
    # 있으므로, 버리지 않고 진단으로 남긴다 -- fatal_rate와 비용이 다른 별도 지표다.
    print(f"free_miss_rate (M2b, 참고, 보수성): {scores['free_miss_rate']:.3f}")
    print(f"range: p50 {scores['range']['abs_p50']:.3f} m | p90 {scores['range']['abs_p90']:.3f} m"
          f" | over {scores['range']['over_mean']:.3f} m | under {scores['range']['under_mean']:.3f} m"
          f" | paired rays {scores['range']['n_paired_rays']}")
    for name, values in scores["rings"].items():
        print(f"  ring {name:10s} iou_free {values['iou_free']:.3f}"
              f"  fatal {values['fatal_rate']:.3f}")


if __name__ == "__main__":
    Fire(main)
