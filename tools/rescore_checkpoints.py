"""저장된 3-class 체크포인트를 free-space 지표로 재채점한다 (스펙 Phase 1).

재학습하지 않고 지표만 바꿔 다시 재는 것이 요점이다. 지표를 수정한 뒤 판정을 다시 내릴 때
학습을 다시 돌리지 않아도 되는 것이 이 도구의 가치다 -- 단, 체크포인트 **선택** 기준인
`iou_free`의 정의를 바꾸면 어느 epoch이 best로 뽑히는지가 달라지므로 그때는 재학습이 필요하다.

**2-head 체크포인트는 더 이상 채점할 수 없다.** Phase 3에서 3-class로 확정하며 2-head 코드를
제거했기 때문이다(`docs/free_space_metric_migration.md` §9). 문서 §6·§8에 기록된 2-head
기준선 숫자는 그 시점의 역사적 값으로 고정되며, 지표를 바꿔도 다시 채점되지 않는다.

실행:
    CUDA_VISIBLE_DEVICES=0 python tools/rescore_checkpoints.py \\
        --checkpoint=runs/robot_bev/ckpt/<run>/model_best-000000030.pth \\
        --train_sequences=raws2,raws3,rawos1,rawos2,rawos4 --val_sequences=raws1,rawos3

`--formulation=binary`로 (D) 정식화 체크포인트도 같은 지표로 채점한다. **정식화를 틀리면
head 채널 수가 안 맞아 `load_checkpoint_state_dict`가 즉시 실패한다** -- 조용히 다른 숫자가
나오는 일은 없다. `--val_sequences`에 시퀀스를 하나만 주면 시퀀스별 분해가 된다.
"""
import sys
from pathlib import Path

import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.baselines import all_free_map, as_batch, constant_free_map  # noqa: E402
from projects.common.binary_metrics import predicted_parts as binary_predicted_parts  # noqa: E402
from projects.common.free_space import decompose, decompose_from_class_index  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    DEFAULT_RING_EDGES_M,
    build_ring_masks,
    fatal_rate,
    free_miss_rate,
    iou_free,
    metrics_per_ring,
    range_error,
    summarize_range_error,
    summarize_range_error_by_gt_range,
    weighted_mean,
)
from projects.common.occupied_metrics import (  # noqa: E402
    summarize_tolerance_f1,
    tolerance_counts,
)
from projects.common.polar import build_ray_index  # noqa: E402
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
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

_COLUMNS = (
    ("name", "체크포인트"), ("split", "val"),
    ("iou_free", "iou_free↑"), ("baseline_iou_free", "baseline iou_free"),
    ("all_free_iou_free", "all-free iou_free"),
    ("fatal_rate", "fatal↓"), ("baseline_fatal_rate", "baseline fatal"),
    ("free_miss_rate", "free_miss↓"),
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

    학습 스크립트의 `load_trunk_weights`와 달리 여기서는 부분 로드를 허용하지 않는다. 이 계약이 없으면
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
    return [
        decompose(*load_masked_labels(sequence_root, sample_id, permanent_blind, invalid))["free"]
        for sequence_root, sample_id in samples
    ]


def score_split(model, loader, vox_util, rays, ring_masks, device, constant_map,
                cell_m, decompose_pred=None) -> dict:
    """`decompose_pred(logits, valid)`가 예측 logits를 free/occupied/unknown으로 나눈다.

    정식화마다 이 규칙만 다르고 나머지 채점은 완전히 같다 -- 그래서 3-class와 binary의
    숫자를 한 표에 놓을 수 있다. 기본값은 3-class의 argmax 분해다.
    """
    if decompose_pred is None:
        def decompose_pred(logits, valid):
            return decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid)

    ious, iou_counts, fatals, fatal_denoms, misses, miss_denoms = [], [], [], [], [], []
    base_ious, base_counts, base_fatals, base_denoms = [], [], [], []
    allfree_ious, allfree_counts = [], []
    range_dicts, ring_dicts, tolerance_dicts = [], [], []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb_camXs"].to(device) - 0.5
            _, _, logits, _, _ = model(
                rgb, batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device), vox_util
            )
            seg_g = batch["seg_bev_g"].to(device)
            vis_g = batch["vis_bev_g"].to(device)
            valid = batch["valid_bev_g"].to(device)

            gt_parts = decompose(seg_g, vis_g, valid)
            # 학습 루프(`three_class_metrics.compute_free_metrics`)와 같은 방식으로 예측을
            # 분해한다 -- 여기가 argmax가 아닌 다른 규칙을 쓰면 재채점 값이 학습 로그의
            # 값과 달라져 두 숫자를 나란히 읽을 수 없다.
            pred_parts = decompose_pred(logits, valid)
            gt, pred = gt_parts["free"], pred_parts["free"]
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

            range_dicts.append(range_error(pred, gt, valid, rays))
            ring_dicts.append(metrics_per_ring(pred, gt, valid, ring_masks))
            tolerance_dicts.append(tolerance_counts(
                pred_parts["occupied"], gt_parts["occupied"], valid, cell_m
            ))

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
        "range": summarize_range_error(range_dicts),
        "range_bins": summarize_range_error_by_gt_range(range_dicts, DEFAULT_RING_EDGES_M),
        "rings": rings,
        "tolerance": summarize_tolerance_f1(tolerance_dicts),
    }


def main(
    checkpoint,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    # 정식화. `three_class`(기본) 또는 `binary`. 학습 때와 같은 값을 줘야 한다.
    formulation="three_class",
    batch_size=4,
    num_workers=4,
    n_theta=None,
    device="cuda",
):
    if formulation not in ("three_class", "binary"):
        raise ValueError(f"formulation은 three_class 또는 binary여야 한다: {formulation}")
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
    model = ThreeClassSegnet(
        Z, Y, X, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
        num_classes=2 if formulation == "binary" else 3,
    ).to(device)
    load_checkpoint_state_dict(model, checkpoint, device)
    model.eval()

    # n_theta는 명시적으로 지정하지 않는 한 `build_ray_index`의 기본값(720, Task 6)을
    # 그대로 따른다 -- 여기서 옛 기본값(360)을 하드코딩해 조용히 되돌리면 안 된다.
    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta=n_theta)
    # binary는 occupied head가 없으므로 예측 free의 경계에서 유도한다 -- 학습 루프와
    # 시각화가 쓰는 것과 **같은 함수**여야 세 곳의 숫자가 갈리지 않는다.
    decompose_pred = None
    if formulation == "binary":
        def decompose_pred(logits, valid, _rays=rays):
            return binary_predicted_parts(logits, valid, _rays)

    scores = score_split(
        model, loader, vox_util,
        rays,
        build_ring_masks(GRID_SPEC), device, constant_map, GRID_SPEC.cell_m,
        decompose_pred=decompose_pred,
    )
    # `--val_sequences=raws1,rawos3`을 Fire가 **tuple**로 파싱하므로 표에 넣기 전에 문자열로
    # 되돌린다. 시퀀스를 하나만 줄 때는 str이라 이 결함이 드러나지 않았다.
    row = {"name": Path(checkpoint).parent.name,
           "split": ",".join(parse_sequence_names(val_sequences)), **scores}
    print(format_markdown_table([row]))
    print()
    r = scores["range"]
    print(f"range: mae {r['mae']:.3f} m | p50 {r['abs_p50']:.3f} m | p90 {r['abs_p90']:.3f} m"
          f" | bias {r['bias']:+.3f} m | over {r['over_mean']:.3f} m"
          f" | under {r['under_mean']:.3f} m")
    # `missed`를 거리 통계 바로 옆에 찍는다 -- 놓친 광선은 위 통계의 표본에서 빠지므로
    # mae만 혼자 읽으면 "장애물을 많이 놓칠수록 좋아 보이는" 방향으로 오독된다.
    print(f"       missed_obstacle_rate {r['missed_obstacle_rate']:.3f}"
          f" ({r['missed_obstacle']}/{r['ok_gt']} rays)"
          f" | paired rays {r['n_paired_rays']}")
    # 경계 정밀도. 면적 `iou_occupied`는 2026-08-21에 뺐다(§23) -- 두께 1셀 표면의 면적 IoU는
    # 한 칸 밀리면 반토막 나서 품질 신호로 읽을 수 없다. binary에서는 head가 없어 "head 대
    # derived" 대조 자체가 같은 숫자를 두 번 찍는 것이기도 했다.
    # 예측 셀 수는 남긴다 -- "몇 셀을 칠했나"가 precision의 해석을 바꾼다(§9의 17배 과잉 예측).
    for name, values in scores["tolerance"].items():
        print(f"  f1@{name:5s} {values['f1']:.3f}"
              f"  precision {values['precision']:.3f}  recall {values['recall']:.3f}"
              f"  | pred {values['n_pred']} gt {values['n_gt']}")
    for name, values in scores["rings"].items():
        print(f"  ring {name:10s} iou_free {values['iou_free']:.3f}"
              f"  fatal {values['fatal_rate']:.3f}"
              f"  range_mae {scores['range_bins'].get(name, {}).get('mae', float('nan')):.3f}")


if __name__ == "__main__":
    Fire(main)
