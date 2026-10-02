"""저장된 3-class 체크포인트를 free-space 지표로 재채점한다 (스펙 Phase 1).

재학습하지 않고 지표만 바꿔 다시 재는 것이 요점이다. 지표를 수정한 뒤 판정을 다시 내릴 때
학습을 다시 돌리지 않아도 되는 것이 이 도구의 가치다 -- 단, 체크포인트 **선택** 기준인
`iou_free`의 정의를 바꾸면 어느 epoch이 best로 뽑히는지가 달라지므로 그때는 재학습이 필요하다.

**2-head 체크포인트는 더 이상 채점할 수 없다.** Phase 3에서 3-class로 확정하며 2-head 코드를
제거했기 때문이다(`docs/archive/free_space_metric_migration.md` §9). 문서 §6·§8에 기록된 2-head
기준선 숫자는 그 시점의 역사적 값으로 고정되며, 지표를 바꿔도 다시 채점되지 않는다.

실행:
    CUDA_VISIBLE_DEVICES=0 python tools/rescore_checkpoints.py \\
        --checkpoint=runs/robot_bev/ckpt/<run>/model_best-000000030.pth \\
        --train_sequences=raws2,raws3,rawos1,rawos2,rawos4 --val_sequences=raws1,rawos3

**지표는 학습 루프와 같은 2026-10-02 집합이다**(`projects/common/metric_spec.py`): free 지표
넷(macro), 링별 넷, `BF@τ`. `--log_dir`과 `--epoch`을 주면 결과를 **학습 로그와 같은 tag로**
TensorBoard에 쓴다 -- 그러면 `summarize_repeats`·`verify_val_predictions`·패키지 생성기가
재채점 결과를 학습 로그처럼 읽는다(학습 없이 분석 파이프라인을 실데이터로 점검할 때 쓴다).
**단, 그 로그에는 그 epoch 하나만 있다** -- epoch 곡선이 필요한 분석(되올림 등)은 재학습해야 한다.

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
from projects.common.bev_occupancy_metrics import (  # noqa: E402
    append_free_metrics,
    summarize_free_metrics,
    summarize_ring_metrics,
    write_epoch_scalars,
)
from projects.common.free_space_metrics import (  # noqa: E402
    build_ring_masks,
    free_scores,
    iou_free,
    metrics_per_ring,
    weighted_mean,
)
from projects.common.metric_spec import FREE_METRICS  # noqa: E402
from projects.common.occupied_metrics import (  # noqa: E402
    boundary_f_scores,
    summarize_boundary_f,
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
from projects.datasets.simplebev_vox import (  # noqa: E402
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_checkpoints  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

_COLUMNS = (
    ("name", "체크포인트"), ("split", "val"),
    ("iou_free", "iou_free↑"), ("baseline_iou_free", "baseline iou_free"),
    ("all_free_iou_free", "all-free iou_free"),
    ("iou_non_free", "iou_non_free↑"), ("precision", "precision↑"), ("recall", "recall↑"),
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

    free_dicts, base_ious, base_counts, allfree_ious, allfree_counts = [], [], [], [], []
    ring_dicts, boundary_dicts = [], []

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

            # 학습 루프와 **같은 함수**(`free_scores`, `metrics_per_ring`, `boundary_f_scores`)
            # 와 같은 epoch 집계(`summarize_*`)를 탄다.
            append_free_metrics(free_dicts, {**free_scores(pred, gt, valid),
                                             "partition_defects": 0})
            for values, counts, arg in ((base_ious, base_counts, base),
                                        (allfree_ious, allfree_counts, allfree)):
                value, count = iou_free(arg, gt, valid)
                values.append(value)
                counts.append(count)
            ring_dicts.append(metrics_per_ring(pred, gt, valid, ring_masks))
            boundary_dicts.append(boundary_f_scores(
                pred_parts["occupied"], gt_parts["occupied"], valid, cell_m
            ))

    free = summarize_free_metrics(free_dicts)
    return {
        **{key: free[key] for key in FREE_METRICS},
        "baseline_iou_free": weighted_mean(base_ious, base_counts),
        "all_free_iou_free": weighted_mean(allfree_ious, allfree_counts),
        "free": free,
        "rings": summarize_ring_metrics(ring_dicts, ring_masks),
        "boundary": summarize_boundary_f(boundary_dicts),
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
    # 주면 결과를 학습 로그와 같은 tag로 `{log_dir}`에 쓴다(epoch 하나). 모듈 docstring 참고.
    log_dir=None,
    epoch=None,
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
    # 표본 규약과 **표본 높이**를 둘 다 체크포인트 옆에서 되찾는다 -- 기본값을 쓰면 옛
    # 런(legacy)을 새 기하로 재채점해 조용히 다른 숫자가 나온다. 높이는 `height.json`에서
    # 오고, 파일이 없으면 옛 기본값(`Y=1`)이라 하위 호환이 유지된다.
    convention, offset = convention_for_checkpoints([checkpoint])
    height = height_config_for_ckpt_dirs([Path(checkpoint).parent])
    Z, Y, X = vox_dims(GRID_SPEC, height["height_bins"])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, val_ds.cameras, device=device,
                                           pixel_convention=convention,
                                           pixel_offset=offset,
                                           height_bins=height["height_bins"],
                                           height_min_m=height["height_min_m"],
                                           height_max_m=height["height_max_m"])
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
    for name, values in scores["boundary"].items():
        print(f"  bf@{name:5s} {values['bf']:.4f}"
              f"  precision {values['precision']:.4f}  recall {values['recall']:.4f}"
              f"  | frames {values['n_frames']}")
    for name, values in scores["rings"].items():
        print(f"  ring {name:10s} " + "  ".join(f"{m} {values[m]:.4f}" for m in FREE_METRICS))

    if log_dir is not None:
        if epoch is None:
            raise SystemExit("--log_dir에는 --epoch이 필요하다(어느 epoch의 값인지)")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        write_epoch_scalars(writer, "val", {"loss": float("nan"), "loss_parts": {},
                                            **{k: scores[k] for k in ("free", "rings", "boundary")}},
                            int(epoch))
        writer.close()
        print(f"\nval scalar를 학습 로그와 같은 tag로 썼다: {log_dir} (epoch {epoch})")


if __name__ == "__main__":
    Fire(main)
