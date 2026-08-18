"""자체 수집 데이터셋 -> 3-class 단일 head Simple-BEV fine-tuning (ROADMAP Phase 4).

SynWoodScape pretrain(`tools/train_synwoodscape.py`)과 지표·로깅을 공유하며
(`projects/common/bev_occupancy_metrics.py`), 다른 것은 세 가지뿐이다:

1. 데이터셋: `RobotBEVDataset` (Double Sphere 3-cam, 시퀀스 단위 split)
2. lifting: `DoubleSphereVoxUtil`
3. 초기화: pretrain 체크포인트에서 시작한다 (`--init_checkpoint`)

2-head 정식화는 제거됐다 -- Phase 3 A/B에서 3-class로 확정했다
(`docs/free_space_metric_migration.md` §8, §9).

실행 예:
    CUDA_VISIBLE_DEVICES=0 python tools/train_robot_bev.py \\
        --train_sequences=raws2,raws3,rawos1,rawos2,rawos4 --val_sequences=raws1,rawos3 \\
        --init_checkpoint=<pretrain best>.pth
"""
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

import saverloader  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.three_class_metrics import (  # noqa: E402
    class_weights_from_labels,
    run_batch as three_class_run_batch,
)
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    DEFAULT_RING_EDGES_M,
    build_ring_masks,
    iou_free,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.bev_occupancy_metrics import (  # noqa: E402
    _Ansi,
    _c,
    _print_banner,
    append_free_metrics,
    empty_epoch_metrics,
    evaluate_split,
    format_epoch_log,
    mean_loss_parts,
    select_checkpoint_score,
    summarize_free_metrics,
    weighted_mean,
    write_epoch_scalars,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    load_masked_labels,
    parse_sequence_names,
    split_samples_by_sequence,
    split_samples_within_sequences,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import (  # noqa: E402
    ThreeClassSegnet,
    head_was_transferred,
    load_trunk_weights,
    unexpected_skips,
)


def compute_label_statistics(samples, permanent_blind, invalid) -> dict:
    """라벨 분포를 **마스킹이 적용된 셀**에서만 실측한다.

    3-class loss는 `valid` 안에서만 채점되고 `vis`가 unknown/관측 클래스를 가르므로,
    마스킹 전 라벨로 비율을 세면 실제 loss가 보는 분포와 어긋난다. 자체 데이터셋에서는 이
    차이가 크다 -- 그리드 전체 obstacle 비율은 23%인데 마스킹 후 관측 영역에서는 5%
    수준이다(온실 통로에서 raycast visibility가 장애물에 닿으며 멈춰 장애물 대부분이 경계
    바깥에 놓이기 때문).
    """
    free = occupied = supervised = total = 0
    for sequence_root, sample_id in samples:
        # 조합은 `free_space.decompose` 하나만 쓴다 -- 여기서 다시 쓰면 loss/지표가 보는
        # 클래스 정의와 이 통계가 세는 정의가 갈라질 수 있다.
        parts = decompose(*load_masked_labels(
            sequence_root, sample_id, permanent_blind, invalid
        ))
        observed = parts["free"] | parts["occupied"]        # == vis & valid
        free += int(parts["free"].sum())
        occupied += int(parts["occupied"].sum())
        supervised += int(observed.sum())
        total += observed.size
    return {
        "trivial_iou": free / max(supervised, 1),
        "supervised_fraction": supervised / max(total, 1),
        "obstacle_fraction": occupied / max(supervised, 1),
    }


def _baseline_iou_free(val_samples, permanent_blind, invalid, constant_map, device):
    """학습 split의 셀별 다수결 free map을 validation 라벨에 한 번만 채점한다."""
    if constant_map is None or not val_samples:
        return float("nan")
    values, counts = [], []
    for sequence_root, sample_id in val_samples:
        triple = load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
        occ, _, valid = triple
        free_gt = torch.from_numpy(decompose(*triple)["free"]).view(1, 1, *occ.shape).to(device)
        valid_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
        value, count = iou_free(as_batch(constant_map, 1, device), free_gt, valid_t)
        values.append(value)
        counts.append(count)
    return weighted_mean(values, counts)


def main(
    exp_name="robot_finetune",
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    val_tail_fraction=0.0,  # val 시퀀스가 없을 때만 쓰는 임시 holdout (시퀀스 뒤쪽 연속 구간)
    init_checkpoint=None,
    num_epochs=60,
    batch_size=8,
    lr=1e-4,  # pretrain(3e-4)보다 낮게 -- 초기값을 크게 흔들지 않는 것이 fine-tuning의 요점
    weight_decay=1e-7,
    num_workers=8,
    encoder_type="res101",
    augment=False,  # 광도 증강. pretrain에서는 +0.006이었지만 적용 여부는 사용자가 결정한다
    val_freq_epochs=1,
    save_freq_epochs=10,
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    log_dir="runs/robot_bev/logs",
    ckpt_dir="runs/robot_bev/ckpt",
    device="cuda",
    # 주기 저장분을 몇 개까지 남길지. **기본값 3은 유효 구간을 지운다** -- `save_freq_epochs=10`
    # 으로 60 epoch을 돌리면 10/20/30/40/50/60에 저장되는데 3개만 남아 40/50/60이 되고,
    # 과적합이 빠른 리그에서 정작 쓸 만한 초기 epoch이 통째로 사라진다(2026-08-18 fine-tuning
    # 진단에서 실제로 겪었다: `docs/finetune_overfitting_diagnosis.md` §5). 체크포인트 하나가
    # 약 487 MB이므로 `num_epochs / save_freq_epochs` 만큼 남기는 것을 기본으로 둔다.
    keep_checkpoints=6,
    n_theta=None,
):
    torch.manual_seed(0)
    np.random.seed(0)

    dataset_root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    sequence_roots = [dataset_root / name for name in names + val_names]
    for root in sequence_roots:
        if not (root / "occupancy_npy").exists():
            raise FileNotFoundError(f"시퀀스를 찾을 수 없다: {root}")

    if val_names:
        train_samples, val_samples = split_samples_by_sequence(sequence_roots, val_names)
        split_note = f"시퀀스 단위 holdout: {','.join(val_names)}"
    elif val_tail_fraction > 0:
        train_samples, val_samples = split_samples_within_sequences(
            sequence_roots, val_tail_fraction
        )
        split_note = (f"시퀀스 뒤쪽 {100 * val_tail_fraction:.0f}% holdout"
                      " (임시 -- 경계가 인접해 낙관적인 숫자다)")
    else:
        train_samples, val_samples = split_samples_by_sequence(sequence_roots, [])
        split_note = "없음"
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    stats = compute_label_statistics(train_samples, permanent_blind, invalid)
    val_stats = compute_label_statistics(val_samples, permanent_blind, invalid)
    train_free_masks = [
        decompose(*load_masked_labels(sequence_root, sample_id, permanent_blind, invalid))["free"]
        for sequence_root, sample_id in train_samples
    ]
    constant_map = constant_free_map(train_free_masks) if train_free_masks else None
    baseline_iou_free = _baseline_iou_free(
        val_samples, permanent_blind, invalid, constant_map, device
    )
    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta=n_theta)
    ring_masks = build_ring_masks(GRID_SPEC)
    class_weights = class_weights_from_labels(
        load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
        for sequence_root, sample_id in train_samples
    )

    _print_banner([
        " robot dataset -> Simple-BEV three-class fine-tuning",
        f" exp_name={exp_name} | encoder={encoder_type} | cameras={','.join(FINETUNE_CAMERA_NAMES)}",
        f" batch_size={batch_size} | lr={lr:.0e} | epochs={num_epochs}",
        f" train sequences={','.join(names) or '-'} ({len(train_samples)} samples)",
        f" val   split={split_note} ({len(val_samples)} samples)",
        f" init_checkpoint={init_checkpoint or 'none (from scratch)'}",
        f" masks: vis=0 on {permanent_blind.sum()} cells | valid=0 on {invalid.sum()} cells",
        f" observed (vis&valid) covers {100 * stats['supervised_fraction']:.2f}% of cells"
        f" (obstacle {100 * stats['obstacle_fraction']:.2f}% inside it)",
        # val 분포를 같이 찍는다 -- 한 시퀀스 안에서도 구간마다 관측 면적과 장애물 비율이
        # 몇 배씩 차이 나므로(raws1은 앞 30장 11.7%/7.4% vs 뒤 8장 36.5%/3.1%), 이게 안
        # 보이면 "val이 안 오른다"의 원인이 모델인지 분포 불일치인지 구분할 수 없다.
        f" val   distribution: covers {100 * val_stats['supervised_fraction']:.2f}%"
        f" (obstacle {100 * val_stats['obstacle_fraction']:.2f}% inside it)"
        + ("  <- train과 크게 다르다" if val_samples and (
            abs(val_stats["supervised_fraction"] - stats["supervised_fraction"]) > 0.05
            or abs(val_stats["obstacle_fraction"] - stats["obstacle_fraction"]) > 0.02
        ) else ""),
        f" class weights (unknown/free/occupied) = {class_weights.tolist()}",
        f" photometric augment (train only) = {bool(augment)}",
        f" trivial 'always drivable' baseline IoU = {stats['trivial_iou']:.3f}  <- compare against this",
        f" constant-map baseline iou_free = {baseline_iou_free:.3f}  <- compare against this",
    ])
    if not val_samples:
        print(_c(_Ansi.YELLOW + _Ansi.BOLD,
                 " [warning] val 시퀀스가 없다 -- 체크포인트 선택 없이 마지막 epoch만 남는다."))

    train_ds = RobotBEVDataset(train_samples, common_root=common_root, augment=augment)
    # 마지막 배치가 1개일 때만 버린다(BatchNorm이 배치 1에서 죽는다). pretrain 쪽은
    # 그냥 drop_last=True인데, 여기서는 시퀀스 하나가 수십 장뿐이라 그러면 한 epoch에서
    # 샘플의 10~20%가 통째로 빠진다.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=len(train_samples) % batch_size == 1,
    )
    val_loader = DataLoader(
        RobotBEVDataset(val_samples, common_root=common_root),  # val은 항상 원본
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    vox_util = build_double_sphere_vox_util(GRID_SPEC, train_ds.cameras, device=device)
    # rand_flip=False: 이 리그의 ROI는 전후 비대칭(전방 4 m / 후방 2 m)이라
    # Simple-BEV의 Z축 flip 증강이 물리적으로 성립하지 않는다.
    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    if init_checkpoint:
        # shape가 맞는 키를 전부 복사한다. 2-head pretrain 체크포인트에서는 출력 head 6개가
        # 형상이 달라 skip되고(674/6), 3-class pretrain 체크포인트에서는 head까지 함께
        # 전이돼 skipped가 0이어야 한다 -- 그 숫자가 곧 "head가 전이됐는지"의 확인이다.
        report = load_trunk_weights(model, init_checkpoint, device)
        print(_c(
            _Ansi.CYAN,
            f" weight transfer: loaded {report['loaded']} tensors, skipped {len(report['skipped'])}"
            + ("  <- 출력 head까지 전이됨" if head_was_transferred(report["skipped"])
               else "  <- 출력 head는 랜덤 초기화"),
        ))
        unexpected = unexpected_skips(report["skipped"])
        if unexpected:
            raise RuntimeError(f"trunk keys were skipped: {unexpected[:5]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_epochs * steps_per_epoch + 10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear",
    )
    class_weights = class_weights.to(device)
    step = lambda batch: three_class_run_batch(  # noqa: E731
        model, batch, vox_util, class_weights, device
    )

    run_name = (f"{exp_name}_{encoder_type}_bs{batch_size}_lr{lr:.0e}"
                f"_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    log_path = Path(log_dir) / run_name
    writer = SummaryWriter(str(log_path))
    ckpt_path = Path(ckpt_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    for tag, samples in (("train", train_samples), ("val", val_samples)):
        (log_path / f"split_{tag}_samples.txt").write_text(
            "".join(f"{root.name}/{sample_id}\n" for root, sample_id in samples)
        )

    global_step = 0
    best_val_score = 0.0
    interrupted = False
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_start = time.time()
            losses, parts_dicts, free_dicts = [], [], []
            for batch in train_loader:
                optimizer.zero_grad()
                loss, parts, free_metrics = step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                parts_dicts.append({k: v.item() for k, v in parts.items()})
                append_free_metrics(free_dicts, free_metrics)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            train = {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "loss_parts": mean_loss_parts(parts_dicts),
                "free": summarize_free_metrics(free_dicts),
            }
            write_epoch_scalars(writer, "train", train, epoch)

            val = empty_epoch_metrics()
            if epoch % val_freq_epochs == 0 and len(val_loader) > 0:
                model.eval()
                val = evaluate_split(step, val_loader, device, rays, ring_masks,
                                     cell_m=GRID_SPEC.cell_m,
                                     range_edges_m=DEFAULT_RING_EDGES_M)
                write_epoch_scalars(writer, "val", val, epoch)

            val_score = select_checkpoint_score(val["free"])
            is_new_best = val_score > best_val_score  # NaN > x는 항상 False
            print(format_epoch_log(
                epoch=epoch, num_epochs=num_epochs, epoch_time=time.time() - epoch_start,
                train_loss=train["loss"], train_loss_parts=train["loss_parts"],
                train_free_metrics=train["free"],
                val_loss=val["loss"], val_loss_parts=val["loss_parts"],
                val_free_metrics=val["free"],
                val_range_metrics=val["range"], val_tolerance_metrics=val["tolerance"],
                baseline_iou_free=baseline_iou_free,
                val_score=val_score, best_val_score=best_val_score, is_new_best=is_new_best,
            ))

            if epoch % save_freq_epochs == 0 or epoch == num_epochs:
                saverloader.save(str(ckpt_path), optimizer, model, epoch,
                                 keep_latest=keep_checkpoints)
            if is_new_best:
                best_val_score = val_score
                saverloader.save(str(ckpt_path), optimizer, model, epoch,
                                 keep_latest=1, model_name="model_best")
    except KeyboardInterrupt:
        interrupted = True
        print("\n" + _c(_Ansi.YELLOW + _Ansi.BOLD,
                        "[interrupted] Ctrl+C. 저장된 체크포인트/로그는 안전하다."))
    finally:
        writer.close()

    if not interrupted:
        _print_banner([" done.", f" logs:  {log_path}", f" ckpts: {ckpt_path}"])


if __name__ == "__main__":
    Fire(main)
