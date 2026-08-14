"""자체 수집 데이터셋 -> two-head Simple-BEV fine-tuning (ROADMAP Phase 4).

SynWoodScape pretrain(`tools/train_synwoodscape.py`)과 지표·로깅을 공유하며
(`projects/common/two_head_metrics.py`), 다른 것은 네 가지뿐이다:

1. 데이터셋: `RobotBEVDataset` (Double Sphere 3-cam, 시퀀스 단위 split)
2. lifting: `DoubleSphereVoxUtil`
3. 초기화: pretrain 체크포인트에서 시작한다 (`--init_checkpoint`)
4. 클래스 비율(`pos_weight`)을 **마스킹 후** 라벨에서 실측한다

실행 예:
    CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py \\
        --train_sequences=raws1 --init_checkpoint=<pretrain best>.pth
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
from projects.common.two_head_metrics import (  # noqa: E402
    _Ansi,
    _c,
    _print_banner,
    format_epoch_log,
    run_batch,
    summarize_deployment_metrics,
    summarize_occupancy_diagnostics,
    weighted_mean,
    write_deployment_metrics,
    write_occupancy_diagnostics,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    load_masked_labels,
    split_samples_by_sequence,
    split_samples_within_sequences,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_two_head import TwoHeadSegnet  # noqa: E402


def compute_label_statistics(samples, permanent_blind, invalid) -> dict:
    """`pos_weight`와 트리비얼 베이스라인을 **마스킹이 적용된 셀**에서만 실측한다.

    occupancy loss는 `vis * valid`로 마스킹되므로, 마스킹 전 라벨로 클래스 비율을 세면
    실제 loss가 보는 분포와 어긋난다. 자체 데이터셋에서는 이 차이가 크다 -- 그리드 전체
    obstacle 비율은 23%인데 마스킹 후 supervised 영역에서는 5% 수준이다(온실 통로에서
    raycast visibility가 장애물에 닿으며 멈춰 장애물 대부분이 경계 바깥에 놓이기 때문).
    """
    pos = neg = supervised = total = 0
    for sequence_root, sample_id in samples:
        occupancy, vis, valid = load_masked_labels(
            sequence_root, sample_id, permanent_blind, invalid
        )
        mask = vis & valid
        pos += int((occupancy & mask).sum())
        neg += int((~occupancy & mask).sum())
        supervised += int(mask.sum())
        total += mask.size
    return {
        "pos_weight": neg / max(pos, 1),
        "trivial_iou": pos / max(supervised, 1),
        "supervised_fraction": supervised / max(total, 1),
        "obstacle_fraction": neg / max(supervised, 1),
    }


def load_initial_weights(model, checkpoint_path, device) -> None:
    """pretrain 체크포인트를 초기값으로 얹는다.

    `Segnet`은 (Z, X)에 대해 완전 합성곱이고 Y=1이 양쪽 같아서, SynWoodScape의
    240x240 그리드에서 학습한 가중치가 로봇의 120x120 그리드에 그대로 들어간다.
    카메라 수(4 -> 3)도 per-camera 파라미터가 없어서 문제되지 않는다.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"체크포인트가 모델과 맞지 않는다: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    model.to(device)


def _evaluate(model, loader, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight):
    losses, occ_losses, vis_losses = [], [], []
    d_ious, o_ious, o_counts, false_highs, false_lows = [], [], [], [], []
    occ_dicts, deploy_dicts = [], []
    with torch.no_grad():
        for batch in loader:
            loss, parts, d_iou, o_iou, o_count, vis_metrics, occ_metrics, deploy = run_batch(
                model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
            )
            losses.append(loss.item())
            occ_losses.append(parts["loss_occ"].item())
            vis_losses.append(parts["loss_vis"].item())
            d_ious.append(d_iou.item())
            o_ious.append(o_iou.item())
            o_counts.append(o_count)
            false_highs.append(vis_metrics["false_high"])
            false_lows.append(vis_metrics["false_low"])
            occ_dicts.append(occ_metrics)
            deploy_dicts.append(deploy)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {
        "loss": mean(losses), "loss_occ": mean(occ_losses), "loss_vis": mean(vis_losses),
        "d_iou": mean(d_ious), "o_iou": weighted_mean(o_ious, o_counts),
        "false_high": mean(false_highs), "false_low": mean(false_lows),
        "occ": summarize_occupancy_diagnostics(occ_dicts),
        "deploy": summarize_deployment_metrics(deploy_dicts),
    }


def main(
    exp_name="robot_finetune",
    train_sequences="raws1",
    val_sequences="",
    val_tail_fraction=0.0,  # val 시퀀스가 없을 때만 쓰는 임시 holdout (시퀀스 뒤쪽 연속 구간)
    init_checkpoint=None,
    num_epochs=60,
    batch_size=8,
    lr=1e-4,  # pretrain(3e-4)보다 낮게 -- 초기값을 크게 흔들지 않는 것이 fine-tuning의 요점
    weight_decay=1e-7,
    num_workers=8,
    encoder_type="res101",
    augment=False,  # 광도 증강. pretrain에서는 +0.006이었지만 적용 여부는 사용자가 결정한다
    pos_weight=None,
    lambda_vis=0.5,
    vis_neg_weight=3.0,
    val_freq_epochs=1,
    save_freq_epochs=10,
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    log_dir="runs/robot_bev/logs",
    ckpt_dir="runs/robot_bev/ckpt",
    device="cuda",
):
    torch.manual_seed(0)
    np.random.seed(0)

    dataset_root = Path(dataset_root)
    names = [n for n in str(train_sequences).split(",") if n]
    val_names = [n for n in str(val_sequences).split(",") if n]
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
    if pos_weight is None:
        pos_weight = stats["pos_weight"]

    _print_banner([
        " robot dataset -> Simple-BEV two-head fine-tuning",
        f" exp_name={exp_name} | encoder={encoder_type} | cameras={','.join(FINETUNE_CAMERA_NAMES)}",
        f" batch_size={batch_size} | lr={lr:.0e} | epochs={num_epochs}",
        f" train sequences={','.join(names) or '-'} ({len(train_samples)} samples)",
        f" val   split={split_note} ({len(val_samples)} samples)",
        f" init_checkpoint={init_checkpoint or 'none (from scratch)'}",
        f" masks: vis=0 on {permanent_blind.sum()} cells | valid=0 on {invalid.sum()} cells",
        f" occupancy loss covers {100 * stats['supervised_fraction']:.2f}% of cells"
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
        f" pos_weight (neg/pos, masked) = {pos_weight:.3f}",
        f" photometric augment (train only) = {bool(augment)}",
        f" trivial 'always drivable' baseline IoU = {stats['trivial_iou']:.3f}  <- compare against this",
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
    model = TwoHeadSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    if init_checkpoint:
        load_initial_weights(model, init_checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_epochs * steps_per_epoch + 10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear",
    )
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

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
            losses, occ_losses, vis_losses = [], [], []
            d_ious, o_ious, o_counts, false_highs, false_lows = [], [], [], [], []
            occ_dicts, deploy_dicts = [], []
            for batch in train_loader:
                optimizer.zero_grad()
                loss, parts, d_iou, o_iou, o_count, vis_metrics, occ_metrics, deploy = run_batch(
                    model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                occ_losses.append(parts["loss_occ"].item())
                vis_losses.append(parts["loss_vis"].item())
                d_ious.append(d_iou.item())
                o_ious.append(o_iou.item())
                o_counts.append(o_count)
                false_highs.append(vis_metrics["false_high"])
                false_lows.append(vis_metrics["false_low"])
                occ_dicts.append(occ_metrics)
                deploy_dicts.append(deploy)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
            train = {
                "loss": mean(losses), "loss_occ": mean(occ_losses), "loss_vis": mean(vis_losses),
                "d_iou": mean(d_ious), "o_iou": weighted_mean(o_ious, o_counts),
                "false_high": mean(false_highs), "false_low": mean(false_lows),
                "occ": summarize_occupancy_diagnostics(occ_dicts),
                "deploy": summarize_deployment_metrics(deploy_dicts),
            }
            for key, tb in (("loss", "loss_epoch"), ("loss_occ", "loss_occ_epoch"),
                            ("loss_vis", "loss_vis_epoch"), ("d_iou", "iou_drivable_epoch"),
                            ("o_iou", "iou_obstacle_epoch"),
                            ("false_high", "visibility_false_high_epoch"),
                            ("false_low", "visibility_false_low_epoch")):
                writer.add_scalar(f"train/{tb}", train[key], epoch)
            write_occupancy_diagnostics(writer, "train", train["occ"], epoch)
            write_deployment_metrics(writer, "train", train["deploy"], epoch)

            val = {
                "loss": float("nan"), "loss_occ": float("nan"), "loss_vis": float("nan"),
                "d_iou": float("nan"), "o_iou": float("nan"),
                "false_high": float("nan"), "false_low": float("nan"),
                "occ": summarize_occupancy_diagnostics([]),
                "deploy": summarize_deployment_metrics([]),
            }
            if epoch % val_freq_epochs == 0 and len(val_loader) > 0:
                model.eval()
                val = _evaluate(model, val_loader, vox_util, pos_weight_tensor, device,
                                lambda_vis, vis_neg_weight)
                for key, tb in (("loss", "loss_epoch"), ("loss_occ", "loss_occ_epoch"),
                                ("loss_vis", "loss_vis_epoch"), ("d_iou", "iou_drivable_epoch"),
                                ("o_iou", "iou_obstacle_epoch"),
                                ("false_high", "visibility_false_high_epoch"),
                                ("false_low", "visibility_false_low_epoch")):
                    writer.add_scalar(f"val/{tb}", val[key], epoch)
                write_occupancy_diagnostics(writer, "val", val["occ"], epoch)
                write_deployment_metrics(writer, "val", val["deploy"], epoch)

            val_score = 0.5 * (val["d_iou"] + val["o_iou"])
            is_new_best = val_score > best_val_score  # NaN > x는 항상 False
            print(format_epoch_log(
                epoch=epoch, num_epochs=num_epochs, epoch_time=time.time() - epoch_start,
                train_loss=train["loss"], train_occ_loss=train["loss_occ"],
                train_vis_loss=train["loss_vis"], train_d_iou=train["d_iou"],
                train_o_iou=train["o_iou"], train_v_false_high=train["false_high"],
                train_v_false_low=train["false_low"], train_occ_metrics=train["occ"],
                val_loss=val["loss"], val_occ_loss=val["loss_occ"], val_vis_loss=val["loss_vis"],
                val_d_iou=val["d_iou"], val_o_iou=val["o_iou"],
                val_v_false_high=val["false_high"], val_v_false_low=val["false_low"],
                val_occ_metrics=val["occ"], val_deploy_metrics=val["deploy"],
                val_score=val_score, best_val_score=best_val_score, is_new_best=is_new_best,
            ))

            if epoch % save_freq_epochs == 0 or epoch == num_epochs:
                saverloader.save(str(ckpt_path), optimizer, model, epoch, keep_latest=3)
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
