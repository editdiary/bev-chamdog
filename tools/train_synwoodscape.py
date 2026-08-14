"""SynWoodScape -> Simple-BEV `Segnet` 학습 스크립트 (ROADMAP Phase 3.3).

Simple-BEV 원본(`train_nuscenes.py`)의 관례를 따라 `Fire`로 `main(...)`의 키워드 인자를
CLI에서 받는다 — config 파일 체계 대신 실행 스크립트(`configs/train_synwoodscape_baseline.sh`)에
인자를 나열한다.

값을 보고 어떻게 튜닝할지는 `docs/training_guide.md`를 참고할 것.

실행 예:
    python tools/train_synwoodscape.py --exp_name=baseline --num_epochs=60 --batch_size=4
"""
import math
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

# Segnet(Encoder_res101/50)이 torchvision.models.resnet*(pretrained=True)로 내부에서 호출하는
# deprecated 인자 경고 -- submodule 코드라 직접 못 고치므로 여기서 억제한다. 동작에는 영향 없음
# (실제로는 pretrained=True와 동일하게 ImageNet 가중치를 불러온다).
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")


class _Ansi:
    """터미널 색 강조용 최소 ANSI 코드 -- 별도 의존성(rich/colorama) 없이 표준 출력에 직접 씀."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_Ansi.RESET}"


def _print_banner(lines) -> None:
    rule = "=" * 60
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))
    for line in lines:
        print(_c(_Ansi.BOLD + _Ansi.CYAN, line))
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))


def _format_deployment_line(metrics):
    """예측 visibility로 마스킹한 occupancy 성능. coverage 없이 IoU만 보면 안 된다 --
    모델이 시야를 좁게 부를수록 IoU는 쉬워지기 때문이다.
    """
    if metrics is None:
        return []
    return [
        (
            f"  deploy| (pred visibility 기준) iou_drivable↑ {metrics['iou_drivable']:.3f} | "
            f"iou_obstacle↑ {metrics['iou_obstacle']:.3f} | "
            f"visible_coverage {metrics['visible_coverage']:.3f}"
        )
    ]


def format_epoch_log(
    *,
    epoch,
    num_epochs,
    epoch_time,
    train_loss,
    train_occ_loss,
    train_vis_loss,
    train_d_iou,
    train_o_iou,
    train_v_false_high,
    train_v_false_low,
    train_occ_metrics=None,
    val_deploy_metrics=None,
    val_loss,
    val_occ_loss,
    val_vis_loss,
    val_d_iou,
    val_o_iou,
    val_v_false_high,
    val_v_false_low,
    val_occ_metrics=None,
    val_score,
    best_val_score,
    is_new_best,
):
    displayed_best = val_score if is_new_best else best_val_score
    checkpoint_note = "new best" if is_new_best else "-"
    return "\n".join([
        (
            f"epoch {epoch:03d}/{num_epochs} | time {epoch_time:6.1f}s | "
            f"val_iou_mean↑ {val_score:.3f} | best_val_iou_mean↑ {displayed_best:.3f} | "
            f"checkpoint: {checkpoint_note}"
            f"{_format_obstacle_bin_summary(val_occ_metrics)}"
        ),
        (
            f"  train | loss_total↓ {train_loss:.4f} | loss_occ↓ {train_occ_loss:.4f} | "
            f"loss_vis↓ {train_vis_loss:.4f} | iou_drivable↑ {train_d_iou:.3f} | "
            f"iou_obstacle↑ {train_o_iou:.3f} | vis_false_high↓ {train_v_false_high:.3f} | "
            f"vis_false_low↓ {train_v_false_low:.3f}"
            f"{_format_occupancy_diag(train_occ_metrics)}"
        ),
        (
            f"  val   | loss_total↓ {val_loss:.4f} | loss_occ↓ {val_occ_loss:.4f} | "
            f"loss_vis↓ {val_vis_loss:.4f} | iou_drivable↑ {val_d_iou:.3f} | "
            f"iou_obstacle↑ {val_o_iou:.3f} | vis_false_high↓ {val_v_false_high:.3f} | "
            f"vis_false_low↓ {val_v_false_low:.3f}"
            f"{_format_occupancy_diag(val_occ_metrics)}"
        ),
        *_format_deployment_line(val_deploy_metrics),
    ])


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

import saverloader  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
from projects.datasets.simplebev_vox import build_vox_util  # noqa: E402
from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OCCUPANCY_GT_ROOT,
    GRID_SPEC,
    SynWoodScapeSimpleBEVDataset,
)
from projects.datasets.synwoodscape_split import discover_all_sample_ids, train_val_split  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.models.fisheye_vox import build_fisheye_vox_util  # noqa: E402
from projects.models.simplebev_two_head import (  # noqa: E402
    TwoHeadSegnet,
    compute_two_head_loss,
    split_two_head_logits,
    visibility_error_rates,
)


def compute_pos_weight(sample_ids, occupancy_gt_root: Path) -> float:
    """`BCEWithLogitsLoss(pos_weight=...)` = neg/pos, valid(관측된) 셀만 대상으로 train split에서 실측."""
    pos = neg = 0
    for sample_id in sample_ids:
        occupancy = np.load(occupancy_gt_root / f"{sample_id}_occupancy.npy").astype(bool)
        visible = np.load(occupancy_gt_root / f"{sample_id}_visible.npy").astype(bool)
        pos += int((occupancy & visible).sum())
        neg += int((~occupancy & visible).sum())
    return neg / max(pos, 1)


def compute_trivial_baseline_iou(sample_ids, occupancy_gt_root: Path) -> float:
    """"항상 drivable로 예측"의 IoU = drivable_fraction. 모델 IoU가 이 값 근처면 학습이 아니라
    다수 클래스를 그냥 외운 것일 수 있다 (`docs/training_guide.md` 참고).
    """
    pos = valid = 0
    for sample_id in sample_ids:
        occupancy = np.load(occupancy_gt_root / f"{sample_id}_occupancy.npy").astype(bool)
        visible = np.load(occupancy_gt_root / f"{sample_id}_visible.npy").astype(bool)
        pos += int((occupancy & visible).sum())
        valid += int(visible.sum())
    return pos / max(valid, 1)


def compute_iou_per_sample(pred, target, valid_g):
    """`pred`/`target`은 이미 0/1(같은 class를 가리키는 binary mask)이어야 한다."""
    intersection = (pred * target * valid_g).sum(dim=[1, 2, 3])
    union = ((pred + target) * valid_g).clamp(0, 1).sum(dim=[1, 2, 3])
    return intersection / (1e-4 + union)


def compute_iou(pred, target, valid_g):
    return compute_iou_per_sample(pred, target, valid_g).mean()


def weighted_mean(values, weights):
    """weight가 0인 항목은 값이 NaN일 수 있으므로(= 셀 수 있는 샘플이 없던 batch) 건너뛴다."""
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total


def compute_drivable_and_obstacle_iou(seg_e_logits, seg_g, valid_g):
    """drivable(다수 클래스) IoU만 보면 "항상 drivable" 트리비얼 해와 구분이 안 된다
    (`docs/training_guide.md` 참고) — obstacle(비주행가능, 소수 클래스) IoU를 항상 같이 본다.

    GT에 obstacle이 하나도 없는 샘플은 intersection이 항상 0이라 IoU가 0으로 고정된다.
    "obstacle 없음"을 완벽하게 맞혀도 0점이라 지표를 왜곡하므로 평균에서 제외하고, 그런
    샘플의 오탐은 `compute_occupancy_diagnostics`의 `empty_false_alarm`으로 따로 본다.
    반환하는 count는 epoch 단위로 batch 간 가중평균을 내기 위한 것이다.
    """
    pred_drivable = torch.sigmoid(seg_e_logits).round()
    pred_obstacle = 1.0 - pred_drivable
    target_obstacle = 1.0 - seg_g

    drivable_iou = compute_iou(pred_drivable, seg_g, valid_g)

    per_sample_iou = compute_iou_per_sample(pred_obstacle, target_obstacle, valid_g)
    has_obstacle = (target_obstacle * valid_g).sum(dim=[1, 2, 3]) > 0
    obstacle_count = int(has_obstacle.sum().item())
    if obstacle_count == 0:
        obstacle_iou = torch.tensor(float("nan"), device=per_sample_iou.device)
    else:
        obstacle_iou = per_sample_iou[has_obstacle].mean()
    return drivable_iou, obstacle_iou, obstacle_count


OBSTACLE_FRACTION_BINS = (
    ("empty", 0.0, 0.0),
    ("tiny", 0.0, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.15),
    ("large", 0.15, 1.0),
)


def _bin_name_for_obstacle_fraction(value: float) -> str:
    if value == 0.0:
        return "empty"
    for name, lo, hi in OBSTACLE_FRACTION_BINS[1:]:
        if lo < value <= hi:
            return name
    return "large"


def compute_occupancy_diagnostics(seg_e_logits, seg_g, valid_g):
    """Occupancy IoU를 해석하기 위한 error 방향과 GT obstacle 비율별 통계를 계산한다."""
    pred_drivable = torch.sigmoid(seg_e_logits).round()
    pred_obstacle = 1.0 - pred_drivable
    target_obstacle = 1.0 - seg_g

    false_obstacle_num = float((pred_obstacle * seg_g * valid_g).sum().item())
    false_obstacle_den = float((seg_g * valid_g).sum().item())
    missed_obstacle_num = float((pred_drivable * target_obstacle * valid_g).sum().item())
    missed_obstacle_den = float((target_obstacle * valid_g).sum().item())
    obstacle_cells = missed_obstacle_den
    valid_cells = float(valid_g.sum().item())

    metrics = {
        "obstacle_frac": obstacle_cells / valid_cells if valid_cells > 0 else float("nan"),
        "false_obstacle": false_obstacle_num / false_obstacle_den if false_obstacle_den > 0 else float("nan"),
        "missed_obstacle": missed_obstacle_num / missed_obstacle_den if missed_obstacle_den > 0 else float("nan"),
        "false_obstacle_num": false_obstacle_num,
        "false_obstacle_den": false_obstacle_den,
        "missed_obstacle_num": missed_obstacle_num,
        "missed_obstacle_den": missed_obstacle_den,
        "obstacle_cells": obstacle_cells,
        "valid_cells": valid_cells,
        "empty_false_alarm_num": 0.0,
        "empty_false_alarm_den": 0.0,
        "empty_false_alarm_samples": 0,
    }
    for name, _, _ in OBSTACLE_FRACTION_BINS:
        metrics[f"obstacle_iou_{name}"] = float("nan")
        metrics[f"obstacle_iou_{name}_sum"] = 0.0
        metrics[f"obstacle_count_{name}"] = 0

    batch_size = seg_g.shape[0]
    for i in range(batch_size):
        valid_i = valid_g[i : i + 1]
        target_i = target_obstacle[i : i + 1]
        pred_i = pred_obstacle[i : i + 1]
        valid_count = float(valid_i.sum().item())
        if valid_count <= 0:
            continue
        frac = float((target_i * valid_i).sum().item()) / valid_count
        bin_name = _bin_name_for_obstacle_fraction(frac)
        metrics[f"obstacle_count_{bin_name}"] += 1
        if bin_name == "empty":
            # GT obstacle이 없으면 IoU는 예측과 무관하게 0이다 -- 대신 오탐량을 센다.
            false_cells = float((pred_i * valid_i).sum().item())
            metrics["empty_false_alarm_num"] += false_cells
            metrics["empty_false_alarm_den"] += valid_count
            metrics["empty_false_alarm_samples"] += int(false_cells > 0)
            continue
        metrics[f"obstacle_iou_{bin_name}_sum"] += float(compute_iou(pred_i, target_i, valid_i).item())

    for name, _, _ in OBSTACLE_FRACTION_BINS:
        count = metrics[f"obstacle_count_{name}"]
        if name != "empty" and count > 0:
            metrics[f"obstacle_iou_{name}"] = metrics[f"obstacle_iou_{name}_sum"] / count
    metrics["empty_false_alarm"] = (
        metrics["empty_false_alarm_num"] / metrics["empty_false_alarm_den"]
        if metrics["empty_false_alarm_den"] > 0
        else float("nan")
    )
    return metrics


def compute_deployment_occupancy_metrics(occ_logits, vis_logits, seg_g, valid_g):
    """배포 조건 그대로의 occupancy 성능: GT visibility가 아니라 **예측** visibility로 마스킹한다.

    개발용 `iou_obstacle`은 GT로 가려진 영역을 빼주므로 occupancy head 자체의 품질을 본다.
    실제 로봇은 GT를 못 쓰고 모델이 "보인다"고 한 영역만 신뢰하게 되므로, 가려진 곳을
    보인다고 착각하면 그 영역의 틀린 occupancy가 그대로 반영돼야 한다.
    시야를 좁게 부를수록 점수는 쉬워지니 `visible_coverage`와 항상 같이 봐야 한다.
    """
    valid = valid_g.float()
    pred_visible = (torch.sigmoid(vis_logits) > 0.5).float() * valid
    drivable_iou, obstacle_iou, obstacle_count = compute_drivable_and_obstacle_iou(
        occ_logits, seg_g, pred_visible
    )
    return {
        "iou_drivable": float(drivable_iou.item()),
        "iou_obstacle": float(obstacle_iou.item()),
        "obstacle_count": obstacle_count,
        "sample_count": int(seg_g.shape[0]),
        "claimed_visible_cells": float(pred_visible.sum().item()),
        "valid_cells": float(valid.sum().item()),
    }


def summarize_deployment_metrics(metric_dicts):
    if not metric_dicts:
        return {"iou_drivable": float("nan"), "iou_obstacle": float("nan"), "visible_coverage": float("nan")}
    claimed = sum(m["claimed_visible_cells"] for m in metric_dicts)
    valid = sum(m["valid_cells"] for m in metric_dicts)
    return {
        "iou_drivable": weighted_mean(
            [m["iou_drivable"] for m in metric_dicts], [m["sample_count"] for m in metric_dicts]
        ),
        "iou_obstacle": weighted_mean(
            [m["iou_obstacle"] for m in metric_dicts], [m["obstacle_count"] for m in metric_dicts]
        ),
        "visible_coverage": claimed / valid if valid > 0 else float("nan"),
    }


def summarize_occupancy_diagnostics(metric_dicts):
    if not metric_dicts:
        result = {
            "obstacle_frac": float("nan"),
            "false_obstacle": float("nan"),
            "missed_obstacle": float("nan"),
        }
        for name, _, _ in OBSTACLE_FRACTION_BINS:
            result[f"obstacle_iou_{name}"] = float("nan")
            result[f"obstacle_count_{name}"] = 0
        result["empty_false_alarm"] = float("nan")
        result["empty_false_alarm_samples"] = 0
        return result

    false_num = sum(m["false_obstacle_num"] for m in metric_dicts)
    false_den = sum(m["false_obstacle_den"] for m in metric_dicts)
    missed_num = sum(m["missed_obstacle_num"] for m in metric_dicts)
    missed_den = sum(m["missed_obstacle_den"] for m in metric_dicts)
    obstacle_cells = sum(m["obstacle_cells"] for m in metric_dicts)
    valid_cells = sum(m["valid_cells"] for m in metric_dicts)
    result = {
        "obstacle_frac": obstacle_cells / valid_cells if valid_cells > 0 else float("nan"),
        "false_obstacle": false_num / false_den if false_den > 0 else float("nan"),
        "missed_obstacle": missed_num / missed_den if missed_den > 0 else float("nan"),
    }
    for name, _, _ in OBSTACLE_FRACTION_BINS:
        count = sum(m[f"obstacle_count_{name}"] for m in metric_dicts)
        result[f"obstacle_count_{name}"] = count
        if name == "empty":  # IoU가 구조적으로 0인 bin -- 아래 false alarm으로 대신 본다
            result[f"obstacle_iou_{name}"] = float("nan")
            continue
        iou_sum = sum(m[f"obstacle_iou_{name}"] * m[f"obstacle_count_{name}"] for m in metric_dicts if m[f"obstacle_count_{name}"] > 0)
        result[f"obstacle_iou_{name}"] = iou_sum / count if count > 0 else float("nan")

    false_alarm_num = sum(m["empty_false_alarm_num"] for m in metric_dicts)
    false_alarm_den = sum(m["empty_false_alarm_den"] for m in metric_dicts)
    result["empty_false_alarm"] = false_alarm_num / false_alarm_den if false_alarm_den > 0 else float("nan")
    result["empty_false_alarm_samples"] = sum(m["empty_false_alarm_samples"] for m in metric_dicts)
    return result


def _format_occupancy_diag(metrics) -> str:
    if metrics is None:
        return ""
    return (
        f" | obst_frac {metrics['obstacle_frac']:.3f} | false_obstacle↓ {metrics['false_obstacle']:.3f} | "
        f"missed_obstacle↓ {metrics['missed_obstacle']:.3f}"
    )


def _format_obstacle_bin_summary(metrics) -> str:
    if metrics is None:
        return ""
    parts = []
    for name, _, _ in OBSTACLE_FRACTION_BINS:
        count = metrics[f"obstacle_count_{name}"]
        if name == "empty":  # IoU 대신 오탐률(false alarm)을 보여준다
            value_text = f"fa {metrics['empty_false_alarm']:.3f}" if count > 0 else "fa -"
        else:
            value = metrics[f"obstacle_iou_{name}"]
            value_text = f"{value:.3f}" if count > 0 else "-"
        parts.append(f"{name}:{value_text}/n{count}")
    return " | val_obst_iou_bins " + " ".join(parts)


def write_occupancy_diagnostics(writer, split: str, metrics, epoch: int) -> None:
    writer.add_scalar(f"{split}/occupancy_obstacle_fraction_epoch", metrics["obstacle_frac"], epoch)
    writer.add_scalar(f"{split}/occupancy_false_obstacle_epoch", metrics["false_obstacle"], epoch)
    writer.add_scalar(f"{split}/occupancy_missed_obstacle_epoch", metrics["missed_obstacle"], epoch)
    for name, _, _ in OBSTACLE_FRACTION_BINS:
        if name != "empty" and metrics[f"obstacle_count_{name}"] > 0:
            writer.add_scalar(f"{split}/occupancy_obstacle_iou_{name}_epoch", metrics[f"obstacle_iou_{name}"], epoch)
        writer.add_scalar(f"{split}/occupancy_obstacle_count_{name}_epoch", metrics[f"obstacle_count_{name}"], epoch)
    if metrics["obstacle_count_empty"] > 0:
        writer.add_scalar(f"{split}/occupancy_empty_false_alarm_epoch", metrics["empty_false_alarm"], epoch)
    writer.add_scalar(f"{split}/occupancy_empty_false_alarm_samples_epoch", metrics["empty_false_alarm_samples"], epoch)


def write_deployment_metrics(writer, split: str, metrics, epoch: int) -> None:
    for key in ("iou_drivable", "iou_obstacle", "visible_coverage"):
        if not math.isnan(metrics[key]):
            writer.add_scalar(f"{split}/deploy_{key}_epoch", metrics[key], epoch)


def run_batch(model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight):
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5  # nuScenes 관례: [0,1] -> [-0.5,0.5]
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    _, _, two_head_bev_e, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
    occ_bev_e, vis_bev_e = split_two_head_logits(two_head_bev_e)

    loss, loss_parts = compute_two_head_loss(
        occ_bev_e,
        vis_bev_e,
        seg_bev_g,
        vis_bev_g,
        valid_bev_g,
        lambda_vis=lambda_vis,
        vis_neg_weight=vis_neg_weight,
        occ_pos_weight=pos_weight_tensor,
    )
    # occupancy는 "라벨이 있고(valid) 실제로 보이는(vis)" 셀에서만 평가한다 -- loss의 w_occ와 동일.
    occ_eval_mask = vis_bev_g * valid_bev_g
    drivable_iou, obstacle_iou, obstacle_count = compute_drivable_and_obstacle_iou(
        occ_bev_e, seg_bev_g, occ_eval_mask
    )
    vis_metrics = visibility_error_rates(torch.sigmoid(vis_bev_e), vis_bev_g, valid_bev_g)
    occ_metrics = compute_occupancy_diagnostics(occ_bev_e, seg_bev_g, occ_eval_mask)
    deploy_metrics = compute_deployment_occupancy_metrics(occ_bev_e, vis_bev_e, seg_bev_g, valid_bev_g)
    return (
        loss,
        loss_parts,
        drivable_iou,
        obstacle_iou,
        obstacle_count,
        vis_metrics,
        occ_metrics,
        deploy_metrics,
    )


def main(
    exp_name="debug",
    num_epochs=60,
    batch_size=4,
    lr=3e-4,
    weight_decay=1e-7,
    num_workers=8,
    val_fraction=0.1,
    split_seed=0,
    encoder_type="res101",
    use_fisheye=True,
    augment=False,
    pos_weight=None,
    lambda_vis=0.5,
    vis_neg_weight=3.0,
    val_freq_epochs=1,
    save_freq_epochs=5,
    log_dir="work_dirs/logs_synwoodscape",
    ckpt_dir="work_dirs/checkpoints_synwoodscape",
    max_samples=None,
    device="cuda",
):
    torch.manual_seed(0)
    np.random.seed(0)

    all_ids = discover_all_sample_ids(DEFAULT_DATASET_ROOT)
    train_ids, val_ids = train_val_split(
        all_ids, DEFAULT_DATASET_ROOT, val_fraction=val_fraction, seed=split_seed
    )
    if max_samples is not None:  # 빠른 smoke run 용 -- 실제 학습에는 쓰지 않는다
        train_ids, val_ids = train_ids[:max_samples], val_ids[: max(1, max_samples // 4)]

    if pos_weight is None:
        pos_weight = compute_pos_weight(train_ids, DEFAULT_OCCUPANCY_GT_ROOT)
    trivial_iou = compute_trivial_baseline_iou(val_ids, DEFAULT_OCCUPANCY_GT_ROOT)

    _print_banner([
        " SynWoodScape -> Simple-BEV training",
        f" exp_name={exp_name} | encoder={encoder_type} | fisheye={use_fisheye}",
        f" batch_size={batch_size} | lr={lr:.0e} | epochs={num_epochs}",
        f" train={len(train_ids)} | val={len(val_ids)}",
        f" pos_weight (neg/pos on train) = {pos_weight:.3f}",
        f" lambda_vis={lambda_vis:.3f} | vis_neg_weight={vis_neg_weight:.3f}",
        f" photometric augment (train only) = {bool(augment)}",
        f" trivial 'always predict drivable' baseline IoU on val = {trivial_iou:.3f}  <- compare against this",
    ])

    train_ds = SynWoodScapeSimpleBEVDataset(train_ids, augment=augment)
    val_ds = SynWoodScapeSimpleBEVDataset(val_ids)  # val은 항상 원본 -- 증강하면 비교가 흔들린다
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    if use_fisheye:
        cameras = [
            load_camera(DEFAULT_DATASET_ROOT / "calibration_data" / f"{name}.json")
            for name in CAMERA_NAMES
        ]
        vox_util = build_fisheye_vox_util(GRID_SPEC, cameras, device=device)
    else:
        vox_util = build_vox_util(GRID_SPEC, device=device)

    # rand_flip=False로 고정한다: Simple-BEV의 forward/backward(Z축) flip 증강은 대칭 grid를
    # 전제하는데 SynWoodScape pretrain grid는 전후 비대칭이라 물리적으로 성립하지 않는다.
    model = TwoHeadSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_epochs * steps_per_epoch + 10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear",
    )
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    # 타임스탬프를 반드시 넣는다: 이게 없으면 같은 exp_name으로 재실행할 때 global_step(epoch)이
    # 1부터 다시 시작하면서 이전 실행의 체크포인트(model-000000001.pth 등)를 그대로 덮어쓰고,
    # keep_latest 정리 로직이 이전 실행분을 지워버린다 (model_best는 keep_latest=1이라
    # 새 실행의 첫 저장에서 즉시 삭제됨). tensorboard 로그는 파일 자체가 지워지진 않지만
    # 같은 폴더에 섞여 들어가 epoch 축이 겹쳐 보인다.
    run_name = f"{exp_name}_{encoder_type}_bs{batch_size}_lr{lr:.0e}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    log_path = Path(log_dir) / run_name
    writer = SummaryWriter(str(log_path))
    ckpt_path = Path(ckpt_dir) / run_name

    # 이번 실행에 실제로 쓰인 train/val sample id를 파일로 남긴다 -- split은 폴더 구조가
    # 아니라 코드(synwoodscape_split.py)로 계산되므로, 이 파일이 없으면 어떤 이미지가
    # train/val인지 육안으로 확인할 방법이 없다.
    # log_path(텐서보드 폴더)에 쓴다 -- ckpt_path에 쓰면 saverloader.load()가 그 폴더의
    # 파일 전체를 "{model_name}-{step}.pth" 패턴으로 가정하고 os.listdir 하다가
    # split_*.txt에서 IndexError로 죽는다(재현 확인함).
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "split_train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (log_path / "split_val_ids.txt").write_text("\n".join(val_ids) + "\n")
    print(f"train/val sample id lists saved to: {log_path}/split_{{train,val}}_ids.txt")

    global_step = 0
    best_val_score = 0.0  # (drivable_iou + obstacle_iou)/2 -- 트리비얼 해로는 못 올라간다
    interrupted = False
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_start = time.time()
            train_losses, train_occ_losses, train_vis_losses = [], [], []
            train_d_ious, train_o_ious, train_v_false_highs, train_v_false_lows = [], [], [], []
            train_o_iou_counts = []
            train_occ_metric_dicts, train_deploy_metric_dicts = [], []
            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_parts, d_iou, o_iou, o_count, vis_metrics, occ_metrics, deploy_metrics = run_batch(
                    model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                train_occ_losses.append(loss_parts["loss_occ"].item())
                train_vis_losses.append(loss_parts["loss_vis"].item())
                train_d_ious.append(d_iou.item())
                train_o_ious.append(o_iou.item())
                train_o_iou_counts.append(o_count)
                train_v_false_highs.append(vis_metrics["false_high"])
                train_v_false_lows.append(vis_metrics["false_low"])
                train_occ_metric_dicts.append(occ_metrics)
                train_deploy_metric_dicts.append(deploy_metrics)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/loss_occ_step", loss_parts["loss_occ"].item(), global_step)
                writer.add_scalar("train/loss_vis_step", loss_parts["loss_vis"].item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_occ_loss = float(np.mean(train_occ_losses)) if train_occ_losses else float("nan")
            train_vis_loss = float(np.mean(train_vis_losses)) if train_vis_losses else float("nan")
            train_d_iou = float(np.mean(train_d_ious)) if train_d_ious else float("nan")
            train_o_iou = weighted_mean(train_o_ious, train_o_iou_counts)
            train_v_false_high = float(np.mean(train_v_false_highs)) if train_v_false_highs else float("nan")
            train_v_false_low = float(np.mean(train_v_false_lows)) if train_v_false_lows else float("nan")
            train_occ_metrics = summarize_occupancy_diagnostics(train_occ_metric_dicts)
            train_deploy_metrics = summarize_deployment_metrics(train_deploy_metric_dicts)
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/loss_occ_epoch", train_occ_loss, epoch)
            writer.add_scalar("train/loss_vis_epoch", train_vis_loss, epoch)
            writer.add_scalar("train/iou_drivable_epoch", train_d_iou, epoch)
            writer.add_scalar("train/iou_obstacle_epoch", train_o_iou, epoch)
            writer.add_scalar("train/visibility_false_high_epoch", train_v_false_high, epoch)
            writer.add_scalar("train/visibility_false_low_epoch", train_v_false_low, epoch)
            write_occupancy_diagnostics(writer, "train", train_occ_metrics, epoch)
            write_deployment_metrics(writer, "train", train_deploy_metrics, epoch)

            val_loss = val_occ_loss = val_vis_loss = val_d_iou = val_o_iou = float("nan")
            val_v_false_high = val_v_false_low = float("nan")
            val_occ_metrics = summarize_occupancy_diagnostics([])
            val_deploy_metrics = summarize_deployment_metrics([])
            if epoch % val_freq_epochs == 0 and len(val_loader) > 0:
                model.eval()
                val_losses, val_occ_losses, val_vis_losses = [], [], []
                val_d_ious, val_o_ious, val_v_false_highs, val_v_false_lows = [], [], [], []
                val_o_iou_counts = []
                val_occ_metric_dicts, val_deploy_metric_dicts = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        loss, loss_parts, d_iou, o_iou, o_count, vis_metrics, occ_metrics, deploy_metrics = run_batch(
                            model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
                        )
                        val_losses.append(loss.item())
                        val_occ_losses.append(loss_parts["loss_occ"].item())
                        val_vis_losses.append(loss_parts["loss_vis"].item())
                        val_d_ious.append(d_iou.item())
                        val_o_ious.append(o_iou.item())
                        val_o_iou_counts.append(o_count)
                        val_v_false_highs.append(vis_metrics["false_high"])
                        val_v_false_lows.append(vis_metrics["false_low"])
                        val_occ_metric_dicts.append(occ_metrics)
                        val_deploy_metric_dicts.append(deploy_metrics)
                val_loss = float(np.mean(val_losses))
                val_occ_loss = float(np.mean(val_occ_losses))
                val_vis_loss = float(np.mean(val_vis_losses))
                val_d_iou = float(np.mean(val_d_ious))
                val_o_iou = weighted_mean(val_o_ious, val_o_iou_counts)
                val_v_false_high = float(np.mean(val_v_false_highs))
                val_v_false_low = float(np.mean(val_v_false_lows))
                val_occ_metrics = summarize_occupancy_diagnostics(val_occ_metric_dicts)
                val_deploy_metrics = summarize_deployment_metrics(val_deploy_metric_dicts)
                writer.add_scalar("val/loss_epoch", val_loss, epoch)
                writer.add_scalar("val/loss_occ_epoch", val_occ_loss, epoch)
                writer.add_scalar("val/loss_vis_epoch", val_vis_loss, epoch)
                writer.add_scalar("val/iou_drivable_epoch", val_d_iou, epoch)
                writer.add_scalar("val/iou_obstacle_epoch", val_o_iou, epoch)
                writer.add_scalar("val/visibility_false_high_epoch", val_v_false_high, epoch)
                writer.add_scalar("val/visibility_false_low_epoch", val_v_false_low, epoch)
                write_occupancy_diagnostics(writer, "val", val_occ_metrics, epoch)
                write_deployment_metrics(writer, "val", val_deploy_metrics, epoch)

            epoch_time = time.time() - epoch_start
            val_score = 0.5 * (val_d_iou + val_o_iou)
            is_new_best = val_score > best_val_score  # NaN > x is always False -- val을 안 돌린 epoch은 자동으로 제외됨

            print(format_epoch_log(
                epoch=epoch,
                num_epochs=num_epochs,
                epoch_time=epoch_time,
                train_loss=train_loss,
                train_occ_loss=train_occ_loss,
                train_vis_loss=train_vis_loss,
                train_d_iou=train_d_iou,
                train_o_iou=train_o_iou,
                train_v_false_high=train_v_false_high,
                train_v_false_low=train_v_false_low,
                train_occ_metrics=train_occ_metrics,
                val_loss=val_loss,
                val_occ_loss=val_occ_loss,
                val_vis_loss=val_vis_loss,
                val_d_iou=val_d_iou,
                val_o_iou=val_o_iou,
                val_v_false_high=val_v_false_high,
                val_v_false_low=val_v_false_low,
                val_occ_metrics=val_occ_metrics,
                val_deploy_metrics=val_deploy_metrics,
                val_score=val_score,
                best_val_score=best_val_score,
                is_new_best=is_new_best,
            ))

            if epoch % save_freq_epochs == 0 or epoch == num_epochs:
                saverloader.save(str(ckpt_path), optimizer, model, epoch, keep_latest=3)
            if is_new_best:
                best_val_score = val_score
                saverloader.save(str(ckpt_path), optimizer, model, epoch, keep_latest=1, model_name="model_best")
    except KeyboardInterrupt:
        # Ctrl+C는 정상적인 중단 방법이다 -- 지금까지 저장된 체크포인트는 그대로 안전하게
        # 남아 있다(에폭이 끝난 시점에만 저장하므로 반쪽짜리 체크포인트는 생기지 않는다).
        # 다만 finally 없이 여기서 그냥 죽으면 writer.close()가 안 불려서 마지막 몇 개
        # tensorboard scalar가(기본 flush_secs만큼) 디스크에 안 쓰인 채 유실될 수 있다.
        interrupted = True
        print("\n" + _c(_Ansi.YELLOW + _Ansi.BOLD, "[interrupted] Training stopped by Ctrl+C. Checkpoints/logs saved so far are safe."))
    finally:
        writer.close()

    if not interrupted:
        _print_banner([
            " done.",
            f" best val (drivable+obstacle)/2 IoU = {best_val_score:.3f}",
            f" trivial 'always predict drivable' baseline was drivable IoU {trivial_iou:.3f}, obstacle IoU 0.0",
        ])
    else:
        print(f"best val (drivable+obstacle)/2 IoU so far = {best_val_score:.3f}")


if __name__ == "__main__":
    Fire(main)
