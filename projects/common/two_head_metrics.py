"""two-head BEV 학습의 데이터셋 비의존 지표·로깅 -- SynWoodScape pretrain과 자체 데이터셋
fine-tuning이 공유한다.

원래는 `tools/train_synwoodscape.py` 안에 있었다. 자체 데이터셋 fine-tuning 스크립트가
같은 지표를 그대로 써야 하는데(그래야 pretrain과 fine-tune 숫자를 나란히 읽을 수 있다),
스크립트끼리 import하면 pretrain 스크립트를 손댈 때 fine-tune이 같이 깨진다. 여기 옮겨서
양쪽이 대등하게 의존하도록 한다.

여기 있는 것은 전부 `(logits, target, mask)` 텐서만 받는다 -- 데이터셋 경로나 라벨 파일
규약에 의존하는 것(`compute_pos_weight` 등)은 각 스크립트에 남는다.
"""
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "third_party/models/simple_bev") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.models.simplebev_two_head import (  # noqa: E402
    compute_two_head_loss,
    split_two_head_logits,
    visibility_error_rates,
)


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

