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
import os
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
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import free_metrics_from_masks  # noqa: E402


class _Ansi:
    """터미널 색 강조용 최소 ANSI 코드 -- 별도 의존성(rich/colorama) 없이 표준 출력에 직접 씀."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _color_enabled() -> bool:
    """색은 **터미널로 직접 출력할 때만** 켠다.

    `... > train.log`나 `| tee`로 로그를 남기는 경우가 많은데, 그 파일에 escape sequence가
    섞이면 나중에 grep/diff가 지저분해진다. 관례대로 `NO_COLOR`로 끄고 `FORCE_COLOR`로
    강제할 수 있다(파이프로 넘기면서 색을 보고 싶을 때: `FORCE_COLOR=1 ... | less -R`).
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _c(color: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"{color}{text}{_Ansi.RESET}"


def _print_banner(lines) -> None:
    """제목과 `<-`로 표시된 줄만 강조한다 -- 전부 같은 색으로 칠하면 강조가 아니라 벽이 된다.

    `<-`는 두 학습 스크립트 모두 "이 숫자와 비교하라"는 뜻으로 쓰는 표시다.
    """
    rule = "=" * 60
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))
    for i, line in enumerate(lines):
        if i == 0:
            print(_c(_Ansi.BOLD + _Ansi.CYAN, line))
        elif "<-" in line:
            print(_c(_Ansi.BOLD + _Ansi.YELLOW, line))
        else:
            print(line)
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))


def _sep() -> str:
    return _c(_Ansi.DIM, " | ")


def _field(label: str, value, fmt: str = ".3f", *, emphasis: str = "") -> str:
    """`라벨(흐리게) 값(또렷하게)`. 라벨은 매 epoch 똑같이 반복되는 부분이라 눈이 값만
    따라가면 되도록 흐리게 깐다. NaN은 숫자처럼 안 보이게 `-`로 눕힌다."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _c(_Ansi.DIM, f"{label} -")
    return f"{_c(_Ansi.DIM, label)} {_c(emphasis, format(value, fmt)) if emphasis else format(value, fmt)}"


def _format_row(tag: str, tag_color: str, fields) -> str:
    return f"  {_c(_Ansi.BOLD + tag_color, tag.ljust(6))}{_c(_Ansi.DIM, '|')} " + _sep().join(fields)


def _format_metric_row(tag, tag_color, loss, occ_loss, vis_loss, d_iou, o_iou,
                       v_false_high, v_false_low, occ_metrics, free_metrics=None):
    """train/val 한 줄. `iou_free`가 주 지표이고, 기존 IoU는 해석용 참고 지표다."""
    fields = []
    if free_metrics is not None:
        fields += [
            _field("iou_free↑", free_metrics["iou_free"], emphasis=_Ansi.BOLD),
            _field("fatal↓", free_metrics["fatal_rate"]),
            _field("free_miss↓", free_metrics["free_miss_rate"]),
        ]
    fields += [
        _field("loss_total↓", loss, ".4f"),
        _field("loss_occ↓", occ_loss, ".4f"),
        _field("loss_vis↓", vis_loss, ".4f"),
        ((_c(_Ansi.DIM, "(ref)") + " ") if free_metrics is not None else "")
        + _field("iou_drivable↑", d_iou),
        _field("iou_obstacle↑", o_iou),
        _field("vis_false_high↓", v_false_high),
        _field("vis_false_low↓", v_false_low),
    ]
    if occ_metrics is not None:
        fields += [
            _field("obst_frac", occ_metrics["obstacle_frac"]),
            _field("false_obstacle↓", occ_metrics["false_obstacle"]),
            _field("missed_obstacle↓", occ_metrics["missed_obstacle"]),
        ]
    return _format_row(tag, tag_color, fields)


def _format_range_line(metrics):
    """광선 통계는 train/val 행을 과도하게 넓히지 않도록 별도 줄에 표시한다."""
    if metrics is None:
        return []
    fields = [
        _field("abs_p50↓", metrics["abs_p50"], emphasis=_Ansi.BOLD),
        _field("abs_p90↓", metrics["abs_p90"]),
        _field("over↓", metrics["over_mean"]),
        _field("under", metrics["under_mean"]),
        _c(_Ansi.DIM, f"rays {metrics['n_paired_rays']} censored {metrics['censored_gt']}"),
    ]
    return [_format_row("range", _Ansi.BLUE, fields)]


def _format_partition_warning(train_free, val_free):
    """free/occupied/unknown 분할이 valid를 덮지 못하면 epoch 로그에서 즉시 경고한다."""
    defects = (train_free or {}).get("partition_defects", 0) + \
              (val_free or {}).get("partition_defects", 0)
    if defects == 0:
        return []
    return [_c(_Ansi.BOLD + _Ansi.RED,
               f"  [BUG] partition defect on {defects} cells"
               " -- free/occupied/unknown does not cover valid exactly")]


def _format_deployment_line(metrics):
    """예측 visibility로 마스킹한 occupancy 성능. coverage 없이 IoU만 보면 안 된다 --
    모델이 시야를 좁게 부를수록 IoU는 쉬워지기 때문이다.
    """
    if metrics is None:
        return []
    fields = [
        _c(_Ansi.DIM, "(pred visibility 기준)") + " "
        + _field("iou_drivable↑", metrics["iou_drivable"], emphasis=_Ansi.BOLD),
        _field("iou_obstacle↑", metrics["iou_obstacle"], emphasis=_Ansi.BOLD),
        _field("visible_coverage", metrics["visible_coverage"]),
    ]
    return [_format_row("deploy", _Ansi.MAGENTA, fields)]


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
    train_free_metrics=None,
    val_deploy_metrics=None,
    val_loss,
    val_occ_loss,
    val_vis_loss,
    val_d_iou,
    val_o_iou,
    val_v_false_high,
    val_v_false_low,
    val_occ_metrics=None,
    val_free_metrics=None,
    val_range_metrics=None,
    baseline_iou_free=None,
    val_score,
    best_val_score,
    is_new_best,
):
    displayed_best = val_score if is_new_best else best_val_score
    # 직전 best 대비 증감. "이번 epoch이 나아졌나"를 숫자 두 개를 눈으로 빼지 않고 보려는 것.
    # best의 초기값이 0.0이라 첫 val epoch에서는 델타가 곧 점수라 무의미하므로 생략한다.
    delta = ""
    if best_val_score > 0 and not math.isnan(val_score):
        diff = val_score - best_val_score
        delta = " " + _c(_Ansi.GREEN if diff > 0 else _Ansi.RED, f"({diff:+.3f})")
    baseline_delta = ""
    val_free_iou = (val_free_metrics or {}).get("iou_free")
    if baseline_iou_free is not None and val_free_iou is not None and not math.isnan(val_free_iou):
        gap = val_free_iou - baseline_iou_free
        baseline_delta = " " + _c(
            _Ansi.GREEN if gap > 0 else _Ansi.RED, f"({gap:+.3f} vs baseline)"
        )
    return "\n".join([
        (
            _c(_Ansi.BOLD + _Ansi.CYAN, f"epoch {epoch:03d}/{num_epochs}") + _sep()
            + _c(_Ansi.DIM, f"time {epoch_time:6.1f}s") + _sep()
            + _field("val_iou_free↑", val_score, emphasis=_Ansi.BOLD) + delta + baseline_delta + _sep()
            + _field("best_val_iou_free↑", displayed_best) + _sep()
            + (_c(_Ansi.BOLD + _Ansi.GREEN, "checkpoint: new best") if is_new_best
               else _c(_Ansi.DIM, "checkpoint: -"))
        ),
        *_format_partition_warning(train_free_metrics, val_free_metrics),
        *_format_obstacle_bin_summary(val_occ_metrics),
        _format_metric_row(
            "train", _Ansi.YELLOW, train_loss, train_occ_loss, train_vis_loss,
            train_d_iou, train_o_iou, train_v_false_high, train_v_false_low,
            train_occ_metrics, train_free_metrics,
        ),
        _format_metric_row(
            "val", _Ansi.CYAN, val_loss, val_occ_loss, val_vis_loss,
            val_d_iou, val_o_iou, val_v_false_high, val_v_false_low,
            val_occ_metrics, val_free_metrics,
        ),
        *_format_range_line(val_range_metrics),
        *_format_deployment_line(val_deploy_metrics),
    ])


def select_checkpoint_score(*, d_iou, o_iou, free_metrics):
    """Return the checkpoint ranking metric, independent of legacy occupancy IoUs.

    ``d_iou`` and ``o_iou`` remain explicit inputs at the selection boundary so callers
    cannot silently substitute their historical mean for the free-space objective.
    """
    del d_iou, o_iou
    return free_metrics["iou_free"]


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


def compute_free_metrics(occ_logits, vis_logits, seg_g, vis_g, valid_g) -> dict:
    """2-head 출력 -> free-space 지표.

    `free`는 두 head의 결합 결과다. 지금까지의 지표는 두 head를 따로 채점해서, 정작
    로봇이 쓰는 결합 결과를 아무도 보지 않았다.

    집계 자체는 `free_metrics_from_masks`에 있다 -- 3-class 경로와 같은 집계기를 써야
    Phase 3의 A/B가 공정하다. 여기서 하는 일은 예측을 free 마스크로 바꾸는 것뿐이다.
    """
    gt = decompose(seg_g, vis_g, valid_g)
    pred = decompose(torch.sigmoid(occ_logits), torch.sigmoid(vis_logits), valid_g)
    return free_metrics_from_masks(pred["free"], gt, valid_g)


_FREE_METRIC_SCALAR_KEYS = (
    "iou_free", "iou_free_count",
    "fatal_rate", "fatal_denom",
    "free_miss_rate", "free_miss_denom",
    "partition_defects",
)


def append_free_metrics(metric_dicts, free_metrics) -> None:
    """epoch 집계에는 스칼라만 보관한다; free mask는 val 배치에서 즉시 소비한다."""
    metric_dicts.append({key: free_metrics[key] for key in _FREE_METRIC_SCALAR_KEYS})


def summarize_free_metrics(dicts) -> dict:
    if not dicts:
        return {"iou_free": float("nan"), "fatal_rate": float("nan"),
                "free_miss_rate": float("nan"), "partition_defects": 0}
    return {
        "iou_free": weighted_mean([d["iou_free"] for d in dicts],
                                  [d["iou_free_count"] for d in dicts]),
        "fatal_rate": weighted_mean([d["fatal_rate"] for d in dicts],
                                    [d["fatal_denom"] for d in dicts]),
        "free_miss_rate": weighted_mean([d["free_miss_rate"] for d in dicts],
                                        [d["free_miss_denom"] for d in dicts]),
        "partition_defects": sum(d["partition_defects"] for d in dicts),
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


def _format_obstacle_bin_summary(metrics):
    """GT obstacle 비율 bin별 val IoU. epoch 헤더에 붙이면 헤더가 터미널 폭을 넘겨 줄바꿈되고,
    그러면 정작 중요한 val_iou_free가 묻힌다 -- 별도 줄로 뺀다. 샘플이 없는 bin(n0)은 흐리게.
    """
    if metrics is None:
        return []
    parts = []
    for name, _, _ in OBSTACLE_FRACTION_BINS:
        count = metrics[f"obstacle_count_{name}"]
        if name == "empty":  # IoU 대신 오탐률(false alarm)을 보여준다
            value_text = f"fa {metrics['empty_false_alarm']:.3f}" if count > 0 else "fa -"
        else:
            value = metrics[f"obstacle_iou_{name}"]
            value_text = f"{value:.3f}" if count > 0 else "-"
        text = f"{name}:{value_text}/n{count}"
        parts.append(_c(_Ansi.DIM, text) if count == 0 else text)
    return [_format_row("bins", _Ansi.BLUE, [_c(_Ansi.DIM, "val_obst_iou_bins") + " " + " ".join(parts)])]


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
    free_metrics = compute_free_metrics(occ_bev_e, vis_bev_e, seg_bev_g, vis_bev_g, valid_bev_g)
    return (
        loss,
        loss_parts,
        drivable_iou,
        obstacle_iou,
        obstacle_count,
        vis_metrics,
        occ_metrics,
        deploy_metrics,
        free_metrics,
    )
