"""반복 실험을 한 표로 집계한다 -- **σ를 측정값으로 만드는 도구다.**

왜 필요한가: 2026-08-21까지 시드가 `train_robot_bev.py`에 0으로 하드코딩돼 있어 같은 config를
두 번 돌리면 같은 결과가 나왔고, **시드 분산을 한 번도 재지 못했다.** 그 상태로 정식화 비교
(+0.0024), 분할 변경(−0.0034), label smoothing(−0.0035) 같은 판정을 전부 "노이즈 대역(0.02)
안"으로 내렸는데 그 0.02는 실측이 아니라 프로젝트 규약이었다. 이 도구가 그 규약을 숫자로
바꾼다. **σ보다 작은 차이는 주장하지 않는다**는 규칙을 쓰려면 σ가 먼저 있어야 한다.

두 가지를 **반드시 함께** 낸다.

- **고정 epoch 값** (`--fixed_epoch`): 사전에 선언한 epoch의 값이라 **선택이 개입하지 않는다.**
  논문의 주 숫자는 이쪽이어야 한다.
- **best epoch 값**: 체크포인트 선택 기준(`iou_free`)의 최대값. 실제로 저장되는 체크포인트가
  내는 값이지만 **max 연산이라 위로 편향된다** -- 반복 수가 늘면 더 커진다.

둘을 나란히 두면 그 편향의 크기가 보인다(실측 plateau 폭이 0.003~0.008이었다).

실행:
    python tools/summarize_repeats.py --log_root=runs/robot_bev_cv/seeds/logs
    python tools/summarize_repeats.py --log_root=runs/robot_bev_cv/loso/logs --group_by=val_sequences
"""
import json
import math
import statistics
import sys
from pathlib import Path

from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

SELECTION_TAG = "val/iou_free_epoch"

# 표에 싣는 지표와 방향. **`iou_free`가 첫 줄이어야 한다** -- 체크포인트 선택 기준이고
# 나머지는 그 선택 아래에서 읽히는 값이다.
REPORTED = (
    ("val/iou_free_epoch", "iou_free", True),
    ("val/fatal_rate_epoch", "fatal_rate", False),
    ("val/free_miss_rate_epoch", "free_miss_rate", False),
    ("val/occupied_f1_10cm_epoch", "f1@10cm", True),
    ("val/occupied_f1_20cm_epoch", "f1@20cm", True),
    ("val/range_mae_epoch", "range_mae", False),
    ("val/range_bias_epoch", "range_bias", None),
    ("val/range_missed_obstacle_rate_epoch", "missed_obstacle", False),
    ("val/loss_epoch", "val_loss", False),
    ("train/iou_free_epoch", "train_iou_free", True),
)


def read_run(run_dir) -> dict:
    """한 런의 scalar 전체와 `config.json`(있으면)을 읽는다."""
    run_dir = Path(run_dir)
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    series = {
        tag: {event.step: event.value for event in acc.Scalars(tag)}
        for tag in acc.Tags()["scalars"]
    }
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {"name": run_dir.name, "config": config, "series": series}


def pick_epochs(series, fixed_epoch=None) -> dict:
    """`{"fixed": epoch, "best": epoch}`. 선택 기준은 `iou_free` 하나다.

    `fixed_epoch`를 주지 않으면 **그 런이 실제로 도달한 마지막 epoch**을 쓴다. 중간에 죽은
    런을 조용히 짧은 값으로 집계하지 않으려면 호출부가 epoch 수를 함께 확인해야 하므로,
    `last_epoch`도 돌려준다.
    """
    selection = series.get(SELECTION_TAG, {})
    if not selection:
        return {"fixed": None, "best": None, "last": None}
    last = max(selection)
    fixed = last if fixed_epoch is None else fixed_epoch
    return {
        "fixed": fixed if fixed in selection else None,
        "best": max(selection, key=selection.get),
        "last": last,
    }


def _mean_sd(values) -> tuple:
    """(평균, 표본표준편차, n). n<2면 sd는 NaN -- 0으로 내놓으면 "분산이 없다"로 오독된다."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def aggregate(runs, fixed_epoch=None) -> dict:
    """런 목록 -> 지표별 `{fixed: (평균, sd, n), best: (...)}`."""
    picks = [pick_epochs(run["series"], fixed_epoch) for run in runs]
    result = {}
    for tag, label, _ in REPORTED:
        columns = {}
        for which in ("fixed", "best"):
            values = [
                run["series"].get(tag, {}).get(pick[which])
                if pick[which] is not None else None
                for run, pick in zip(runs, picks)
            ]
            columns[which] = _mean_sd(values)
        result[label] = columns
    result["_epochs"] = {
        "fixed": [p["fixed"] for p in picks],
        "best": [p["best"] for p in picks],
        "last": [p["last"] for p in picks],
    }
    return result


def group_runs(runs, group_by=None) -> dict:
    """`group_by`가 `config.json`의 키면 그 값으로 묶는다. 없으면 전체를 한 그룹으로 둔다."""
    if not group_by:
        return {"all": list(runs)}
    grouped = {}
    for run in runs:
        key = run["config"].get(group_by)
        grouped.setdefault("(config 없음)" if key is None else str(key), []).append(run)
    return grouped


def _format_value(mean, sd, n) -> str:
    if n == 0:
        return "     n/a"
    if n == 1:
        return f"{mean:.4f}      "
    return f"{mean:.4f}±{sd:.4f}"


def format_table(label, runs, summary) -> str:
    lines = [f"\n=== {label}  ({len(runs)} 런) ==="]
    epochs = summary["_epochs"]
    lines.append(f"  fixed epoch {epochs['fixed']}   best epoch {epochs['best']}"
                 f"   last epoch {epochs['last']}")
    if any(e is None for e in epochs["fixed"]):
        lines.append("  [경고] fixed epoch에 값이 없는 런이 있다 -- 중간에 죽었거나"
                     " num_epochs가 다르다")
    lines.append(f"  {'지표':<16} {'@fixed':>16} {'@best':>16}   {'best−fixed':>11}")
    for _, name, higher in REPORTED:
        fixed_mean, fixed_sd, fixed_n = summary[name]["fixed"]
        best_mean, best_sd, best_n = summary[name]["best"]
        arrow = "" if higher is None else ("↑" if higher else "↓")
        delta = (f"{best_mean - fixed_mean:+.4f}"
                 if fixed_n and best_n else "n/a")
        lines.append(f"  {name + arrow:<16} {_format_value(fixed_mean, fixed_sd, fixed_n):>16}"
                     f" {_format_value(best_mean, best_sd, best_n):>16}   {delta:>11}")
    return "\n".join(lines)


def format_per_run(runs, fixed_epoch=None) -> str:
    """런별 원자료. 평균만 보고 넘어가면 한 런이 죽어 있는 것을 놓친다."""
    lines = ["\n=== 런별 `iou_free` ==="]
    lines.append(f"  {'런':<52} {'seed':>5} {'@fixed':>8} {'@best':>8} {'best ep':>8}")
    for run in runs:
        pick = pick_epochs(run["series"], fixed_epoch)
        selection = run["series"].get(SELECTION_TAG, {})
        fixed = selection.get(pick["fixed"])
        best = selection.get(pick["best"])
        seed = run["config"].get("seed", "?")
        lines.append(
            f"  {run['name'][:52]:<52} {str(seed):>5}"
            f" {'n/a' if fixed is None else f'{fixed:.4f}':>8}"
            f" {'n/a' if best is None else f'{best:.4f}':>8}"
            f" {str(pick['best']):>8}"
        )
    return "\n".join(lines)


def main(log_root, fixed_epoch=None, group_by=None, pattern="*"):
    """`log_root` 아래의 런 폴더들을 집계한다.

    `group_by`에 `config.json`의 키(예: `val_sequences`, `label_smoothing`)를 주면 그 값으로
    묶어 표를 따로 낸다 -- LOSO fold별, 또는 손잡이별 대조에 쓴다.
    """
    log_root = Path(log_root)
    if not log_root.is_dir():
        raise FileNotFoundError(f"로그 루트가 없다: {log_root}")
    run_dirs = sorted(d for d in log_root.glob(pattern) if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"런 폴더가 없다: {log_root}/{pattern}")

    runs = [read_run(d) for d in run_dirs]
    alive = [r for r in runs if r["series"].get(SELECTION_TAG)]
    if len(alive) != len(runs):
        print(f"[경고] scalar가 없는 런 {len(runs) - len(alive)}개를 건너뛴다")

    for label, group in group_runs(alive, group_by).items():
        print(format_table(label, group, aggregate(group, fixed_epoch)))
    print(format_per_run(alive, fixed_epoch))


if __name__ == "__main__":
    Fire(main)
