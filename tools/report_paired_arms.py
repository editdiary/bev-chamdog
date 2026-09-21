"""대조군과 각 팔을 **시드끼리 짝지어** 비교한다 -- 캠페인 `02_projection_and_prior`.

**왜 짝짓나.** 이 캠페인은 대조군을 다시 돌리지 않고 `01_overall`의 고정 split 5런을
재사용한다(원장 §6.0). 같은 시드끼리 빼면 초기화·데이터 순서에서 오는 런 간 산포
(σ_seed ≈ 0.0018)가 차이에서 상쇄되므로, 평균만 비교할 때보다 훨씬 작은 효과를 볼 수 있다.

**판정 규약**(원장 §6.4).

- 주 지표는 `iou_free`와 constant-map baseline 대비 **margin**이다. 대조군과 팔이 같은
  val split을 보므로 baseline이 같고, margin의 차이는 `iou_free`의 차이와 정확히 같다 --
  그래도 함께 싣는다. 절대값만으로는 통로 폭에 좌우되는 수치라 읽을 수 없기 때문이다.
- **σ_seed보다 작은 평균 차이는 우열로 쓰지 않는다.**
- n=5에서 p값을 만들지 않는다. 대신 **평균 차이·차이의 표준편차·부호가 일치한 시드 수**를
  싣는다. 5개 중 5개가 같은 방향이면 그 자체가 읽을 만한 증거이고, 3/5이면 아니다.

실행:
    python tools/report_paired_arms.py \
        --control=runs/paper_final/01_overall/fixed_split/logs \
        --arms=source_prior=runs/paper_final/02_projection_and_prior/source_prior/logs,\
pinhole120=runs/paper_final/02_projection_and_prior/pinhole120/logs
"""
import json
import math
import statistics
import sys
from pathlib import Path

from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.summarize_repeats import REPORTED, pick_epochs, read_run  # noqa: E402

# `01_overall`의 고정 split constant-map baseline. 같은 val split이므로 모든 팔이 공유한다.
# 출처: `runs/paper_final/01_overall/fixed_split/analysis/constant_map_baseline.json`.
_BASELINE_PATH = ("runs/paper_final/01_overall/fixed_split/analysis/"
                  "constant_map_baseline.json")
# `01_overall`에서 실측한 시드 간 산포. 이보다 작은 차이는 우열로 쓰지 않는다.
SIGMA_SEED = 0.0018


def _seed_of(name: str):
    """런 이름 끝의 `_s{N}`. 없으면 None."""
    tail = name.rsplit("_s", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def load_arm(log_root, fixed_epoch):
    """`{seed: {metric: value}}`. 해당 epoch에 도달하지 못한 런은 값이 None이 된다."""
    log_root = Path(log_root)
    if not log_root.exists():
        raise SystemExit(f"로그 디렉터리가 없다: {log_root}")
    out = {}
    for run_dir in sorted(d for d in log_root.iterdir() if d.is_dir()):
        seed = _seed_of(run_dir.name)
        if seed is None:
            continue
        run = read_run(run_dir)
        pick = pick_epochs(run["series"], fixed_epoch)["fixed"]
        out[seed] = {
            "_run": run_dir.name,
            "_epoch": pick,
            **{label: (run["series"].get(tag, {}).get(pick) if pick is not None else None)
               for tag, label, _ in REPORTED},
        }
    return out


def _stats(values):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def compare(control, arm):
    """짝지은 차이. 두 쪽에 다 있는 시드만 쓴다."""
    seeds = sorted(set(control) & set(arm))
    rows = {}
    for _, label, higher_is_better in REPORTED:
        diffs = [
            (arm[s][label] - control[s][label])
            if arm[s][label] is not None and control[s][label] is not None else None
            for s in seeds
        ]
        clean = [d for d in diffs if d is not None and math.isfinite(d)]
        mean, sd, n = _stats(diffs)
        positive = sum(1 for d in clean if d > 0)
        rows[label] = {
            "higher_is_better": higher_is_better,
            "control": _stats([control[s][label] for s in seeds])[:2],
            "arm": _stats([arm[s][label] for s in seeds])[:2],
            "paired_diff_mean": mean,
            "paired_diff_sd": sd,
            "n": n,
            "per_seed": {s: d for s, d in zip(seeds, diffs)},
            "sign_agreement": f"{max(positive, len(clean)-positive)}/{len(clean)}",
            "arm_better_seeds": (positive if higher_is_better
                                 else (len(clean) - positive) if higher_is_better is False
                                 else None),
        }
    return {"seeds": seeds, "metrics": rows}


def _fmt(value, width=8, places=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f"{'-':>{width}}"
    return f"{value:>{width}.{places}f}"


def format_table(name, result, baseline=None):
    lines = [f"\n=== {name}  (시드 {result['seeds']}, 짝지은 차이) ===",
             f"  {'지표':<16}{'대조군':>10}{'팔':>10}{'차이(팔−대조)':>16}"
             f"{'차이 sd':>10}{'부호 일치':>10}  판정"]
    for label, row in result["metrics"].items():
        if label.startswith("_"):
            continue
        mean = row["paired_diff_mean"]
        hib = row["higher_is_better"]
        if mean is None or not math.isfinite(mean):
            verdict = "-"
        elif abs(mean) < SIGMA_SEED:
            verdict = f"차이 없음 (|Δ| < σ_seed {SIGMA_SEED})"
        elif hib is None:
            verdict = "방향 규약 없음"
        else:
            better = (mean > 0) == hib
            verdict = "**팔이 낫다**" if better else "대조군이 낫다"
        lines.append(
            f"  {label:<16}{_fmt(row['control'][0], 10)}{_fmt(row['arm'][0], 10)}"
            f"{_fmt(mean, 16)}{_fmt(row['paired_diff_sd'], 10)}"
            f"{row['sign_agreement']:>10}  {verdict}")
    if baseline is not None:
        ctl = result["metrics"]["iou_free"]["control"][0]
        arm = result["metrics"]["iou_free"]["arm"][0]
        lines.append(f"  margin(iou_free − constant-map {baseline:.4f}): "
                     f"대조군 {ctl - baseline:+.4f} | 팔 {arm - baseline:+.4f}")
    return "\n".join(lines)


def main(control, arms, fixed_epoch: int = 40, json_out=None):
    baseline = None
    path = _REPO_ROOT / _BASELINE_PATH
    if path.exists():
        baseline = float(json.loads(path.read_text())["constant_map_baseline_iou_free"])

    control_runs = load_arm(control, fixed_epoch)
    print(f"대조군 {control}: 시드 {sorted(control_runs)}")
    if baseline is not None:
        print(f"constant-map baseline iou_free = {baseline:.4f}  (고정 split 공통)")

    if isinstance(arms, str):
        arms = [a for a in arms.split(",") if a]
    payload = {"control": str(control), "fixed_epoch": fixed_epoch,
               "baseline_iou_free": baseline, "sigma_seed": SIGMA_SEED, "arms": {}}
    for entry in arms:
        name, _, log_root = str(entry).partition("=")
        result = compare(control_runs, load_arm(log_root, fixed_epoch))
        payload["arms"][name] = {"log_root": log_root, **result}
        print(format_table(name, result, baseline))

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                             default=str) + "\n")
        print(f"\nJSON: {json_out}")


if __name__ == "__main__":
    Fire(main)
