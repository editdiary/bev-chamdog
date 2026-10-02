"""대조군과 각 팔을 **시드끼리 짝지어** 비교한다 -- 캠페인 `02_projection_and_prior`.

**왜 짝짓나.** 이 캠페인은 대조군을 다시 돌리지 않고 `01_overall`의 고정 split 5런을
재사용한다(원장 §6.0). 같은 시드끼리 빼면 초기화·데이터 순서에서 오는 런 간 산포
(σ_seed ≈ 0.0018)가 차이에서 상쇄되므로, 평균만 비교할 때보다 훨씬 작은 효과를 볼 수 있다.

**판정 규약**(원장 §6.4).

- 주 지표는 `iou_free`와 constant-map baseline 대비 **margin**이다. 대조군과 팔이 같은
  val split을 보므로 baseline이 같고, margin의 차이는 `iou_free`의 차이와 정확히 같다 --
  그래도 함께 싣는다. 절대값만으로는 통로 폭에 좌우되는 수치라 읽을 수 없기 때문이다.
- n=5에서 p값을 만들지 않는다. 대신 **평균 차이·차이의 표준편차·평균의 표준오차·부호가
  일치한 시드 수**를 싣는다.

**판정은 문턱 둘을 모두 넘어야 한다.**

1. **실용 문턱** `|평균 차이| > σ_seed`(0.0018). `01_overall`에서 실측한 런 간 산포보다
   작은 차이는 재현 가능한 크기가 아니다.
2. **통계 문턱** `|평균 차이| >= 2 x SE`, 여기서 `SE = (차이의 표준편차)/sqrt(n)`.

**둘째가 없으면 안 되는 이유**(2026-09-21에 실제로 겪었다). σ_seed는 *한 런*의 산포이지
*짝지은 차이*의 산포가 아니다. `source_prior`의 `iou_free`는 평균 차이 −0.0020으로 σ_seed를
살짝 넘지만, 차이 자체가 시드마다 0.0036씩 흔들려서 `|평균|/SE`가 1.24밖에 안 된다 --
즉 우열을 말할 수 없는데 첫째 문턱만 보면 "대조군이 낫다"로 찍힌다.

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
# **라벨과 val split만으로 정해지는 값이라 학습을 다시 해도 바뀌지 않는다.** 그래도
# 다른 트리에서 돌릴 때는 `--baseline_path`로 그쪽 것을 가리켜 대조할 수 있게 열어 둔다.
_BASELINE_PATH = ("runs/paper_final/01_overall/fixed_split/analysis/"
                  "constant_map_baseline.json")
# `01_overall`에서 실측한 시드 간 산포. 이보다 작은 차이는 우열로 쓰지 않는다.
# **대조군을 다시 학습하면 이 값도 다시 재야 한다** -- 판정 규칙이 여기 걸려 있다.
# `--sigma_seed`로 넘길 수 있고, 넘기지 않으면 2026-09-18 캠페인의 실측값을 쓴다.
SIGMA_SEED = 0.0018
# 짝지은 평균이 그 평균의 표준오차의 몇 배여야 우열로 쓰나. n=5에서 2는 느슨한 편이지만
# p값을 만들지 않기로 했으므로 보수적인 눈금 하나로 둔다.
MIN_SE_RATIO = 2.0


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
        # 짝지은 **평균**의 불확실성이다. σ_seed(한 런의 산포)와는 다른 양이다.
        se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float("nan")
        rows[label] = {
            "higher_is_better": higher_is_better,
            "control": _stats([control[s][label] for s in seeds])[:2],
            "arm": _stats([arm[s][label] for s in seeds])[:2],
            "paired_diff_mean": mean,
            "paired_diff_sd": sd,
            "paired_diff_se": se,
            "abs_mean_over_se": abs(mean) / se if se and math.isfinite(se) and se > 0 else float("inf"),
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


def format_table(name, result, baseline=None, sigma_seed=SIGMA_SEED):
    lines = [f"\n=== {name}  (시드 {result['seeds']}, 짝지은 차이) ===",
             f"  {'지표':<16}{'대조군':>10}{'팔':>10}{'차이(팔−대조)':>15}"
             f"{'차이 sd':>10}{'|Δ|/SE':>8}{'부호':>7}  판정"]
    for label, row in result["metrics"].items():
        if label.startswith("_"):
            continue
        mean = row["paired_diff_mean"]
        hib = row["higher_is_better"]
        ratio = row.get("abs_mean_over_se", float("nan"))
        if mean is None or not math.isfinite(mean):
            verdict = "-"
        elif abs(mean) < sigma_seed:
            verdict = f"차이 없음 (|Δ| < σ_seed {sigma_seed})"
        elif not (math.isfinite(ratio) and ratio >= MIN_SE_RATIO):
            verdict = f"불확실 (|Δ|/SE {ratio:.1f} < {MIN_SE_RATIO})"
        elif hib is None:
            verdict = "방향 규약 없음"
        else:
            better = (mean > 0) == hib
            verdict = "**팔이 낫다**" if better else "대조군이 낫다"
        lines.append(
            f"  {label:<16}{_fmt(row['control'][0], 10)}{_fmt(row['arm'][0], 10)}"
            f"{_fmt(mean, 15)}{_fmt(row['paired_diff_sd'], 10)}"
            f"{_fmt(ratio, 8, 1)}{row['sign_agreement']:>7}  {verdict}")
    if baseline is not None:
        ctl = result["metrics"]["iou_free"]["control"][0]
        arm = result["metrics"]["iou_free"]["arm"][0]
        lines.append(f"  margin(iou_free − constant-map {baseline:.4f}): "
                     f"대조군 {ctl - baseline:+.4f} | 팔 {arm - baseline:+.4f}")
    return "\n".join(lines)


def main(control, arms, fixed_epoch: int = 100, json_out=None,
         sigma_seed: float = SIGMA_SEED, baseline_path: str = _BASELINE_PATH):
    sigma_seed = float(sigma_seed)
    baseline = None
    path = Path(baseline_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if path.exists():
        baseline = float(json.loads(path.read_text())["constant_map_baseline_iou_free"])

    control_runs = load_arm(control, fixed_epoch)
    print(f"대조군 {control}: 시드 {sorted(control_runs)}")
    if baseline is not None:
        print(f"constant-map baseline iou_free = {baseline:.4f}  (고정 split 공통)")
    print(f"판정 눈금: σ_seed = {sigma_seed:.4f} | |Δ|/SE ≥ {MIN_SE_RATIO}")

    if isinstance(arms, str):
        arms = [a for a in arms.split(",") if a]
    payload = {"control": str(control), "fixed_epoch": fixed_epoch,
               "baseline_iou_free": baseline, "sigma_seed": sigma_seed,
               "min_se_ratio": MIN_SE_RATIO, "arms": {}}
    for entry in arms:
        name, _, log_root = str(entry).partition("=")
        result = compare(control_runs, load_arm(log_root, fixed_epoch))
        payload["arms"][name] = {"log_root": log_root, **result}
        print(format_table(name, result, baseline, sigma_seed))

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                             default=str) + "\n")
        print(f"\nJSON: {json_out}")


if __name__ == "__main__":
    Fire(main)
