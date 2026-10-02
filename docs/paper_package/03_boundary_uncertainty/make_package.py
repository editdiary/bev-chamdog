"""`03_boundary_uncertainty`의 논문용 `data/`와 `provenance/`를 만든다. **이 스크립트가 정본이다.**

CSV를 손으로 고치지 않는다 -- 다음 재생성 때 사라지고, 그 사이에 문서와 데이터가 갈린다.

**무결성 게이트.** 확률맵 재채점 결과에 실패가 하나라도 있으면 **아무것도 쓰지 않고** 멈춘다
(`01_overall`·`02_projection_and_prior`의 `make_package.py`와 같은 계약).

**원자료는 `runs/loss_effect/`다**(30런 = 6조건 x 5시드, 2026-09-02와 2026-09-21).

**[2026-10-02] 두 가지가 바뀌었다.** (1) 광선 보조항을 뺐다 -- 사다리는 `A_ce -> B_perset
-> C_hard -> C_soft` 넷이고 보조항 부록 표(`appendix_aux_range_term.csv`)는 없어졌다.
(2) 지표가 `projects/common/metric_spec.py`로 바뀌었다(precision/recall/BF@τ, 프레임 macro).
동작점 분석은 "같은 free_miss에서 fatal" 대신 **"같은 recall에서 precision"**으로 읽는다.
**옛 `runs/loss_effect/` 런은 새 지표 tag가 없어 이 스크립트가 거부한다** -- 다음 캠페인에서
다시 학습한 루트를 `--root`로 넘긴다(되올림 분석이 epoch 곡선을 써서 재채점으로는 안 된다).
해석 정본은 [`docs/loss_effect_results.md`](../../loss_effect_results.md)이고 여기는 숫자만 만든다.

서버에서만 돈다:
    cd docs/paper_package/03_boundary_uncertainty && python make_package.py
"""
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.metric_spec import EXPORTED_METRICS, PAPER_METRICS, require_tags  # noqa: E402
from tools.summarize_repeats import read_run  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = _REPO_ROOT / "runs/loss_effect"
SEEDS = (0, 1, 2, 3, 4)
FIXED_EPOCH = 40

# 사다리. **[2026-10-02 사용자 결정] 광선 보조항을 아예 뺀다 -- soft-boundary BCE까지 넷이다.**
# (2026-09-22에는 `E_cumulative`를 다섯째 칸으로 두는 안이었고, 옛 `D_range`·`E_cumulative`는
# `C_soft`에서 갈라지는 대체 팔이었다 -- `loss_effect_results.md` §16.)
PAPER_LADDER = ("A_ce", "B_perset", "C_hard", "C_soft")
CELLS = PAPER_LADDER

CELL_DESIGN = {
    "A_ce": {
        "label": "Weighted CE (baseline)",
        "aggregation": "inverse-frequency class weights",
        "boundary_target": "hard (all cells)",
        "aux_range_term": "none",
        "lambda_b": 0.0, "sigma_m": 0.10, "lambda_r": 0.0,
        "step_from": "", "knob_changed": "",
    },
    "B_perset": {
        "label": "Per-set mean (no boundary term)",
        "aggregation": "per-set mean",
        "boundary_target": "none (band unsupervised)",
        "aux_range_term": "none",
        "lambda_b": 0.0, "sigma_m": 0.10, "lambda_r": 0.0,
        "step_from": "A_ce", "knob_changed": "aggregation + band supervision removed",
    },
    "C_hard": {
        "label": "+ boundary term, hard target",
        "aggregation": "per-set mean",
        "boundary_target": "hard 0/1",
        "aux_range_term": "none",
        "lambda_b": 0.5, "sigma_m": 0.001, "lambda_r": 0.0,
        "step_from": "B_perset", "knob_changed": "band supervision on (hard)",
    },
    "C_soft": {
        "label": "+ soft boundary target (proposed)",
        "aggregation": "per-set mean",
        "boundary_target": "soft (Gaussian, sigma=0.10 m)",
        "aux_range_term": "none",
        "lambda_b": 0.5, "sigma_m": 0.10, "lambda_r": 0.0,
        "step_from": "C_hard", "knob_changed": "band target hard -> soft",
    },
}

# 지표 정본은 `metric_spec`. 시드별 원자료는 보조 지표(BF의 P/R, 링별)까지, 표는 일곱 개만.
METRICS = PAPER_METRICS

# 되올림·국소화 그림에 쓰는 곡선. **`val/loss_epoch`은 조건마다 다른 함수라 곡선으로 겹쳐
# 그리면 안 된다** -- 아래 셋은 loss와 무관하게 같은 식으로 기록되는 진단 CE다.
CURVE_TAGS = {
    "val/ce_all_epoch": "ce_all",
    "val/ce_boundary_epoch": "ce_boundary",
    "val/ce_confident_epoch": "ce_confident",
    **{tag: label for tag, label, _ in PAPER_METRICS},
}

# 짝지은 비교. (팔, 대조군, 역할). **같은 시드끼리 짝지으므로 런 간 산포가 차이에서 상쇄된다.**
PAIRS = (
    ("C_soft", "A_ce", "main: proposed vs baseline"),
    ("C_soft", "C_hard", "mechanism: soft vs hard band target (only knob)"),
    ("C_hard", "A_ce", "mechanism: aggregation only (both hard)"),
    ("B_perset", "A_ce", "aggregation + band removal"),
)


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {path.relative_to(HERE)}  ({len(rows)}행)")


def _round(value, places=6):
    if value is None:
        return ""
    try:
        return "" if not math.isfinite(value) else round(value, places)
    except TypeError:
        return ""


def _mean_sd(values):
    clean = [v for v in values if v is not None and isinstance(v, (int, float))
             and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def _stat(values):
    """`{mean, sd, n}` 딕셔너리에서 셋을 꺼낸다(RESULTS.json 형식)."""
    return values.get("mean"), values.get("sd"), values.get("n")


def _load(root: Path):
    """`{cell: {seed: {metric: value@FIXED_EPOCH, _series, _run}}}`."""
    out = {}
    for cell in CELLS:
        out[cell] = {}
        for seed in SEEDS:
            run_dir = root / "logs" / f"{cell}_s{seed}"
            if not run_dir.exists():
                raise SystemExit(f"런 로그가 없다: {run_dir}")
            run = read_run(run_dir)
            series = run["series"]
            require_tags(series, run_dir.name, EXPORTED_METRICS)
            if FIXED_EPOCH not in series.get("val/iou_free_epoch", {}):
                raise SystemExit(f"{run_dir.name}이 epoch {FIXED_EPOCH}에 도달하지 못했다")
            out[cell][seed] = {
                "_run": run_dir.name, "_series": series, "_config": run["config"],
                **{label: series.get(tag, {}).get(FIXED_EPOCH)
                   for tag, label, _ in EXPORTED_METRICS},
            }
    return out


def _paired(runs, arm, control, metric):
    diffs = [runs[arm][s][metric] - runs[control][s][metric] for s in SEEDS]
    mean, sd, n = _mean_sd(diffs)
    se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float("nan")
    positive = sum(1 for d in diffs if d > 0)
    return {
        "diffs": diffs, "mean": mean, "sd": sd, "se": se,
        "ratio": abs(mean) / se if se and math.isfinite(se) and se > 0 else float("inf"),
        "agreement": f"{max(positive, n - positive)}/{n}",
    }


def main(root=DEFAULT_ROOT, out_dir=HERE):
    root, out_dir = Path(root), Path(out_dir)
    data, prov = out_dir / "data", out_dir / "provenance"

    # --- 무결성 게이트 ------------------------------------------------------------
    verify_path = root / "analysis/verify_predictions.json"
    if not verify_path.exists():
        raise SystemExit(f"무결성 결과가 없다: {verify_path}")
    integrity = json.loads(verify_path.read_text())
    if integrity.get("failures"):
        raise SystemExit(f"무결성 실패가 있어 아무것도 쓰지 않는다: {integrity['failures']}")
    max_diff = max(r["abs_diff"] for r in integrity["rows"])
    print(f"무결성 게이트 통과: {len(integrity['rows'])}건 / 파일 {integrity['n_files']}개, "
          f"최대 절대차 {max_diff:.2e} (허용치 {integrity['tol']})")

    bundle = json.loads((root / "analysis/RESULTS.json").read_text())
    runs = _load(root)

    print("\ndata/ 생성")

    # --- 1. 시드별 원자료 ----------------------------------------------------------
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            for _, label, _ in EXPORTED_METRICS:
                rows.append([cell, CELL_DESIGN[cell]["label"], seed,
                             runs[cell][seed]["_run"], FIXED_EPOCH, label,
                             _round(runs[cell][seed][label])])
    _write(data / "per_seed_metrics.csv",
           ["cell", "cell_label", "seed", "run_name", "epoch", "metric", "value"], rows)

    # --- 2. 논문 Table 1: 사다리 정확도 ------------------------------------------------
    # **이 표의 요지는 "거의 안 움직인다"다.** 그게 이 절의 주장(정확도가 아니라 학습
    # 거동이 바뀐다)의 전제이므로 감추지 않고 본문에 싣는다.
    rows = []
    for cell in CELLS:
        design = CELL_DESIGN[cell]
        row = [cell, design["label"], design["aggregation"], design["boundary_target"],
               design["aux_range_term"]]
        for _, label, _ in METRICS:
            mean, sd, _ = _mean_sd([runs[cell][s][label] for s in SEEDS])
            row += [_round(mean), _round(sd)]
        rows.append(row)
    header = ["cell", "cell_label", "aggregation", "boundary_target", "aux_range_term"]
    for _, label, _ in METRICS:
        header += [f"{label}_mean", f"{label}_sd"]
    _write(data / "table1_ladder_accuracy.csv", header, rows)

    # --- 3. 논문 Table 2: 목적함수 수렴과 모델 선택 (주장 ①③) -----------------------------
    axis1 = bundle["axes"]["1_objective_convergence"]
    rows = []
    for cell in CELLS:
        rebound = axis1["rebound_excess_pct"].get(cell, {})
        min_ep = axis1["val_loss_min_epoch"].get(cell, {})
        regret_iou = axis1["regret_iou_free"].get(cell, {})
        regret_f1 = axis1["regret_bf_10cm"].get(cell, {})
        gap = bundle["per_cell"][cell].get("delta_epoch_loss_vs_iou", {})
        rows.append([
            cell, CELL_DESIGN[cell]["label"], CELL_DESIGN[cell]["boundary_target"],
            *[_round(v, 4) for v in _stat(rebound)[:2]],
            *[_round(v, 3) for v in _stat(min_ep)[:2]],
            *[_round(v, 3) for v in _stat(gap)[:2]],
            *[_round(v, 6) for v in _stat(regret_iou)[:2]],
            *[_round(v, 6) for v in _stat(regret_f1)[:2]],
        ])
    _write(data / "table2_objective_convergence.csv",
           ["cell", "cell_label", "boundary_target",
            "rebound_excess_pct_mean", "rebound_excess_pct_sd",
            "val_loss_min_epoch_mean", "val_loss_min_epoch_sd",
            "epoch_gap_loss_vs_iou_mean", "epoch_gap_loss_vs_iou_sd",
            "regret_iou_free_mean", "regret_iou_free_sd",
            "regret_bf_10cm_mean", "regret_bf_10cm_sd"], rows)

    # --- 4. 논문 Table 3: 되올림의 경계 국소화 (주장 ②) ------------------------------------
    # 눈금은 **loss와 무관한 진단 CE**다. 각 런의 `val/loss_epoch`은 조건마다 다른 함수라
    # 나란히 놓을 수 없다 -- 그 함정이 `loss_effect_results.md` §16.6에 적혀 있다.
    axis7 = bundle["axes"]["7_boundary_localization"]
    rows = []
    for cell in CELLS:
        c = axis7["cells"].get(cell)
        if c is None:
            continue
        rows.append([
            cell, CELL_DESIGN[cell]["label"], CELL_DESIGN[cell]["boundary_target"],
            _round(100 * c["frac_boundary_cells"], 3),
            *[_round(v, 3) for v in _stat(c["boundary_ce_rise_pct"])[:2]],
            *[_round(v, 3) for v in _stat(c["confident_ce_rise_pct"])[:2]],
            _round(c["contribution_boundary"], 6), _round(c["contribution_confident"], 6),
            _round(100 * c["boundary_share"], 3),
        ])
    _write(data / "table3_boundary_localization.csv",
           ["cell", "cell_label", "boundary_target", "boundary_cells_pct",
            "boundary_ce_rise_pct_mean", "boundary_ce_rise_pct_sd",
            "confident_ce_rise_pct_mean", "confident_ce_rise_pct_sd",
            "contribution_boundary", "contribution_confident",
            "boundary_share_of_rise_pct"], rows)

    # --- 5. 논문 Table 4: 동작점 (주장 ④ + 안전 반증) ---------------------------------------
    # `tau_star`는 **같은 목표 recall을 만드는 문턱**이고 그 시드 간 산포가 이 축의 주장이다.
    # `precision_at_matched_operating_point`는 그 자리에서의 precision이고, **여기서 차이가
    # 사라지면 tau=0.5의 precision 차이는 동작점 이동이다**(§6.2·§16.4).
    axis5 = bundle["axes"]["5_operating_point"]
    rows = []
    for cell in CELLS:
        c = axis5["cells"].get(cell)
        if c is None:
            continue
        tau, fat = c["tau_star"], c["precision_at_matched_operating_point"]
        prec_tau50, prec_tau50_sd, _ = _mean_sd([runs[cell][s]["precision"] for s in SEEDS])
        rows.append([
            cell, CELL_DESIGN[cell]["label"],
            _round(axis5["target_recall"], 6),
            *[_round(v, 6) for v in _stat(tau)[:2]],
            *[_round(v, 6) for v in _stat(fat)[:2]],
            _round(prec_tau50), _round(prec_tau50_sd),
        ])
    _write(data / "table4_operating_point.csv",
           ["cell", "cell_label", "target_recall",
            "tau_star_mean", "tau_star_sd",
            "precision_at_matched_recall_mean", "precision_at_matched_recall_sd",
            "precision_at_tau_0.5_mean", "precision_at_tau_0.5_sd"], rows)

    # --- 5b. 논문 Table 5: 여러 동작점에서의 문턱 산포와 같은 recall에서의 precision --------
    # RESULTS.json의 축 5는 앵커가 하나다. 주장 ④(문턱 재현성)와 안전 반증은 **한 점이 아니라
    # 구간에서** 성립해야 하므로 여기서 다섯 앵커로 다시 만든다. 계산은 τ 스윕 원시 행만
    # 쓰므로 자족적이다 -- 다른 도구를 먼저 돌려야 하는 의존이 없다.
    sweep_rows = list(csv.DictReader((root / "analysis/threshold_sweep_rows.csv").open()))
    curves = {}
    for row in sweep_rows:
        cell, seed = row["cell"], int(row["seed"])
        if cell not in CELLS:
            continue
        # x축이 recall이다. 정렬하면 증가 순서가 된다(recall은 τ에 대해 감소한다).
        curves.setdefault((cell, seed), []).append(
            (float(row["recall"]), float(row["tau"]), float(row["precision"]),
             float(row["bf_10cm"]), float(row["iou_free"])))
    for key in curves:
        curves[key].sort()

    covered = [(min(p[0] for p in pts), max(p[0] for p in pts)) for pts in curves.values()]
    lo, hi = max(c[0] for c in covered), min(c[1] for c in covered)
    if not lo < hi:
        raise SystemExit(f"모든 런이 함께 덮는 recall 구간이 없다: [{lo}, {hi}]")
    anchors = [lo + (hi - lo) * i / 4 for i in range(5)]

    def _interp(points, target, column):
        xs = [p[0] for p in points]
        ys = [p[column] for p in points]
        if target < xs[0] or target > xs[-1]:
            return None
        for i in range(1, len(xs)):
            if xs[i] >= target:
                if xs[i] == xs[i - 1]:
                    return ys[i]
                w = (target - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + w * (ys[i] - ys[i - 1])
        return ys[-1]

    # **`bf_10cm`과 `iou_free`도 같은 앵커에서 낸다.** tau=0.5 표(Table 1)의 차이 중 어디까지가
    # 동작점 이동이고 어디부터가 진짜인지, 이 열들이 없으면 가를 수 없다.
    ANCHORED = ((1, "tau_star"), (2, "precision"), (3, "bf_10cm"), (4, "iou_free"))
    rows = []
    for cell in CELLS:
        for anchor in anchors:
            row = [cell, CELL_DESIGN[cell]["label"], _round(anchor)]
            n_cov = 0
            for column, _name in ANCHORED:
                mean, sd, n = _mean_sd(
                    [_interp(curves[(cell, s)], anchor, column) for s in SEEDS])
                n_cov = max(n_cov, n)
                row += [_round(mean), _round(sd)]
            rows.append(row[:3] + [n_cov] + row[3:])
    header = ["cell", "cell_label", "recall_anchor", "n_seeds_covering"]
    for _column, name in ANCHORED:
        header += [f"{name}_at_anchor_mean", f"{name}_at_anchor_sd"]
    _write(data / "table5_threshold_dispersion.csv", header, rows)

    # --- 6. 짝지은 차이 (주 통계) ---------------------------------------------------------
    # 판정 규약(캠페인 공통): **|평균 Δ| > sigma_seed(대조군)** 이고 동시에 **|평균 Δ| >= 2*SE**.
    # n=5에서 p값은 만들지 않는다.
    rows = []
    for arm, control, role in PAIRS:
        for _, label, higher_is_better in METRICS:
            p = _paired(runs, arm, control, label)
            ctrl_mean, ctrl_sd, _ = _mean_sd([runs[control][s][label] for s in SEEDS])
            arm_mean, arm_sd, _ = _mean_sd([runs[arm][s][label] for s in SEEDS])
            passes = (math.isfinite(p["mean"]) and math.isfinite(ctrl_sd)
                      and abs(p["mean"]) > ctrl_sd
                      and math.isfinite(p["se"]) and abs(p["mean"]) >= 2 * p["se"])
            rows.append([
                arm, control, role, label,
                "" if higher_is_better is None else ("higher" if higher_is_better else "lower"),
                _round(ctrl_mean), _round(ctrl_sd), _round(arm_mean), _round(arm_sd),
                _round(p["mean"]), _round(p["sd"]), _round(p["se"]),
                _round(p["ratio"], 3), p["agreement"],
                "pass" if passes else "not-decidable",
            ] + [_round(d) for d in p["diffs"]])
    _write(data / "paired_differences.csv",
           ["arm", "control", "role", "metric", "better_direction",
            "control_mean", "control_sd", "arm_mean", "arm_sd",
            "paired_diff_mean", "paired_diff_sd", "paired_diff_se", "abs_mean_over_se",
            "sign_agreement", "verdict"] + [f"diff_seed{s}" for s in SEEDS], rows)

    # --- 8. Figure 1 입력: 되올림의 경계 국소화 ------------------------------------------
    rows = []
    for cell in CELLS:
        for tag, label in CURVE_TAGS.items():
            if label not in ("ce_all", "ce_boundary", "ce_confident"):
                continue
            for epoch in range(1, FIXED_EPOCH + 1):
                mean, sd, n = _mean_sd(
                    [runs[cell][s]["_series"].get(tag, {}).get(epoch) for s in SEEDS])
                if n == 0:
                    continue
                rows.append([cell, CELL_DESIGN[cell]["label"], label, epoch,
                             _round(mean), _round(sd), n])
    _write(data / "figure_rebound_localization.csv",
           ["cell", "cell_label", "metric", "epoch", "mean", "sd", "n"], rows)

    # --- 9. Figure 2 입력: 동작점 문턱의 시드 간 산포 ---------------------------------------
    rows = []
    for cell in CELLS:
        c = axis5["cells"].get(cell)
        if c is None:
            continue
        for seed, value in zip(SEEDS, c["tau_star"]["values"]):
            rows.append([cell, CELL_DESIGN[cell]["label"], seed,
                         _round(axis5["target_recall"], 6), _round(value)])
    _write(data / "figure_operating_point.csv",
           ["cell", "cell_label", "seed", "target_recall", "tau_star"], rows)

    # --- 10. Figure 3 입력: 같은 recall에서의 precision (안전 반증) -------------------------
    sweep = list(csv.DictReader((root / "analysis/threshold_sweep_rows.csv").open()))
    rows = []
    for row in sweep:
        if row["cell"] not in CELLS:
            continue
        rows.append([row["cell"], CELL_DESIGN[row["cell"]]["label"], int(row["seed"]),
                     _round(float(row["tau"]), 4),
                     _round(float(row["recall"])), _round(float(row["precision"])),
                     _round(float(row["iou_free"])), _round(float(row["bf_10cm"]))])
    _write(data / "figure_safety_utility_curve.csv",
           ["cell", "cell_label", "seed", "tau", "recall", "precision", "iou_free",
            "bf_10cm"], rows)

    # --- 11. 전체 검증 곡선 (부록) ---------------------------------------------------------
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            for tag, label in CURVE_TAGS.items():
                for epoch, value in sorted(runs[cell][seed]["_series"].get(tag, {}).items()):
                    rows.append([cell, runs[cell][seed]["_run"], seed, epoch, label,
                                 _round(value)])
    _write(data / "val_curves.csv", ["cell", "run", "seed", "epoch", "metric", "value"], rows)

    # --- provenance ------------------------------------------------------------------
    print("\nprovenance/ 생성")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    env = bundle["environment"]
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "environment_and_config.json").write_text(json.dumps({
        "experiment": "03_boundary_uncertainty",
        "git_commit": commit,
        "runtime_of_analysis": {"python": platform.python_version(),
                                "conda_env": Path(sys.prefix).name},
        "runtime_of_training": {k: env.get(k) for k in
                                ("python", "conda_env", "torch", "cuda", "numpy", "device",
                                 "numpy_elision_broken")},
        "protocol": {
            "split": "fixed (train raws2,raws3,rawos1,rawos2,rawos4 = 192 frames; "
                     "val raws1,rawos3 = 75 frames)",
            "seeds": list(SEEDS),
            "epoch": FIXED_EPOCH,
            "epoch_note": "pre-declared fixed epoch, not the argmax checkpoint -- the argmax "
                          "is itself a random variable and inflates the seed spread",
            "checkpoint_selection_rule_for_reference": bundle["experiment"][
                "checkpoint_selection_rule"],
        },
        "cells": CELL_DESIGN,
        "paper_ladder": list(PAPER_LADDER),
        "ladder_note": "[2026-10-02] The auxiliary ray term is dropped from the paper; the "
                       "ladder ends at the soft-boundary BCE (C_soft). Metrics are frame-macro "
                       "IoU_free / IoU_non-free / Precision / Recall / BF@{0.1,0.2,0.3} m.",
        "metric_definitions": "projects/common/metric_spec.py",
        "statistics": {
            "practical_floor": "|paired mean diff| > sigma_seed of the control at fixed epoch",
            "statistical_floor": "|paired mean diff| >= 2 * SE, SE = sd(diffs)/sqrt(5)",
            "note": "no p-values at n=5; both floors must be cleared to claim a winner",
            "sigma_run": "0.0009 iou_free -- rerun noise floor from 18 matched pairs "
                         "(runs/archive/loss_effect_v1); cuDNN nondeterminism",
        },
        # **이 목록은 2026-09-21까지의 옛 지표·옛 런에서 나온 결론이다.** 다음 캠페인 결과로
        # 하나씩 다시 확인한 뒤 고친다 -- 데이터에서 자동으로 나오는 값이 아니다.
        "claims_as_of": "2026-09-21 (runs/loss_effect, old metrics) -- re-verify after rerun",
        "claims": {
            "supported": ["objective convergence (rebound removed)",
                          "rebound is localized to the boundary band",
                          "objective realigns with quality (usable for checkpoint selection)",
                          "operating-point threshold reproducibility"],
            "not_supported": ["accuracy improvement",
                              "safety improvement (operating-point shift, retracted)",
                              "retraining reproducibility of final metrics",
                              "prediction geometry / cell-decision stability"],
            "unanswerable_here": ["sequence-level generalization (val has 2 sequences; "
                                  "LOSO in 01_overall is the canonical answer)"],
        },
    }, indent=2, ensure_ascii=False) + "\n")
    print("  provenance/environment_and_config.json")

    (prov / "integrity.json").write_text(json.dumps({
        "tolerance": integrity["tol"],
        "tau": integrity["tau"],
        "n_files": integrity["n_files"],
        "n_checks": len(integrity["rows"]),
        "n_failures": 0,
        "max_abs_diff": max_diff,
        "method": "exported pre-threshold probability maps re-scored on an independent path "
                  "and compared against the training logs",
    }, indent=2, ensure_ascii=False) + "\n")
    print("  provenance/integrity.json")

    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            cfg = runs[cell][seed]["_config"]
            rows.append([cell, runs[cell][seed]["_run"], seed, cfg.get("loss"),
                         cfg.get("lambda_b"), cfg.get("sigma_m"), cfg.get("lambda_r"),
                         cfg.get("range_loss_mode") or ("arc_huber"
                                                        if float(cfg.get("lambda_r") or 0) > 0
                                                        else ""),
                         cfg.get("delta_m"), cfg.get("num_epochs"), cfg.get("encoder_type")])
    _write(prov / "run_manifest.csv",
           ["cell", "run_name", "seed", "loss", "lambda_b", "sigma_m", "lambda_r",
            "range_loss_mode", "delta_m", "num_epochs", "encoder_type"], rows)
    print("\n완료.")


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        main()
    else:
        fire.Fire(main)
