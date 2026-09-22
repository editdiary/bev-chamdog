"""`02b_native_source_prior`의 논문용 `data/`와 `provenance/`를 만든다. **이 스크립트가 정본이다.**

CSV를 손으로 고치지 않는다 -- 다음 재생성 때 사라지고, 그 사이에 문서와 데이터가 갈린다.

## 이 실험이 묻는 것

`02`의 `source_prior`는 SynWoodScape를 **타깃 과제에 맞춰 가공해서** 썼다(전방8/후방4/
횡±6 m, 셀 0.05 m로 타깃과 동일). 그러면 재는 것은 "도메인 **내용**의 불일치"뿐이다.
남이 공개한 주행 BEV 모델을 그대로 가져다 쓰는 상황은 그렇지 않다 -- 범위도 해상도도
그쪽 관행을 따른다. 이 팔은 그 상황을 만든다(30 m, 200×200, 셀 0.15 m).

그래서 표를 **세 줄**로 낸다. 두 사전학습 팔이 같은 대조군에 대해 얼마씩 움직였는지를
나란히 놓아야 "공간 스케일 불일치"의 몫이 보인다.

| 줄 | 사전학습 | source 격자 |
|---|---|---|
| `control` | 없음 (scratch) | -- |
| `source_prior` | 가공한 SynWoodScape | 240×240, 0.05 m (타깃과 동일) |
| `source_prior_native` | **원본 기하** | 200×200, **0.15 m** |

**무결성 게이트.** 확률맵 재채점에 실패가 하나라도 있으면 **아무것도 쓰지 않고** 멈춘다
(`01`·`02`와 같은 계약).

서버에서만 돈다:
    cd docs/paper_package/02b_native_source_prior && python make_package.py
    python make_package.py --root=.../runs/paper_final_cumulative/02b_native_source_prior \
        --control=.../runs/paper_final_cumulative/01_overall/fixed_split/logs
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

from tools.summarize_repeats import read_run  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = _REPO_ROOT / "runs/paper_final/02b_native_source_prior"
DEFAULT_CONTROL = _REPO_ROOT / "runs/paper_final/01_overall/fixed_split/logs"
#: 가공한 기하로 사전학습한 기존 팔. **보조 비교**로만 쓴다(§유보 참고).
DEFAULT_ADAPTED = _REPO_ROOT / "runs/paper_final/02_projection_and_prior/source_prior/logs"
DEFAULT_BASELINE_PATH = (_REPO_ROOT / "runs/paper_final/01_overall/fixed_split/"
                         "analysis/constant_map_baseline.json")
SEEDS = (0, 1, 2, 3, 4)
ARM = "source_prior_native"
PRETRAIN_ARM = "source_pretrain_native"
FIXED_EPOCH = 40

#: 두 source 구성의 기하. 학습과 무관하게 정해지는 값이라 상수로 둔다.
#: 근거는 `projects/bev_gt/grid.py`의 `SYNWOODSCAPE_NATIVE_GRID_SPEC` 주석.
SOURCE_GEOMETRY = {
    "source_prior": {
        "label": "SynWoodScape pretrain (adapted)",
        "grid": "240x240", "cell_m": 0.05,
        "front_m": 8.0, "rear_m": 4.0, "half_width_m": 6.0,
        "manual_refinement": True,
        "note": "cropped and resampled to the target task's cell size",
    },
    ARM: {
        "label": "SynWoodScape pretrain (native)",
        "grid": "200x200", "cell_m": 0.15,
        "front_m": 15.0, "rear_m": 15.0, "half_width_m": 15.0,
        "manual_refinement": False,
        "note": "the source's own 30 m coverage at a driving-standard cell size",
    },
    "control": {
        "label": "Target only (scratch)",
        "grid": "", "cell_m": None,
        "front_m": None, "rear_m": None, "half_width_m": None,
        "manual_refinement": None, "note": "no pretraining",
    },
}

METRICS = (
    ("val/iou_free_epoch", "iou_free", True),
    ("val/fatal_rate_epoch", "fatal_rate", False),
    ("val/free_miss_rate_epoch", "free_miss_rate", False),
    ("val/occupied_f1_10cm_epoch", "occupied_f1_10cm", True),
    ("val/occupied_f1_20cm_epoch", "occupied_f1_20cm", True),
    ("val/occupied_f1_40cm_epoch", "occupied_f1_40cm", True),
    ("val/range_mae_epoch", "range_mae", False),
    ("val/range_bias_epoch", "range_bias", None),
    ("val/range_missed_obstacle_rate_epoch", "range_missed_obstacle_rate", False),
)
CURVE_TAGS = {"val/iou_free_epoch": "iou_free",
              "val/fatal_rate_epoch": "fatal_rate",
              "val/free_miss_rate_epoch": "free_miss_rate"}


def _shown(path: Path):
    try:
        return path.relative_to(HERE)
    except ValueError:
        return path


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {_shown(path)}  ({len(rows)}행)")


def _round(value, places=6):
    return "" if value is None or not math.isfinite(value) else round(value, places)


def _mean_sd(values):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def _load(pattern, required=True):
    """`{seed: {metric: value at FIXED_EPOCH}}`. 없으면 `None`(보조 비교는 없어도 된다)."""
    out = {}
    for seed in SEEDS:
        run_dir = Path(str(pattern).format(seed=seed))
        if not run_dir.exists():
            if required:
                raise SystemExit(f"런 로그가 없다: {run_dir}")
            return None
        series = read_run(run_dir)["series"]
        out[seed] = {"_run": run_dir.name, "_series": series,
                     **{label: series.get(tag, {}).get(FIXED_EPOCH)
                        for tag, label, _ in METRICS}}
    return out


def _paired(runs, arm, label):
    """짝지은 차이. 대조군과 팔이 **같은 시드끼리** 짝지어져 런 간 산포가 상쇄된다."""
    diffs = [runs[arm][s][label] - runs["control"][s][label] for s in SEEDS]
    mean, sd, n = _mean_sd(diffs)
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    positive = sum(1 for d in diffs if d > 0)
    ratio = abs(mean) / se if se and math.isfinite(se) and se > 0 else float("inf")
    return diffs, mean, sd, se, ratio, f"{max(positive, n - positive)}/{n}"


def main(root=DEFAULT_ROOT, out_dir=HERE, control=DEFAULT_CONTROL,
         adapted=DEFAULT_ADAPTED, baseline_path=DEFAULT_BASELINE_PATH,
         sigma_seed: float = None):
    root, out_dir = Path(root), Path(out_dir)
    control, adapted = Path(control), Path(adapted)
    data, prov = out_dir / "data", out_dir / "provenance"

    # --- 무결성 게이트 ------------------------------------------------------------
    verify_path = root / ARM / "analysis/verify_predictions.json"
    if not verify_path.exists():
        raise SystemExit(f"무결성 결과가 없다: {verify_path}\n"
                         f"  먼저: bash configs/paper_final_native_prior_analysis.sh")
    integrity = json.loads(verify_path.read_text())
    if integrity.get("failures"):
        raise SystemExit(f"무결성 실패가 있어 아무것도 쓰지 않는다: {integrity['failures']}")
    max_diff = max(r["abs_diff"] for r in integrity["rows"])
    n_checks = len(integrity["rows"])
    print(f"무결성 게이트 통과: {n_checks}건, 최대 절대차 {max_diff:.2e} "
          f"(허용치 {integrity['tol']})")

    baseline = float(json.loads(
        Path(baseline_path).read_text())["constant_map_baseline_iou_free"])

    runs = {"control": _load(control / "final_s{seed}"),
            ARM: _load(root / ARM / "logs" / (ARM + "_s{seed}"))}
    # **보조 비교.** 없으면 그 줄만 빠진다 -- 주 비교(scratch 대비)는 그대로 성립한다.
    adapted_runs = _load(adapted / "source_prior_s{seed}", required=False)
    if adapted_runs is not None:
        runs["source_prior"] = adapted_runs
    else:
        print(f"  (보조 비교 생략: {adapted} 에 5시드가 없다)")
    arms = [a for a in ("source_prior", ARM) if a in runs]

    # 판정 눈금. 넘기지 않으면 대조군 5시드의 iou_free 표준편차로 잡는다 --
    # **대조군을 다시 학습했으면 그 값이 곧 새 눈금이다.**
    if sigma_seed is None:
        sigma_seed = _mean_sd([runs["control"][s]["iou_free"] for s in SEEDS])[1]
    sigma_seed = float(sigma_seed)
    print(f"판정 눈금: σ_seed = {sigma_seed:.4f} (대조군 5시드 iou_free sd) | |Δ|/SE ≥ 2.0")

    print("\ndata/ 생성")

    # --- 1. 시드별 원자료 ----------------------------------------------------------
    rows = []
    for arm in ["control"] + arms:
        for seed in SEEDS:
            for _, label, _ in METRICS:
                rows.append([arm, SOURCE_GEOMETRY[arm]["label"], seed,
                             runs[arm][seed]["_run"], FIXED_EPOCH, label,
                             _round(runs[arm][seed][label])])
    _write(data / "per_seed_metrics.csv",
           ["arm", "arm_label", "seed", "run_name", "epoch", "metric", "value"], rows)

    # --- 2. 짝지은 차이 (주 통계) ---------------------------------------------------
    rows = []
    for arm in arms:
        for _, label, hib in METRICS:
            diffs, mean, sd, se, ratio, agree = _paired(runs, arm, label)
            rows.append([
                arm, SOURCE_GEOMETRY[arm]["label"], label,
                "" if hib is None else ("higher" if hib else "lower"),
                _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[0]),
                _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[1]),
                _round(_mean_sd([runs[arm][s][label] for s in SEEDS])[0]),
                _round(_mean_sd([runs[arm][s][label] for s in SEEDS])[1]),
                _round(mean), _round(sd), _round(se), _round(ratio, 3), agree,
            ] + [_round(d) for d in diffs])
    _write(data / "paired_differences.csv",
           ["arm", "arm_label", "metric", "better_direction",
            "control_mean", "control_sd", "arm_mean", "arm_sd",
            "paired_diff_mean", "paired_diff_sd", "paired_diff_se", "abs_mean_over_se",
            "sign_agreement"] + [f"diff_seed{s}" for s in SEEDS], rows)

    # --- 3. 논문 표: 세 줄 비교 ------------------------------------------------------
    # 판정은 두 바닥을 **둘 다** 넘어야 한다: |Δ̄| > σ_seed 그리고 |Δ̄| ≥ 2·SE.
    def _verdict(mean, ratio, hib):
        if mean is None or not math.isfinite(mean):
            return "-"
        if abs(mean) < sigma_seed:
            return f"no difference (|d| < sigma_seed {sigma_seed:.4f})"
        if not (math.isfinite(ratio) and ratio >= 2.0):
            return f"inconclusive (|d|/SE {ratio:.1f} < 2)"
        return "arm better" if (mean > 0) == hib else "control better"

    rows = []
    for arm in ["control"] + arms:
        g = SOURCE_GEOMETRY[arm]
        mean_iou, sd_iou, _ = _mean_sd([runs[arm][s]["iou_free"] for s in SEEDS])
        row = [arm, g["label"], g["grid"], g["cell_m"] if g["cell_m"] else "",
               g["front_m"] if g["front_m"] else "",
               "" if g["manual_refinement"] is None else g["manual_refinement"],
               _round(mean_iou), _round(sd_iou), _round(mean_iou - baseline)]
        if arm == "control":
            rows.append(row + ["", "", "", "", ""])
            continue
        _, mean, sd, se, ratio, agree = _paired(runs, arm, "iou_free")
        rows.append(row + [_round(mean), _round(sd), _round(ratio, 3), agree,
                           _verdict(mean, ratio, True)])
    _write(data / "table1_native_vs_adapted.csv",
           ["arm", "arm_label", "source_grid", "source_cell_m", "source_front_m",
            "manual_refinement", "iou_free_mean", "iou_free_sd", "margin_vs_constant_map",
            "paired_diff_mean", "paired_diff_sd", "abs_mean_over_se", "sign_agreement",
            "verdict"], rows)

    # --- 4. 전 지표 판정표 ----------------------------------------------------------
    rows = []
    for _, label, hib in METRICS:
        _, mean, sd, se, ratio, agree = _paired(runs, ARM, label)
        rows.append([label, "" if hib is None else ("higher" if hib else "lower"),
                     _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[0]),
                     _round(_mean_sd([runs[ARM][s][label] for s in SEEDS])[0]),
                     _round(mean), _round(se), _round(ratio, 3), agree,
                     "-" if hib is None else _verdict(mean, ratio, hib)])
    _write(data / "table2_native_prior_all_metrics.csv",
           ["metric", "better_direction", "target_only_mean", "native_pretrained_mean",
            "paired_diff_mean", "paired_diff_se", "abs_mean_over_se", "sign_agreement",
            "verdict"], rows)

    # --- 5. 초기 에폭 궤적 (사전학습이 무엇을 주나) --------------------------------------
    rows = []
    for arm in ["control"] + arms:
        for tag, label in CURVE_TAGS.items():
            for epoch in range(1, FIXED_EPOCH + 1):
                mean, sd, n = _mean_sd(
                    [runs[arm][s]["_series"].get(tag, {}).get(epoch) for s in SEEDS])
                rows.append([arm, SOURCE_GEOMETRY[arm]["label"], label, epoch,
                             _round(mean), _round(sd), n])
    _write(data / "figure_early_epochs.csv",
           ["arm", "arm_label", "metric", "epoch", "mean", "sd", "n"], rows)

    # --- 6. source 도메인 사전학습 곡선 -------------------------------------------------
    # 사전학습이 실제로 수렴했는지 확인용. target 확률맵이 없으므로 scalar만 남는다.
    rows = []
    pretrain = _load(root / PRETRAIN_ARM / "logs" / "swscape_native_pretrain_s{seed}",
                     required=False)
    if pretrain is None:
        print(f"  (사전학습 곡선 생략: {root / PRETRAIN_ARM} 에 5시드가 없다)")
    else:
        for seed in SEEDS:
            series = pretrain[seed]["_series"]
            for tag, label in CURVE_TAGS.items():
                for epoch, value in sorted(series.get(tag, {}).items()):
                    rows.append([pretrain[seed]["_run"], seed, epoch, label, _round(value)])
        _write(data / "source_pretrain_curves.csv",
               ["run", "seed", "epoch", "metric", "value"], rows)

    # --- 6b. 두 사전학습의 source 도메인 최종 성능 --------------------------------------
    # **"학습 실패가 아니라 전이 실패"의 근거다.** 이 숫자가 없으면 "사전학습이 그냥
    # 안 된 것 아니냐"는 반박을 막을 수 없다. 두 팔은 서로 다른 과제를 풀므로
    # 값끼리 비교하지 않는다 -- 각자 자기 과제에서 수렴했는지만 본다.
    rows = []
    for arm, pat, log_root in (
        ("source_pretrain", "swscape_binary_pretrain_s{seed}",
         Path(str(adapted).replace("/source_prior/logs", "/source_pretrain/logs"))),
        (PRETRAIN_ARM, "swscape_native_pretrain_s{seed}", root / PRETRAIN_ARM / "logs"),
    ):
        runs_here = _load(log_root / pat, required=False)
        if runs_here is None:
            print(f"  (source 최종 성능 생략: {log_root} 에 5시드가 없다)")
            continue
        vals = [runs_here[s]["iou_free"] for s in SEEDS]
        mean, sd, n = _mean_sd(vals)
        label = ("SynWoodScape pretrain (adapted)" if arm == "source_pretrain"
                 else SOURCE_GEOMETRY[ARM]["label"])
        grid = "240x240" if arm == "source_pretrain" else SOURCE_GEOMETRY[ARM]["grid"]
        cell = 0.05 if arm == "source_pretrain" else SOURCE_GEOMETRY[ARM]["cell_m"]
        rows.append([arm, label, grid, cell, FIXED_EPOCH, _round(mean), _round(sd), n])
    if rows:
        _write(data / "source_pretrain_summary.csv",
               ["arm", "label", "source_grid", "source_cell_m", "epoch",
                "source_iou_free_mean", "source_iou_free_sd", "n_seeds"], rows)

    # --- 7. 전체 검증 곡선 (부록) -----------------------------------------------------
    rows = []
    for arm in ["control"] + arms:
        for seed in SEEDS:
            for tag, label in CURVE_TAGS.items():
                for epoch, value in sorted(runs[arm][seed]["_series"].get(tag, {}).items()):
                    rows.append([arm, runs[arm][seed]["_run"], seed, epoch, label,
                                 _round(value)])
    _write(data / "val_curves.csv", ["arm", "run", "seed", "epoch", "metric", "value"], rows)

    # --- 8. 두 source 구성의 기하 --------------------------------------------------------
    rows = []
    for arm in arms:
        g = SOURCE_GEOMETRY[arm]
        rows.append([arm, g["label"], g["grid"], g["cell_m"], g["front_m"], g["rear_m"],
                     g["half_width_m"], g["manual_refinement"], g["note"]])
    _write(data / "source_geometry.csv",
           ["arm", "label", "grid", "cell_m", "front_m", "rear_m", "half_width_m",
            "manual_refinement", "note"], rows)

    # --- 9. source 라벨 통계 -------------------------------------------------------------
    stats_path = _REPO_ROOT / "dataset/synwoodscape_native_roi_15_15_15_h08/manifest.json"
    if stats_path.exists():
        m = json.loads(stats_path.read_text())
        _write(data / "source_label_stats.csv",
               ["dataset", "n_samples", "grid", "cell_m", "observed_pct_of_grid",
                "free_pct_of_observed", "ego_cells", "manual_refinement",
                "visibility_rule"],
               [[m["name"], m["n_samples"], "x".join(str(v) for v in m["grid"]["shape"]),
                 m["grid"]["cell_m"], _round(100 * m["mean_visible_fraction"], 2),
                 _round(100 * m["mean_free_fraction_of_observed"], 2), m["ego_cells"],
                 m["occupancy"]["manual_refinement"], m["visibility"]["rule"]]])

    # --- provenance ----------------------------------------------------------------------
    print("\nprovenance/ 생성")
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    prov.mkdir(parents=True, exist_ok=True)
    (prov / "environment_and_config.json").write_text(json.dumps({
        "experiment": "02b_native_source_prior",
        "git_commit": commit,
        "runtime": {"python": platform.python_version(),
                    "torch": __import__("torch").__version__,
                    "numpy": __import__("numpy").__version__,
                    "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q",
                    "conda_env": Path(sys.prefix).name},
        "protocol": {
            "split": "fixed (train raws2,raws3,rawos1,rawos2,rawos4 = 192 frames; "
                     "val raws1,rawos3 = 75 frames)",
            "seeds": list(SEEDS),
            "epoch": FIXED_EPOCH,
            "control": f"reused from {_rel(control)} (paired by seed)",
            "secondary_comparison": _rel(adapted) if adapted_runs is not None else None,
            "constant_map_baseline_iou_free": baseline,
        },
        "source_geometry": SOURCE_GEOMETRY,
        "statistics": {
            "practical_floor": f"|paired mean diff| > sigma_seed ({sigma_seed:.6f}, "
                               f"sd of the control's five seeds)",
            "statistical_floor": "|paired mean diff| >= 2 * SE, SE = sd(diffs)/sqrt(5)",
            "note": "no p-values at n=5; both floors must be cleared to claim a winner",
        },
        "caveats": [
            "the native-geometry labels have no manual refinement -- the refined labels "
            "exist only at ROI 8/4/+-6, so the gap against the adapted arm mixes geometry "
            "with label quality",
            "the SynWoodScape BEV is a pinhole view from 15 m, not an orthographic "
            "projection; a surface at height h is magnified by 15/(15-h)",
        ],
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"  {_shown(prov / 'environment_and_config.json')}")

    (prov / "integrity.json").write_text(json.dumps({
        "tolerance": integrity["tol"], "tau": integrity["tau"],
        "n_files": integrity["n_files"], "n_checks": n_checks,
        "n_failures": 0, "max_abs_diff": max_diff,
        "method": "exported pre-threshold probability maps re-scored on an independent path "
                  "and compared against the training logs",
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"  {_shown(prov / 'integrity.json')}")

    rows = [[r["arm"], r["run_name"], r["seed"], r["script"],
             r["environment"].get("RANGE_LOSS_MODE", ""),
             r["environment"].get("LAMBDA_R", ""),
             r["environment"].get("INIT_CHECKPOINT", ""),
             r["environment"].get("NUM_EPOCHS", "")]
            for r in manifest["runs"]]
    _write(prov / "run_manifest.csv",
           ["arm", "run_name", "seed", "script", "range_loss_mode", "lambda_r",
            "init_checkpoint", "num_epochs"], rows)
    print("\n완료.")


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        main()
    else:
        fire.Fire(main)
