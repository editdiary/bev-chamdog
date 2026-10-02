"""`02_projection_and_prior`의 논문용 `data/`와 `provenance/`를 만든다. **이 스크립트가 정본이다.**

CSV를 손으로 고치지 않는다 -- 다음 재생성 때 사라지고, 그 사이에 문서와 데이터가 갈린다.

**무결성 게이트.** 세 팔의 확률맵 재채점 결과 중 하나라도 실패가 있으면 **아무것도 쓰지
않고** 멈춘다(`01_overall/make_package.py`와 같은 계약).

서버에서만 돈다(결과 bundle과 캘리브레이션·라벨을 읽는다):
    cd docs/99_paper_results/02_projection_and_prior && python make_package.py   (리허설: --seeds=0 --fixed_epoch=1)
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
from projects.common.paired_stats import (  # noqa: E402
    CAPTION_NOTE, paired_test, paper_metric_table, verdict)
from tools.summarize_repeats import read_run  # noqa: E402

HERE = Path(__file__).resolve().parent
# 캠페인 v3(2026-10-02~, `docs/paper_campaign_v3_plan.md`). 옛 캠페인 결과는 `docs/paper_package/`에 있다.
CAMPAIGN_ROOT = _REPO_ROOT / "runs/99_full_campaign"
DEFAULT_ROOT = CAMPAIGN_ROOT / "02_projection_and_prior"
CONTROL_LOGS = CAMPAIGN_ROOT / "01_overall/fixed_split/logs"
DEFAULT_BASELINE_PATH = (CAMPAIGN_ROOT / "01_overall/fixed_split/"
                        "analysis/constant_map_baseline.json")
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("pinhole120", "pinhole150", "source_prior")
FIXED_EPOCH = 100   # [2026-10-02] 학습 길이와 같다 (tools/paper_final_epochs.py)

# 팔마다 붙는 설계 수치. 커버리지와 각해상도는 `tools/measure_projection_tradeoff.py`가
# 캘리브레이션만으로 계산한 값이다(학습과 무관하므로 여기 상수로 둔다).
ARM_DESIGN = {
    "control":    {"label": "DS-native (frozen)", "hfov_deg": None, "coverage_pct": 99.8,
                   "front_px_per_deg": 2.55, "edge_px_per_deg": 1.96},
    "pinhole150": {"label": "Undistort + pinhole 150deg", "hfov_deg": 150.0, "coverage_pct": 96.7,
                   "front_px_per_deg": 1.20, "edge_px_per_deg": 17.87},
    "pinhole120": {"label": "Undistort + pinhole 120deg", "hfov_deg": 120.0, "coverage_pct": 93.1,
                   "front_px_per_deg": 2.58, "edge_px_per_deg": 10.32},
    "source_prior": {"label": "SynWoodScape pretrain", "hfov_deg": None, "coverage_pct": 99.8,
                     "front_px_per_deg": 2.55, "edge_px_per_deg": 1.96},
}

# 지표 정본은 `projects/common/metric_spec.py`(2026-10-02 개편, 전부 프레임 macro).
# 시드별 원자료와 짝지은 차이는 보조 지표(BF의 P/R, 링별)까지 전부, 본문 Table 2는 일곱 개만.
METRICS = EXPORTED_METRICS
CURVE_TAGS = {tag: label for tag, label, _ in PAPER_METRICS}
CURVE_EPOCHS = tuple(range(1, FIXED_EPOCH + 1))


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    try:
        shown = path.relative_to(HERE)
    except ValueError:
        shown = path
    print(f"  {shown}  ({len(rows)}행)")


def _rel(path):
    """저장소 안이면 저장소 기준 상대 경로, 아니면 그대로."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _round(value, places=6):
    return "" if value is None or not math.isfinite(value) else round(value, places)


def _mean_sd(values):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def _load(log_dir_pattern):
    """`{seed: {metric: value at FIXED_EPOCH}}`."""
    out = {}
    for seed in SEEDS:
        run_dir = Path(str(log_dir_pattern).format(seed=seed))
        if not run_dir.exists():
            raise SystemExit(f"런 로그가 없다: {run_dir}")
        series = read_run(run_dir)["series"]
        require_tags(series, run_dir.name, METRICS)
        out[seed] = {"_run": run_dir.name, "_series": series,
                     **{label: series.get(tag, {}).get(FIXED_EPOCH)
                        for tag, label, _ in METRICS}}
    return out


def main(root=DEFAULT_ROOT, out_dir=HERE, control=CONTROL_LOGS,
         baseline_path=DEFAULT_BASELINE_PATH, seeds=None, fixed_epoch=None):
    """`seeds`/`fixed_epoch`은 리허설(1시드·1 epoch)용이다 -- 기본값이 본 실험이다."""
    global SEEDS, FIXED_EPOCH, CURVE_EPOCHS
    if seeds is not None:
        SEEDS = tuple(int(s) for s in (seeds if isinstance(seeds, (list, tuple))
                                       else str(seeds).split(",")))
    if fixed_epoch is not None:
        FIXED_EPOCH = int(fixed_epoch)
        CURVE_EPOCHS = tuple(range(1, FIXED_EPOCH + 1))
    root, out_dir, control = Path(root), Path(out_dir), Path(control)
    data, prov = out_dir / "data", out_dir / "provenance"

    # --- 무결성 게이트 ------------------------------------------------------------
    integrity = {}
    for arm in ARMS:
        path = root / arm / "analysis/verify_predictions.json"
        if not path.exists():
            raise SystemExit(f"무결성 결과가 없다: {path}")
        integrity[arm] = json.loads(path.read_text())
    failed = {a: p["failures"] for a, p in integrity.items() if p.get("failures")}
    if failed:
        raise SystemExit(f"무결성 실패가 있어 아무것도 쓰지 않는다: {failed}")
    max_diff = max(r["abs_diff"] for p in integrity.values() for r in p["rows"])
    n_checks = sum(len(p["rows"]) for p in integrity.values())
    print(f"무결성 게이트 통과: {n_checks}건, 최대 절대차 {max_diff:.2e} "
          f"(허용치 {integrity[ARMS[0]]['tol']})")

    baseline = float(json.loads(
        Path(baseline_path).read_text())["constant_map_baseline_iou_free"])

    runs = {"control": _load(control / "final_s{seed}")}
    for arm in ARMS:
        runs[arm] = _load(root / arm / "logs" / (arm + "_s{seed}"))

    print("\ndata/ 생성")

    # --- 1. 시드별 원자료 ----------------------------------------------------------
    rows = []
    for arm, per_seed in runs.items():
        for seed in SEEDS:
            for _, label, _ in METRICS:
                rows.append([arm, ARM_DESIGN[arm]["label"], seed,
                             per_seed[seed]["_run"], FIXED_EPOCH, label,
                             _round(per_seed[seed][label])])
    _write(data / "per_seed_metrics.csv",
           ["arm", "arm_label", "seed", "run_name", "epoch", "metric", "value"], rows)

    # --- 2. 짝지은 차이 (주 통계) ---------------------------------------------------
    # 대조군과 팔이 **같은 시드끼리** 짝지어지므로 런 간 산포가 차이에서 상쇄된다.
    # 판정에 쓰는 노이즈 눈금은 σ_seed가 아니라 **짝지은 평균의 표준오차**다.
    paired_rows = []
    for arm in ARMS:
        for _, label, higher_is_better in METRICS:
            diffs = [runs[arm][s][label] - runs["control"][s][label] for s in SEEDS]
            mean, sd, n = _mean_sd(diffs)
            se = sd / math.sqrt(n) if n > 1 else float("nan")
            positive = sum(1 for d in diffs if d > 0)
            test = paired_test(diffs)
            paired_rows.append([
                arm, ARM_DESIGN[arm]["label"], label,
                "" if higher_is_better is None else ("higher" if higher_is_better else "lower"),
                _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[0]),
                _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[1]),
                _round(_mean_sd([runs[arm][s][label] for s in SEEDS])[0]),
                _round(_mean_sd([runs[arm][s][label] for s in SEEDS])[1]),
                _round(mean), _round(sd), _round(se),
                _round(abs(mean) / se if se and math.isfinite(se) and se > 0 else float("inf"), 3),
                f"{max(positive, n - positive)}/{n}",
                _round(test["ci_low"]), _round(test["ci_high"]), _round(test["p"], 5),
                test["significant"], verdict(test, higher_is_better),
            ] + [_round(d) for d in diffs])
    # 판정(2026-10-02): 짝지은 양측 t-검정의 95 % 신뢰구간이 0을 포함하지 않으면 유의.
    _write(data / "paired_differences.csv",
           ["arm", "arm_label", "metric", "better_direction",
            "control_mean", "control_sd", "arm_mean", "arm_sd",
            "paired_diff_mean", "paired_diff_sd", "paired_diff_se", "abs_mean_over_se",
            "sign_agreement", "ci95_low", "ci95_high", "p_value", "significant", "verdict"]
           + [f"diff_seed{s}" for s in SEEDS], paired_rows)

    # --- 3. 논문 Table 1: 축 A ------------------------------------------------------
    def _summary_row(arm):
        iou_mean, iou_sd, _ = _mean_sd([runs[arm][s]["iou_free"] for s in SEEDS])
        design = ARM_DESIGN[arm]
        row = [design["label"],
               "" if design["hfov_deg"] is None else f"{design['hfov_deg']:.0f}",
               design["coverage_pct"], design["front_px_per_deg"],
               _round(iou_mean), _round(iou_sd), _round(iou_mean - baseline)]
        if arm == "control":
            return row + ["", "", "", "", "", "", ""]
        diffs = [runs[arm][s]["iou_free"] - runs["control"][s]["iou_free"] for s in SEEDS]
        mean, sd, n = _mean_sd(diffs)
        test = paired_test(diffs)
        return row + [_round(mean), _round(sd), _round(test["ci_low"]), _round(test["ci_high"]),
                      _round(test["p"], 5), test["significant"], test["sign_agreement"]]

    _write(data / "table1_projection.csv",
           ["arm", "hfov_deg", "bev_coverage_pct", "front_px_per_deg",
            "iou_free_mean", "iou_free_sd", "margin_vs_constant_map",
            "paired_diff_mean", "paired_diff_sd", "ci95_low", "ci95_high", "p_value",
            "significant", "sign_agreement"],
           [_summary_row(a) for a in ("control", "pinhole150", "pinhole120")])

    # --- 4. 논문 Table 2: 축 B ------------------------------------------------------
    rows = []
    for _, label, higher_is_better in PAPER_METRICS:
        diffs = [runs["source_prior"][s][label] - runs["control"][s][label] for s in SEEDS]
        mean, sd, n = _mean_sd(diffs)
        se = sd / math.sqrt(n)
        ratio = abs(mean) / se if se > 0 else float("inf")
        positive = sum(1 for d in diffs if d > 0)
        test = paired_test(diffs)
        rows.append([label,
                     "" if higher_is_better is None else ("higher" if higher_is_better else "lower"),
                     _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[0]),
                     _round(_mean_sd([runs["control"][s][label] for s in SEEDS])[1]),
                     _round(_mean_sd([runs["source_prior"][s][label] for s in SEEDS])[0]),
                     _round(_mean_sd([runs["source_prior"][s][label] for s in SEEDS])[1]),
                     _round(mean), _round(se), _round(ratio, 3),
                     f"{max(positive, n - positive)}/{n}",
                     _round(test["ci_low"]), _round(test["ci_high"]), _round(test["p"], 5),
                     test["significant"], verdict(test, higher_is_better)])
    _write(data / "table2_source_prior.csv",
           ["metric", "better_direction", "target_only_mean", "target_only_sd",
            "pretrained_mean", "pretrained_sd", "paired_diff_mean", "paired_diff_se",
            "abs_mean_over_se", "sign_agreement", "ci95_low", "ci95_high", "p_value",
            "significant", "verdict"], rows)

    # --- 4b. 논문 표(지표 mean ± SD, 대조군과 유의하면 *) -----------------------------------
    # 사용자 결정(2026-10-02): 표에는 성능 지표만, 유의한 칸에 `*`. 통계값 전부는 위 CSV에 있다.
    paper = [name for _, name, _ in PAPER_METRICS]
    labels = {a: ARM_DESIGN[a]["label"] for a in ARM_DESIGN}
    for stem, arms_here in (("table_paper_projection", ("control", "pinhole150", "pinhole120")),
                            ("table_paper_source_prior", ("control", "source_prior"))):
        header, rows = paper_metric_table({a: runs[a] for a in arms_here}, "control", paper,
                                          SEEDS, labels)
        _write(data / f"{stem}.csv", header, rows)
    (data / "table_paper_caption.txt").write_text(
        CAPTION_NOTE + " Control: DS-native projection trained from scratch.\n")

    # --- 5. Figure 1 입력: 커버리지 대 정확도 -----------------------------------------
    rows = []
    for arm in ("control", "pinhole150", "pinhole120"):
        iou_mean, iou_sd, _ = _mean_sd([runs[arm][s]["iou_free"] for s in SEEDS])
        d = ARM_DESIGN[arm]
        rows.append([arm, d["label"], "" if d["hfov_deg"] is None else d["hfov_deg"],
                     d["coverage_pct"], d["front_px_per_deg"], d["edge_px_per_deg"],
                     _round(iou_mean), _round(iou_sd), _round(iou_mean - baseline)])
    _write(data / "figure_coverage_vs_accuracy.csv",
           ["arm", "label", "hfov_deg", "bev_coverage_pct", "front_px_per_deg",
            "edge_px_per_deg", "iou_free_mean", "iou_free_sd", "margin"], rows)

    # --- 6. Figure 2 입력: 초기 에폭 궤적 (축 B 기전) ----------------------------------
    rows = []
    for arm in ("control", "source_prior"):
        for tag, label in CURVE_TAGS.items():
            for epoch in CURVE_EPOCHS:
                vals = [runs[arm][s]["_series"].get(tag, {}).get(epoch) for s in SEEDS]
                mean, sd, n = _mean_sd(vals)
                rows.append([arm, ARM_DESIGN[arm]["label"], label, epoch,
                             _round(mean), _round(sd), n])
    _write(data / "figure_early_epochs.csv",
           ["arm", "arm_label", "metric", "epoch", "mean", "sd", "n"], rows)

    # --- 7. 전체 검증 곡선 (부록) -----------------------------------------------------
    rows = []
    for arm, per_seed in runs.items():
        for seed in SEEDS:
            for tag, label in CURVE_TAGS.items():
                for epoch, value in sorted(per_seed[seed]["_series"].get(tag, {}).items()):
                    rows.append([arm, per_seed[seed]["_run"], seed, epoch, label, _round(value)])
    _write(data / "val_curves.csv", ["arm", "run", "seed", "epoch", "metric", "value"], rows)

    # --- 8. 설계 근거: 화각 절충표 -----------------------------------------------------
    from projects.bev_gt.grid import ROBOT_GRID_SPEC, cell_centers_m
    from projects.geometry.double_sphere import (
        FINETUNE_CAMERA_NAMES, load_cameras, load_ego_T_cams)
    from tools.measure_projection_tradeoff import (
        SAMPLE_HEIGHTS, _ds_px_per_deg, ds_visible, pinhole_visible)
    import numpy as np

    calib = _REPO_ROOT / "dataset/sj_datasets/common/calibration/calib.yaml"
    cameras, ego_T_cams = load_cameras(calib), load_ego_T_cams(calib)
    fwd, lat = cell_centers_m(ROBOT_GRID_SPEC)
    fg, lg = np.meshgrid(fwd, lat, indexing="ij")

    def _union(fn, names=FINETUNE_CAMERA_NAMES):
        m = np.zeros(fg.shape, bool)
        for z in SAMPLE_HEIGHTS:
            for n in names:
                m |= fn(n, z)
        return m

    ds_all = _union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z))
    base_ppd = _ds_px_per_deg(cameras["front"], 0.5, 512)
    rows = [["ds_native", "", _round(100 * ds_all.mean(), 2), _round(base_ppd, 3),
             _round(_ds_px_per_deg(cameras["front"], 90, 512), 3), 1.0]]
    for hfov in (90, 100, 110, 120, 130, 140, 150, 160, 170):
        m = _union(lambda n, z: pinhole_visible(ego_T_cams[n], fg, lg, z, hfov, 512, 288)) & ds_all
        f = 256.0 / np.tan(np.deg2rad(hfov / 2.0))
        ppd0 = f * np.deg2rad(1.0)
        rows.append(["pinhole", hfov, _round(100 * m.sum() / ds_all.sum(), 2), _round(ppd0, 3),
                     _round(ppd0 / np.cos(np.deg2rad(hfov / 2.0)) ** 2, 3),
                     _round(ppd0 / base_ppd, 3)])
    _write(data / "projection_tradeoff.csv",
           ["projection", "hfov_deg", "bev_coverage_pct", "front_px_per_deg",
            "edge_px_per_deg", "front_px_per_deg_vs_fisheye"], rows)

    # --- 9. 설계 근거: 도메인 prior와 카메라 커버리지 ------------------------------------
    front_only = _union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z), ("front",))
    rng = np.hypot(fg, lg)
    rows = []
    for label, mask in (("front+left+right", ds_all), ("front_only", front_only)):
        row = [label, int(mask.sum()), _round(100 * mask.mean(), 2)]
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
            band = (rng >= lo) & (rng < hi)
            row.append(_round(100 * mask[band].mean(), 2))
        rows.append(row)
    _write(data / "camera_coverage.csv",
           ["camera_set", "cells_visible", "coverage_pct",
            "coverage_0_1m_pct", "coverage_1_2m_pct", "coverage_2_3m_pct",
            "coverage_3_4m_pct"], rows)

    import glob as _glob

    def _prior(occ_paths, vis_of):
        free = visible = cells = 0
        for path in occ_paths:
            occ = np.load(path).astype(bool)
            vis = np.load(vis_of(path)).astype(bool)
            free += int((occ & vis).sum()); visible += int(vis.sum()); cells += int(occ.size)
        return len(occ_paths), 100 * free / cells, 100 * free / visible, 100 * visible / cells

    sw = sorted(_glob.glob(str(_REPO_ROOT / "dataset/synwoodscape_2head_roi_8_4_6_h08/*_occupancy.npy")))
    rb = sorted(_glob.glob(str(_REPO_ROOT / "dataset/sj_datasets/raw*/occupancy_npy/*.npy")))
    rows = [
        ["synwoodscape_source", "240x240 / front 8 / rear 4 / lateral +-6 m / 5 cm",
         *[_round(v, 2) for v in _prior(sw, lambda p: p.replace("_occupancy.npy", "_visible.npy"))]],
        ["robot_target", "120x120 / front 4 / rear 2 / lateral +-3 m / 5 cm",
         *[_round(v, 2) for v in _prior(rb, lambda p: p.replace("/occupancy_npy/", "/visibility_npy/"))]],
    ]
    _write(data / "domain_prior.csv",
           ["domain", "grid", "n_samples", "free_pct_of_grid", "free_pct_of_observed",
            "observed_pct_of_grid"], rows)

    # --- provenance ------------------------------------------------------------------
    print("\nprovenance/ 생성")
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    prov.mkdir(parents=True, exist_ok=True)
    (prov / "environment_and_config.json").write_text(json.dumps({
        "experiment": "02_projection_and_prior",
        "git_commit": commit,
        "runtime": {"python": platform.python_version(),
                    "torch": __import__("torch").__version__,
                    "numpy": __import__("numpy").__version__,
                    "gpu": (__import__("torch").cuda.get_device_name(0)
                            if __import__("torch").cuda.is_available() else None),
                    "conda_env": Path(sys.prefix).name},
        "protocol": {
            "split": "fixed (train raws2,raws3,rawos1,rawos2,rawos4 = 192 frames; "
                     "val raws1,rawos3 = 75 frames)",
            "seeds": list(SEEDS),
            "epoch": FIXED_EPOCH,
            # 저장소 기준 상대 경로로 적는다 -- 절대 경로를 적으면 서버가 바뀔 때
            # provenance만 달라져 diff가 시끄러워진다.
            "control": f"reused from {_rel(control)} (paired by seed)",
            "constant_map_baseline_iou_free": baseline,
        },
        "arms": {a: ARM_DESIGN[a] for a in ("control",) + ARMS},
        "statistics": {
            "rule": "two-sided paired t-test over seeds (df = n-1); significant iff the 95% "
                    "confidence interval of the mean paired difference excludes 0",
            "definition": "projects/common/paired_stats.py",
            "multiple_comparisons": "each metric tested separately, no correction",
        },
    }, indent=2, ensure_ascii=False) + "\n")
    print("  provenance/environment_and_config.json")

    (prov / "integrity.json").write_text(json.dumps({
        "tolerance": integrity[ARMS[0]]["tol"],
        "tau": integrity[ARMS[0]]["tau"],
        "n_checks": n_checks,
        "n_failures": 0,
        "max_abs_diff": max_diff,
        "per_arm": {a: {"n_files": p["n_files"], "n_rows": len(p["rows"]),
                        "max_abs_diff": max(r["abs_diff"] for r in p["rows"])}
                    for a, p in integrity.items()},
        "method": "exported pre-threshold probability maps re-scored on an independent path "
                  "and compared against the training logs",
    }, indent=2, ensure_ascii=False) + "\n")
    print("  provenance/integrity.json")

    rows = []
    for run in manifest["runs"]:
        env = run["environment"]
        rows.append([run["arm"], run["run_name"], run["seed"], run["script"],
                     env.get("PROJECTION", "ds_native"), env.get("PINHOLE_HFOV_DEG", ""),
                     env.get("INIT_CHECKPOINT", ""), env.get("NUM_EPOCHS", "")])
    _write(prov / "run_manifest.csv",
           ["arm", "run_name", "seed", "script", "projection", "pinhole_hfov_deg",
            "init_checkpoint", "num_epochs"], rows)
    print("\n완료.")


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        main()
    else:
        fire.Fire(main)
