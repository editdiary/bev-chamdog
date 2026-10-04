#!/usr/bin/env python
"""실험 04 Edge Deployment 패키지 생성기 -- 보드가 돌려준 `results/`에서 `data/`·`provenance/`를 만든다.

**이 스크립트가 정본이다.** `data/`의 CSV를 손으로 고치지 않는다.

입력(보드 -> 서버, `tools/jetson/README.md`의 절차):
    runs/jetson_bundle/results/
    ├── bench.json                 지연·연속 실행·구간 시각·환경
    ├── <run>/<구성>_prob_free.npz  모델 × 구성의 검증 예측
    ├── results_tegrastats.log     전력(없으면 전력 표를 건너뛴다)
    ├── build_report.json          엔진 빌드 기록·층 정밀도 요약
    └── engines/<run>/*_layers.json
    runs/jetson_bundle/manifest.json   서버 쪽 출처·검증

채점은 논문의 다른 실험과 **같은 지표 함수**다(`tools/jetson/score.py`). 유의성은 같은 시드끼리
짝지은 양측 t-검정의 95 % 신뢰구간(`projects/common/paired_stats.py`), 대조군은 `pytorch_fp32`.

    python make_package.py
    python make_package.py --results=<results 폴더> --out_dir=<출력 폴더>
"""
import calendar
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "jetson"))

from projects.common.paired_stats import CAPTION_NOTE, paired_test, paper_metric_table, verdict  # noqa: E402
from projects.common.polar import build_ray_index  # noqa: E402
from projects.datasets.robot_simplebev import GRID_SPEC  # noqa: E402
from score import score  # noqa: E402

DEFAULT_RESULTS = REPO / "runs/jetson_bundle/results"
DEFAULT_LABELS = REPO / "runs/99_full_campaign/01_overall/fixed_split/analysis/predictions/labels.npz"
CONFIGS = ("pytorch_fp32", "trt_fp16", "trt_int8")
LABELS = {"pytorch_fp32": "PyTorch fp32 (whole model)",
          "trt_fp16": "TensorRT fp16 encoder + PyTorch fp32 BEV stage",
          "trt_int8": "TensorRT INT8 encoder + PyTorch fp32 BEV stage"}
METRICS = ("iou_free", "iou_non_free", "precision", "recall", "bf_10cm", "bf_20cm", "bf_30cm")
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def _r(v, places=6):
    return "" if v is None else round(float(v), places)


def _write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path.relative_to(path.parents[1])}  ({len(rows)}행)")


def _mean_sd(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def parse_tegrastats(path: Path, utc_offset_s: int):
    """줄마다 `(epoch 초, {레일: mW})`. tegrastats 시각은 보드 지역 시각(초 단위)이다."""
    rows = []
    pat = re.compile(r"(\w+) (\d+)mW/(\d+)mW")
    for line in path.read_text(errors="ignore").splitlines():
        if len(line) < 19:
            continue
        try:
            local = time.strptime(line[:19], "%m-%d-%Y %H:%M:%S")
        except ValueError:
            continue
        epoch = calendar.timegm(local) - utc_offset_s
        rails = {name: int(cur) for name, cur, _ in pat.findall(line) if name in RAILS}
        gr3d = re.search(r"GR3D_FREQ (\d+)%", line)
        rows.append((epoch, rails, int(gr3d.group(1)) if gr3d else None))
    return rows


def power_in(rows, start, end):
    """구간 안 표본의 레일별 중앙값(mW)과 합. 경계 1초는 버린다(초 단위 시각이라 걸친다)."""
    inside = [r for r in rows if start + 1 <= r[0] <= end - 1]
    if not inside:
        return None
    out = {rail: statistics.median(r[1][rail] for r in inside if rail in r[1]) for rail in RAILS
           if any(rail in r[1] for r in inside)}
    out["total"] = sum(out.values())
    gr = [r[2] for r in inside if r[2] is not None]
    out["gr3d_util_median"] = statistics.median(gr) if gr else None
    out["n_samples"] = len(inside)
    return out


def main(results=str(DEFAULT_RESULTS), out_dir=str(HERE), labels=str(DEFAULT_LABELS)):
    results, out = Path(results), Path(out_dir)
    data, prov = out / "data", out / "provenance"
    data.mkdir(parents=True, exist_ok=True)
    prov.mkdir(parents=True, exist_ok=True)
    bench = json.loads((results / "bench.json").read_text())
    runs = bench["runs"]
    configs = [c for c in CONFIGS if c in bench["configs"]]
    seeds = list(range(len(runs)))
    with np.load(labels, allow_pickle=False) as blob:
        lab = {k: np.array(blob[k]) for k in blob.files}
    rays = build_ray_index(GRID_SPEC)
    print(f"모델 {runs} | 구성 {configs} | 보드 {bench['env'].get('device')} TensorRT {bench['env'].get('tensorrt')}")

    # === 모델 × 구성: 정확도 + 지연 ======================================================
    groups = {c: {} for c in configs}           # 논문 표용 {구성: {seed 순번: {지표: 값}}}
    per_seed_rows = []
    for k, run in enumerate(runs):
        for c in configs:
            with np.load(results / run / f"{c}_prob_free.npz", allow_pickle=False) as blob:
                prob, ids = np.array(blob["prob_free"]), list(blob["sample_ids"])
            if ids != list(lab["sample_ids"]):
                raise SystemExit(f"!! {run}/{c}: 프레임 순서가 라벨과 다르다")
            m = score(prob, lab, rays)
            sp = bench["speed"][run][c]
            m.update(end_to_end_ms=sp["end_to_end"]["median_of_round_medians"],
                     encoder_ms=sp["encoder"]["median_of_round_medians"],
                     bev_stage_ms=sp["bev_stage"]["median_of_round_medians"],
                     fps=sp["fps"], encoder_share=sp["encoder_share_of_end_to_end"],
                     end_to_end_p99_ms=statistics.median(r["p99"] for r in sp["end_to_end"]["rounds"]),
                     decision_flip_rate=bench["accuracy_vs_server"][run][c]["decision_flip_rate_vs_server"])
            groups[c][k] = m
            per_seed_rows.append([run, k, c] + [_r(m[x]) for x in METRICS] +
                                 [_r(m[x], 4) for x in ("end_to_end_ms", "encoder_ms", "bev_stage_ms",
                                                        "end_to_end_p99_ms", "fps", "encoder_share")] +
                                 [_r(m["decision_flip_rate"], 8)])
    _write(data / "per_seed.csv",
           ["run", "seed_index", "config", *METRICS, "end_to_end_ms", "encoder_ms", "bev_stage_ms",
            "end_to_end_p99_ms", "fps", "encoder_share", "decision_flip_rate_vs_server_fp32"],
           per_seed_rows)

    # 시드별 차이(대조 = 같은 시드의 pytorch_fp32) -- INT8 손실이 모델마다 얼마나 다른지가 여기서 보인다.
    rows = []
    for c in configs:
        if c == "pytorch_fp32":
            continue
        for k, run in enumerate(runs):
            rows.append([run, c] + [_r(groups[c][k][m] - groups["pytorch_fp32"][k][m]) for m in METRICS]
                        + [_r(groups[c][k]["decision_flip_rate"], 8)])
    _write(data / "per_seed_diff_vs_fp32.csv",
           ["run", "config", *(f"d_{m}" for m in METRICS), "decision_flip_rate_vs_server_fp32"], rows)

    # === Table 1: 속도 (시드 5개의 평균 ± SD) =============================================
    base_ms = {k: groups["pytorch_fp32"][k]["end_to_end_ms"] for k in seeds} if "pytorch_fp32" in groups else None
    rows = []
    for c in configs:
        row = [c, LABELS[c]]
        for key in ("end_to_end_ms", "end_to_end_p99_ms", "fps", "encoder_ms", "bev_stage_ms", "encoder_share"):
            mean, sd = _mean_sd(groups[c][k][key] for k in seeds)
            row += [_r(mean, 4), _r(sd, 4)]
        speedups = [base_ms[k] / groups[c][k]["end_to_end_ms"] for k in seeds] if base_ms else []
        mean, sd = _mean_sd(speedups)
        mem = [bench["speed"][r][c].get("engine_device_memory_mb") for r in runs]
        row += [_r(mean, 4), _r(sd, 4), _r(_mean_sd(mem)[0], 1)]
        rows.append(row)
    _write(data / "table1_speed.csv",
           ["config", "label", "end_to_end_ms_mean", "end_to_end_ms_sd", "end_to_end_p99_ms_mean",
            "end_to_end_p99_ms_sd", "fps_mean", "fps_sd", "encoder_ms_mean", "encoder_ms_sd",
            "bev_stage_ms_mean", "bev_stage_ms_sd", "encoder_share_mean", "encoder_share_sd",
            "speedup_vs_pytorch_fp32_mean", "speedup_vs_pytorch_fp32_sd", "engine_device_memory_mb"],
           rows)

    # === Table 2: 정확도 + 짝지은 차이 (대조 = pytorch_fp32) ==============================
    rows = []
    for c in configs:
        for metric in METRICS:
            mean, sd = _mean_sd(groups[c][k][metric] for k in seeds)
            if c == "pytorch_fp32":
                rows.append([c, metric, _r(mean), _r(sd)] + [""] * 9)
                continue
            t = paired_test([groups[c][k][metric] - groups["pytorch_fp32"][k][metric] for k in seeds])
            rows.append([c, metric, _r(mean), _r(sd), _r(t["mean"]), _r(t["sd"]), _r(t["ci_low"]),
                         _r(t["ci_high"]), _r(t["p"], 5), t["significant"], t["sign_agreement"],
                         verdict(t, True), _r(_mean_sd(groups[c][k]["decision_flip_rate"] for k in seeds)[0], 8)])
    _write(data / "table2_accuracy_paired.csv",
           ["config", "metric", "mean", "sd", "paired_diff_vs_fp32_mean", "paired_diff_sd", "ci95_low",
            "ci95_high", "p_value", "significant", "sign_agreement", "verdict", "decision_flip_rate_mean"],
           rows)

    # === 논문 표: 속도 열 + 지표 열(`*`) =================================================
    header, acc_rows = paper_metric_table(groups, "pytorch_fp32", list(METRICS), seeds,
                                          {c: LABELS[c] for c in configs})
    speed_cells = {}
    for c in configs:
        ms, ms_sd = _mean_sd(groups[c][k]["end_to_end_ms"] for k in seeds)
        fps, fps_sd = _mean_sd(groups[c][k]["fps"] for k in seeds)
        speed_cells[LABELS[c]] = [f"{ms:.1f} ± {ms_sd:.1f}", f"{fps:.1f} ± {fps_sd:.1f}"]
    _write(data / "table_paper_edge.csv", ["condition", "latency_ms", "fps"] + header[1:],
           [[r[0]] + speed_cells[r[0]] + r[1:] for r in acc_rows])
    (data / "table_paper_caption.txt").write_text(
        "Latency is the median per-frame time (3 cameras, 512x288, batch 1) on Jetson AGX Orin "
        "(MAXN, clocks locked), from the GPU-resident input tensor to the free-space probability "
        "map; image decoding and resizing are excluded. " + CAPTION_NOTE +
        " Control: the whole model in PyTorch fp32.\n")

    # === 연속 실행 + 전력 ===============================================================
    rows = []
    for c in configs:
        s = bench["sustain"].get(c)
        if s:
            rows.append([c, s["seconds"], s["calls"], _r(s["first_10pct"]["median"], 4),
                         _r(s["last_10pct"]["median"], 4), _r(s["last_10pct"]["median"] / s["first_10pct"]["median"], 4),
                         _r(s["all"]["p99"], 4)])
    _write(data / "sustained_run.csv",
           ["config", "seconds", "calls", "first_10pct_median_ms", "last_10pct_median_ms",
            "last_over_first", "all_p99_ms"], rows)

    teg = results / "results_tegrastats.log"
    if teg.exists():
        samples = parse_tegrastats(teg, bench["env"].get("utc_offset_s", 0))
        idle = next(s for s in bench["segments"] if s["kind"] == "idle")
        idle_p = power_in(samples, idle["start"], idle["end"])
        rows = [["idle", "", *(_r(idle_p.get(r), 1) for r in RAILS), _r(idle_p["total"], 1), "", "",
                 idle_p["gr3d_util_median"], idle_p["n_samples"]]] if idle_p else []
        for c in configs:
            seg = next((s for s in bench["segments"] if s["kind"] == "sustain" and s.get("config") == c), None)
            p = power_in(samples, seg["start"], seg["end"]) if seg else None
            if not p:
                continue
            ms = bench["sustain"][c]["all"]["median"]
            energy = p["total"] * ms / 1000.0                  # mW × s = mJ
            rows.append([c, _r(ms, 4), *(_r(p.get(r), 1) for r in RAILS), _r(p["total"], 1),
                         _r(energy, 2), _r((p["total"] - idle_p["total"]) * ms / 1000.0, 2) if idle_p else "",
                         p["gr3d_util_median"], p["n_samples"]])
        _write(data / "power.csv",
               ["config", "sustain_median_ms", *(f"{r}_mW" for r in RAILS), "total_mW",
                "energy_per_frame_mJ", "energy_above_idle_per_frame_mJ", "gr3d_util_median_pct", "n_samples"],
               rows)
    else:
        print("  (tegrastats 기록 없음 -- power.csv 생략)")

    # === 층 정밀도 =======================================================================
    build = json.loads((results / "build_report.json").read_text()) if (results / "build_report.json").exists() else {}
    rows = []
    for run, engines in build.get("models", {}).items():
        for prec, e in engines.items():
            conv = e.get("conv_by_output_precision", {})
            total = sum(conv.values())
            rows.append([run, prec, e.get("size_mb"), e.get("build_seconds"), total,
                         conv.get("int8", 0), conv.get("fp16", 0), conv.get("fp32", 0)])
    if rows:
        _write(data / "engine_layers.csv",
               ["run", "precision", "engine_mb", "build_seconds", "conv_layers", "conv_int8",
                "conv_fp16", "conv_fp32"], rows)

    # === 출처 ===========================================================================
    manifest_path = results.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    (prov / "environment.json").write_text(json.dumps({
        "board": bench["env"], "tensorrt_build": {k: v for k, v in build.items() if k != "models"},
        "protocol": {k: bench[k] for k in ("iters", "warmup", "rounds", "idle_seconds", "sustain_seconds")},
        "bundle": {k: manifest.get(k) for k in ("runs", "root", "input_shape", "output_shape",
                                                "camera_names", "n_val_frames", "n_calib_frames",
                                                "calib_sequences", "onnx_opset", "server")},
        "bundle_checks": {r: m.get("checks") for r, m in manifest.get("models", {}).items()},
    }, indent=2, ensure_ascii=False))
    print("완료")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
