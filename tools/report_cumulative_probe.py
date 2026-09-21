"""누적형 ray loss(`cumulative_l1`)를 대조군과 **시드끼리 짝지어** 판정한다.

## 왜 별도 도구인가

이 probe의 산출물은 `runs/cumulative_ray_loss`에 있고 대조군(`C_soft`·`D_range`)은
`runs/loss_effect`에 있다. 기존 집계 도구는 전부 **한 root 안의 런들**을 전제하므로
(`report_ablation.py`·`summarize_repeats.py`·`report_threshold_sweep.py`) 두 트리를 짝지어
읽지 못한다. 대조군을 다시 돌리는 것은 같은 config를 두 번 학습하는 낭비이고, 재실행
노이즈(σ_run ≈ 0.0009)를 새로 끼얹는 일이기도 하다.

## 무엇을 내나

1. **고정 epoch 40이 주 판정이다.** 선택 checkpoint(val `iou_free` 최고점)는 max 연산이라
   위로 편향되므로 보조로만 싣는다.
2. **짝지은 차이.** 같은 시드끼리 빼면 초기화·데이터 순서에서 오는 산포가 상쇄된다.
   평균 차이·차이의 표준편차·**부호가 일치한 시드 수**를 함께 낸다. n=5에서 p값은 만들지
   않는다.
3. **같은 `free_miss`에서의 `fatal`.** τ=0.5 한 점의 `fatal` 차이는 representation 개선과
   동작점 이동을 구분하지 못한다(`loss_effect_results.md` §6.2). 저장된 확률맵을 τ로 쓸어
   곡선을 만들고 같은 `free_miss`에서 비교한다. **체크포인트를 다시 forward하지 않는다** --
   `export_val_predictions.py`가 남긴 `p(free)`가 정본 재료이고, 그것을
   `verify_val_predictions.py`가 학습 로그와 대조해 검증했다.

## 미리 돌려야 하는 것

    python tools/export_run_scalars.py --log_root=runs/cumulative_ray_loss/logs \
        --out_dir=runs/cumulative_ray_loss/analysis
    python tools/export_val_predictions.py --log_root=runs/cumulative_ray_loss \
        --cells=E_cumulative --seeds=0,1,2,3,4
    python tools/verify_val_predictions.py --root=runs/cumulative_ray_loss

실행:

    python tools/report_cumulative_probe.py --out_dir=runs/cumulative_ray_loss/analysis
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.free_space_metrics import (  # noqa: E402
    fatal_rate,
    free_miss_rate,
    iou_free,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.datasets.robot_simplebev import GRID_SPEC  # noqa: E402
from tools.build_results_bundle import _rebound  # noqa: E402
from tools.summarize_repeats import read_run  # noqa: E402
from tools.verify_val_predictions import numpy_metrics  # noqa: E402

# 표에 싣는 지표. **`iou_free`가 첫 줄이고 `fatal`/`free_miss`가 바로 뒤다** -- 셋을 떼어
# 읽으면 동작점 이동을 개선으로 오독한다.
METRICS = (
    ("val/iou_free_epoch", "iou_free", True),
    ("val/fatal_rate_epoch", "fatal", False),
    ("val/free_miss_rate_epoch", "free_miss", False),
    ("val/occupied_f1_10cm_epoch", "f1@10cm", True),
    ("val/occupied_f1_20cm_epoch", "f1@20cm", True),
    ("val/range_mae_epoch", "range_mae", False),
    ("val/range_bias_epoch", "range_bias", None),
    ("val/range_missed_obstacle_rate_epoch", "missed_obstacle", False),
    # ray 보조항의 진단. `C_soft`는 λ_R=0이라 항을 계산하지 않으므로 비어 있고,
    # `range_cumulative_mae`는 누적형 런에만 있다 -- 없는 칸은 빈칸으로 둔다.
    ("val/range_arc_mae_epoch", "range_arc_mae", False),
    ("val/range_cumulative_mae_epoch", "range_cumulative_mae", False),
    ("val/loss_epoch", "val_loss", False),
    ("train/iou_free_epoch", "train_iou_free", True),
)

SELECTION_TAG = "val/iou_free_epoch"

# τ 격자와 앵커 규약은 `report_threshold_sweep.py`와 같다 -- 두 표를 나란히 읽어야 한다.
DEFAULT_TAUS = (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
N_AUTO_ANCHORS = 5


def _csv_list(value):
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _split_arm(token, default_root):
    """`"C_soft"` -> `("C_soft", default_root)`, `"E_cumulative@runs/x"` -> `("E_cumulative", "runs/x")`.

    대조군이 **여러 산출물 트리에 흩어져 있을 수 있다** -- 이 probe는 `runs/loss_effect`의
    사다리와 `runs/cumulative_ray_loss`의 누적형을 동시에 대조군으로 세운다.
    """
    cell, sep, root = str(token).partition("@")
    return cell.strip(), Path(root.strip()) if sep else Path(default_root)


def _mean_sd(values):
    """`(평균, 표본 표준편차, n)`. 값이 없으면 `(None, None, 0)`."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None, None, 0
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), sd, len(vals)


def _fmt(mean, sd, n, digits=5):
    if mean is None:
        return "—"
    if n < 2:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


# ── 1. TensorBoard 쪽 집계 ──────────────────────────────────────────────────────

def collect_arm(log_root: Path, cell: str, seeds, fixed_epoch: int) -> dict:
    """`{seed: {"fixed": {...}, "best": {...}, "selected_epoch": n, ...}}`.

    해당 epoch에 도달하지 못한 런은 값이 `None`이 된다 -- **조용히 짧은 런을 집계하지
    않는다.** 호출부가 `complete`를 보고 거른다.
    """
    out = {}
    for seed in seeds:
        run_dir = Path(log_root) / f"{cell}_s{seed}"
        if not run_dir.exists():
            continue
        run = read_run(run_dir)
        series = run["series"]
        selection = series.get(SELECTION_TAG, {})
        if not selection:
            continue
        best_epoch = max(selection, key=selection.get)
        last_epoch = max(selection)
        lambda_b = float(run["config"].get("lambda_b", 0) or 0)
        reb = _rebound(series.get("val/loss_epoch", {}),
                       series.get("val/entropy_boundary_epoch", {}), lambda_b)
        entry = {"selected_epoch": int(best_epoch), "last_epoch": int(last_epoch),
                 "complete": fixed_epoch in selection,
                 "rebound_excess_pct": reb["excess_pct"],
                 "rebound_raw_pct": reb["raw_pct"],
                 "val_loss_min_epoch": reb["min_epoch"],
                 "config": run["config"]}
        for which, epoch in (("fixed", fixed_epoch), ("best", best_epoch)):
            entry[which] = {name: series.get(tag, {}).get(epoch)
                            for tag, name, _ in METRICS}
        out[seed] = entry
    return out


def paired_delta(arm, control, metric, which):
    """같은 시드끼리의 차이. `(시드별 차이 dict, 평균, SD, 부호 일치 수, 방향)`."""
    diffs = {}
    for seed in sorted(set(arm) & set(control)):
        a, c = arm[seed][which].get(metric), control[seed][which].get(metric)
        if a is None or c is None:
            continue
        diffs[seed] = a - c
    if not diffs:
        return diffs, None, None, 0, 0
    values = list(diffs.values())
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    sign = 1 if mean > 0 else -1
    agree = sum(1 for v in values if (v > 0) == (mean > 0) and v != 0)
    return diffs, mean, sd, agree, sign


# ── 2. 저장된 확률맵의 τ 스윕 ───────────────────────────────────────────────────

def _load_npz(path: Path) -> dict:
    """**즉시 통째로 읽고 파일을 닫는다** -- 지연 로딩된 `NpzFile`을 여러 개 들고 있으면
    첫 읽기가 다른 데이터를 돌려주는 것을 겪었다(`projects/common/npsafe.py`)."""
    with np.load(path, allow_pickle=False) as blob:
        return {k: np.array(blob[k]) for k in blob.files}


def load_labels(pred_dirs):
    """`labels.npz`를 하나 읽고, 여러 트리의 것이 **같은지** 확인한다.

    확률맵을 두 트리에서 가져와 한 곡선 위에 올리므로, 프레임 순서가 다르면 비교가 조용히
    무의미해진다.
    """
    ref, ref_dir = None, None
    for pred_dir in pred_dirs:
        lab = _load_npz(Path(pred_dir) / "labels.npz")
        if ref is None:
            ref, ref_dir = lab, pred_dir
            continue
        if list(lab["sample_ids"]) != list(ref["sample_ids"]):
            raise SystemExit(f"프레임 순서가 다르다: {ref_dir} 대 {pred_dir}")
        for key in ("free", "occupied", "valid"):
            if not np.array_equal(lab[key], ref[key]):
                raise SystemExit(f"라벨이 다르다({key}): {ref_dir} 대 {pred_dir}")
    return ref


def score_prob_map(prob, tau, gt, gt_parts, valid, rays):
    """학습 루프와 **같은 함수**로 τ에서의 지표를 낸다(`verify_val_predictions`와 동일 경로)."""
    pred = (prob > tau) & valid
    out = {"iou_free": float(iou_free(pred, gt, valid)[0]),
           "fatal": float(fatal_rate(pred, gt, valid)[0]),
           "free_miss": float(free_miss_rate(pred, gt, valid)[0])}
    out.update({k: float(v) for k, v in
                numpy_metrics(pred, gt_parts, valid, rays, GRID_SPEC.cell_m).items()})
    out["missed_obs"] = out.pop("missed_obstacle_rate")
    return out


def sweep(sources, seeds, which, taus):
    """`{(cell, seed, tau): metrics}`. `sources`는 `{cell: predictions 디렉터리}`."""
    missing = [str(d) for d in sorted({str(d) for d in sources.values()})
               if not (Path(d) / "labels.npz").exists()]
    if missing:
        print(f"!! 확률맵 디렉터리가 없다: {missing}."
              " `tools/export_val_predictions.py`를 먼저 돌린다.")
        return {}, {}
    labels = load_labels(sorted({str(d) for d in sources.values()}))
    gt = torch.from_numpy(labels["free"]).unsqueeze(1)
    valid = torch.from_numpy(labels["valid"]).unsqueeze(1)
    gt_parts = {"free": gt, "occupied": torch.from_numpy(labels["occupied"]).unsqueeze(1)}
    rays = build_ray_index(GRID_SPEC)

    rows, checkpoints = {}, {}
    for cell, pred_dir in sources.items():
        for seed in seeds:
            path = Path(pred_dir) / f"{cell}_s{seed}__{which}.npz"
            if not path.exists():
                print(f"!! 확률맵이 없어 건너뛴다: {path}")
                continue
            blob = _load_npz(path)
            if list(blob["sample_ids"]) != list(labels["sample_ids"]):
                raise SystemExit(f"{path.name}의 프레임 순서가 labels.npz와 다르다")
            checkpoints[(cell, seed)] = str(blob["checkpoint"])
            prob = torch.from_numpy(blob["prob_free"].astype(np.float32)).unsqueeze(1)
            for tau in taus:
                rows[(cell, seed, tau)] = score_prob_map(prob, tau, gt, gt_parts, valid, rays)
            print(f"  {cell}_s{seed}__{which} ({blob['checkpoint']})", flush=True)
    return rows, checkpoints


def _curve(rows, cell, seeds, taus, metric):
    """τ별 시드 평균. 한 시드라도 빠지면 그 τ는 nan이다."""
    out = []
    for tau in taus:
        vals = [rows[(cell, s, tau)][metric] for s in seeds if (cell, s, tau) in rows]
        out.append(float(np.mean(vals)) if len(vals) == len(seeds) else float("nan"))
    return np.array(out)


def _interp_monotone(x, xs, ys):
    """`np.interp`는 `xs`가 증가하지 않으면 **경고 없이 틀린 값**을 준다. 검사한 뒤 쓴다."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return float("nan")
    if np.any(np.diff(xs) < 0):
        return float("nan")
    if not xs.min() <= x <= xs.max():
        return float("nan")      # clamp된 값을 비교로 오독하지 않게 한다
    return float(np.interp(x, xs, ys))


def _auto_anchors(curves, n=N_AUTO_ANCHORS):
    """모든 칸의 `free_miss` 곡선이 **동시에 덮는** 구간을 균등 분할한다."""
    finite = [c[np.isfinite(c)] for c in curves]
    if any(len(c) == 0 for c in finite):
        return ()
    lo, hi = max(c.min() for c in finite), min(c.max() for c in finite)
    if not lo < hi:
        return ()
    pad = 0.01 * (hi - lo)
    return tuple(np.linspace(lo + pad, hi - pad, n))


# ── 3. 보고 ────────────────────────────────────────────────────────────────────

def main(probe_root="runs/cumulative_ray_loss",
         control_root="runs/loss_effect",
         probe_cell="E_cumulative",
         control_cells="C_soft,D_range",
         seeds="0,1,2,3,4",
         fixed_epoch=40,
         taus=DEFAULT_TAUS,
         which="last,best",
         out_dir=None):
    probe_root, control_root = Path(probe_root), Path(control_root)
    control_cells = _csv_list(control_cells)
    seeds = [int(s) for s in _csv_list(seeds)]
    taus = tuple(float(t) for t in (taus if isinstance(taus, (list, tuple))
                                    else _csv_list(taus)))
    which_list = _csv_list(which)
    out_dir = Path(out_dir) if out_dir else probe_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_roots = {probe_cell: probe_root}
    arms = {probe_cell: collect_arm(probe_root / "logs", probe_cell, seeds, fixed_epoch)}
    control_names = []
    for token in control_cells:
        cell, root = _split_arm(token, control_root)
        arm_roots[cell] = root
        control_names.append(cell)
        arms[cell] = collect_arm(root / "logs", cell, seeds, fixed_epoch)
    control_cells = control_names
    order = [probe_cell, *control_cells]

    print(f"=== 누적형 ray loss probe -- 고정 epoch {fixed_epoch}이 주 판정 ===")
    print(f"probe  {probe_root}/logs/{probe_cell}_s*")
    for cell in control_cells:
        print(f"대조군 {arm_roots[cell]}/logs/{cell}_s*")
    incomplete = [f"{c}_s{s}" for c in order for s, e in arms[c].items() if not e["complete"]]
    for cell in order:
        got = sorted(arms[cell])
        print(f"  {cell:14s} n={len(got)} seeds={got}"
              + (f"  선택 epoch={[arms[cell][s]['selected_epoch'] for s in got]}" if got else ""))
    if incomplete:
        print(f"  !! epoch {fixed_epoch}에 도달하지 못한 런: {incomplete}")

    # === 표 1: 고정 epoch / 선택 checkpoint ==========================================
    for which_epoch, title in (("fixed", f"고정 epoch {fixed_epoch} (주 판정)"),
                               ("best", "선택 checkpoint (보조 -- max 연산이라 위로 편향)")):
        print(f"\n=== {title} -- 평균 ± 시드 간 표준편차 ===")
        width = max(22, *(len(c) for c in order))
        print("  " + "지표".ljust(22) + "".join(c.rjust(width) for c in order))
        for _, name, _ in METRICS:
            cols = []
            for cell in order:
                cols.append(_fmt(*_mean_sd([arms[cell][s][which_epoch].get(name)
                                            for s in sorted(arms[cell])])).rjust(width))
            print("  " + name.ljust(22) + "".join(cols))
        if which_epoch == "fixed":
            for key in ("rebound_excess_pct", "rebound_raw_pct"):
                cols = [_fmt(*_mean_sd([arms[c][s][key] for s in sorted(arms[c])]),
                             digits=3).rjust(width) for c in order]
                print("  " + key.ljust(22) + "".join(cols))

    # === 표 2: 짝지은 차이 ===========================================================
    paired = {}
    for which_epoch in ("fixed", "best"):
        print(f"\n=== 짝지은 차이 ({which_epoch}) -- {probe_cell} − 대조군, 같은 시드끼리 ===")
        print("  n=5에서 p값을 만들지 않는다. **부호 일치 시드 수**가 판정 근거다.")
        header = f"  {'지표':<22}{'대조군':>10}{'평균 Δ':>14}{'SD(Δ)':>12}{'부호 일치':>10}"
        print(header)
        for _, name, better_high in METRICS:
            for control in control_cells:
                diffs, mean, sd, agree, _ = paired_delta(
                    arms[probe_cell], arms[control], name, which_epoch)
                if mean is None:
                    continue
                paired[(which_epoch, name, control)] = {
                    "per_seed": {str(k): v for k, v in diffs.items()},
                    "mean": mean, "sd": sd, "n": len(diffs), "n_same_sign": agree,
                    "better_high": better_high,
                }
                mark = ""
                if better_high is not None:
                    good = (mean > 0) == bool(better_high)
                    mark = "  좋아짐" if good else "  나빠짐"
                print(f"  {name:<22}{control:>10}{mean:>+14.5f}{sd:>12.5f}"
                      f"{f'{agree}/{len(diffs)}':>10}{mark}")

    # === 표 3: 저장된 확률맵의 τ 스윕 ================================================
    sources = {cell: root / "analysis" / "predictions" for cell, root in arm_roots.items()}

    sweep_rows_path = out_dir / "threshold_sweep_rows.csv"
    all_rows, sweep_summary = [], {}
    for which_map in which_list:
        label = {"last": f"고정 epoch {fixed_epoch}", "best": "선택 checkpoint"}.get(
            which_map, which_map)
        print(f"\n=== τ 스윕 ({which_map} = {label}) -- 저장된 확률맵에서 ===")
        rows, checkpoints = sweep(sources, seeds, which_map, taus)
        live = [c for c in order if any((c, s, taus[0]) in rows for s in seeds)]
        for cell in live:
            for seed in seeds:
                for tau in taus:
                    if (cell, seed, tau) not in rows:
                        continue
                    all_rows.append({"which": which_map, "cell": cell, "seed": seed,
                                     "tau": tau,
                                     "checkpoint": checkpoints.get((cell, seed), ""),
                                     **rows[(cell, seed, tau)]})

        fm = {c: _curve(rows, c, seeds, taus, "free_miss") for c in live}
        print("  " + "cell".rjust(14) + "τ".rjust(7)
              + "".join(k.rjust(12) for k in ("free_miss", "fatal", "missed_obs",
                                              "iou_free", "f1@10cm")))
        for cell in live:
            for i, tau in enumerate(taus):
                vals = [_curve(rows, cell, seeds, taus, k)[i]
                        for k in ("free_miss", "fatal", "missed_obs", "iou_free", "f1@10cm")]
                print("  " + cell.rjust(14) + f"{tau:.2f}".rjust(7)
                      + "".join(f"{v:.4f}".rjust(12) for v in vals))

        # probe나 대조군 중 하나라도 확률맵이 없으면 비교 자체가 성립하지 않는다.
        if probe_cell not in live or len(live) < 2:
            print(f"  !! 비교할 곡선이 모자란다(있는 칸: {live}) -- "
                  "`export_val_predictions.py`를 먼저 돌린다.")
            continue
        anchors = _auto_anchors([fm[c] for c in live])
        print(f"\n  --- 같은 free_miss에서의 비교 (이것이 safety 판정 표다) ---")
        if not anchors:
            print("  !! 칸들의 free_miss 구간이 겹치지 않는다 -- τ 격자를 넓혀야 한다")
        else:
            print(f"  앵커는 모든 칸이 덮는 구간 [{min(anchors):.4f}, {max(anchors):.4f}]에서"
                  f" {len(anchors)}점 자동 유도.")
            for metric in ("fatal", "missed_obs"):
                print(f"\n  [{metric}]" + "free_miss".rjust(12)
                      + "".join(c.rjust(12) for c in live)
                      + "".join(f"{probe_cell[:4]}−{c[:6]}".rjust(16) for c in live
                                if c != probe_cell))
                curves = {c: _curve(rows, c, seeds, taus, metric) for c in live}
                for anchor in anchors:
                    vals = {c: _interp_monotone(anchor, fm[c], curves[c]) for c in live}
                    deltas = [vals[probe_cell] - vals[c] for c in live if c != probe_cell]
                    key = (which_map, metric, round(float(anchor), 6))
                    sweep_summary[str(key)] = {"anchor": float(anchor),
                                               "values": {c: vals[c] for c in live},
                                               "deltas": deltas}
                    print("  " + " " * len(f"  [{metric}]") + f"{anchor:.4f}".rjust(12)
                          + "".join(f"{vals[c]:.4f}".rjust(12) for c in live)
                          + "".join(f"{d:+.4f}".rjust(16) for d in deltas))
            print("  (nan = 그 칸의 곡선이 이 free_miss를 덮지 않는다)")
            print("\n  읽는 법: 차이가 대조군의 시드 σ 안이면 '곡선이 겹친다'이고, 그때"
                  " τ=0.5의\n  fatal 차이는 representation 개선이 아니라 동작점 이동이다.")

            # 시드마다 목표 free_miss를 맞추는 τ*와 그 자리의 fatal
            print(f"\n  --- 목표 free_miss를 맞추는 τ*와 그 자리의 fatal (시드별) ---")
            print("  " + "q(free_miss)".rjust(14)
                  + "".join(f"{c} τ*".rjust(16) for c in live)
                  + "".join(f"{c} fatal@τ*".rjust(18) for c in live))
            for anchor in anchors:
                tau_cols, fatal_cols = [], []
                for cell in live:
                    stars, fatals = [], []
                    for seed in seeds:
                        if (cell, seed, taus[0]) not in rows:
                            continue
                        fm_s = [rows[(cell, seed, t)]["free_miss"] for t in taus]
                        ft_s = [rows[(cell, seed, t)]["fatal"] for t in taus]
                        stars.append(_interp_monotone(anchor, fm_s, taus))
                        fatals.append(_interp_monotone(anchor, fm_s, ft_s))
                    ok_t = [v for v in stars if np.isfinite(v)]
                    ok_f = [v for v in fatals if np.isfinite(v)]
                    # **모든 시드가 이 q를 덮어야 비교가 된다** -- 일부만 덮으면 표본이
                    # 달라져 σ가 작아진 것인지 안정적인 것인지 갈리지 않는다.
                    if len(ok_t) < len(stars) or len(ok_f) < len(fatals) or not ok_t:
                        tau_cols.append("n/a"); fatal_cols.append("n/a"); continue
                    tau_cols.append(f"{np.mean(ok_t):.3f}±{np.std(ok_t, ddof=1):.3f}")
                    fatal_cols.append(f"{np.mean(ok_f):.4f}±{np.std(ok_f, ddof=1):.4f}")
                print("  " + f"{anchor:.4f}".rjust(14)
                      + "".join(c.rjust(16) for c in tau_cols)
                      + "".join(c.rjust(18) for c in fatal_cols))
            print("  (n/a = 시드 중 일부의 곡선이 이 free_miss를 덮지 않는다)")

    # === 산출물 ====================================================================
    if all_rows:
        keys = ["which", "cell", "seed", "tau", "checkpoint", "free_miss", "fatal",
                "missed_obs", "iou_free", "f1@10cm", "f1@20cm", "range_mae", "range_bias"]
        with sweep_rows_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"\n  τ 스윕 원시 행 -> {sweep_rows_path}")

    results_csv = out_dir / "RESULTS.csv"
    with results_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("cell", "seed", "epoch_kind", "epoch", "metric", "value"))
        for cell in order:
            for seed in sorted(arms[cell]):
                entry = arms[cell][seed]
                for kind, epoch in (("fixed", fixed_epoch),
                                    ("best", entry["selected_epoch"])):
                    for _, name, _ in METRICS:
                        value = entry[kind].get(name)
                        if value is None:
                            continue
                        writer.writerow((cell, seed, kind, epoch, name, repr(float(value))))
                for key in ("rebound_excess_pct", "rebound_raw_pct"):
                    if entry[key] is not None:
                        writer.writerow((cell, seed, "run", entry["last_epoch"], key,
                                         repr(float(entry[key]))))
    print(f"  런별 지표 -> {results_csv}")

    payload = {
        "probe_root": str(probe_root), "control_root": str(control_root),
        "arm_roots": {c: str(r) for c, r in arm_roots.items()},
        "probe_cell": probe_cell, "control_cells": control_cells,
        "seeds": seeds, "fixed_epoch": int(fixed_epoch), "taus": list(taus),
        "incomplete_runs": incomplete,
        "selected_epochs": {c: {str(s): arms[c][s]["selected_epoch"]
                                for s in sorted(arms[c])} for c in order},
        "summary": {
            which_epoch: {
                cell: {name: dict(zip(("mean", "sd", "n"),
                                      _mean_sd([arms[cell][s][which_epoch].get(name)
                                                for s in sorted(arms[cell])])))
                       for _, name, _ in METRICS}
                for cell in order}
            for which_epoch in ("fixed", "best")},
        "rebound": {cell: dict(zip(("mean", "sd", "n"),
                                   _mean_sd([arms[cell][s]["rebound_excess_pct"]
                                             for s in sorted(arms[cell])])))
                    for cell in order},
        "paired": {f"{w}|{m}|{c}": v for (w, m, c), v in paired.items()},
        "threshold_anchors": sweep_summary,
    }
    results_json = out_dir / "RESULTS.json"
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  요약 -> {results_json}")
    return 0


if __name__ == "__main__":
    Fire(main)
