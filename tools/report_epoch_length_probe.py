"""epoch 길이 사전 실험(`configs/epoch_length_probe.sh`)의 곡선과 요약. **이 스크립트가 정본이다.**

`runs/99_epoch_exp/{ep040,ep100}/logs/{A_ce,C_soft}_s0`의 TensorBoard를 읽어
`runs/99_epoch_exp/analysis/`에 셋을 쓴다.

    curves.csv    런 x epoch x 지표 (긴 형식)
    summary.csv   런마다 마지막 epoch · 최근 10 epoch 평균 · best epoch · 진단 CE 되올림
    curves.png    여섯 칸: val iou_free / val BF@0.10 / 그 런이 최적화한 손실의 세 영역
                  (경계에서 먼 free · 먼 non-free · 경계 대역) / 경계 대역의 공통 눈금 hard CE

**손실 로그는 모든 실험이 같은 7개다**(2026-10-02, `metric_spec` 손실 절). `loss_*`는 모델이 실제로
받은 손실의 영역별 값이라 런마다 함수가 다르고(soft-BCE의 경계 항은 하한으로 수렴한다 -- 점선,
값은 `config.json`의 `label_constants`), `bce_*`는 모든 런에 같은 함수라 논문에 싣는다.
2026-10-02 이전에 기록된 런에는 이 tag가 없다.

**시드 하나다** -- 런 사이 차이를 우열로 읽지 않는다(사전 실험).

실행:  python tools/report_epoch_length_probe.py
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fire import Fire  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from projects.common.metric_spec import (  # noqa: E402
    PAPER_LOSS_TERMS, PAPER_METRICS, TRAINING_LOSS_TERMS)

CELLS = {"A_ce": ("BCE (weighted)", "#eb6834"), "C_soft": ("soft-BCE (proposed)", "#2a78d6")}
EPOCHS = {40: "--", 100: "-"}
DIAGNOSTIC = ("bce_free", "bce_non_free", "bce_boundary")
PANELS = (("iou_free", "val IoU_free  ↑"), ("bf_10cm", "val BF@0.10 m  ↑"),
          ("loss_free", "val loss_free  ↓\n(own objective, d > 0.30 m)"),
          ("loss_non_free", "val loss_non_free  ↓\n(own objective, d < −0.30 m)"),
          ("loss_boundary", "val loss_boundary  ↓\n(own objective, |d| ≤ 0.30 m; dotted = floor)"),
          ("bce_boundary", "val bce_boundary  ↓\n(same BCE for every run, |d| ≤ 0.30 m)"))


def _floor(run_dir):
    """경계 항의 하한 -- 라벨만의 상수라 `config.json`의 `label_constants`에 있다."""
    import json
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    return float(cfg.get("label_constants", {}).get("val", {}).get("loss_boundary_floor", 0.0))


def _load(path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return {t: {e.step: e.value for e in acc.Scalars(t)} for t in acc.Tags()["scalars"]}


def main(root="runs/99_epoch_exp"):
    root = Path(root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    names = [n for _, n, _ in PAPER_METRICS] + list(TRAINING_LOSS_TERMS + PAPER_LOSS_TERMS)
    runs = {(c, e): _load(root / f"ep{e:03d}" / "logs" / f"{c}_s0") for e in EPOCHS for c in CELLS}

    with (out / "curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "num_epochs", "epoch", "metric", "value"])
        for (c, e), s in runs.items():
            for n in names + ["train_iou_free"]:
                tag = "train/iou_free_epoch" if n == "train_iou_free" else f"val/{n}_epoch"
                for ep, v in sorted(s.get(tag, {}).items()):
                    w.writerow([c, e, ep, n, v])

    rows = []
    for (c, e), s in runs.items():
        last = max(s["val/iou_free_epoch"])
        iou = s["val/iou_free_epoch"]
        best = max(iou, key=iou.get)
        row = {"cell": c, "num_epochs": e, "best_iou_epoch": best, "best_iou_free": iou[best]}
        for n in [n for _, n, _ in PAPER_METRICS]:
            row[f"{n}_last"] = s[f"val/{n}_epoch"][last]
            row[f"{n}_last10_mean"] = statistics.fmean(
                s[f"val/{n}_epoch"][k] for k in range(last - 9, last + 1))
        row["train_iou_free_last"] = s["train/iou_free_epoch"][last]
        for n in DIAGNOSTIC:
            curve = s[f"val/{n}_epoch"]
            m = min(curve, key=curve.get)
            row[f"{n}_min_epoch"] = m
            row[f"{n}_rise_pct"] = 100 * (curve[last] / curve[m] - 1)
        rows.append(row)
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4))
    for ax, (metric, label) in zip(axes.flat, PANELS):
        for (c, e), s in runs.items():
            curve = s[f"val/{metric}_epoch"]
            xs = sorted(curve)
            ax.plot(xs, [curve[x] for x in xs], color=CELLS[c][1], linestyle=EPOCHS[e],
                    linewidth=1.6, label=f"{CELLS[c][0]}, {e} ep")
            if metric == "loss_boundary":
                floor = _floor(root / f"ep{e:03d}" / "logs" / f"{c}_s0")
                if floor > 0:
                    ax.axhline(floor, color=CELLS[c][1], linestyle=":", linewidth=1.0)
        ax.set_title(label, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("epoch", fontsize=9, color="#52514e")
        ax.grid(True, color="#d8d7d2", linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5, colors="#52514e")
    # 정확도 두 칸은 초기 급상승이 눈금을 잡아먹으므로 후반부가 보이게 아래를 자른다.
    for ax, metric in zip(axes.flat[:2], ("iou_free", "bf_10cm")):
        late = [v for s in runs.values() for ep, v in s[f"val/{metric}_epoch"].items() if ep >= 10]
        ax.set_ylim(min(late) - 0.01, max(late) + 0.005)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Epoch-length probe (seed 0, one run each) — OneCycle LR stretches with the run length",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)
    for p in ("curves.csv", "summary.csv", "curves.png"):
        print(f"  {out / p}")


if __name__ == "__main__":
    Fire(main)
