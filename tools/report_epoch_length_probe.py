"""epoch 길이 사전 실험(`configs/epoch_length_probe.sh`)의 곡선과 요약. **이 스크립트가 정본이다.**

`runs/99_epoch_exp/{ep040,ep100}/logs/{A_ce,C_soft}_s0`의 TensorBoard를 읽어
`runs/99_epoch_exp/analysis/`에 셋을 쓴다.

    curves.csv    런 x epoch x 지표 (긴 형식)
    summary.csv   런마다 마지막 epoch · 최근 10 epoch 평균 · best epoch · 진단 CE 되올림
    curves.png    네 칸: val iou_free / val BF@10cm / 진단 hard CE(전체) / 진단 hard CE(경계 대역)

**진단 CE를 그리는 이유.** 각 런의 `val/loss_epoch`은 자기 손실 함수의 값이라 BCE 런과 soft 런을
한 축에 놓을 수 없다. `val/ce_*`는 손실과 무관하게 hard 0/1 target에 대한 CE라 같은 눈금이다
(`binary_metrics.region_ce_diagnostics`).

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
from projects.common.metric_spec import PAPER_METRICS  # noqa: E402

CELLS = {"A_ce": ("BCE (weighted)", "#eb6834"), "C_soft": ("soft-BCE (proposed)", "#2a78d6")}
EPOCHS = {40: "--", 100: "-"}
DIAGNOSTIC = ("ce_all", "ce_boundary", "ce_confident")
PANELS = (("iou_free", "val IoU_free  ↑"), ("bf_10cm", "val BF@0.10 m  ↑"),
          ("ce_all", "val hard CE, all cells  ↓\n(loss-independent)"),
          ("ce_boundary", "val hard CE, boundary band  ↓\n(|d| ≤ 0.15 m)"))


def _load(path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return {t: {e.step: e.value for e in acc.Scalars(t)} for t in acc.Tags()["scalars"]}


def main(root="runs/99_epoch_exp"):
    root = Path(root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    names = [n for _, n, _ in PAPER_METRICS] + list(DIAGNOSTIC) + ["loss"]
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

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (metric, label) in zip(axes.flat, PANELS):
        for (c, e), s in runs.items():
            curve = s[f"val/{metric}_epoch"]
            xs = sorted(curve)
            ax.plot(xs, [curve[x] for x in xs], color=CELLS[c][1], linestyle=EPOCHS[e],
                    linewidth=1.6, label=f"{CELLS[c][0]}, {e} ep")
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
