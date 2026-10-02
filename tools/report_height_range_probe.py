"""높이 범위 사전 실험(`configs/height_range_probe.sh`)의 곡선과 요약. **이 스크립트가 정본이다.**

`runs/99_height_exp/{설정}/logs/C_soft_s0`의 TensorBoard를 읽어 `runs/99_height_exp/analysis/`에 쓴다.

    curves.csv    런 x epoch x 지표 (긴 형식)
    summary.csv   런마다 마지막 epoch · 최근 10 epoch 평균 · best epoch · 논문용 BCE 되올림 · 학습 시간
    curves.png    여섯 칸: val iou_free / precision / recall / BF@0.10 / bce_boundary / loss_total

**시드 하나다** -- 런 사이 차이를 우열로 읽지 않는다(사전 실험). 시드 간 sd 참고값(5시드, 옛 캠페인):
iou_free 약 0.001, BF@0.10 약 0.006.

실행:  python tools/report_height_range_probe.py
"""
import csv
import json
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

# 이름 -> (그림 범례, 색). 확정값을 파랑으로 둔다(다른 그림과 같은 뜻).
RUNS = {
    "y4_-0.25_1.75": ("Y=4, samples 0 ... 1.5 m (frozen)", "#2a78d6"),
    "y9_-2.25_2.25": ("Y=9, samples -2.0 ... 2.0 m", "#eb6834"),
    "y9_-1.25_3.25": ("Y=9, samples -1.0 ... 3.0 m", "#a555c7"),
}
PANELS = (("iou_free", "val IoU_free  ↑"), ("precision", "val precision  ↑"),
          ("recall", "val recall  ↑"), ("bf_10cm", "val BF@0.10 m  ↑"),
          ("bce_boundary", "val bce_boundary  ↓\n(same BCE for every run, |d| ≤ 0.30 m)"),
          ("loss_total", "val loss_total  ↓\n(the objective the model was trained on)"))


def _load(path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    series = {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags}
    walls = [e.wall_time for e in acc.Scalars("val/iou_free_epoch")]
    first = acc.Scalars("train/loss_step")[0].wall_time if "train/loss_step" in tags else walls[0]
    return series, (walls[-1] - first) / 60.0


def main(root="runs/99_height_exp"):
    root = Path(root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    names = [n for _, n, _ in PAPER_METRICS] + list(TRAINING_LOSS_TERMS + PAPER_LOSS_TERMS)
    runs, minutes, configs = {}, {}, {}
    for tag in RUNS:
        d = root / tag / "logs" / "C_soft_s0"
        if not d.exists():
            print(f"  (없음: {d})")
            continue
        runs[tag], minutes[tag] = _load(d)
        configs[tag] = json.loads((d / "config.json").read_text())

    with (out / "curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "epoch", "metric", "value"])
        for tag, s in runs.items():
            for n in names:
                for ep, v in sorted(s.get(f"val/{n}_epoch", {}).items()):
                    w.writerow([tag, ep, n, v])

    rows = []
    for tag, s in runs.items():
        iou = s["val/iou_free_epoch"]
        last, best = max(iou), max(iou, key=iou.get)
        cfg = configs[tag]
        row = {"run": tag, "height_bins": cfg.get("height_bins"),
               "height_min_m": cfg.get("height_min_m"), "height_max_m": cfg.get("height_max_m"),
               "epochs": last, "train_minutes": round(minutes[tag], 1),
               "best_iou_epoch": best, "best_iou_free": iou[best]}
        for _, n, _ in PAPER_METRICS:
            row[f"{n}_last"] = s[f"val/{n}_epoch"][last]
            row[f"{n}_last10_mean"] = statistics.fmean(
                s[f"val/{n}_epoch"][k] for k in range(last - 9, last + 1))
        for n in PAPER_LOSS_TERMS:
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
        for tag, s in runs.items():
            curve = s[f"val/{metric}_epoch"]
            xs = sorted(curve)
            ax.plot(xs, [curve[x] for x in xs], color=RUNS[tag][1], linewidth=1.5,
                    label=RUNS[tag][0])
        ax.set_title(label, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("epoch", fontsize=9, color="#52514e")
        ax.grid(True, color="#d8d7d2", linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5, colors="#52514e")
        if metric in ("iou_free", "precision", "recall", "bf_10cm"):
            # 초기 급상승이 눈금을 잡아먹으므로 후반부가 보이게 아래를 자른다.
            late = [v for s in runs.values() for ep, v in s[f"val/{metric}_epoch"].items() if ep >= 15]
            ax.set_ylim(min(late) - 0.01, max(late) + 0.005)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Height-range probe (soft-BCE, seed 0, one run each, 100 epochs)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)
    for p in ("curves.csv", "summary.csv", "curves.png"):
        print(f"  {out / p}")


if __name__ == "__main__":
    Fire(main)
