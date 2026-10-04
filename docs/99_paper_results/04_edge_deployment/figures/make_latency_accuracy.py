#!/usr/bin/env python
"""Figure 1 -- **인코더만 TensorRT로 바꾸면 얼마나 빨라지고, 무엇을 잃는가** (Jetson AGX Orin).

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/table1_speed.csv`와 `../data/table2_accuracy_paired.csv`, matplotlib 외 의존성이 없다.

**(a)** 구성별 한 프레임 지연을 인코더와 BEV 단계로 나눈 막대. BEV 단계는 세 구성에서 같다(PyTorch fp32)
-- 인코더를 줄일수록 BEV 단계의 몫이 커진다는 것이 한눈에 보인다. 막대 끝에 FPS.

**(b)** PyTorch fp32 대비 지표 변화(시드 5개로 짝지은 평균, 막대 = 95 % 신뢰구간). 속이 찬 점은
신뢰구간이 0을 빼는(유의한) 칸이다.

실행:
    python make_latency_accuracy.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-04-fig1"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
SPEED = HERE.parent / "data" / "table1_speed.csv"
ACCURACY = HERE.parent / "data" / "table2_accuracy_paired.csv"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
ENC_COLOR = "#2a78d6"
BEV_COLOR = "#b9b8b2"

CONFIG_STYLE = {
    "pytorch_fp32": ("PyTorch fp32", "#52514e", "o"),
    "trt_fp16": ("TensorRT fp16 encoder", "#2a78d6", "o"),
    "trt_int8": ("TensorRT INT8 encoder", "#eb6834", "s"),
}
ORDER = ("pytorch_fp32", "trt_fp16", "trt_int8")
METRIC_LABELS = (("iou_free", "IoU$_{free}$"), ("precision", "Precision"), ("recall", "Recall"),
                 ("bf_10cm", "BF\n@0.10 m"), ("bf_20cm", "BF\n@0.20 m"), ("bf_30cm", "BF\n@0.30 m"))


def load():
    with SPEED.open(encoding="utf-8") as fh:
        speed = {r["config"]: r for r in csv.DictReader(fh)}
    acc = {}
    with ACCURACY.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["paired_diff_vs_fp32_mean"]:
                acc[(r["config"], r["metric"])] = (float(r["paired_diff_vs_fp32_mean"]),
                                                   float(r["ci95_low"]), float(r["ci95_high"]),
                                                   r["significant"] == "True")
    return speed, acc


def build(speed, acc):
    fig, (ax_lat, ax_acc) = plt.subplots(1, 2, figsize=(10.4, 4.3),
                                         gridspec_kw={"width_ratios": [1.05, 1.0]})

    # --- (a) 지연 분해 ------------------------------------------------------------------
    ys = list(range(len(ORDER)))[::-1]
    for y, cfg in zip(ys, ORDER):
        r = speed[cfg]
        enc, bev = float(r["encoder_ms_mean"]), float(r["bev_stage_ms_mean"])
        total, fps = float(r["end_to_end_ms_mean"]), float(r["fps_mean"])
        ax_lat.barh(y, enc, color=ENC_COLOR, height=0.56, edgecolor="#fcfcfb", linewidth=1.5)
        ax_lat.barh(y, bev, left=enc, color=BEV_COLOR, height=0.56, edgecolor="#fcfcfb", linewidth=1.5)
        ax_lat.annotate(f"{total:.1f} ms  ·  {fps:.1f} FPS", xy=(enc + bev, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
    ax_lat.set_yticks(ys)
    ax_lat.set_yticklabels([CONFIG_STYLE[c][0] for c in ORDER], fontsize=9.5, color=INK)
    ax_lat.set_xlabel("per-frame latency  [ms]   (3 cameras, 512×288, batch 1)",
                      fontsize=9.5, color=INK_MUTED)
    ax_lat.set_xlim(0, float(speed["pytorch_fp32"]["end_to_end_ms_mean"]) * 1.42)
    ax_lat.set_title("(a)  Where the time goes", fontsize=10, color=INK, pad=9, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENC_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=BEV_COLOR)]
    # 범례는 축 아래로 -- 축 안에 두면 막대 끝의 지연·FPS 글자와 겹친다(2026-10-04 실측).
    ax_lat.legend(handles, ["image encoder (ResNet-101)", "BEV stage (PyTorch fp32)"],
                  loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=8.8)
    ax_lat.grid(True, axis="x", color=GRID, linewidth=0.7)

    # --- (b) 지표 변화 ------------------------------------------------------------------
    xs = list(range(len(METRIC_LABELS)))
    for offset, cfg in ((-0.14, "trt_fp16"), (0.14, "trt_int8")):
        label, color, marker = CONFIG_STYLE[cfg]
        for x, (metric, _) in zip(xs, METRIC_LABELS):
            mean, lo, hi, sig = acc[(cfg, metric)]
            ax_acc.errorbar([x + offset], [mean], yerr=[[mean - lo], [hi - mean]], color=color,
                            marker=marker, markersize=6.5, capsize=3.5, elinewidth=1.5,
                            markerfacecolor=color if sig else "#fcfcfb", markeredgewidth=1.5,
                            linestyle="none", zorder=3)
        ax_acc.plot([], [], color=color, marker=marker, linestyle="none", label=label)
    ax_acc.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels([lbl for _, lbl in METRIC_LABELS], fontsize=9, color=INK)
    ax_acc.set_ylabel("change vs PyTorch fp32\n(paired over 5 seeds, 95 % CI)",
                      fontsize=9.5, color=INK_MUTED)
    ax_acc.set_title("(b)  What the speed costs", fontsize=10, color=INK, pad=9, loc="left")
    ax_acc.legend(loc="lower left", frameon=False, fontsize=8.8,
                  title="filled = CI excludes 0", title_fontsize=8.3)
    ax_acc.grid(True, axis="y", color=GRID, linewidth=0.7)

    for ax in (ax_lat, ax_acc):
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="pdf,png,svg")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    fig = build(*load())
    for fmt in args.formats.split(","):
        out = HERE / f"fig1_latency_accuracy.{fmt}"
        fig.savefig(out, dpi=args.dpi, metadata=_NO_TIMESTAMP.get(fmt, {}))
        print(f"  figures/{out.name}")


if __name__ == "__main__":
    main()
