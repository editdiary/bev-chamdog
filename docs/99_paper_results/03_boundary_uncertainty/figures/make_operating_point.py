#!/usr/bin/env python
"""Figure 2 -- **동작점 곡선은 그대로이고, 그 위에 서기 위한 문턱만 재현 가능해진다.**

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/table5_threshold_dispersion.csv`와 `../data/table1_ladder_accuracy.csv`
둘, matplotlib 외 의존성이 없다.

**[2026-10-02] 축을 recall/precision(프레임 macro)으로 바꿨다**(옛 free-miss = 1 − recall,
fatal = 1 − precision, 둘 다 micro). 주석의 숫자와 위치는 **데이터에서 계산한다** -- 예전에는
옛 데이터 기준으로 하드코딩돼 있었다. 아래 서술의 수치(±0.07~0.10 등)는 옛 결과이고 다음
캠페인 결과로 다시 확인해야 한다.

**왼쪽(a).** 목표 동작점(recall)을 맞추려면 결정 문턱 $\\tau$를 얼마로 놔야 하는가,
그리고 그 값이 **재학습마다 얼마나 흔들리는가**. 세로 막대가 시드 5개의 ±1 표준편차다.
기존 CE는 ±0.07~0.10으로 흔들려서 다시 학습할 때마다 문턱을 다시 잡아야 한다. soft target은
±0.006~0.011이다 -- 구간 전체에서 **6~16배 좁다.**

**오른쪽(b).** 같은 free-miss에서의 fatal, 즉 안전-효용 곡선 자체다. **세 곡선이 겹친다.**
그래서 $\\tau = 0.5$ 한 점에서 보이는 fatal 차이(속이 빈 표식들이 서로 다른 자리에 있다)는
**곡선이 좋아진 것이 아니라 그 위에서 동작점이 옮겨 간 것**이다.

**이 그림은 주장 하나를 세우고 하나를 반증한다.** 논문에서 `fatal` 개선을 단독으로 쓰지
않는 근거가 오른쪽 panel이다.

실행:
    python make_operating_point.py
    python make_operating_point.py --formats=pdf --dpi=600
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-03-fig2"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
DISPERSION = HERE.parent / "data" / "table5_threshold_dispersion.csv"
LADDER = HERE.parent / "data" / "table1_ladder_accuracy.csv"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

CELL_STYLE = {
    "C_soft": ("Soft boundary target (proposed)", "#2a78d6", "-", "o"),
    "A_ce": ("Weighted CE (baseline)", "#eb6834", "--", "s"),
    "C_hard": ("Hard boundary target", "#a555c7", "-.", "^"),
}
ORDER = ("A_ce", "C_hard", "C_soft")


def load():
    curves = defaultdict(list)
    with DISPERSION.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] not in CELL_STYLE:
                continue
            curves[row["cell"]].append((
                float(row["recall_anchor"]), float(row["tau_star_at_anchor_mean"]),
                float(row["tau_star_at_anchor_sd"] or 0.0), float(row["precision_at_anchor_mean"]),
                float(row["precision_at_anchor_sd"] or 0.0)))
    for cell in curves:
        curves[cell].sort()

    tau50 = {}
    with LADDER.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] not in CELL_STYLE:
                continue
            tau50[row["cell"]] = (float(row["recall_mean"]),
                                  float(row["precision_mean"]))
    return curves, tau50


def build(curves, tau50):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax_tau, ax_curve = axes

    # --- (a) 목표 동작점을 만드는 문턱과 그 산포 --------------------------------------
    for cell in ORDER:
        label, color, dash, marker = CELL_STYLE[cell]
        pts = curves[cell]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax_tau.errorbar(xs, ys, yerr=es, color=color, linestyle=dash, linewidth=2.0,
                        marker=marker, markersize=5.5, markeredgecolor="#fcfcfb",
                        markeredgewidth=1.0, capsize=4, capthick=1.4,
                        elinewidth=1.6, label=label, zorder=3)
    ax_tau.set_title("(a)  Threshold needed for a target safety level",
                     fontsize=10, color=INK, pad=9, loc="left")
    ax_tau.set_ylabel("decision threshold  $\\tau^\\ast$\n(bars: ±1 SD over 5 seeds)",
                      fontsize=9.5, color=INK_MUTED)
    # 흔들림 폭을 숫자로 한 번 못박는다 -- 막대만으로는 배수가 안 읽힌다. 가운데 앵커의
    # 값을 데이터에서 읽고, 글자는 점에서 **화면 좌표로** 띄운다(축 범위가 바뀌어도 맞는다).
    for cell, dy in (("A_ce", -34), ("C_soft", 30)):
        x, y, sd = curves[cell][len(curves[cell]) // 2][:3]
        color = CELL_STYLE[cell][1]
        ax_tau.annotate(f"±{sd:.3f}", xy=(x, y), xytext=(14, dy),
                        textcoords="offset points", fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=1.1))

    # --- (b) 안전-효용 곡선 자체 -------------------------------------------------------
    for cell in ORDER:
        label, color, dash, marker = CELL_STYLE[cell]
        pts = curves[cell]
        xs = [p[0] for p in pts]
        ys = [p[3] for p in pts]
        # 여기서는 곡선만 그린다 -- 표식은 tau=0.5가 어디에 떨어지는가 하나만 쓴다.
        ax_curve.plot(xs, ys, color=color, linestyle=dash, linewidth=2.0,
                      label=label, zorder=3, solid_capstyle="round")
        fm, fatal = tau50[cell]
        ax_curve.plot([fm], [fatal], marker=marker, markersize=10.0,
                      markerfacecolor="none", markeredgecolor=color,
                      markeredgewidth=2.0, linestyle="none", zorder=4)
    ax_curve.set_title("(b)  The safety–utility curve itself",
                       fontsize=10, color=INK, pad=9, loc="left")
    ax_curve.set_ylabel("precision  ↑\n(of cells called free, fraction right)",
                        fontsize=9.5, color=INK_MUTED)
    # 점에서 화면 좌표로 띄운다 -- 데이터 좌표로 적으면 지표·런이 바뀔 때마다 엉뚱한 데 간다.
    # 아래로 띄운다 -- 위로 띄우면 C_soft의 τ=0.5 표식(가장 높은 점)과 겹친다(2026-10-03).
    ax_curve.annotate("open marks:\nwhere $\\tau = 0.5$\nlands",
                      xy=(tau50["A_ce"][0], tau50["A_ce"][1]),
                      xytext=(14, -40), textcoords="offset points", fontsize=8.8,
                      color=INK_MUTED, ha="left", va="top",
                      arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.0,
                                      connectionstyle="arc3,rad=0.25"))

    for ax in axes:
        ax.set_xlabel("recall   (of truly free cells, fraction found)",
                      fontsize=9.5, color=INK_MUTED)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    handles = [Line2D([], [], color=CELL_STYLE[c][1], linestyle=CELL_STYLE[c][2],
                      marker=CELL_STYLE[c][3], markersize=5.5, linewidth=2.0,
                      markeredgecolor="#fcfcfb", label=CELL_STYLE[c][0]) for c in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.05), labelcolor=INK,
               handlelength=2.6, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", default="pdf,png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stem", default="fig2_operating_point")
    args = parser.parse_args()

    fig = build(*load())
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        out = HERE / f"{args.stem}.{fmt}"
        fig.savefig(out, format=fmt, dpi=args.dpi, bbox_inches="tight",
                    metadata=_NO_TIMESTAMP.get(fmt, {}))
        print(f"  {out.relative_to(HERE.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
