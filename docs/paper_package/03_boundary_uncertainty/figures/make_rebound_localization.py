#!/usr/bin/env python
"""Figure 1 -- **되올림은 경계 대역에만 있고, soft target이 그것을 없앤다.**

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/figure_rebound_localization.csv` 하나, matplotlib 외 의존성이 없다.

**무엇을 보이나.** 왼쪽은 경계 대역(`|d| <= 0.15 m`, 유효 셀의 12.7 %)의 validation CE,
오른쪽은 그 바깥이다. 기존 CE(`A_ce`)와 hard 대역 감독(`C_hard`)은 왼쪽에서 최저점 뒤로
크게 되올라가는데 오른쪽은 거의 평평하다 -- **오차 증가가 경계에 몰려 있다.** 대역 target을
soft로 바꾸면(`C_soft`) 왼쪽의 되올림이 사라진다.

**눈금이 왜 CE인가.** 각 조건의 `val/loss`는 서로 다른 함수라 한 그림에 겹쳐 그릴 수 없다.
여기 쓴 CE는 **loss 종류와 무관하게 같은 식으로 기록되는 진단값**이고 gradient에 들어가지
않는다. 그래서 세 조건을 같은 축에 놓을 수 있다.

**`C_hard`가 이 그림의 핵심이다.** `C_soft`와 집계·가중치·δ가 전부 같고 **대역 target만
hard**다. 그런데 곡선이 기존 CE 쪽에 붙는다 -- 원인이 집계 방식이 아니라 hard target임을
가른다.

띠는 시드 5개의 ±1 표준편차다.

실행:
    python make_rebound_localization.py
    python make_rebound_localization.py --formats=pdf --dpi=600
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-03-fig1"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "figure_rebound_localization.csv"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

# 검증된 3색(대비·색각 이상 통과). 파랑 = 확정 설정으로 실험 2와 같은 뜻을 유지한다.
CELL_STYLE = {
    "C_soft": ("Soft boundary target (proposed)", "#2a78d6", "-"),
    "A_ce": ("Weighted CE (baseline)", "#eb6834", "--"),
    "C_hard": ("Hard boundary target", "#a555c7", "-."),
}
ORDER = ("A_ce", "C_hard", "C_soft")

PANELS = (
    ("ce_boundary", "(a)  Boundary band   $|d| \\leq 0.15$ m\n12.7 % of valid cells"),
    ("ce_confident", "(b)  Outside the band\n87.3 % of valid cells"),
)


def load(path: Path = DATA) -> dict:
    series = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] not in CELL_STYLE:
                continue
            series[(row["cell"], row["metric"])][int(row["epoch"])] = (
                float(row["mean"]), float(row["sd"]) if row["sd"] != "" else 0.0)
    return series


def build(series: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    for ax, (metric, title) in zip(axes, PANELS):
        for cell in ORDER:
            points = series.get((cell, metric))
            if not points:
                continue
            label, color, dash = CELL_STYLE[cell]
            epochs = sorted(points)
            mean = [points[e][0] for e in epochs]
            sd = [points[e][1] for e in epochs]
            ax.fill_between(epochs, [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)],
                            color=color, alpha=0.14, linewidth=0)
            ax.plot(epochs, mean, color=color, linestyle=dash, linewidth=2.0,
                    label=label, solid_capstyle="round")
            # 최저점을 찍어 "되올림이 어디서 시작하나"를 직접 읽게 한다.
            best = min(epochs, key=lambda e: points[e][0])
            ax.plot([best], [points[best][0]], marker="o", markersize=5.0,
                    color=color, markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=5)

        ax.set_title(title, fontsize=10, color=INK, pad=9, loc="left")
        ax.set_xlabel("epoch", fontsize=9.5, color=INK_MUTED)
        ax.set_xlim(1, 40)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    axes[0].set_ylabel("validation cross-entropy  ↓\n(diagnostic, not the training loss)",
                       fontsize=9.5, color=INK_MUTED)
    axes[1].set_ylabel("validation cross-entropy  ↓", fontsize=9.5, color=INK_MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.045),
               labelcolor=INK, handlelength=2.4, columnspacing=2.0)
    # 최저점 표식이 무엇인지 한 줄로 밝힌다 -- 범례 항목을 늘리지 않는다.
    fig.text(0.5, -0.115, "● marks each curve's minimum; the rise after it is the rebound.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", default="pdf,png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stem", default="fig1_rebound_localization")
    args = parser.parse_args()

    fig = build(load())
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        out = HERE / f"{args.stem}.{fmt}"
        fig.savefig(out, format=fmt, dpi=args.dpi, bbox_inches="tight",
                    metadata=_NO_TIMESTAMP.get(fmt, {}))
        print(f"  {out.relative_to(HERE.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
