#!/usr/bin/env python
"""Figure 3 -- **soft target이 치르는 값: 경계가 조금 흐려진다.**

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/table1_ladder_accuracy.csv`와 `../data/table5_threshold_dispersion.csv`,
matplotlib 외 의존성이 없다.

**[2026-10-02] 지표를 BF@τ(프레임 macro, τ = 10/20/30 cm)와 recall 축으로 바꿨다**(옛
`F1@τ`는 micro, τ = 10/20/40 cm, 옛 x축은 free-miss = 1 − recall). 주석의 숫자는 데이터에서
계산한다. 아래 서술의 수치는 옛 결과이고 다음 캠페인 결과로 다시 확인해야 한다.

**왜 이 그림이 필요한가.** 이 절은 "경계의 불확실성을 모델링한다"고 주장하는데 **경계
정밀도 지표 `F1@10cm`은 오히려 내려간다**(−0.012, 5/5 시드). 감추면 심사자가 먼저 찾는다.
그래서 정면으로 싣고, 그것이 **설계가 의도한 그대로의 자국**임을 보인다.

**왼쪽(a).** 허용 오차를 10 → 20 → 40 cm로 늘리면 손해가 −0.012 → −0.007 → −0.001로
**줄어들다 사라진다.** 경계가 **끊긴** 것이 아니라 **번진** 것이라는 뜻이다. 끊겼다면
허용 오차를 넓혀도 회복되지 않는다.

**오른쪽(b).** 같은 free-miss에서 봐도 `F1@10cm` 격차는 남는다(−0.012 ~ −0.048).
즉 이것은 §Figure 2의 `fatal`과 달리 **동작점 이동이 아니라 실재하는 비용**이다.
같은 자리에서 `IoU_free`는 −0.0005 ~ −0.0030으로 사실상 같다 -- **비용은 경계에만 있다.**

**한 문장으로.** 대역을 soft로 감독하면 그 대역에서 확률 경사가 완만해진다. 목적함수가
정상화되는 것과 경계가 번지는 것은 **같은 손잡이의 양면**이다.

실행:
    python make_boundary_precision_cost.py
    python make_boundary_precision_cost.py --formats=pdf --dpi=600
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-03-fig3"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "data" / "table1_ladder_accuracy.csv"
DISPERSION = HERE.parent / "data" / "table5_threshold_dispersion.csv"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

CELL_STYLE = {
    "C_soft": ("Soft boundary target (proposed)", "#2a78d6", "-", "o"),
    "A_ce": ("Weighted CE (baseline)", "#eb6834", "--", "s"),
    "C_hard": ("Hard boundary target", "#a555c7", "-.", "^"),
}
ORDER = ("A_ce", "C_hard", "C_soft")
TOLERANCES = ((10, "bf_10cm"), (20, "bf_20cm"), (30, "bf_30cm"))


def load():
    at_tau50 = {}
    with LADDER.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] not in CELL_STYLE:
                continue
            at_tau50[row["cell"]] = {
                tol: (float(row[f"{col}_mean"]), float(row[f"{col}_sd"] or 0.0))   # 1시드면 빈 칸
                for tol, col in TOLERANCES}

    anchored = defaultdict(list)
    with DISPERSION.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] not in CELL_STYLE:
                continue
            anchored[row["cell"]].append((
                float(row["recall_anchor"]),
                float(row["bf_10cm_at_anchor_mean"]),
                float(row["bf_10cm_at_anchor_sd"] or 0.0)))
    for cell in anchored:
        anchored[cell].sort()
    return at_tau50, anchored


def build(at_tau50, anchored):
    fig, (ax_tol, ax_anchor) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # --- (a) 허용 오차를 넓히면 손해가 사라진다 ------------------------------------------
    xs = [tol for tol, _ in TOLERANCES]
    for cell in ORDER:
        label, color, dash, marker = CELL_STYLE[cell]
        ys = [at_tau50[cell][tol][0] for tol in xs]
        es = [at_tau50[cell][tol][1] for tol in xs]
        ax_tol.errorbar(xs, ys, yerr=es, color=color, linestyle=dash, linewidth=2.0,
                        marker=marker, markersize=6.0, markeredgecolor="#fcfcfb",
                        markeredgewidth=1.0, capsize=4, capthick=1.4, elinewidth=1.6,
                        label=label, zorder=3)
    # 격차를 숫자로 적는다 -- 두 곡선 사이를 눈으로 재게 하지 않는다. 곡선을 따라 붙이면
    # 어느 위치에서든 눈금이나 표식과 부딪히므로 **빈 왼쪽 위에 한 덩이로** 둔다.
    gaps = "\n".join(
        f"  {tol} cm:  {at_tau50['C_soft'][tol][0] - at_tau50['A_ce'][tol][0]:+.3f}"
        for tol in xs)
    ax_tol.text(0.035, 0.96, "proposed − baseline\n" + gaps, transform=ax_tol.transAxes,
                ha="left", va="top", fontsize=8.8, color=INK_MUTED, linespacing=1.45)
    ax_tol.set_title("(a)  The cost shrinks as the tolerance widens",
                     fontsize=10, color=INK, pad=9, loc="left")
    ax_tol.set_xlabel("boundary tolerance  [cm]", fontsize=9.5, color=INK_MUTED)
    ax_tol.set_ylabel("boundary F-measure  BF  ↑\n(at $\\tau = 0.5$, ±1 SD over 5 seeds)",
                      fontsize=9.5, color=INK_MUTED)
    ax_tol.set_xticks(xs)

    # --- (b) 같은 동작점에서도 남는다 = 동작점 이동이 아니다 -------------------------------
    for cell in ORDER:
        label, color, dash, marker = CELL_STYLE[cell]
        pts = anchored[cell]
        ax_anchor.errorbar([p[0] for p in pts], [p[1] for p in pts],
                           yerr=[p[2] for p in pts], color=color, linestyle=dash,
                           linewidth=2.0, marker=marker, markersize=5.5,
                           markeredgecolor="#fcfcfb", markeredgewidth=1.0,
                           capsize=4, capthick=1.4, elinewidth=1.6, zorder=3)
    ax_anchor.set_title("(b)  It survives matching the operating point",
                        fontsize=10, color=INK, pad=9, loc="left")
    ax_anchor.set_xlabel("recall   (of truly free cells, fraction found)",
                         fontsize=9.5, color=INK_MUTED)
    ax_anchor.set_ylabel("boundary F-measure  $BF_{10\\,\\mathrm{cm}}$  ↑",
                         fontsize=9.5, color=INK_MUTED)
    # 격차 범위는 데이터에서 계산한다(같은 앵커끼리 proposed − baseline).
    gaps_at = [s[1] - a[1] for s, a in zip(anchored["C_soft"], anchored["A_ce"])]
    # 화살표를 쓰지 않는다 -- 어느 각도로 그어도 곡선을 가로지른다. 빈 오른쪽 아래에 둔다.
    ax_anchor.text(0.97, 0.06,
                   f"proposed − baseline at matched recall:  "
                   f"{min(gaps_at):+.3f} to {max(gaps_at):+.3f}",
                   transform=ax_anchor.transAxes, ha="right", va="bottom",
                   fontsize=8.8, color=INK_MUTED)

    for ax in (ax_tol, ax_anchor):
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
    parser.add_argument("--stem", default="fig3_boundary_precision_cost")
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
