#!/usr/bin/env python
"""Figure 1 -- 어안이 이기는 이유는 해상도가 아니라 **커버리지**다.

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/figure_coverage_vs_accuracy.csv`와 `../data/projection_tradeoff.csv`,
matplotlib 외 의존성이 없다.

**무엇을 보이나.** 가로축이 BEV 격자 커버리지, 세로축이 `iou_free`다. 세 팔이 거의 직선
위에 놓이는 것이 첫째 주장이고 -- 커버리지가 정확도를 예측한다 -- 핀홀 120°가 150°보다
**정면 해상도는 2.2배 높은데도** 더 나쁜 것이 둘째 주장이다. 즉 이 과제에서 시야가
해상도보다 중요하다. 점 옆의 괄호가 정면 각해상도이고, 그것이 둘째 주장의 근거다.

세로 오차막대는 시드 5개의 표준편차다. 대조군과는 **시드끼리 짝지어** 비교하므로 실제
판정 노이즈는 이 막대보다 훨씬 작다(짝지은 |Δ|/SE는 21~27이다) -- 캡션에 적는다.

실행:
    python make_coverage_vs_accuracy.py
    python make_coverage_vs_accuracy.py --formats=pdf --dpi=600
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 재생성해도 바이트가 같게 만든다 -- 근거는 `01_overall/figures/make_margin_inversion.py`.
matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-02-fig1"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
ARMS_CSV = HERE.parent / "data" / "figure_coverage_vs_accuracy.csv"
SWEEP_CSV = HERE.parent / "data" / "projection_tradeoff.csv"

#: dataviz 기본 팔레트 slot 1/2 -- 6가지 검사를 통과한 조합이다.
FISHEYE_COLOR = "#2a78d6"   # slot 1 blue  = 확정 설정(DS-native)
PINHOLE_COLOR = "#eb6834"   # slot 2 orange = 대조 팔(가상 핀홀)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"


def load_arms(path: Path = ARMS_CSV) -> list:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["coverage"] = float(row["bev_coverage_pct"])
        row["iou"] = float(row["iou_free_mean"])
        row["sd"] = float(row["iou_free_sd"])
        row["ppd"] = float(row["front_px_per_deg"])
        row["is_fisheye"] = row["arm"] == "control"
    return sorted(rows, key=lambda r: r["coverage"])


def load_sweep(path: Path = SWEEP_CSV) -> list:
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["projection"] == "pinhole"]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)


def draw(ax, arms) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.6, alpha=0.7)

    # 세 점을 잇는 옅은 선 -- "커버리지가 정확도를 예측한다"는 추세를 눈에 보이게 한다.
    # 회귀선이 아니라 안내선이므로 회색이고 얇다(점 3개로 기울기를 주장하지 않는다).
    ax.plot([r["coverage"] for r in arms], [r["iou"] for r in arms],
            color=GRID, linewidth=1.2, zorder=1)

    for row in arms:
        color = FISHEYE_COLOR if row["is_fisheye"] else PINHOLE_COLOR
        ax.errorbar(row["coverage"], row["iou"], yerr=row["sd"],
                    fmt="o", markersize=9, color=color, ecolor=color,
                    elinewidth=1.6, capsize=3, capthick=1.6,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=3)

    # 직접 라벨. 텍스트는 **본문 잉크 색**을 쓴다 -- 계열 색을 글자에 입히지 않는다.
    labels = {
        "control":    ("Fisheye, native\n(Double Sphere)", -14, 10, "right"),
        "pinhole150": ("Pinhole 150°", 4, -16, "left"),
        "pinhole120": ("Pinhole 120°", 6, 4, "left"),
    }
    for row in arms:
        text, dx, dy, ha = labels[row["arm"]]
        ax.annotate(f"{text}\n({row['ppd']:.2f} px/deg forward)",
                    (row["coverage"], row["iou"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center",
                    fontsize=8, color=INK, linespacing=1.35)

    ax.set_xlabel("BEV grid coverage  [% of 120×120 cells seen by some camera]",
                  fontsize=9, color=INK_MUTED, labelpad=7)
    ax.set_ylabel("free-space IoU  (val, epoch 40, mean of 5 seeds)",
                  fontsize=9, color=INK_MUTED, labelpad=7)
    # 커버리지는 100 %를 넘을 수 없다 -- 축이 그 너머로 가면 눈금이 뜻을 잃는다.
    ax.set_xlim(91.5, 100.9)
    lo = min(r["iou"] - r["sd"] for r in arms)
    hi = max(r["iou"] + r["sd"] for r in arms)
    pad = (hi - lo) * 0.55
    ax.set_ylim(lo - pad, hi + pad * 0.8)

    ax.set_title("Rectifying the fisheye costs coverage, and coverage is what matters",
                 fontsize=10.5, color=INK, pad=12, loc="left")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=7, color=FISHEYE_COLOR,
               markeredgecolor="white", markeredgewidth=1.2, label="Fisheye, used natively"),
        Line2D([], [], marker="o", linestyle="", markersize=7, color=PINHOLE_COLOR,
               markeredgecolor="white", markeredgewidth=1.2, label="Undistorted to a pinhole"),
    ], loc="lower right", frameon=False, fontsize=8, labelcolor=INK_MUTED,
        handletextpad=0.5, borderaxespad=0.8)


def main(formats="pdf,png,svg", dpi: int = 400, stem="fig1_coverage_vs_accuracy"):
    arms = load_arms()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style(ax)
    draw(ax, arms)
    fig.tight_layout()
    for fmt in [f.strip() for f in str(formats).split(",") if f.strip()]:
        out = HERE / f"{stem}.{fmt}"
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight",
                    metadata=_NO_TIMESTAMP.get(fmt, {}))
        print(f"  {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        main()
    else:
        fire.Fire(main)
