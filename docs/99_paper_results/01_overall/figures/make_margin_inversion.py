#!/usr/bin/env python
"""Figure 1 -- constant-map 기준선 대비 마진이 raw IoU 순위를 뒤집는다.

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/figure_margin_inversion.csv` 하나뿐이고 matplotlib 외 의존성이 없다.

dumbbell 하나로 두 가지를 동시에 보인다. fold를 모델 `iou_free` 내림차순으로 세우면
(i) 막대 길이(=마진)가 세로 순서와 어긋나는 것이 "순위가 뒤집힌다"는 주장이고,
(ii) 주황 점이 넓게 흩어지는데 파랑 점은 모여 있는 것이 "기준선은 장면에 휘둘리지만
모델은 그렇지 않다"는 주장이다. 뒤에 깔린 옅은 띠가 각 계열의 fold 간 범위다.

실행:
    python make_margin_inversion.py
    python make_margin_inversion.py --formats=pdf,png --dpi=600
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 재생성해도 바이트가 같게 만든다. 기본값은 PDF/SVG에 생성 시각과 무작위 element id를
# 박아서, 내용이 같아도 돌릴 때마다 git이 변경으로 잡는다. 그러면 "스크립트가 정본"이
# 무의미해진다 -- 진짜 수정과 잡음을 구별할 수 없기 때문이다.
matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-fig1"
#: 포맷별로 지워야 하는 타임스탬프 키가 다르다.
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "figure_margin_inversion.csv"

#: dataviz 기본 팔레트 slot 1/2. 6가지 검사를 모두 통과한 조합이다
#: (최악 all-pairs CVD ΔE 24.7, 정상시야 ΔE 33.6, 대비 3:1 이상).
MODEL_COLOR = "#2a78d6"     # slot 1 blue
BASELINE_COLOR = "#eb6834"  # slot 2 orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

#: 데이터의 한국어 요인값을 논문 영문 라벨로 옮긴다.
WIDTH_EN = {"좁음": "narrow", "넓음": "wide"}
LIGHTING_EN = {"햇빛": "sunlit", "가림막": "shaded"}


def load(path: Path = DATA) -> list:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["baseline"] = float(row["constant_map_baseline"])
        row["model"] = float(row["model_iou_free"])
        row["margin"] = float(row["margin"])
        row["width_en"] = WIDTH_EN.get(row["corridor_width"], row["corridor_width"])
        row["lighting_en"] = LIGHTING_EN.get(row["lighting"], row["lighting"])
        row["is_extrap"] = row["extrapolation"].strip().lower() in ("true", "1", "yes")
    return rows


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)


def panel_dumbbell(ax, rows) -> None:
    rows = sorted(rows, key=lambda r: r["model"])  # 아래에서 위로 오름차순 = 위가 최고
    ys = range(len(rows))

    # 계열별 fold 간 범위를 옅은 띠로 깐다. 띠 너비의 차이가 곧 "기준선이 5.7배 더
    # 흔들린다"는 수치의 시각적 대응물이다.
    for values, color in ((["baseline"], BASELINE_COLOR), (["model"], MODEL_COLOR)):
        xs = [r[values[0]] for r in rows]
        ax.axvspan(min(xs), max(xs), color=color, alpha=0.09, linewidth=0, zorder=0)
    for y, row in zip(ys, rows):
        ax.plot([row["baseline"], row["model"]], [y, y],
                color=INK_MUTED, linewidth=2, solid_capstyle="round", zorder=1)
        ax.plot(row["baseline"], y, marker="o", markersize=8, color=BASELINE_COLOR,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.plot(row["model"], y, marker="o", markersize=8, color=MODEL_COLOR,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        # 직접 라벨은 마진 하나만. 모든 점에 숫자를 붙이지 않는다.
        ax.annotate(f"+{row['margin']:.3f}", xy=(row["model"], y),
                    xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=INK)

    labels = [f"{r['fold']}†" if r["is_extrap"] else r["fold"] for r in rows]
    sub = [f"{r['lighting_en']} · {r['width_en']}" for r in rows]
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{a}\n{b}" for a, b in zip(labels, sub)], fontsize=8)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if row["is_extrap"] else INK_MUTED)

    ax.set_xlim(0.32, 1.00)
    ax.set_ylim(-0.55, len(rows) + 0.35)
    ax.set_xlabel("IoU$_{\\mathrm{free}}$", fontsize=9, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)

    # 띠 너비를 숫자로도 적어 둔다. 그림만 보고도 캡션 없이 읽히게.
    base = [r["baseline"] for r in rows]
    model = [r["model"] for r in rows]
    top = len(rows) + 0.05
    ax.annotate("constant-map spread across folds",
                xy=((min(base) + max(base)) / 2, top),
                ha="center", va="center", fontsize=7.5, color=INK_MUTED)
    ax.annotate("ours", xy=((min(model) + max(model)) / 2, top),
                ha="center", va="center", fontsize=7.5, color=INK_MUTED)
    _style(ax)


def build(rows):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    fig.patch.set_facecolor("white")
    panel_dumbbell(ax, rows)

    handles = [
        Line2D([], [], marker="o", markersize=8, linestyle="none", color=BASELINE_COLOR,
               markeredgecolor="white", markeredgewidth=1.2, label="Constant-map baseline"),
        Line2D([], [], marker="o", markersize=8, linestyle="none", color=MODEL_COLOR,
               markeredgecolor="white", markeredgewidth=1.2, label="Ours (5-seed mean)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), labelcolor=INK_MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main(formats: str = "pdf,png,svg", dpi: int = 400, out_dir: str = str(HERE)):
    rows = load()
    fig = build(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in (f.strip() for f in formats.split(",") if f.strip()):
        path = out_dir / f"fig1_margin_inversion.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                    metadata=_NO_TIMESTAMP.get(fmt, {}))
        print(f"  {path.name}")
    plt.close(fig)

    # 표에 적을 수치를 콘솔에도 찍어 그림과 본문이 갈리지 않게 한다.
    base = [r["baseline"] for r in rows]
    model = [r["model"] for r in rows]
    def sd(xs):
        if len(xs) < 2:                       # 1 fold 리허설
            return float("nan")
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    ratio = sd(base) / sd(model) if sd(model) and sd(model) == sd(model) else float("nan")
    print(f"\n  기준선 fold 간 SD {sd(base):.4f} / 모델 {sd(model):.4f} = {ratio:.1f}배")
    top_iou = max(rows, key=lambda r: r["model"])
    top_margin = max(rows, key=lambda r: r["margin"])
    print(f"  iou 1위 {top_iou['fold']} (margin {top_iou['margin']:+.3f}) / "
          f"margin 1위 {top_margin['fold']} (iou {top_margin['model']:.3f})")


if __name__ == "__main__":
    # fire가 있으면 CLI 플래그를 받고, 없으면 기본값으로 돈다. 노트북에서 matplotlib
    # 하나만 있어도 그림이 나와야 하므로 fire를 필수 의존성으로 두지 않는다.
    try:
        import fire
    except ImportError:
        main()
    else:
        fire.Fire(main)
