#!/usr/bin/env python
"""Figure 2 -- 합성 사전학습이 물려주는 것은 **"격자 전체를 확신하는 습관"**이다.

**이 스크립트가 그림의 정본이다.** 내보낸 PDF/PNG/SVG를 손으로 고치지 않는다.
입력은 `../data/figure_early_epochs.csv` 하나, matplotlib 외 의존성이 없다.

**[2026-10-02] 지표를 precision/recall(프레임 macro)로 바꿨다.** 옛 그림은 `fatal_rate`
(= 1 − precision)·`free_miss_rate`(= 1 − recall)의 micro 값이었다. 아래 서사와 주석 문구는
옛 두 캠페인의 결과이고, **다음 캠페인 결과로 다시 확인해야 한다**(2026-09-23 재학습에서 이미
기전 주장이 좁혀졌다 -- 원장 §5).

**무엇을 보이나.** 1에폭에서 사전학습 모델은 자유공간을 **더 많이 찾으면서**(recall 높음)
**더 자주 틀린다**(precision 낮음). 두 가지가 동시에 일어나는 방식은 하나뿐이다 -- 자유공간을
더 넓게 선언하는 것이다. 그것이 라벨이 격자의 94 %를 덮는 도메인에서 온 prior이고,
라벨이 28 %만 덮는 우리 도메인에서는 감독되지 않는 영역까지 확신하게 만든다.

**그리고 2에폭이면 사라진다.** 그래서 이 그림의 결론은 "사전학습이 해롭다"가 아니라
**"prior는 실재하지만 얕다"**이다. 최종 `iou_free`에 차이가 없다는 표와 모순되지 않는다.

띠는 시드 5개의 ±1 표준편차다.

실행:
    python make_source_prior_mechanism.py
    python make_source_prior_mechanism.py --formats=pdf --dpi=600
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["svg.hashsalt"] = "bev-chamdog-02-fig2"
_NO_TIMESTAMP = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "figure_early_epochs.csv"

TARGET_ONLY_COLOR = "#2a78d6"   # slot 1 blue  = 확정 설정
PRETRAINED_COLOR = "#eb6834"    # slot 2 orange = 사전학습
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

LAST_EPOCH = 10   # prior는 2~3에폭이면 풀린다. 40까지 그리면 그 구간이 안 보인다.
PANELS = (
    ("precision", "precision  ↑\n(of cells called free, fraction right)"),
    ("recall", "recall  ↑\n(of truly free cells, fraction found)"),
)
ARM_STYLE = {
    "control": ("Target-only (frozen setting)", TARGET_ONLY_COLOR),
    "source_prior": ("SynWoodScape pretrained", PRETRAINED_COLOR),
}


def load(path: Path = DATA) -> dict:
    series = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            epoch = int(row["epoch"])
            if epoch > LAST_EPOCH:
                continue
            series[(row["arm"], row["metric"])][epoch] = (
                float(row["mean"]), float(row["sd"]))
    return series


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)


def draw_panel(ax, series, metric, ylabel) -> None:
    for arm, (_, color) in ARM_STYLE.items():
        points = series[(arm, metric)]
        epochs = sorted(points)
        means = [points[e][0] for e in epochs]
        sds = [points[e][1] for e in epochs]
        ax.fill_between(epochs, [m - s for m, s in zip(means, sds)],
                        [m + s for m, s in zip(means, sds)],
                        color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(epochs, means, color=color, linewidth=2.0, zorder=3)
        ax.plot(epochs, means, "o", color=color, markersize=4.5,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    ax.set_xlabel("training epoch", fontsize=9, color=INK_MUTED, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_MUTED, labelpad=7, linespacing=1.4)
    ax.set_xlim(0.5, LAST_EPOCH + 0.5)
    ax.set_xticks([1, 2, 3, 5, 10])


def main(formats="pdf,png,svg", dpi: int = 400, stem="fig2_source_prior_mechanism"):
    series = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, (metric, ylabel) in zip(axes, PANELS):
        _style(ax)
        draw_panel(ax, series, metric, ylabel)

    # 1에폭에 주석 하나. 두 패널이 **반대 방향**으로 벌어지는 것이 이 그림의 전부다.
    axes[0].annotate("pretrained model calls\nfar more cells free",
                     xy=(1, series[("source_prior", "precision")][1][0]),
                     xytext=(2.6, series[("source_prior", "precision")][1][0] - 0.02),
                     fontsize=8, color=INK, linespacing=1.35,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9))
    axes[1].annotate("… so it finds more,\nbut it is wrong more often",
                     xy=(1, series[("source_prior", "recall")][1][0]),
                     xytext=(2.6, series[("source_prior", "recall")][1][0] + 0.022),
                     fontsize=8, color=INK, linespacing=1.35,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9))

    fig.suptitle("Synthetic pretraining transfers a habit of being confident everywhere "
                 "— and it washes out by epoch 3",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=1.02)
    fig.legend(handles=[
        Line2D([], [], color=color, linewidth=2.0, label=label)
        for label, color in ARM_STYLE.values()
    ], loc="lower center", ncol=2, frameon=False, fontsize=8.5,
        labelcolor=INK_MUTED, bbox_to_anchor=(0.5, -0.06))
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
