"""논문용 구조도. 저장된 실측 예제만 읽으며 학습·추론을 다시 실행하지 않는다.

실행: conda run -n bev-chamdog python tools/render_model_architecture.py
출력: docs/figures/model_architecture.{svg,pdf,png}, training_objective.{svg,pdf,png}
구조·예제 출처: docs/figures/architecture_assets/provenance.json
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
ASSETS = OUT / "architecture_assets"
INK, MUTED, RULE = "#172C42", "#536579", "#D3DCE5"
BLUE, BLUE_BG = "#3268A8", "#EDF3FA"
GOLD, GOLD_BG = "#A66A18", "#FCF4E6"
GREEN, GREEN_BG = "#208673", "#E7F3EF"
PURPLE, PURPLE_BG = "#7956A1", "#F4EFF9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.unicode_minus": False,
})


def canvas(height):
    fig = plt.figure(figsize=(14.4, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1440), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=15, color=INK, bold=False, ha="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                   weight="bold" if bold else "normal", linespacing=1.45, **kw)


def box(ax, x, y, w, h, edge=BLUE, fill=BLUE_BG, dashed=False, radius=5, lw=1.5):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        edgecolor=edge, facecolor=fill, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid")
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK, dashed=False, lw=1.6, connection="arc3"):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=connection,
        linestyle=(0, (4, 3)) if dashed else "solid", shrinkA=0, shrinkB=0)
    ax.add_patch(patch)


def route(ax, points, color=BLUE, dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.4, ls="--" if dashed else "-", solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color, dashed, lw=1.4)


def photo(ax, filename, x, y, w, h):
    ax.imshow(np.asarray(Image.open(ASSETS / filename)), extent=(x, x+w, y+h, y), zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=RULE, lw=0.7, zorder=4))


def plane(ax, cx, cy, width=122, height=42):
    # Four horizontal sampling planes, not opaque solid voxels.
    pts = np.array([[cx,cy-height/2], [cx+width/2,cy],
                    [cx,cy+height/2], [cx-width/2,cy]])
    ax.add_patch(Polygon(pts, facecolor=GOLD_BG, edgecolor=GOLD, lw=1.1, zorder=3))
    for t in (.25, .5, .75):
        for p, q in ((pts[0]*(1-t)+pts[3]*t, pts[1]*(1-t)+pts[2]*t),
                     (pts[0]*(1-t)+pts[1]*t, pts[3]*(1-t)+pts[2]*t)):
            ax.plot([p[0],q[0]], [p[1],q[1]], color=GOLD, alpha=.42, lw=.6, zorder=4)


def save(fig, name):
    fig.canvas.draw()
    # Every text object must fit inside the exported page.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for label in ax.texts:
            b = label.get_window_extent(renderer)
            if b.x0 < bounds.x0 - 1 or b.y0 < bounds.y0 - 1 or b.x1 > bounds.x1 + 1 or b.y1 > bounds.y1 + 1:
                raise RuntimeError(f"그림 바깥으로 나온 라벨: {label.get_text()}")
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def architecture():
    fig, ax = canvas(760)
    text(ax, 20, 26, "(a)  Image-to-BEV inference", 20, bold=True, ha="left")
    for x, col, bg, label in ((845, BLUE, BLUE_BG, "Learned module"),
                              (1130, GOLD, GOLD_BG, "Fixed geometry / fusion")):
        box(ax, x, 16, 20, 20, col, bg, radius=2)
        text(ax, x+30, 26, label, 14, ha="left")

    # Calibrations are model inputs; labels never enter this path.
    box(ax, 431, 57, 282, 40, GOLD, GOLD_BG)
    text(ax, 572, 77, "DS intrinsics + extrinsics", 13, GOLD)
    route(ax, [(713,77),(731,77),(731,144),(715,144)], GOLD)

    for x, title in ((87,"Fisheye images"), (285,"Shared encoder"),
                     (572,"Geometric lifting"), (833,"BEV compression"),
                     (1050,"BEV decoder"), (1317,"Free-space map")):
        text(ax, x, 113, title, 14, bold=True)

    for i, name in enumerate(("front", "left", "right")):
        y = 139+i*77
        photo(ax, name+".png", 21, y, 132, 74.25)
        ax.add_patch(Rectangle((22,y+52),52,21,facecolor="white",alpha=.92,lw=0,zorder=5))
        text(ax, 26, y+62, name, 12, ha="left", zorder=6)
    text(ax, 87, 386, "3 cameras · RGB", 14)
    text(ax, 87, 409, "288 × 512 each", 14, MUTED)

    box(ax, 188, 150, 194, 195)
    text(ax, 285, 181, "ResNet-101", 18, BLUE, True)
    text(ax, 285, 245, "Trunk to layer3\n↑2 + concat\n1 × 1 conv", 14)
    text(ax, 285, 319, "Stride 8 · shared", 13, BLUE)
    arrow(ax, (154,247), (184,247))
    text(ax, 285, 386, "3 × 128 × 36 × 64", 14)
    text(ax, 285, 409, "Camera features", 14, MUTED)

    box(ax, 427, 132, 286, 232, GOLD, "#FFFCF7")
    text(ax, 570, 155, "Double Sphere projection", 13.5, GOLD, True)
    # Draw bottom-to-top so the four sampled heights remain distinguishable.
    for i in range(4):
        cy = 278 - 27*i
        plane(ax, 544, cy)
        text(ax, 643, cy, f"{i*.5:.1f} m", 13, GOLD)
    text(ax, 571, 311, "Bilinear feature sampling", 13, GOLD)
    text(ax, 571, 344, "Camera masked mean", 13.5, GOLD, True)
    arrow(ax, (384,247), (423,247))
    text(ax, 572, 386, "128 × 120 × 4 × 120", 14)
    text(ax, 572, 409, "C × Z × Y × X; after fusion", 13.5, MUTED)

    box(ax, 751, 150, 164, 195)
    text(ax, 833, 180, "Fold height", 16, BLUE, True)
    text(ax, 833, 207, "4 × 128 → 512", 13)
    arrow(ax, (833,223), (833,244), BLUE)
    text(ax, 833, 264, "3 × 3 conv", 16, BLUE, True)
    text(ax, 833, 289, "512 → 128 ch", 14)
    text(ax, 833, 320, "IN + GELU", 14)
    arrow(ax, (715,247), (747,247))
    text(ax, 833, 386, "128 × 120 × 120", 14)
    text(ax, 833, 409, "2D BEV features", 14, MUTED)

    box(ax, 955, 150, 193, 195)
    text(ax, 1051, 180, "ResNet-18", 16, BLUE, True)
    text(ax, 1051, 212, "U-Net + additive skips", 12)
    text(ax, 1051, 257, "120 → 60 → 30 → 15\n15 → 30 → 60 → 120", 11.8)
    text(ax, 1051, 307, "Binary head", 14, BLUE, True)
    text(ax, 1051, 332, "3 × 3 → 1 × 1", 12)
    arrow(ax, (917,247), (951,247))
    text(ax, 1051, 386, "2 × 120 × 120", 14)
    text(ax, 1051, 409, "not-free / free logits", 14, MUTED)

    raw_mask = np.asarray(Image.open(ASSETS / "prediction.png")) > 0
    ax.imshow(raw_mask, cmap=ListedColormap(["#E5E9ED", GREEN]),
              extent=(1228,1402,311,137), interpolation="nearest", vmin=0, vmax=1)
    ax.add_patch(Rectangle((1228,137),174,174,fill=False,edgecolor=RULE,lw=1.2))
    # Ego is at row 80: forward 4 m, rear 2 m. This is a location marker only.
    arrow(ax,(1315,137+174*80/120+8),(1315,137+174*80/120-8),INK,lw=1.5)
    text(ax, 1315, 332, "argmax over 2 logits", 13.5)
    arrow(ax, (1150,247), (1224,247))
    for x, col, name in ((1231,GREEN,"free"),(1306,"#E5E9ED","not-free")):
        ax.add_patch(Rectangle((x,350),12,12,facecolor=col,edgecolor=RULE,lw=.5))
        text(ax,x+18,356,name,12,ha="left")
    text(ax, 1315, 386, "120 × 120 · 5 cm", 14)
    text(ax, 1315, 409, "6 m × 6 m coverage", 14, MUTED)

    # Architectural details: concats in image space, adds in BEV space.
    box(ax, 20, 444, 650, 264, RULE, "white", radius=4, lw=1)
    box(ax, 690, 444, 730, 264, RULE, "white", radius=4, lw=1)
    text(ax, 37, 467, "(b)  Image encoder", 17, bold=True, ha="left")
    text(ax, 647, 467, "per camera", 13, MUTED, ha="right")
    stages = [(39, 522, 120, "Stem / L1–2", "512 ch"),
              (196, 522, 104, "Layer3", "1024 ch"),
              (337, 522, 82, "↑2, cat", "1536 ch"),
              (455, 522, 91, "Conv ×2", "512 ch"),
              (578, 522, 73, "1 × 1", "128 ch")]
    for x,y,w,name,shape in stages:
        box(ax,x,y,w,61)
        text(ax,x+w/2,y+19,name,11,BLUE,True)
        text(ax,x+w/2,y+43,shape,11)
    for x,s in ((99,"36 × 64"),(248,"18 × 32"),(377,"36 × 64"),(500,"36 × 64"),(614,"36 × 64")):
        text(ax,x,604,s,11,MUTED)
    for left,right in zip(stages,stages[1:]):
        arrow(ax,(left[0]+left[2]+3,553),(right[0]-4,553),BLUE)
    route(ax,[(99,522),(99,499),(377,499),(377,519)],BLUE)
    text(ax, 337, 638, "cat: concatenate layer2 and upsampled layer3 features", 12.2)
    text(ax, 337, 667, "Fusion conv: 3 × 3 + IN + ReLU; ImageNet initialization", 12.2, MUTED)
    text(ax, 337, 690, "IN: instance normalization", 12, MUTED)

    text(ax, 707, 467, "(c)  BEV decoder", 17, bold=True, ha="left")
    text(ax, 1397, 467, "C × Z × X", 13, MUTED, ha="right")
    # Left-to-right down/up path, with explicit '+' on the upsampling blocks.
    nodes = [(723,530,62,80,"Input","128","120²"),
             (820,546,62,64,"Stem/L1","64","60²"),
             (917,561,62,49,"L2","128","30²"),
             (1014,575,62,35,"L3","256","15²"),
             (1111,561,62,49,"↑2 +","128","30²"),
             (1208,546,62,64,"↑2 +","64","60²"),
             (1305,530,62,80,"↑2 +","128","120²")]
    for i,(x,y,w,h,name,ch,spatial) in enumerate(nodes):
        box(ax,x,y,w,h,BLUE,"white" if i >= 4 else BLUE_BG)
        text(ax,x+w/2,y+h/2,ch,14,BLUE,True)
        text(ax,x+w/2,632,name,12)
        text(ax,x+w/2,655,spatial,12,MUTED)
    for left,right in zip(nodes,nodes[1:]):
        arrow(ax,(left[0]+left[2]+2,594),(right[0]-3,594),BLUE)
    for left,right,y in ((0,6,491),(1,5,509),(2,4,529)):
        a,b = nodes[left],nodes[right]
        route(ax,[(a[0]+31,a[1]),(a[0]+31,y),(b[0]+31,y),(b[0]+31,b[1]-3)],BLUE)
    text(ax,1055,684,"↑2: bilinear upsample → 1 × 1 conv + IN;  +: add skip",12.5,MUTED)
    text(ax,20,735,"Batch dimension omitted. Heights are relative to ground; Z/X are BEV axes, Y is height.",13,MUTED,ha="left")
    save(fig,"model_architecture")


def training_objective():
    fig, ax = canvas(510)
    text(ax,20,27,"Training-only supervision",20,bold=True,ha="left")
    text(ax,1420,27,"Same inference architecture for all loss ablations",14,MUTED,ha="right")
    box(ax,20,75,208,84,BLUE,BLUE_BG)
    text(ax,124,101,"Binary logits",16,BLUE,True)
    text(ax,124,132,r"$p=\mathrm{softmax}(z)_{\mathrm{free}}$",16)
    box(ax,20,288,208,82,RULE,"#F6F8FA")
    text(ax,124,314,"BEV labels",16,bold=True)
    text(ax,124,344,"Free mask + validity",14)

    box(ax,275,73,380,298,PURPLE,PURPLE_BG,dashed=True)
    text(ax,465,99,"Boundary-aware target",17,PURPLE,True)
    # Draw the true truncated-Gaussian target, not an arbitrary sigmoid.
    gx,gy,gw,gh = 341,151,244,101
    ax.plot([gx,gx+gw],[gy+gh,gy+gh],color=MUTED,lw=1)
    ax.plot([gx,gx],[gy,gy+gh],color=MUTED,lw=1)
    delta,sigma = .30,.10
    d=np.linspace(-.4,.4,401)
    target=np.clip((ndtr(d/sigma)-ndtr(-delta/sigma))/(ndtr(delta/sigma)-ndtr(-delta/sigma)),0,1)
    ax.add_patch(Rectangle((gx+gw*.125,gy),gw*.75,gh,facecolor="#E6DAF0",lw=0,zorder=1.5))
    ax.plot(gx+(d+.4)/.8*gw,gy+(1-target)*gh,color=PURPLE,lw=2.4)
    for value,label in ((-.3,"−0.30"),(0,"0"),(.3,"+0.30")):
        text(ax,gx+(value+.4)/.8*gw,gy+gh+17,label,11.5,MUTED)
    for value in (0,1): text(ax,gx-16,gy+(1-value)*gh,str(value),12,MUTED)
    text(ax,465,134,"Truncated-Gaussian CDF",14)
    text(ax,465,288,"Signed distance d [m]  (+ inside free)",12.5)
    text(ax,465,319,r"$\delta=0.30\,\mathrm{m},\quad \sigma=0.10\,\mathrm{m}$",16)
    text(ax,465,349,"Hard targets outside the boundary band",12.5)
    arrow(ax,(230,329),(271,329),PURPLE,True)

    box(ax,707,74,337,121,PURPLE,PURPLE_BG,dashed=True)
    text(ax,875,101,"Per-region supervision",15,PURPLE,True)
    text(ax,875,139,r"$\frac{1}{2}L_F+\frac{1}{2}L_N+0.5L_B$",22)
    text(ax,875,175,"Hard BCE / hard BCE / soft BCE",13)
    route(ax,[(228,115),(253,115),(253,59),(876,59),(876,71)],PURPLE,True)
    arrow(ax,(657,135),(703,135),PURPLE,True)

    box(ax,707,246,337,125,PURPLE,PURPLE_BG,dashed=True)
    text(ax,875,267,"Radial free length",15,PURPLE,True)
    text(ax,875,308,r"$\hat a_\theta=\Delta r\,\sum_{r\in V_\theta} p_{\theta,r}$",18)
    text(ax,875,353,"Dead-zone Huber loss",13)
    route(ax,[(124,161),(124,215),(253,215),(253,426),(690,426),(690,280),(703,280)],PURPLE,True)
    route(ax,[(124,372),(124,391),(680,391),(680,335),(703,335)],PURPLE,True)
    text(ax,865,397,r"$\delta_R=0.15\,\mathrm{m},\quad \beta=0.15\,\mathrm{m}$",14)
    text(ax,437,450,"GT and prediction share the ray sampler",12,MUTED)

    box(ax,1100,159,316,143,PURPLE,"white",dashed=True)
    text(ax,1258,191,"Total training objective",15,PURPLE,True)
    text(ax,1258,231,r"$L=L_{\mathrm{cell}}+0.3L_{\mathrm{range}}$",21)
    text(ax,1258,272,"Backpropagate through the model",12.5)
    route(ax,[(1046,135),(1075,135),(1075,202),(1096,202)],PURPLE,True)
    route(ax,[(1046,309),(1075,309),(1075,265),(1096,265)],PURPLE,True)
    text(ax,20,487,"Losses use valid cells; permanent blind cells receive hard not-free targets. Range loss uses eligible GT rays.",13,MUTED,ha="left")
    save(fig,"training_objective")


def main():
    meta = json.loads((ASSETS / "provenance.json").read_text())
    assert meta["height"]["height_bins"] == 4
    assert meta["tensor_shapes"]["decoder.segmentation_head"] == [1,2,120,120]
    architecture()
    training_objective()
    for stem in ("model_architecture", "training_objective"):
        for ext in ("svg", "pdf", "png"):
            p = OUT / f"{stem}.{ext}"
            print(f"{p.relative_to(ROOT)}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
