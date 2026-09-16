"""논문용 구조도. 저장된 실측 예제만 읽으며 학습·추론을 다시 실행하지 않는다.

실행: conda run -n bev-chamdog python tools/render_model_architecture.py
기본 출력: docs/figures/model_architecture.{svg,pdf,png}
목적함수 재생성은 --figure objective, 두 그림은 --figure all로 명시한다.
구조·예제 출처: docs/figures/architecture_assets/provenance.json
"""
from pathlib import Path
import argparse
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
        if suffix == "svg":
            path = OUT / f"{name}.{suffix}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def tapered_module(ax, x0, x1, cy, left_h, right_h, label, size=16):
    """A shallow 3D trapezoid: narrowing encoder or widening decoder."""
    front = np.array([[x0,cy-left_h/2], [x1,cy-right_h/2],
                      [x1,cy+right_h/2], [x0,cy+left_h/2]])
    shift = np.array([9,-9])
    ax.add_patch(Polygon([front[0],front[1],front[1]+shift,front[0]+shift],
                         facecolor="#D9E6F5",edgecolor=BLUE,lw=1.3))
    ax.add_patch(Polygon([front[1],front[2],front[2]+shift,front[1]+shift],
                         facecolor="#B9D0ED",edgecolor=BLUE,lw=1.3))
    ax.add_patch(Polygon(front,facecolor=BLUE_BG,edgecolor=BLUE,lw=1.8))
    if label:
        text(ax,(x0+x1)/2,cy,label,size,BLUE,True)


def feature_stack(ax, x, cy, count=3, height=83, width=37, colors=None):
    """Upright sheets represent feature channels, not metric BEV planes."""
    colors = colors or ["#D9E6F5", "#E7EFF9", "#F5F8FD"]
    for i in range(count):
        xx = x+(count-1-i)*7
        yy = cy-height/2-(count-1-i)*6
        pts=np.array([[xx,yy],[xx+width,yy+10],
                      [xx+width,yy+height+10],[xx,yy+height]])
        ax.add_patch(Polygon(pts,facecolor=colors[i % len(colors)],
                             edgecolor=BLUE,lw=1.1,zorder=4))
        for t in (.25,.5,.75):
            p=pts[0]*(1-t)+pts[3]*t
            q=pts[1]*(1-t)+pts[2]*t
            ax.plot([p[0],q[0]],[p[1],q[1]],color=BLUE,alpha=.2,lw=.6,zorder=5)


def bev_sheet(ax,cx,cy):
    pts=np.array([[cx,cy-32],[cx+46,cy],[cx,cy+32],[cx-46,cy]])
    lower=pts+np.array([0,9])
    ax.add_patch(Polygon([pts[1],pts[2],pts[3],lower[3],lower[2],lower[1]],
                         facecolor="#B9D0ED",edgecolor=BLUE,lw=1.1,zorder=3))
    ax.add_patch(Polygon(pts,facecolor=BLUE_BG,edgecolor=BLUE,lw=1.3,zorder=4))
    for t in (.2,.4,.6,.8):
        for p,q in ((pts[0]*(1-t)+pts[3]*t,pts[1]*(1-t)+pts[2]*t),
                    (pts[0]*(1-t)+pts[1]*t,pts[3]*(1-t)+pts[2]*t)):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=BLUE,alpha=.35,lw=.6,zorder=5)


def architecture():
    fig, ax = canvas(490)
    text(ax,22,30,"Fisheye-to-BEV free-space prediction",21,bold=True,ha="left")
    for x,col,bg,label in ((895,BLUE,BLUE_BG,"Learned"),
                           (1110,GOLD,GOLD_BG,"Fixed geometry")):
        ax.add_patch(Polygon([(x,20),(x+22,25),(x+22,39),(x,44)],
                             facecolor=bg,edgecolor=col,lw=1.4))
        text(ax,x+34,32,label,14,ha="left")

    # Short stage names sit above silhouettes; details sit below them.
    for x,title in ((84,"3 cameras"),(256,"Shared encoder"),
                    (531,"Multi-height lifting"),(795,"BEV fusion"),
                    (1036,"BEV decoder"),(1330,"Free-space map")):
        text(ax,x,100,title,14,bold=True)

    for i,name in enumerate(("front","left","right")):
        y=145+i*80
        photo(ax,name+".png",22,y,125,70.3)
        ax.add_patch(Rectangle((23,y+48),50,21,facecolor="white",alpha=.92,lw=0,zorder=5))
        text(ax,27,y+58,name,11.5,ha="left",zorder=6)
        arrow(ax,(150,y+35),(181,220+40*i),lw=1.4)

    # One shared encoder, three camera-specific feature maps.
    tapered_module(ax,185,324,260,210,86,"ResNet\n101",17)
    feature_stack(ax,351,260)
    arrow(ax,(335,260),(347,260),BLUE,lw=1.4)
    text(ax,265,421,"ResNet-101 · stride 8",12.5,MUTED)
    text(ax,382,335,"Per-view\nfeatures",11.5,MUTED)

    # DS calibration supplies fixed sampling coordinates at four physical heights.
    box(ax,451,138,169,34,GOLD,GOLD_BG,radius=4,lw=1)
    text(ax,535,155,"DS calibration",12.5,GOLD)
    arrow(ax,(535,174),(535,195),GOLD,lw=1.3)
    for i in range(4):
        cy=315-32*i
        plane(ax,535,cy,width=148,height=51)
    arrow(ax,(407,260),(455,260),GOLD)
    text(ax,535,365,"0 / 0.5 / 1.0 / 1.5 m",11.5,GOLD)
    text(ax,535,421,"Double Sphere · 4 heights",12.5,MUTED)

    # Fuse cameras first; then concatenate height slices along channels.
    arrow(ax,(612,260),(657,260),GOLD)
    circle=plt.Circle((676,260),19,facecolor=GOLD_BG,edgecolor=GOLD,lw=1.5,zorder=5)
    ax.add_patch(circle)
    text(ax,676,260,r"$\mu$",21,GOLD,zorder=6)
    text(ax,676,319,"Masked\ncamera mean",11.5,GOLD)
    arrow(ax,(697,260),(726,260),GOLD)
    feature_stack(ax,730,260,count=4,height=103,width=22,
                  colors=["#B9D0ED","#CDDEF1","#E0EAF7","#F0F5FC"])
    arrow(ax,(777,260),(787,260),BLUE,lw=1.4)
    tapered_module(ax,791,839,260,104,52,"",13)
    text(ax,817,330,"Conv",12,BLUE)
    arrow(ax,(850,260),(858,260),BLUE,lw=1.3)
    bev_sheet(ax,906,260)
    text(ax,906,330,"BEV\nfeatures",11.5,MUTED)
    text(ax,802,421,"Fold height → conv",12.5,MUTED)

    # The widening shape is a decoder icon, not a claim of monotonic upsampling.
    arrow(ax,(953,260),(965,260),BLUE,lw=1.4)
    tapered_module(ax,969,1107,260,78,210,"BEV\nU-Net",17)
    feature_stack(ax,1145,260,count=2,height=112,width=17,
                  colors=["#D6E9E3","#ECF0F4"])
    arrow(ax,(1119,260),(1140,260),BLUE,lw=1.4)
    text(ax,1162,354,"Binary\nhead",11.5,BLUE)
    text(ax,1040,421,"ResNet-18 + binary head",12,MUTED)

    # This is the same raw prediction used in the verified first figure.
    raw_mask=np.asarray(Image.open(ASSETS/"prediction.png"))>0
    ax.imshow(raw_mask,cmap=ListedColormap(["#E5E9ED",GREEN]),
              extent=(1245,1415,345,175),interpolation="nearest",vmin=0,vmax=1)
    ax.add_patch(Rectangle((1245,175),170,170,fill=False,edgecolor=RULE,lw=1.2))
    arrow(ax,(1181,260),(1239,260))
    text(ax,1210,232,"argmax",10.5,MUTED)
    ego_y=175+170*80/120
    arrow(ax,(1330,ego_y+8),(1330,ego_y-8),INK,lw=1.5)
    for x,col,name in ((1247,GREEN,"free"),(1324,"#E5E9ED","not-free")):
        ax.add_patch(Rectangle((x,364),11,11,facecolor=col,edgecolor=RULE,lw=.5))
        text(ax,x+16,370,name,11.5,ha="left")
    text(ax,1330,421,"120 × 120 · 5 cm",12,MUTED)

    text(ax,84,421,"512 × 288",12,MUTED)
    ax.plot([22,1415],[454,454],color=RULE,lw=.8)
    text(ax,22,473,"Images → image features → multi-height features → 2D BEV features → free-space prediction",
         13,MUTED,ha="left")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("architecture", "objective", "all"), default="architecture")
    args = parser.parse_args()
    meta = json.loads((ASSETS / "provenance.json").read_text())
    assert meta["height"]["height_bins"] == 4
    assert meta["tensor_shapes"]["decoder.segmentation_head"] == [1,2,120,120]
    stems = []
    if args.figure in ("architecture", "all"):
        architecture()
        stems.append("model_architecture")
    if args.figure in ("objective", "all"):
        training_objective()
        stems.append("training_objective")
    for stem in stems:
        for ext in ("svg", "pdf", "png"):
            p = OUT / f"{stem}.{ext}"
            print(f"{p.relative_to(ROOT)}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
