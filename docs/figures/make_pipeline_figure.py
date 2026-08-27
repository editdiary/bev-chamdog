"""파이프라인 모식도(`bev_pipeline.svg`)를 생성한다. **그림의 정본은 이 스크립트다.**

왜 스크립트인가: lifting 단계의 아이소메트릭 복셀 격자와 광선은 좌표를 손으로 적으면
고치기가 사실상 불가능하다. 배치를 바꾸려면 아래 좌표 상수(`LF_X`, `N`, `BASE` 등)만
건드리고 다시 돌린다.

출력 둘:

- `bev_pipeline.svg`           독립 파일. 색이 리터럴이라 브라우저·Inkscape·Word에서
                               그대로 열린다. **논문 figure에 쓸 쪽이다.**
- `bev_pipeline_themed.svg`    색이 CSS 변수(`var(--ink)` 등)라 페이지 테마를 따른다.
                               `bev_pipeline.html`에 인라인으로 박혀 있는 판본.

숫자의 출처: 형상과 파라미터 수는 저장소 코드를 실행해 확인한 값이고
`docs/paper_experiment_compendium.md` §1.3과 일치한다. 수용영역 118셀은
디코더를 선형화해 gradient가 닿는 범위를 측정한 값이다.

실행:
    python docs/figures/make_pipeline_figure.py
"""
import math
import re
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# 독립 SVG용 리터럴 팔레트 (페이지의 라이트 테마와 같은 값)
LITERAL = {
    "var(--ink)": "#16201C", "var(--ink-mid)": "#4A574F", "var(--ink-soft)": "#78847C",
    "var(--rule)": "#D6DDD8", "var(--surface)": "#FFFFFF", "var(--surface-sub)": "#EDF1EE",
    "var(--learned)": "#1B565D", "var(--learned-bg)": "#E6F0F0",
    "var(--geom)": "#9C6119", "var(--geom-bg)": "#F6EDDE",
}
GROUND = "#F7F8F6"

import math

W, H = 1620, 520
P = []          # SVG 조각들
def add(s): P.append(s)

# ── 팔레트(CSS 변수 참조) ────────────────────────────────────────────────
INK, MID, SOFT = "var(--ink)", "var(--ink-mid)", "var(--ink-soft)"
RULE, SURF, SUB = "var(--rule)", "var(--surface)", "var(--surface-sub)"
LRN, LRNBG = "var(--learned)", "var(--learned-bg)"
GEO, GEOBG = "var(--geom)", "var(--geom-bg)"

def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def txt(x, y, t, size=11, fill=MID, anchor="middle", weight=400, mono=False, cls=""):
    fam = "mono" if mono else "sans"
    add(f'<text x="{x:.1f}" y="{y:.1f}" class="f-{fam} {cls}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(t)}</text>')

def box(x, y, w, h, fill=SURF, stroke=LRN, sw=1.6, dash=None, r=3):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def arrow(x1, y1, x2, y2, stroke=INK, sw=1.6, dash=None, head="a-ink"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}"{d} marker-end="url(#{head})"/>')

# ── 아이소메트릭 평면 ────────────────────────────────────────────────────
AX, BY = 1.62, 0.82        # i-j -> 화면 x, y 계수
def iso(ox, oy, i, j, k=0.0, lift=0.0):
    return ox + (i - j) * AX, oy + (i + j) * BY - k * lift

def plane(ox, oy, n, k, lift, fill, stroke, sw=1.0, op=1.0, grid=0):
    pts = [iso(ox, oy, 0, 0, k, lift), iso(ox, oy, n, 0, k, lift),
           iso(ox, oy, n, n, k, lift), iso(ox, oy, 0, n, k, lift)]
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<polygon points="{p}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')
    for t in range(1, grid):
        f = n * t / grid
        for a, b in (((f, 0), (f, n)), ((0, f), (n, f))):
            x1, y1 = iso(ox, oy, *a, k, lift); x2, y2 = iso(ox, oy, *b, k, lift)
            add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{stroke}" stroke-width=".5" stroke-opacity=".45"/>')
    return pts

# ── defs ────────────────────────────────────────────────────────────────
add('<defs>')
for name, col in (("a-ink", INK), ("a-geo", GEO), ("a-lrn", LRN), ("a-soft", SOFT)):
    add(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{col}"/></marker>')
add(f'<clipPath id="bevclip"><rect x="0" y="0" width="150" height="150" rx="2"/></clipPath>')
add('</defs>')

BASE = 210          # 주 흐름의 세로 중심
SHAPE_Y = 452       # 형상 주석 레일
STAGE_Y = 62        # 단계 라벨 레일

def stage_label(cx, no, name, learned, param):
    col = LRN if learned else GEO
    bg = LRNBG if learned else GEOBG
    add(f'<rect x="{cx-13:.1f}" y="{STAGE_Y-27}" width="26" height="19" rx="2.5" fill="{bg}"/>')
    txt(cx, STAGE_Y - 13, no, 12, col, weight=700, mono=True)
    txt(cx, STAGE_Y + 6, name, 12.5, INK, weight=600)
    txt(cx, STAGE_Y + 22, param, 10, col, mono=True)

def shape(cx, s, sub=None):
    txt(cx, SHAPE_Y, s, 10.5, MID, mono=True)
    if sub: txt(cx, SHAPE_Y + 15, sub, 9.5, SOFT)

# ═══ 0. 어안 카메라 3대 ═══════════════════════════════════════════════
CAM_X = 42
stage_label(CAM_X + 46, "0", "어안 입력", False, "학습 없음")
for idx, nm in enumerate(("front", "left", "right")):
    y = BASE - 78 + idx * 56
    box(CAM_X, y, 92, 46, SUB, RULE, 1.2)
    add(f'<circle cx="{CAM_X+46}" cy="{y+23}" r="17" fill="none" stroke="{SOFT}" stroke-width="1.1"/>')
    add(f'<circle cx="{CAM_X+46}" cy="{y+23}" r="8" fill="none" stroke="{SOFT}" '
        f'stroke-width=".8" stroke-opacity=".7"/>')
    for a in range(0, 360, 45):
        r1, r2 = 17, 22
        x1 = CAM_X+46 + r1*math.cos(math.radians(a)); y1 = y+23 + r1*math.sin(math.radians(a))*.62
        x2 = CAM_X+46 + r2*math.cos(math.radians(a)); y2 = y+23 + r2*math.sin(math.radians(a))*.62
        add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{SOFT}" '
            f'stroke-width=".7" stroke-opacity=".5"/>')
    txt(CAM_X + 46, y + 40, nm, 9, SOFT, mono=True)
shape(CAM_X + 46, "3 x 288 x 512", "1280x720 -> 512x288")

# ═══ 1. 2D 인코더 ════════════════════════════════════════════════════
EN_X = 196
arrow(CAM_X + 100, BASE, EN_X - 8, BASE)
txt(CAM_X + 118, BASE - 9, "x3", 9.5, SOFT, mono=True)
stage_label(EN_X + 47, "1", "2D 인코더", True, "37.04 M")
# ResNet-101 stage들을 폭 감소 슬래브로
slabs = [(0, 132, "c1"), (20, 132, "L1"), (40, 108, "L2"), (66, 78, "L3")]
for dx, hh, lab in slabs:
    box(EN_X + dx, BASE - hh/2, 15, hh, LRNBG, LRN, 1.2, r=2)
    txt(EN_X + dx + 7.5, BASE + hh/2 + 12, lab, 8.5, SOFT, mono=True)
# FPN 되올림 화살표: L3 -> L2와 concat
add(f'<path d="M {EN_X+73} {BASE-46} C {EN_X+73} {BASE-76}, {EN_X+48} {BASE-76}, '
    f'{EN_X+48} {BASE-58}" fill="none" stroke="{LRN}" stroke-width="1.3" '
    f'marker-end="url(#a-lrn)"/>')
txt(EN_X + 60, BASE - 84, "x2 up + concat", 8.5, LRN, mono=True)
box(EN_X + 88, BASE - 34, 15, 68, SURF, LRN, 1.2, r=2)
txt(EN_X + 95.5, BASE + 46, "1x1", 8.5, SOFT, mono=True)
txt(EN_X + 47, BASE + 82, "ResNet-101, layer4 제거", 9.5, SOFT)
txt(EN_X + 47, BASE + 96, "최종 stride 8", 9.5, SOFT)
shape(EN_X + 52, "128 x 36 x 64", "카메라마다 특징맵 한 장")

# ═══ 2. 특징맵 3장 ═══════════════════════════════════════════════════
FM_X = 336
arrow(EN_X + 107, BASE, FM_X - 6, BASE)
for idx in range(3):
    o = idx * 9
    plane(FM_X + 26 - o, BASE - 6 + o, 26, 0, 0,
          LRNBG if idx == 2 else SURF, LRN, 1.1, .95, grid=0)
txt(FM_X + 26, BASE + 74, "S=3", 9.5, SOFT, mono=True)

# ═══ 3. lifting (중심) ═══════════════════════════════════════════════
LF_X, LF_Y = 520, 148
arrow(FM_X + 92, BASE, LF_X - 78, BASE, GEO, 1.8, "5 3", "a-geo")
stage_label(LF_X + 12, "2", "lifting", False, "학습 파라미터 0")
N = 54
LIFT = 26
heights = ("1.5 m", "1.0 m", "0.5 m", "0 m")
for k in range(4):                                   # 위 -> 아래 순으로 그려 겹침 정리
    kk = 3 - k
    plane(LF_X, LF_Y + 96, N, kk, LIFT, GEOBG, GEO, 1.1, .55 if kk else .8, grid=6)
    x, y = iso(LF_X, LF_Y + 96, 0, N, kk, LIFT)
    txt(x - 12, y + 4, heights[k], 9, GEO, anchor="end", mono=True)
# 카메라 원점과 광선
cx, cy = LF_X - 74, LF_Y + 96 + N * BY * 0.55
add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{GEO}"/>')
txt(cx, cy + 20, "cam", 9, GEO, mono=True)
for i, j, k in ((10, 44, 0), (26, 30, 1), (44, 12, 3), (38, 42, 2), (16, 16, 3)):
    px, py = iso(LF_X, LF_Y + 96, i, j, k, LIFT)
    add(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" stroke="{GEO}" '
        f'stroke-width=".9" stroke-opacity=".55" stroke-dasharray="3 2.5"/>')
    add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.8" fill="{GEO}"/>')
txt(LF_X + 12, LF_Y + 200, "격자 셀이 자기 좌표의 픽셀을 찾아 읽는다 (pull)", 9.5, SOFT)
txt(LF_X + 12, LF_Y + 214, "Double Sphere 투영 + grid_sample bilinear", 9.5, SOFT)
shape(LF_X + 12, "128 x 120 x 4 x 120", "단계 3: 본 카메라만 평균 (2대가 62 %)")

# ═══ 4. 높이 접기 + compressor ═══════════════════════════════════════
CP_X = 792
arrow(LF_X + 106, BASE, CP_X - 10, BASE, GEO, 1.8, "5 3", "a-geo")
stage_label(CP_X + 44, "4", "높이 접기 + compressor", True, "0.59 M")
for k in range(4):                                   # 4장이 한 장으로
    y0 = BASE - 52 + k * 15
    add(f'<line x1="{CP_X}" y1="{y0}" x2="{CP_X+30}" y2="{BASE+16}" stroke="{GEO}" '
        f'stroke-width="1" stroke-opacity=".7"/>')
    add(f'<rect x="{CP_X-26}" y="{y0-4}" width="26" height="8" rx="1.5" fill="{GEOBG}" '
        f'stroke="{GEO}" stroke-width=".9"/>')
txt(CP_X - 13, BASE - 66, "Y=4", 9, GEO, mono=True)
box(CP_X + 30, BASE + 4, 62, 24, LRNBG, LRN, 1.4, r=2)
txt(CP_X + 61, BASE + 20, "512 -> 128", 9.5, LRN, mono=True)
txt(CP_X + 61, BASE + 44, "3x3 conv", 9.5, SOFT, mono=True)
txt(CP_X + 44, BASE + 78, "높이를 채널로 접는다.", 9.5, SOFT)
txt(CP_X + 44, BASE + 92, "높이를 섞는 유일한 학습 층.", 9.5, SOFT)
shape(CP_X + 44, "128 x 120 x 120", "여기부터 완전히 2D")

# ═══ 5. BEV 디코더 U-Net ═════════════════════════════════════════════
UN_X = 960
arrow(CP_X + 96, BASE, UN_X - 8, BASE)
stage_label(UN_X + 116, "5", "BEV 디코더 U-Net", True, "3.37 M")
# (dx, 막대높이, 해상도라벨)
down = [(0, 112, "120"), (38, 78, "60"), (76, 50, "30"), (114, 32, "15")]
up = [(152, 50, "30"), (190, 78, "60"), (228, 112, "120")]
for dx, hh, lab in down:
    box(UN_X + dx, BASE - hh/2, 16, hh, LRNBG, LRN, 1.2, r=2)
    txt(UN_X + dx + 8, BASE + hh/2 + 12, lab, 8.5, SOFT, mono=True)
for dx, hh, lab in up:
    box(UN_X + dx, BASE - hh/2, 16, hh, SURF, LRN, 1.2, r=2)
    txt(UN_X + dx + 8, BASE + hh/2 + 12, lab, 8.5, SOFT, mono=True)
for (d1, h1, _lab), (d2, _h2, _l2) in zip(down[:3], reversed(up)):
    yy = BASE - h1/2 - 10
    add(f'<path d="M {UN_X+d1+8} {yy} C {UN_X+d1+8} {yy-26}, {UN_X+d2+8} {yy-26}, '
        f'{UN_X+d2+8} {yy}" fill="none" stroke="{LRN}" stroke-width="1.1" '
        f'stroke-dasharray="4 3" stroke-opacity=".8" marker-end="url(#a-lrn)"/>')
txt(UN_X + 122, BASE - 118, "skip 연결 (5 cm 경계를 되살린다)", 9.5, LRN)
txt(UN_X + 122, BASE + 78, "수용영역 118 셀 = 5.90 m (격자 폭의 98 %)", 9.5, SOFT)
txt(UN_X + 122, BASE + 92, "유효(질량 90 %) 43 셀 = 2.15 m", 9.5, SOFT)
shape(UN_X + 122, "2 x 120 x 120", "free / not-free logit")

# ═══ 6. 출력 BEV ═════════════════════════════════════════════════════
OU_X, OU_Y = 1400, BASE - 75
arrow(UN_X + 252, BASE, OU_X - 10, BASE, GEO, 1.8, "5 3", "a-geo")
stage_label(OU_X + 75, "6", "free-space 마스크", False, "학습 파라미터 0")
box(OU_X, OU_Y, 150, 150, SUB, RULE, 1.4, r=2)
add(f'<g clip-path="url(#bevclip)" transform="translate({OU_X} {OU_Y})">')
# ego는 전방 4 m / 후방 2 m -> 아래에서 1/3 지점
ego = (75, 100)
free = [(ego[0], ego[1]), (18, 74), (12, 40), (40, 16), (74, 8), (108, 18), (132, 44),
        (138, 72), (120, 96), (96, 108)]
pp = " ".join(f"{x},{y}" for x, y in free)
add(f'<polygon points="{pp}" fill="{GEO}" fill-opacity=".22" stroke="{GEO}" stroke-width="1.6"/>')
for ox, oy, rw, rh in ((6, 20, 22, 10), (120, 26, 24, 9), (30, 4, 40, 8), (128, 96, 18, 22)):
    add(f'<rect x="{ox}" y="{oy}" width="{rw}" height="{rh}" rx="1.5" fill="{INK}" '
        f'fill-opacity=".38"/>')
add(f'<circle cx="{ego[0]}" cy="{ego[1]}" r="4.5" fill="{INK}"/>')
add(f'<path d="M {ego[0]} {ego[1]-7} l 0 -11 m -4 4 l 4 -4 l 4 4" fill="none" stroke="{INK}" '
    f'stroke-width="1.4"/>')
for t in range(1, 6):
    add(f'<line x1="0" y1="{t*25}" x2="150" y2="{t*25}" stroke="{RULE}" stroke-width=".5"/>')
    add(f'<line x1="{t*25}" y1="0" x2="{t*25}" y2="150" stroke="{RULE}" stroke-width=".5"/>')
add('</g>')
txt(OU_X + 75, OU_Y + 168, "120 x 120, 5 cm 셀", 9.5, SOFT, mono=True)
txt(OU_X + 75, OU_Y + 182, "전방 4 m / 후방 2 m / 좌우 ±3 m", 9.5, SOFT)
shape(OU_X + 75, "free 마스크 120 x 120", "occupied는 경계에서 유도")

# ═══ 레일과 범례 ═════════════════════════════════════════════════════
add(f'<line x1="24" y1="{SHAPE_Y-24}" x2="{W-24}" y2="{SHAPE_Y-24}" stroke="{RULE}" '
    f'stroke-width="1"/>')
txt(24, SHAPE_Y - 32, "텐서 형상 (batch 축 생략)", 9, SOFT, anchor="start", mono=True)

LG_Y = 22
add(f'<line x1="{W-330}" y1="{LG_Y}" x2="{W-300}" y2="{LG_Y}" stroke="{INK}" stroke-width="1.8"/>')
txt(W - 294, LG_Y + 4, "학습하는 단계", 10, MID, anchor="start")
add(f'<line x1="{W-190}" y1="{LG_Y}" x2="{W-160}" y2="{LG_Y}" stroke="{GEO}" stroke-width="1.8" '
    f'stroke-dasharray="5 3"/>')
txt(W - 154, LG_Y + 4, "파라미터 0 (기하만)", 10, MID, anchor="start")

svg_body = "\n".join(P)
HEADER = (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
          f'xmlns="http://www.w3.org/2000/svg" role="img" '
          f'aria-label="어안 카메라 3대에서 BEV free-space 마스크까지의 파이프라인 모식도">')

themed = f"{HEADER}\n{svg_body}\n</svg>\n"
(OUT_DIR / "bev_pipeline_themed.svg").write_text(themed, encoding="utf-8")

# 독립 판본: CSS 변수를 리터럴로, class를 font-family 속성으로 바꾼다
solo = themed
for var, hexv in LITERAL.items():
    solo = solo.replace(var, hexv)
solo = solo.replace('class="f-mono "', 'font-family="DejaVu Sans Mono, Consolas, monospace"')
solo = solo.replace('class="f-sans "', 'font-family="Noto Sans KR, Malgun Gothic, DejaVu Sans, sans-serif"')
solo = re.sub(r'class="f-(?:mono|sans) ?"', '', solo)
solo = solo.replace(HEADER, HEADER.replace('<svg ', f'<svg style="background:{GROUND}" ', 1), 1)
solo = '<?xml version="1.0" encoding="UTF-8"?>\n' + solo
(OUT_DIR / "bev_pipeline.svg").write_text(solo, encoding="utf-8")

for name in ("bev_pipeline.svg", "bev_pipeline_themed.svg"):
    print(f"  {name:28s} {(OUT_DIR / name).stat().st_size:>7,} bytes")
