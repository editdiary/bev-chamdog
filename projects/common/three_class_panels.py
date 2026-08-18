"""3-class BEV 예측 시각화 패널.

`bev_panels.py`(2-head 시절의 occupancy/visibility 패널)를 대체한다. 왜 새로 만들었는가:

1. **패널 중복.** 옛 배치는 occupancy·visibility를 GT/pred 각각 그려 4칸을 썼는데, 3-class
   분해가 그 둘을 이미 담고 있다(`vis = free | occupied`, 관측 영역 안에서 `drivable = free`).
   같은 정보를 세 번 보는 셈이라 눈이 갈 곳을 잃는다.
2. **글자가 안 보였다.** 패널 7개를 한 줄로 늘어놓아 폭이 3400 px를 넘는데 PIL 기본 폰트
   (약 11 px)로 글자를 얹으니 화면에 맞춰 축소하면 읽을 수 없었다. 여기서는 TrueType을
   실제 크기로 얹고, 패널을 2x2로 접어 폭을 절반으로 줄인다.
3. **오차 지도가 없었다.** GT와 pred를 나란히 놓고 눈으로 뺄셈을 하게 했다. 어디가 어떻게
   틀렸는지를 직접 칠하면 `fatal_rate`/`free_miss_rate`가 그림의 어느 색인지 바로 보인다.

색은 지표 이름과 1:1로 대응시킨다 -- 그림에서 본 색을 로그의 숫자로 바로 연결하는 것이
이 패널의 목적이다.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# GT 라벨 팔레트. `invalid`는 `unknown`과 반드시 달라야 한다 -- 하나는 배포 때 사라지는
# 수집 아티팩트(카트 손잡이·미는 사람)고 다른 하나는 진짜 미관측이다.
CLASS_COLOURS = {
    "free": (60, 200, 90),
    "occupied": (220, 60, 60),
    "unknown": (28, 28, 34),
    "invalid": (150, 60, 190),
}

# 오차 지도 팔레트. 앞의 세 개가 지표 이름과 그대로 대응한다.
ERROR_COLOURS = {
    "fatal": (255, 140, 0),        # pred free & not GT free -> `fatal_rate`의 분자
    "miss": (70, 130, 255),        # GT free & not pred free -> `free_miss_rate`의 분자
    "occ_unknown": (230, 210, 60),  # 둘 다 non-free인데 occupied/unknown이 뒤바뀐 셀
    "correct": (45, 55, 45),        # 맞은 셀은 어둡게 눕혀 오차만 도드라지게 한다
    "invalid": (90, 40, 115),
}
RANGE_COLOURS = {"gt": (255, 255, 255), "pred": (255, 235, 100)}

_BG = (250, 250, 250)
_FG = (20, 20, 20)

# 라벨이 한국어라 **한글 글리프가 있는 폰트가 먼저 와야 한다.** DejaVu에는 한글이 없어서
# 처음 만들었을 때 라벨이 전부 tofu(□□)로 나왔다.
#
# Noto Sans CJK는 `.ttc` 묶음이고 KR face가 index 1(mono는 6)에 있다. 다만 JP face(index 0)도
# 한글 글리프를 갖고 있어 **이 패널의 라벨만 보면 결과가 같다** -- index를 지정하는 것은
# 한국어 문서에 맞는 face를 고르는 선택이고, 렌더링 정상 여부를 가르지는 않는다.
_FONT_CANDIDATES = {
    (False, False): [("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),
                     ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)],
    (False, True): [("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1),
                    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
    (True, False): [("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 6),
                    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0)],
    (True, True): [("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 6),
                   ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 0)],
}


def load_font(size: int, *, mono: bool = False, bold: bool = False):
    """요청한 크기의 TrueType. 후보를 순서대로 시도하고 전부 실패하면 기본 폰트로 떨어진다.

    기본 폰트로 떨어지면 **크기 지정이 무시된다**(약 11 px 고정). 그게 옛 패널에서 글자가
    읽히지 않던 원인이므로 여기서 조용히 넘어가지 않도록 후보를 명시해 둔다.
    """
    for path, index in _FONT_CANDIDATES[(mono, bold)]:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default()


def _as_2d(mask) -> np.ndarray:
    array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return array.reshape(array.shape[-2], array.shape[-1]).astype(bool)


def render_classes(parts, valid) -> np.ndarray:
    """3-class 분해 -> `(H, W, 3)` uint8. `valid` 밖은 `invalid` 색으로 남는다."""
    valid_2d = _as_2d(valid)
    image = np.empty((*valid_2d.shape, 3), np.uint8)
    image[...] = CLASS_COLOURS["invalid"]
    image[valid_2d] = CLASS_COLOURS["unknown"]
    for name in ("unknown", "occupied", "free"):
        image[_as_2d(parts[name])] = CLASS_COLOURS[name]
    return image


def render_errors(pred_parts, gt_parts, valid) -> np.ndarray:
    """어디가 어떻게 틀렸는지.

    세 오류 범주는 **서로소다** -- `fatal`은 `pred_free & ~gt_free`, `miss`는
    `~pred_free & gt_free`, `occ_unknown`은 `~pred_free & ~gt_free`를 요구하므로 한 셀이 둘에
    걸릴 수 없다. 따라서 칠하는 순서는 결과에 영향을 주지 않는다. 조건을 손대 겹침이 생기면
    지도가 순서에 의존하게 되므로, 테스트가 (pred 클래스 × GT 클래스) 9가지 조합 전부에
    대해 색을 못박아 그 순간을 잡는다.
    """
    valid_2d = _as_2d(valid)
    pred_free, gt_free = _as_2d(pred_parts["free"]), _as_2d(gt_parts["free"])

    image = np.empty((*valid_2d.shape, 3), np.uint8)
    image[...] = ERROR_COLOURS["invalid"]
    image[valid_2d] = ERROR_COLOURS["correct"]
    # 둘 다 non-free인데 occupied/unknown이 뒤바뀐 셀. free 판정에는 영향이 없지만
    # `iou_occupied`/`iou_unknown`이 이걸 센다.
    swapped = valid_2d & ~pred_free & ~gt_free & (
        _as_2d(pred_parts["occupied"]) != _as_2d(gt_parts["occupied"])
    )
    image[swapped] = ERROR_COLOURS["occ_unknown"]
    image[valid_2d & ~pred_free & gt_free] = ERROR_COLOURS["miss"]
    image[valid_2d & pred_free & ~gt_free] = ERROR_COLOURS["fatal"]
    return image


def draw_range_profile(image, r_m, status, rays, grid_spec, colour, ray_ok: int) -> np.ndarray:
    """`r(theta)`를 격자 위 점렬로 찍는다 -- M3 오차를 눈으로 보게 하는 것이 목적이다.

    **확대 전에** 찍는다. 그래야 한 셀이 `upscale`×`upscale` 블록이 되어 실제로 보인다.
    """
    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    thetas = np.linspace(0.0, 2 * np.pi, len(r_m), endpoint=False)
    for i, theta in enumerate(thetas):
        if status[i] != ray_ok:
            continue
        radius_cells = r_m[i] / grid_spec.cell_m
        row = int(round(origin_row - radius_cells * np.cos(theta)))
        col = int(round(origin_col - radius_cells * np.sin(theta)))
        if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
            image[row, col] = colour
    return image


def upscale(image: np.ndarray, factor: int) -> Image.Image:
    height, width = image.shape[:2]
    return Image.fromarray(image).resize((width * factor, height * factor), Image.NEAREST)


_SWATCH_W, _LEGEND_SPACING = 30, 26


def _legend_item_width(text, font) -> int:
    return _SWATCH_W + int(font.getlength(text)) + _LEGEND_SPACING


def _wrap_legend(legend, available_w: int, font) -> list:
    """범례 항목을 폭에 맞춰 여러 줄로 접는다. 항목 하나가 폭보다 넓으면 그 줄에 혼자 둔다."""
    rows, row, used = [], [], 0
    for item in legend:
        width = _legend_item_width(item[0], font)
        if row and used + width > available_w:
            rows.append(row)
            row, used = [], 0
        row.append(item)
        used += width
    if row:
        rows.append(row)
    return rows


def _paste_labelled(canvas, draw, image, box, label, font):
    x, y = box
    draw.text((x, y), label, fill=_FG, font=font)
    canvas.paste(image, (x, y + font.size + 6))


def build_panel(*, camera_images, camera_names, bev_grid, headline, metric_lines,
                legend, cam_thumb_wh=(400, 225)) -> Image.Image:
    """카메라 행 + BEV 격자 + 지표/범례를 한 장으로 합친다.

    `bev_grid`는 `[[(label, PIL.Image), ...], ...]` 행렬이다. 한 줄로 길게 늘이지 않고 격자로
    접는 이유: 폭이 화면을 넘으면 사용자가 전체를 축소해서 보게 되고, 그 순간 셀도 글자도
    같이 작아져 아무것도 안 보인다(옛 배치의 실제 문제였다).
    """
    title_font = load_font(30, bold=True)
    label_font = load_font(22, bold=True)
    text_font = load_font(20, mono=True)
    legend_font = load_font(19)

    thumb_w, thumb_h = cam_thumb_wh
    gap, pad = 14, 16
    label_h = label_font.size + 6

    bev_w = max(sum(image.width for _, image in row) + gap * (len(row) - 1) for row in bev_grid)
    bev_h = sum(max(image.height for _, image in row) + label_h + gap for row in bev_grid)
    cams_w = thumb_w * len(camera_names) + gap * (len(camera_names) - 1)

    header_h = title_font.size + 14
    metrics_h = int(text_font.size * 1.45) * len(metric_lines) + 10
    total_w = max(bev_w, cams_w) + pad * 2
    # 범례를 폭에 맞춰 줄로 접는다 -- 한 줄로 두면 항목이 늘어날 때 오른쪽이 조용히 잘린다.
    legend_rows = _wrap_legend(legend, total_w - pad * 2, legend_font)
    legend_h = int(legend_font.size * 1.6) * len(legend_rows) + 8
    total_h = (header_h + thumb_h + gap + bev_h + metrics_h + legend_h + pad * 2)

    canvas = Image.new("RGB", (total_w, total_h), _BG)
    draw = ImageDraw.Draw(canvas)

    y = pad
    draw.text((pad, y), headline, fill=_FG, font=title_font)
    y += header_h

    for i, (name, image) in enumerate(zip(camera_names, camera_images)):
        x = pad + i * (thumb_w + gap)
        thumb = Image.fromarray(image).resize((thumb_w, thumb_h))
        canvas.paste(thumb, (x, y))
        draw.text((x + 8, y + 6), name, fill=(255, 240, 0), font=label_font)
    y += thumb_h + gap

    for row in bev_grid:
        x = pad
        for label, image in row:
            _paste_labelled(canvas, draw, image, (x, y), label, label_font)
            x += image.width + gap
        y += max(image.height for _, image in row) + label_h + gap

    for line in metric_lines:
        draw.text((pad, y), line, fill=_FG, font=text_font)
        y += int(text_font.size * 1.45)
    y += 10

    for row in legend_rows:
        x = pad
        for text, colour in row:
            draw.rectangle([x, y + 3, x + 22, y + 3 + legend_font.size - 4], fill=colour,
                           outline=(60, 60, 60))
            draw.text((x + 30, y), text, fill=_FG, font=legend_font)
            x += _legend_item_width(text, legend_font)
        y += int(legend_font.size * 1.6)
    return canvas
