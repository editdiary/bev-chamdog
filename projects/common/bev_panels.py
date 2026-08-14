"""BEV 예측 시각화 패널 렌더링 -- SynWoodScape와 자체 데이터셋 시각화 도구가 공유한다.

원래는 `tools/visualize_predictions.py` 안에 있었다. 두 데이터셋의 패널을 나란히 놓고
읽으려면 색·배치가 같아야 하는데, 도구끼리 import하면 한쪽을 손댈 때 다른 쪽이 깨진다.

카메라 썸네일 크기는 인자로 받는다 -- SynWoodScape는 4:3(1280x966), 자체 리그는
16:9(1280x720)라 같은 값을 쓰면 한쪽이 찌그러진다.
"""
import numpy as np
from PIL import Image, ImageDraw

# `tools/build_hybrid_occupancy.py`의 GT combined.png와 같은 팔레트 -- 나란히 비교하기 쉽게 유지.
COLOR_UNKNOWN = (128, 128, 128)
COLOR_DRIVABLE = (76, 175, 80)
COLOR_OBSTACLE = (192, 57, 43)
COLOR_VISIBLE = (33, 150, 243)

CELL_UPSCALE = 2
CAM_THUMB_W, CAM_THUMB_H = 220, 166


def occupancy_to_image(occupancy: np.ndarray, observed: np.ndarray,
                       upscale: int = CELL_UPSCALE) -> Image.Image:
    image = np.full(occupancy.shape + (3,), COLOR_UNKNOWN, dtype=np.uint8)
    image[observed & (occupancy == 1)] = COLOR_DRIVABLE
    image[observed & (occupancy == 0)] = COLOR_OBSTACLE
    h, w = occupancy.shape
    return Image.fromarray(image).resize((w * upscale, h * upscale), Image.NEAREST)


def visibility_to_image(visible: np.ndarray, upscale: int = CELL_UPSCALE) -> Image.Image:
    image = np.full(visible.shape + (3,), COLOR_UNKNOWN, dtype=np.uint8)
    image[visible] = COLOR_VISIBLE
    h, w = visible.shape
    return Image.fromarray(image).resize((w * upscale, h * upscale), Image.NEAREST)


def compute_iou(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    intersection = (pred * target * valid).sum()
    union = np.clip(pred + target, 0, 1) * valid
    return float(intersection / max(union.sum(), 1e-4))


def format_obstacle_iou(pred_obstacle: np.ndarray, target_obstacle: np.ndarray,
                        valid: np.ndarray) -> str:
    """GT에 obstacle이 없는 샘플은 IoU가 0으로 고정되므로(완벽히 맞혀도 0) 숫자를 감춘다.
    학습 쪽이 이런 샘플을 평균에서 빼는 것과 같은 이유다.
    """
    if (target_obstacle * valid).sum() == 0:
        return "n/a (no GT obstacle)"
    return f"{compute_iou(pred_obstacle, target_obstacle, valid):.3f}"


def build_panel(rgb_camXs_01: np.ndarray, camera_names, bev_images, title_lines,
                cam_thumb_wh=(CAM_THUMB_W, CAM_THUMB_H)) -> Image.Image:
    thumb_w, thumb_h = cam_thumb_wh
    cams_row = Image.new("RGB", (thumb_w * len(camera_names), thumb_h))
    draw_cams = ImageDraw.Draw(cams_row)
    for i, name in enumerate(camera_names):
        arr = (rgb_camXs_01[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        thumb = Image.fromarray(arr).resize((thumb_w, thumb_h))
        cams_row.paste(thumb, (i * thumb_w, 0))
        draw_cams.text((i * thumb_w + 5, 5), name, fill=(255, 255, 0))

    gap = 12
    label_h = 22
    bev_row_w = sum(image.width for _, image in bev_images) + gap * (len(bev_images) - 1)
    bev_row_h = max(image.height for _, image in bev_images) + label_h
    bev_row = Image.new("RGB", (bev_row_w, bev_row_h), (255, 255, 255))
    draw_bev = ImageDraw.Draw(bev_row)
    x = 0
    for label, image in bev_images:
        draw_bev.text((x + 5, 4), label, fill=(0, 0, 0))
        bev_row.paste(image, (x, label_h))
        x += image.width + gap

    text_h = 20 * (len(title_lines) + 1)
    total_w = max(cams_row.width, bev_row.width)
    total_h = cams_row.height + bev_row.height + text_h
    panel = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    panel.paste(cams_row, (0, 0))
    panel.paste(bev_row, (0, cams_row.height))
    draw = ImageDraw.Draw(panel)
    for i, line in enumerate(title_lines):
        draw.text((5, cams_row.height + bev_row.height + i * 20), line, fill=(0, 0, 0))
    return panel
