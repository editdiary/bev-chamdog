"""BEV grid cell을 "어느 카메라가 보는가"로 색칠해서 4-cam 가시성 판정을 눈으로 확인한다.

`projects.bev_gt.visibility.compute_visible_mask`는 4대 카메라 판정을 OR로 합쳐 True/False
하나로만 돌려주므로(그래서 `build_visibility_mask.py`가 만드는 `*_visible.png`는 안 보이는
곳이 전부 같은 빨강이라 "어떤 카메라가 어디를 보는지" 구분이 안 된다), 여기서는
`compute_per_camera_visible_masks`로 카메라별 판정을 따로 얻어 각 카메라에 고유 색을 준다.
겹치는 셀(여러 카메라가 동시에 보는 곳)은 기여하는 카메라 색의 평균으로 섞는다.
`outputs/occupancy_gt/`와는 별도 경로(`outputs/debug_geometry/`)에 저장한다.

Run: python tools/visualize_camera_visibility.py --samples 00000 --spec synwoodscape_pretrain
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.grid import ROBOT_GRID_SPEC, SYNWOODSCAPE_PRETRAIN_GRID_SPEC  # noqa: E402
from projects.bev_gt.visibility import compute_per_camera_visible_masks  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/debug_geometry/camera_visibility")

GRID_SPECS = {
    "robot": ROBOT_GRID_SPEC,
    "synwoodscape_pretrain": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
}
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")
CAMERA_COLORS = {
    "FV": (230, 25, 75),   # 빨강 (전방)
    "MVL": (60, 180, 75),  # 초록 (좌측)
    "MVR": (0, 130, 200),  # 파랑 (우측)
    "RV": (245, 200, 15),  # 노랑 (후방)
}
NOT_VISIBLE_COLOR = (35, 35, 35)


def load_cameras() -> dict:
    return {
        name: load_camera(DATASET_ROOT / "calibration_data" / f"{name}.json")
        for name in CAMERA_NAMES
    }


def build_composite(per_camera_masks: dict) -> np.ndarray:
    """카메라별 mask -> (n_rows, n_cols, 3) uint8. 겹치는 셀은 기여 카메라 색의 평균."""
    shape = next(iter(per_camera_masks.values())).shape
    color_sum = np.zeros(shape + (3,), dtype=np.float64)
    contributor_count = np.zeros(shape, dtype=np.int32)

    for camera_name, mask in per_camera_masks.items():
        color_sum[mask] += CAMERA_COLORS[camera_name]
        contributor_count += mask

    composite = np.full(shape + (3,), NOT_VISIBLE_COLOR, dtype=np.uint8)
    seen = contributor_count > 0
    composite[seen] = (color_sum[seen] / contributor_count[seen, None]).astype(np.uint8)
    return composite


def build_panel(per_camera_masks: dict) -> np.ndarray:
    """카메라별 mask를 각자 고유 색으로 따로 그린 2x2 패널 (겹침 없이 단독 시야 확인용)."""
    shape = next(iter(per_camera_masks.values())).shape
    rows, cols = shape
    gap = max(2, rows // 40)
    panel = np.zeros((rows * 2 + gap, cols * 2 + gap, 3), dtype=np.uint8)

    positions = {"FV": (0, 0), "MVL": (0, 1), "MVR": (1, 0), "RV": (1, 1)}
    for camera_name, mask in per_camera_masks.items():
        tile = np.zeros((rows, cols, 3), dtype=np.uint8)
        tile[mask] = CAMERA_COLORS[camera_name]
        r, c = positions[camera_name]
        row_off = r * (rows + gap)
        col_off = c * (cols + gap)
        panel[row_off : row_off + rows, col_off : col_off + cols] = tile

    return panel


def add_legend(image: Image.Image, extra_top_px: int = 36) -> Image.Image:
    """이미지 위쪽에 카메라 이름-색 범례를 붙인다."""
    width, height = image.size
    canvas = Image.new("RGB", (width, height + extra_top_px), (255, 255, 255))
    canvas.paste(image, (0, extra_top_px))
    draw = ImageDraw.Draw(canvas)

    x = 8
    for camera_name in CAMERA_NAMES:
        color = CAMERA_COLORS[camera_name]
        draw.rectangle([x, 8, x + 20, 28], fill=color)
        draw.text((x + 24, 12), camera_name, fill=(0, 0, 0))
        x += 24 + 8 * len(camera_name) + 24
    draw.rectangle([x, 8, x + 20, 28], fill=NOT_VISIBLE_COLOR)
    draw.text((x + 24, 12), "not visible", fill=(0, 0, 0))

    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000"])
    parser.add_argument("--spec", default="synwoodscape_pretrain", choices=list(GRID_SPECS))
    parser.add_argument("--scale", type=int, default=4, help="nearest-neighbor 확대 배율")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cameras = load_cameras()
    grid_spec = GRID_SPECS[args.spec]

    for sample_idx in args.samples:
        depth_maps = {
            name: np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{name}.npy")
            for name in CAMERA_NAMES
        }
        per_camera_masks = compute_per_camera_visible_masks(grid_spec, cameras, depth_maps)

        composite = build_composite(per_camera_masks)
        composite_img = Image.fromarray(composite).resize(
            (composite.shape[1] * args.scale, composite.shape[0] * args.scale), Image.NEAREST
        )
        composite_img = add_legend(composite_img)
        composite_path = OUTPUT_DIR / f"{sample_idx}_{args.spec}_camera_composite.png"
        composite_img.save(composite_path)

        panel = build_panel(per_camera_masks)
        panel_img = Image.fromarray(panel).resize(
            (panel.shape[1] * args.scale, panel.shape[0] * args.scale), Image.NEAREST
        )
        panel_img = add_legend(panel_img)
        panel_path = OUTPUT_DIR / f"{sample_idx}_{args.spec}_camera_panel.png"
        panel_img.save(panel_path)

        for camera_name, mask in per_camera_masks.items():
            print(f"[{sample_idx}/{args.spec}] {camera_name} visible_fraction={mask.mean():.3f}")
        print(f"  -> {composite_path}")
        print(f"  -> {panel_path}")


if __name__ == "__main__":
    main()
