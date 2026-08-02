"""각 어안 카메라의 RGB 이미지 위에 그 카메라 자신의 depth map을 색으로 덮어 그린다.

`visibility_mask`가 "투영된 점의 range_m vs depth_map[row,col]"를 비교해 가려짐을 판정하는데,
그 depth_map 자체가 그 카메라 시야에서 정확히 어떤 실제 3D 구조를 담고 있는지 눈으로 바로
확인하기 위한 도구다. `outputs/occupancy_gt/`와는 별도 경로(`outputs/debug_geometry/`)에 저장한다
— occupancy/visibility 산출물과 섞이지 않게.

가까운 거리(0m 부근)를 빨강, 먼 거리(`--max-depth-m`, 기본 15m)를 파랑/보라로 칠하는
turbo 계열 colormap을 쓴다. `--max-depth-m`보다 먼 픽셀(하늘 등, sky=1000.0 포함)은 원본
RGB를 그대로 남겨 depth 색과 섞이지 않게 한다.

Run: python tools/visualize_depth_overlay.py --samples 00000 --cameras FV MVL MVR RV
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/debug_geometry/depth_overlay")
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")


def build_overlay(sample_idx: str, camera_name: str, max_depth_m: float, alpha: float) -> np.ndarray:
    rgb = np.asarray(
        Image.open(DATASET_ROOT / "rgb_images" / f"{sample_idx}_{camera_name}.png").convert("RGB")
    )
    depth = np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{camera_name}.npy")

    in_range = depth <= max_depth_m
    normalized = np.clip(depth / max_depth_m, 0.0, 1.0)
    colored = (matplotlib.colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)

    overlay = rgb.copy()
    blended = (alpha * colored + (1 - alpha) * rgb).astype(np.uint8)
    overlay[in_range] = blended[in_range]
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000"])
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_NAMES), choices=CAMERA_NAMES)
    parser.add_argument("--max-depth-m", type=float, default=15.0, help="이 거리보다 먼 픽셀은 원본 RGB 유지")
    parser.add_argument("--alpha", type=float, default=0.55, help="depth 색상 블렌딩 비율")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample_idx in args.samples:
        for camera_name in args.cameras:
            overlay = build_overlay(sample_idx, camera_name, args.max_depth_m, args.alpha)
            out_path = OUTPUT_DIR / f"{sample_idx}_{camera_name}_depth_overlay.png"
            Image.fromarray(overlay).save(out_path)
            print(f"[{sample_idx}/{camera_name}] max_depth_m={args.max_depth_m} -> {out_path}")


if __name__ == "__main__":
    main()
