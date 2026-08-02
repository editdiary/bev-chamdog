"""Phase 1 검증: SynWoodScape LiDAR 포인트를 4대 fisheye 카메라에 재투영해
   투영 함수의 정확성을 육안 오버레이 + semantic consistency rate로 확인한다.

Run: python tools/verify_fisheye_projection.py --samples 00000 00001 00002
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as `python tools/verify_fisheye_projection.py` from the repo
# root without needing PYTHONPATH set: when invoked as a script, sys.path[0]
# is this file's directory (tools/), not the repo root, so `projects` is not
# importable unless we add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.common.metrics import class_consistency_rate  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.geometry.frames import world_points_to_ego  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
CAMERAS = ["FV", "MVL", "MVR", "RV"]
OUTPUT_DIR = Path("outputs/fisheye_verification")

# readme.txt 팔레트에서 뽑은 대표 색상 (BGR, cv2 저장용). 없는 class는 회색으로 표시한다.
PALETTE_BGR = {
    6: (50, 234, 157), 7: (128, 64, 128), 8: (232, 35, 244), 1: (70, 70, 70),
    9: (35, 142, 107), 10: (142, 0, 0), 12: (0, 220, 220),
}
DEFAULT_COLOR_BGR = (128, 128, 128)


def verify_sample(sample_idx: str) -> None:
    with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
        lidar = pickle.load(f)
    points_ego = world_points_to_ego(lidar["points"], np.asarray(lidar["transform"]))
    labels = lidar["labels"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cam_name in CAMERAS:
        cam = load_camera(DATASET_ROOT / "calibration_data" / f"{cam_name}.json")
        pixels = cam.project_3d_to_2d(points_ego)

        valid = ~np.isnan(pixels[:, 0])
        in_bounds = (
            valid
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < cam.width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < cam.height)
        )
        n_in_bounds = int(in_bounds.sum())
        if n_in_bounds == 0:
            print(f"[{sample_idx}/{cam_name}] no points landed in the image, skipping")
            continue

        image_path = DATASET_ROOT / "rgb_images" / f"{sample_idx}_{cam_name}.png"
        image = cv2.imread(str(image_path))
        gt_path = DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{cam_name}.png"
        gt_labels_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        # Rounding a float pixel coordinate just under the bound (e.g. 1279.6
        # when cam.width == 1280) can round up to exactly `cam.width`, which
        # is out of range for indexing. Clip after rounding as a safety net.
        px = np.clip(pixels[in_bounds, 0].round().astype(int), 0, cam.width - 1)
        py = np.clip(pixels[in_bounds, 1].round().astype(int), 0, cam.height - 1)
        point_labels = labels[in_bounds]
        gt_at_pixel = gt_labels_image[py, px]

        rate = class_consistency_rate(point_labels, gt_at_pixel)
        print(f"[{sample_idx}/{cam_name}] n_in_bounds={n_in_bounds} semantic_consistency_rate={rate:.3f}")

        overlay = image.copy()
        for u, v, label in zip(px, py, point_labels):
            color = PALETTE_BGR.get(int(label), DEFAULT_COLOR_BGR)
            cv2.circle(overlay, (int(u), int(v)), radius=2, color=color, thickness=-1)
        out_path = OUTPUT_DIR / f"{sample_idx}_{cam_name}_overlay.png"
        cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    for sample_idx in args.samples:
        verify_sample(sample_idx)


if __name__ == "__main__":
    main()
