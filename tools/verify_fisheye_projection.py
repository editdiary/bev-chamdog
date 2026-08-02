"""Phase 1 검증: SynWoodScape LiDAR 포인트를 4대 fisheye 카메라에 재투영해
   투영 함수의 정확성을 육안 오버레이 + semantic consistency rate로 확인한다.

Run: python tools/verify_fisheye_projection.py --samples 00000 00001 00002

세 가지 지표를 함께 보고한다.

- `semantic_consistency_rate`: 이미지 안에 떨어진 모든 포인트에 대한 라벨 일치율.
  LiDAR는 ego 기준 z=2.0에 있고 카메라는 z=0.9~1.0이라, **LiDAR에는 보이지만 카메라에는
  가려지는** 포인트가 구조적으로 20~35% 존재한다. 그 포인트들은 가림 물체의 class 위에
  떨어지므로 이 지표는 투영이 완벽해도 0.9에 도달할 수 없다.
- `depth_agreement`: LiDAR 포인트의 카메라까지 거리가 그 픽셀의 depth map 값과 5% 이내로
  일치하는 비율. **투영 기하 정확도에 가장 민감한 지표다** — 카메라 위치가 10 cm만 틀려도
  0.72 → 0.22처럼 크게 떨어진다. 나머지는 (진짜 가림) + (잔여 오차)다.
- `visible_consistency_rate`: 위 depth 일치 포인트만 골라 계산한 라벨 일치율.
  단, 이 필터 자체가 "depth와 맞는 포인트"를 고르는 것이라 **자기선택 편향**이 있어
  translation 오차에 둔감하다(0.5 m 틀려도 0.87~0.99). 기하 정확도 판단은
  `depth_agreement`를 먼저 보고, 이 값은 "보이는 포인트에서는 라벨이 맞는가"의 확인용으로 쓴다.
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
from projects.geometry.frames import lidar_points_to_ego  # noqa: E402
from projects.geometry.reprojection import project_points_to_image, visibility_mask  # noqa: E402

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
    points_ego = lidar_points_to_ego(lidar["points"])
    labels = lidar["labels"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cam_name in CAMERAS:
        cam = load_camera(DATASET_ROOT / "calibration_data" / f"{cam_name}.json")
        projected = project_points_to_image(cam, points_ego)
        n_in_bounds = len(projected)
        if n_in_bounds == 0:
            print(f"[{sample_idx}/{cam_name}] no points landed in the image, skipping")
            continue

        image_path = DATASET_ROOT / "rgb_images" / f"{sample_idx}_{cam_name}.png"
        image = cv2.imread(str(image_path))
        gt_path = DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{cam_name}.png"
        gt_labels_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        depth_map = np.load(DATASET_ROOT / "depth_maps" / "raw_data" / f"{sample_idx}_{cam_name}.npy")

        px, py = projected.col, projected.row
        point_labels = labels[projected.index]
        gt_at_pixel = gt_labels_image[py, px]
        visible = visibility_mask(projected, depth_map)

        rate = class_consistency_rate(point_labels, gt_at_pixel)
        visible_rate = class_consistency_rate(point_labels, gt_at_pixel, valid_mask=visible)
        print(
            f"[{sample_idx}/{cam_name}] n_in_bounds={n_in_bounds} "
            f"semantic_consistency_rate={rate:.3f} "
            f"depth_agreement={visible.mean():.3f} "
            f"visible_consistency_rate={visible_rate:.3f} (n_visible={int(visible.sum())})"
        )

        overlay = image.copy()
        for u, v, label in zip(px, py, point_labels):
            color = PALETTE_BGR.get(int(label), DEFAULT_COLOR_BGR)
            cv2.circle(overlay, (int(u), int(v)), radius=2, color=color, thickness=-1)
        out_path = OUTPUT_DIR / f"{sample_idx}_{cam_name}_overlay.png"
        cv2.imwrite(str(out_path), overlay)

        # 가시 포인트만 그린 오버레이 — 육안 확인용(가려진 포인트가 화면을 덮지 않는다)
        visible_overlay = image.copy()
        for u, v, label in zip(px[visible], py[visible], point_labels[visible]):
            color = PALETTE_BGR.get(int(label), DEFAULT_COLOR_BGR)
            cv2.circle(visible_overlay, (int(u), int(v)), radius=2, color=color, thickness=-1)
        cv2.imwrite(str(OUTPUT_DIR / f"{sample_idx}_{cam_name}_overlay_visible.png"), visible_overlay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    for sample_idx in args.samples:
        verify_sample(sample_idx)


if __name__ == "__main__":
    main()
