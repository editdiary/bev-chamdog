"""Phase 2 1회성 보정: 여러 샘플에 걸쳐 BEV 이미지의 스케일(m/px)과
   ego-vehicle 원점 일관성을 확인하고, LiDAR 포인트를 이용해 전방/좌우 부호를 찾는다.

Run: python tools/calibrate_bev_scale.py --samples 00000 00010 00100 00200 00300
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Allow running as `python tools/calibrate_bev_scale.py` from the repo root
# without needing PYTHONPATH set: when invoked as a script, sys.path[0] is
# this file's directory (tools/), not the repo root, so `projects` is not
# importable unless we add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.bev_calibration import (  # noqa: E402
    ego_vehicle_bbox_center,
    estimate_scale_m_per_px,
    load_box_3d_annotations,
    load_instance_bev_image,
)
from projects.geometry.frames import lidar_points_to_ego  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")


def collect_scale_estimates(sample_indices):
    scales_x, scales_y, centers = [], [], []
    for idx in sample_indices:
        instance_image = load_instance_bev_image(
            DATASET_ROOT / f"_unuserd/instance_annotations/gtLabels/{idx}_BEV.png"
        )
        box_3d_by_instance = load_box_3d_annotations(
            DATASET_ROOT / f"_unuserd/box_3d_annotations/{idx}.pkl"
        )
        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )

        center = ego_vehicle_bbox_center(semantic_bev)
        if center is not None:
            centers.append(center)

        for instance_id in np.unique(instance_image):
            if instance_id == 0:
                continue
            result = estimate_scale_m_per_px(instance_image, box_3d_by_instance, int(instance_id))
            if result is not None:
                scales_x.append(result[0])
                scales_y.append(result[1])

    return np.array(scales_x), np.array(scales_y), np.array(centers)


def find_axis_sign(sample_indices, meters_per_pixel, origin_px, image_shape):
    """Search sign_forward/sign_lateral (row axis = forward, col axis = lateral; see docstring above)
    against the pixel class each LiDAR point lands on, aggregated over all samples."""
    height, width = image_shape
    candidates = [(sf, sl) for sf in (1, -1) for sl in (1, -1)]
    totals = {c: [0, 0] for c in candidates}  # (matches, total)

    for idx in sample_indices:
        with open(DATASET_ROOT / f"lidar_data/{idx}.pkl", "rb") as f:
            lidar = pickle.load(f)
        points_ego = lidar_points_to_ego(lidar["points"])
        labels = lidar["labels"]

        radius_xy = np.linalg.norm(points_ego[:, :2], axis=1)
        half_footprint_m = height / 2 * meters_per_pixel
        keep = (radius_xy > 1.0) & (radius_xy < half_footprint_m)
        if not keep.any():
            continue
        ego_x, ego_y = points_ego[keep, 0], points_ego[keep, 1]
        point_labels = labels[keep]

        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )

        for sign_forward, sign_lateral in candidates:
            row = origin_px[1] + (sign_forward * ego_x) / meters_per_pixel
            col = origin_px[0] + (sign_lateral * ego_y) / meters_per_pixel
            row_i, col_i = row.round().astype(int), col.round().astype(int)
            valid = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
            if not valid.any():
                continue
            gt_classes = semantic_bev[row_i[valid], col_i[valid]]
            matches = (gt_classes == point_labels[valid]).sum()
            totals[(sign_forward, sign_lateral)][0] += int(matches)
            totals[(sign_forward, sign_lateral)][1] += int(valid.sum())

    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00010", "00100", "00200", "00300"])
    args = parser.parse_args()

    scales_x, scales_y, centers = collect_scale_estimates(args.samples)
    print(f"scale_x: mean={scales_x.mean():.5f} std={scales_x.std():.5f} n={len(scales_x)}")
    print(f"scale_y: mean={scales_y.mean():.5f} std={scales_y.std():.5f} n={len(scales_y)}")
    print(f"ego bbox centers: {centers.tolist()}")

    meters_per_pixel = float(np.concatenate([scales_x, scales_y]).mean())
    origin_px = tuple(centers.mean(axis=0)) if len(centers) else (511.5, 511.5)

    sample_image = load_instance_bev_image(
        DATASET_ROOT / f"semantic_annotations/gtLabels/{args.samples[0]}_BEV.png"
    )
    totals = find_axis_sign(args.samples, meters_per_pixel, origin_px, sample_image.shape)

    print("\naxis sign search (sign_forward, sign_lateral) -> match_rate (n samples):")
    for signs, (matches, total) in totals.items():
        rate = matches / total if total else float("nan")
        print(f"  {signs}: {rate:.3f} (n={total})")

    print(
        "\n지침: 위에서 가장 높은 match_rate가 다른 후보들보다 뚜렷하게 크면 "
        "그 (sign_forward, sign_lateral)을 Task 9의 SIGN_FORWARD/SIGN_LATERAL 상수로 쓴다. "
        "차이가 뚜렷하지 않으면 rgb_images/<idx>_BEV.png(컬러)를 열어 차량 전방(헤드라이트 방향)이 "
        "이미지의 어느 쪽인지 육안으로 확인해 부호를 정한다."
    )


if __name__ == "__main__":
    main()
