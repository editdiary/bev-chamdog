"""Phase 2 1회성 보정 확인: BEV 이미지의 원점·축 부호·스케일을 실데이터로 재확인한다.

파이프라인이 실제로 쓰는 상수는 `projects/bev_gt/bev_crop.py`에 있고, 이 스크립트는
그 상수를 **검증**하는 용도다 (값을 새로 제안하지 않는다).

네 가지를 확인한다.

1. **원점** — ego 차량(semantic class 24) 블롭의 픽셀 bbox 중심이 여러 샘플에서
   (511.5, 511.5)로 일관되는지 본다.
2. **축 부호** — LiDAR 지면 포인트를 BEV 픽셀로 보내 semantic class 일치율을 4개 부호
   후보에 대해 비교한다. `SIGN_FORWARD`/`SIGN_LATERAL`이 최고여야 한다.
3. **스케일** — LiDAR **지면** 포인트로 m/px를 sweep해 `BEV_METERS_PER_PIXEL`(=15/512)
   근방에서 정점이 나오는지 본다.
4. (참고) 차량 실루엣 기반 추정치를 함께 찍어 **왜 그것을 쓰면 안 되는지**를 수치로 보여준다.

⚠️ **차량 실루엣(3D 박스 ÷ 픽셀 bbox) 기반 스케일 추정은 구조적으로 편향된다 — 이 값으로
`BEV_METERS_PER_PIXEL`을 되돌리지 말 것.**
BEV 카메라는 ego (0,0,15)의 **핀홀**(pitch=−90, FOV 90°)이라 높이 h인 면은 15/(15−h)배로
확대되어 찍힌다. 차량 실루엣은 지붕(h≈1.5 m)까지의 윤곽이므로 지면보다 크게 나오고, 그만큼
m/px가 **작게** 나온다 (실측: 실루엣 0.0285~0.0288 vs 참값 0.029297).
게다가 이 방법을 임의 yaw의 다른 instance들에 적용하면 축정렬 픽셀 bbox가 yaw에 오염되어
x/y 축 값이 갈라진다(예전 이 스크립트의 출력 ~0.031 / ~0.019). 그래서 아래 [4]는
**ego 자신의 instance 하나만**(자기 프레임에서 축정렬이므로 yaw 오염이 없다) 참고로 찍고,
그 값을 스케일 후보로 제안하지 않는다.
파이프라인 상수는 기하 도출(15·tan45°/512 = 15/512, 전체 1024px = 30.0 m)에서 오고,
그 교차검증은 아래 [3] + `tests/bev_gt/test_bev_crop.py`의
`test_bev_scale_beats_the_biased_vehicle_silhouette_estimate_on_ground_points`가 담당한다.

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
    box_corners_to_ego,
    ego_vehicle_bbox_center,
    load_box_3d_annotations,
    load_instance_bev_image,
    silhouette_scale_m_per_px,
)
from projects.bev_gt.bev_crop import (  # noqa: E402
    BEV_METERS_PER_PIXEL,
    BEV_ORIGIN_PX,
    SIGN_FORWARD,
    SIGN_LATERAL,
    ego_to_bev_pixel,
)
from projects.geometry.frames import lidar_points_to_ego, parse_vehicle_transform  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")

# 지면 위 class만 쓴다 (핀홀 확대가 없는 z≈0 표면):
# road line(6), road(7), sidewalk(8), ground(14), terrain(20)
GROUND_CLASS_IDS = (6, 7, 8, 14, 20)
# ego 차체 self-hit과 원거리(1024px=30m 밖) 포인트를 배제하는 반경 창
MIN_RADIUS_M, MAX_RADIUS_M = 2.5, 13.0
MAX_GROUND_HEIGHT_M = 0.15

# ego 차량 자신의 instance/semantic class id (SynWoodScape 팔레트: 24 = ego-vehicle)
EGO_INSTANCE_ID = 24
BEV_CAMERA_HEIGHT_M = 15.0  # readme.txt: BEV camera at z=15, pitch=-90


def collect_ego_bbox_centers(sample_indices):
    """ego 차량(semantic class 24) blob의 픽셀 bbox 중심을 샘플별로 모은다."""
    centers = []
    for idx in sample_indices:
        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )
        center = ego_vehicle_bbox_center(semantic_bev)
        if center is not None:
            centers.append(center)
    return np.array(centers)


def load_ground_points(sample_indices):
    """샘플별 (지면 LiDAR 포인트 ego 좌표, per-point label, BEV semantic 이미지)."""
    for idx in sample_indices:
        with open(DATASET_ROOT / f"lidar_data/{idx}.pkl", "rb") as f:
            lidar = pickle.load(f)
        points_ego = lidar_points_to_ego(lidar["points"])
        labels = np.asarray(lidar["labels"])

        radius_xy = np.linalg.norm(points_ego[:, :2], axis=1)
        keep = (
            np.isin(labels, GROUND_CLASS_IDS)
            & (np.abs(points_ego[:, 2]) < MAX_GROUND_HEIGHT_M)
            & (radius_xy > MIN_RADIUS_M)
            & (radius_xy < MAX_RADIUS_M)
        )
        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )
        yield points_ego[keep], labels[keep], semantic_bev


def class_match_counts(points_ego, labels, semantic_bev, **mapping_kwargs):
    """포인트를 BEV 픽셀로 보내 semantic class가 일치하는 개수 / 이미지 안에 든 개수."""
    if not len(points_ego):
        return 0, 0
    row, col = ego_to_bev_pixel(points_ego[:, 0], points_ego[:, 1], **mapping_kwargs)
    row_i, col_i = np.round(row).astype(int), np.round(col).astype(int)
    height, width = semantic_bev.shape
    inside = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
    if not inside.any():
        return 0, 0
    gt_classes = semantic_bev[row_i[inside], col_i[inside]]
    return int((gt_classes == labels[inside]).sum()), int(inside.sum())


def sweep(samples, variants, **fixed_kwargs):
    """variants: {label: mapping_kwargs}. 모든 샘플에 걸쳐 누적한 일치율을 돌려준다."""
    matches = {label: 0 for label in variants}
    totals = {label: 0 for label in variants}
    for points_ego, labels, semantic_bev in load_ground_points(samples):
        for label, kwargs in variants.items():
            m, t = class_match_counts(
                points_ego, labels, semantic_bev, **{**fixed_kwargs, **kwargs}
            )
            matches[label] += m
            totals[label] += t
    return {label: (matches[label], totals[label]) for label in variants}


def report_silhouette_bias(sample_indices):
    """ego 실루엣 기반 m/px를 찍고, 왜 그것이 참값보다 낮은지 배율로 설명한다.

    ego instance만 쓰고, 3D 박스는 반드시 **ego 프레임으로 되돌린 뒤** 치수를 잰다. 예전
    버전은 world 프레임 코너의 축정렬 extent를 그대로 써서 actor의 world yaw가 섞였고,
    그래서 샘플마다 x/y가 뒤바뀐(0.028↔0.062, 0.028↔0.014) 값이 나왔다.
    """
    print("\n[4] (참고만) 차량 실루엣 기반 추정 — ⚠️ 스케일 상수로 쓰면 안 된다")
    ratios, roof_heights = [], []
    for idx in sample_indices:
        instance_image = load_instance_bev_image(
            DATASET_ROOT / f"_unuserd/instance_annotations/gtLabels/{idx}_BEV.png"
        )
        box_3d_by_instance = load_box_3d_annotations(
            DATASET_ROOT / f"_unuserd/box_3d_annotations/{idx}.pkl"
        )
        world_T_ego = parse_vehicle_transform(DATASET_ROOT / f"vehicle_data/rgb_images/{idx}.txt")
        corners_ego = box_corners_to_ego(box_3d_by_instance, EGO_INSTANCE_ID, world_T_ego)
        result = silhouette_scale_m_per_px(instance_image, corners_ego, EGO_INSTANCE_ID)
        if result is None:
            print(f"  {idx}: ego instance {EGO_INSTANCE_ID} 없음")
            continue
        lateral, forward = result
        ratios.extend(r / BEV_METERS_PER_PIXEL for r in result)
        roof_heights.append(float(corners_ego[:, 2].max()))
        print(
            f"  {idx}: 좌우기준={lateral:.5f} 전후기준={forward:.5f} "
            f"(참값 15/512 대비 {lateral / BEV_METERS_PER_PIXEL:.3f}배 / "
            f"{forward / BEV_METERS_PER_PIXEL:.3f}배)"
        )
    if not ratios:
        print("  ego instance를 찾은 샘플이 없어 편향 시연을 건너뛴다.")
        return
    roof_height = max(roof_heights)
    roof_magnification = BEV_CAMERA_HEIGHT_M / (BEV_CAMERA_HEIGHT_M - roof_height)
    print(
        f"  실루엣이 항상 참값보다 작게(관측 {min(ratios):.3f}~{max(ratios):.3f}배) 나오는 이유: "
        f"BEV 핀홀(높이 {BEV_CAMERA_HEIGHT_M:g} m)에서 높이 h인 면은 15/(15−h)배로 확대되어 찍힌다. "
        f"ego 박스의 최상단은 z={roof_height:.3f} m이므로 확대율이 1.000~{roof_magnification:.3f}배이고, "
        f"참값을 그 범위로 나눈 구간 "
        f"[{BEV_METERS_PER_PIXEL / roof_magnification:.5f}, {BEV_METERS_PER_PIXEL:.5f}] 안에 "
        "관측값이 들어간다. 즉 편향의 크기·방향이 핀홀 모델로 정량적으로 설명된다.\n"
        "  → 스케일 상수는 [3]의 지면 포인트 sweep과 기하 도출(15/512)에서만 가져온다."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00010", "00100", "00200", "00300"])
    args = parser.parse_args()

    print("파이프라인 상수 (projects/bev_gt/bev_crop.py):")
    print(f"  BEV_METERS_PER_PIXEL = 15/512 = {BEV_METERS_PER_PIXEL:.9f}  (1024px = 30.0 m)")
    print(f"  BEV_ORIGIN_PX        = {BEV_ORIGIN_PX}")
    print(f"  SIGN_FORWARD, SIGN_LATERAL = {SIGN_FORWARD}, {SIGN_LATERAL}")

    # --- 1. 원점 ---
    centers = collect_ego_bbox_centers(args.samples)
    print("\n[1] ego bbox 중심 (원점 확인)")
    print(f"  샘플별: {centers.tolist()}")
    if len(centers):
        print(f"  mean={tuple(np.round(centers.mean(axis=0), 3))} max_dev_px={np.abs(centers - np.array(BEV_ORIGIN_PX)).max():.3f}")

    # --- 2. 축 부호 ---
    sign_variants = {
        (sf, sl): {"sign_forward": sf, "sign_lateral": sl} for sf in (1, -1) for sl in (1, -1)
    }
    sign_results = sweep(args.samples, sign_variants)
    print("\n[2] 축 부호 (sign_forward, sign_lateral) -> 지면 LiDAR class 일치율")
    ranked = sorted(sign_results.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1)))
    for signs, (m, t) in ranked:
        marker = "  <-- 파이프라인 상수" if signs == (SIGN_FORWARD, SIGN_LATERAL) else ""
        print(f"  {signs}: {m / max(t, 1):.4f} (n={t}){marker}")
    best_signs = ranked[0][0]
    if best_signs == (SIGN_FORWARD, SIGN_LATERAL):
        print("  → 파이프라인 상수가 최고. 일치.")
    else:
        print(f"  → ⚠️ 최고가 {best_signs}다. bev_crop.py의 부호를 재검토할 것.")

    # --- 3. 스케일 sweep ---
    candidates = np.round(np.arange(0.0270, 0.0320 + 1e-9, 0.0002), 6)
    scale_results = sweep(
        args.samples, {mpp: {"meters_per_pixel": float(mpp)} for mpp in candidates}
    )
    print("\n[3] 스케일 sweep (지면 LiDAR class 일치율) — 차량 실루엣이 아니라 지면 포인트 기반")
    for mpp in candidates:
        m, t = scale_results[mpp]
        rate = m / max(t, 1)
        bar = "#" * int(round((rate - 0.85) * 200)) if rate > 0.85 else ""
        print(f"  {mpp:.4f} -> {rate:.4f} {bar}")
    best_mpp = max(candidates, key=lambda mpp: scale_results[mpp][0] / max(scale_results[mpp][1], 1))
    print(f"  sweep 최고: {best_mpp:.4f}")
    print(f"  파이프라인 상수 15/512 = {BEV_METERS_PER_PIXEL:.6f}")
    if abs(best_mpp - BEV_METERS_PER_PIXEL) <= 0.0003:
        print("  → sweep 정점이 15/512와 0.0003 이내로 일치. 상수 확인됨.")
    else:
        print("  → ⚠️ sweep 정점이 15/512에서 멀다. 재조사할 것.")

    # --- 4. (참고) 차량 실루엣 추정 = 편향된 값. 스케일 후보로 쓰지 말 것 ---
    report_silhouette_bias(args.samples)


if __name__ == "__main__":
    main()
