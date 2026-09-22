"""SynWoodScape **원본 기하**(30 m × 30 m, 200×200, 0.15 m/cell) 학습 라벨을 만든다.

`dataset/synwoodscape_2head_roi_8_4_6_h08/`과 **같은 파일 이름 규약**으로 내보내므로
`SynWoodScapeSimpleBEVDataset`이 루트만 바꿔서 그대로 읽는다.

    <sample_id>_occupancy.npy   uint8, drivable=1 / non-drivable=0
    <sample_id>_visible.npy     bool,  gather-column(z=0~0.8 m) 가시성 & ~ego

## 무엇이 기존 라벨과 다른가 -- 둘이고, 둘 다 의도한 것이다

1. **범위와 셀 크기.** 기존은 전방8/후방4/±6 m를 0.05 m/cell로(240×240) 잘랐고, 여기는
   소스가 실제로 주는 **30 m 전체**를 0.15 m/cell로(200×200) 쓴다. 이 실험이 재려는 축이다.
2. **⚠ 수동 보정이 없다.** 기존 라벨은 `dataset/annotated_roi_8-4-6_semantic_crop/`에서
   사람이 보정한 것이고, 그 보정은 **ROI 8/4/±6으로 자른 뒤에** 이뤄졌다. 넓힌 영역에는
   보정본이 없으므로 여기서는 **시맨틱 라벨의 자동 크롭**을 그대로 쓴다.
   **그래서 이 라벨로 학습한 팔과 기존 팔의 차이에는 기하와 라벨 품질이 섞여 있다.**
   기하만 분리하려면 "보정 없이 8/4/±6"을 한 팔 더 만들어야 한다.

## 알아 둘 기하 한계

BEV 소스는 정사영이 아니라 **z=15 m의 핀홀**이라 높이 h인 면이 15/(15−h)배로 확대된다.
횡방향 15 m에서 1.5 m 높이 물체는 바깥으로 약 1.7 m 밀린다. **넓힐수록 주변부 라벨이
지저분해진다** -- ±6 m일 때보다 확실히 나쁘다. 결과 해석에 이 한 줄이 필요하다.

## ego 제외

기존 파이프라인은 `canonical_ego_mask.png`(다수결로 만든 정준 실루엣)를 썼는데 그것은
240×240 전용이다. 여기서는 같은 방식으로 **이 그리드에서 다시 만든다** -- 샘플마다 semantic
class 24(ego-vehicle)를 크롭해 다수결(≥50 %)을 취한다.

Run:
    python tools/build_synwoodscape_native_labels.py
    python tools/build_synwoodscape_native_labels.py --limit 5 --out_root /tmp/probe
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.bev_crop import (  # noqa: E402
    BEV_METERS_PER_PIXEL,
    BEV_ORIGIN_PX,
    SIGN_FORWARD,
    SIGN_LATERAL,
    crop_bev_occupancy,
    ego_to_bev_pixel,
)
from projects.bev_gt.grid import (  # noqa: E402
    SYNWOODSCAPE_NATIVE_GRID_SPEC,
    cell_centers_m,
)
from projects.bev_gt.visibility import (  # noqa: E402
    COLUMN_HEIGHTS_M,
    compute_column_visible_mask,
)
from projects.geometry.fisheye import load_camera  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
DEFAULT_OUT_ROOT = Path("dataset/synwoodscape_native_roi_15_15_15_h08")
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")
EGO_CLASS_ID = 24
EGO_VOTE_FRACTION = 0.5


def _sample_ids(root: Path):
    return sorted(p.name[:5] for p in (root / "semantic_annotations/gtLabels").glob("*_BEV.png"))


def _semantic(root: Path, sample_id: str) -> np.ndarray:
    return np.array(Image.open(root / "semantic_annotations/gtLabels" / f"{sample_id}_BEV.png"))


def _crop_class(semantic: np.ndarray, grid_spec, class_id: int) -> np.ndarray:
    """특정 semantic class만 뽑아 ego 격자로 크롭한다. 좌표 매핑은 `crop_bev_occupancy`와 같다.

    `crop_bev_occupancy`를 재사용하지 않는 이유는 그것이 drivable 여부로 이미 이진화해
    돌려주기 때문이다. 대신 **매핑 상수와 `ego_to_bev_pixel`을 같은 모듈에서 가져와** 두
    경로가 갈라지지 않게 한다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    rows, _ = ego_to_bev_pixel(forward_m, 0.0, BEV_METERS_PER_PIXEL, BEV_ORIGIN_PX,
                               SIGN_FORWARD, SIGN_LATERAL)
    _, cols = ego_to_bev_pixel(0.0, lateral_m, BEV_METERS_PER_PIXEL, BEV_ORIGIN_PX,
                               SIGN_FORWARD, SIGN_LATERAL)
    height, width = semantic.shape
    rows = np.clip(np.round(rows).astype(int), 0, height - 1)
    cols = np.clip(np.round(cols).astype(int), 0, width - 1)
    return semantic[np.ix_(rows, cols)] == class_id


def main(dataset_root=DATASET_ROOT, out_root=DEFAULT_OUT_ROOT, limit=None):
    dataset_root, out_root = Path(dataset_root), Path(out_root)
    grid_spec = SYNWOODSCAPE_NATIVE_GRID_SPEC
    sample_ids = _sample_ids(dataset_root)
    if limit:
        sample_ids = sample_ids[:int(limit)]
    print(f"샘플 {len(sample_ids)}개 · 격자 {grid_spec.n_rows}x{grid_spec.n_cols} "
          f"({grid_spec.cell_m} m/cell, 전방 {grid_spec.front_m} / 후방 {grid_spec.rear_m} / "
          f"좌우 ±{grid_spec.half_width_m} m)")

    cameras = {name: load_camera(dataset_root / "calibration_data" / f"{name}.json")
               for name in CAMERA_NAMES}

    # --- 1단계: 정준 ego 실루엣 (다수결) -------------------------------------------------
    print("\n[1/2] 정준 ego 실루엣을 다수결로 만든다")
    ego_votes = np.zeros((grid_spec.n_rows, grid_spec.n_cols), dtype=np.int32)
    for i, sample_id in enumerate(sample_ids, 1):
        ego_votes += _crop_class(_semantic(dataset_root, sample_id), grid_spec, EGO_CLASS_ID)
        if i % 100 == 0:
            print(f"  {i}/{len(sample_ids)}")
    canonical_ego = ego_votes >= (EGO_VOTE_FRACTION * len(sample_ids))
    print(f"  ego 셀 {int(canonical_ego.sum())}개 ({100 * canonical_ego.mean():.2f} %)")

    # --- 2단계: occupancy + visibility --------------------------------------------------
    print("\n[2/2] occupancy와 gather-column 가시성을 만든다")
    out_root.mkdir(parents=True, exist_ok=True)
    stats = []
    for i, sample_id in enumerate(sample_ids, 1):
        semantic = _semantic(dataset_root, sample_id)
        occupancy = crop_bev_occupancy(semantic, grid_spec)
        depths = {name: np.load(dataset_root / "depth_maps/raw_data" / f"{sample_id}_{name}.npy")
                  for name in CAMERA_NAMES}
        raw_visible = compute_column_visible_mask(grid_spec, cameras, depths)
        visible = raw_visible & ~canonical_ego

        np.save(out_root / f"{sample_id}_occupancy.npy", occupancy.astype(np.uint8))
        np.save(out_root / f"{sample_id}_visible.npy", visible)
        stats.append({
            "sample_id": sample_id,
            "raw_visible_fraction": float(raw_visible.mean()),
            "visible_fraction": float(visible.mean()),
            "free_fraction_of_observed": float(
                (occupancy.astype(bool) & visible).sum() / max(int(visible.sum()), 1)),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(sample_ids)}")

    np.save(out_root / "canonical_ego_mask.npy", canonical_ego)
    mean_vis = float(np.mean([s["visible_fraction"] for s in stats]))
    mean_free = float(np.mean([s["free_fraction_of_observed"] for s in stats]))
    manifest = {
        "name": out_root.name,
        "source": str(dataset_root),
        "grid": {"front_m": grid_spec.front_m, "rear_m": grid_spec.rear_m,
                 "half_width_m": grid_spec.half_width_m, "cell_m": grid_spec.cell_m,
                 "shape": [grid_spec.n_rows, grid_spec.n_cols]},
        "occupancy": {"rule": "semantic drivable classes -> 1, else 0",
                      "manual_refinement": False,
                      "note": "the refined labels exist only for ROI 8/4/+-6; this root is the "
                              "automatic crop of the semantic BEV"},
        "visibility": {"rule": "gather column visibility, z=0..0.8 m, ego excluded",
                       "height_samples": len(COLUMN_HEIGHTS_M),
                       "reproduction_check": "99.0% per-cell agreement with the recorded "
                                             "visibility_h08/raw_visible on ROI 8/4/+-6"},
        "n_samples": len(sample_ids),
        "mean_visible_fraction": mean_vis,
        "mean_free_fraction_of_observed": mean_free,
        "ego_cells": int(canonical_ego.sum()),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\n완료. 관측 {100 * mean_vis:.1f} % · 관측 셀 중 free {100 * mean_free:.1f} %")
    print(f"  -> {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.dataset_root, args.out_root, args.limit)
