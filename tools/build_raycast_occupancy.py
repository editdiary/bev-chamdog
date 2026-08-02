"""Phase 2 재작성: grid cell 가설검정(`visibility.py`) 대신 depth+semantic map을 직접
언프로젝션하는 raycasting으로 occupancy와 observed(=visible) mask를 한 번에 만든다.

`build_occupancy_gt.py`(BEV god's-eye 라벨) + `build_visibility_mask.py`(grid cell을
z=0으로 가정하고 5%/0.20m 오차 이내인지 가설검정) 조합은 두 값의 출처가 달라
(하나는 top-down 라벨, 하나는 근거리 시야각에서 노이즈에 민감한 가설검정) 서로 안 맞는
경우가 있었다. 이 스크립트는 `projects.bev_gt.raycast_occupancy`로 두 값을 **같은 소스**
(카메라 광선이 실제로 부딪힌 지점)에서 함께 만든다 — occupancy는 광선이 맞은 지점의 실제
semantic 라벨, observed(=visible)는 광선이 그 cell에 도달했는가 그 자체다.

`occupancy.npy`/`visible.npy`(학습용) 외에 `combined.png`(drivable=초록/obstacle=빨강/
unknown=회색 3색 한 장, 사람이 눈으로 검증할 때 보는 파일)와 `summary.csv`(샘플별 지표)를
같이 만든다.

`--review-only`를 주면 `combined.png`만 만든다 — npy/occupancy.png/visible.png는
combined.png와 정보가 겹치거나(occupancy.png, visible.png는 combined.png를 만드는 두 배열을
그대로 흑백으로 덤프한 것) 검증 전에는 아직 필요 없는(npy, 실제 학습 입력) 파일이라 사람이
눈으로 라벨을 검토하는 단계에서는 굳이 같이 만들 이유가 없다.

Run: python tools/build_raycast_occupancy.py --samples 00000 00001 00002
Run: python tools/build_raycast_occupancy.py --all --output-dir outputs/raycast_occupancy_gt --review-only --specs robot
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.grid import ROBOT_GRID_SPEC, SYNWOODSCAPE_PRETRAIN_GRID_SPEC  # noqa: E402
from projects.bev_gt.raycast_occupancy import build_raycast_occupancy  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
DEFAULT_OUTPUT_DIR = Path("outputs/occupancy_gt")

GRID_SPECS = {
    "robot": ROBOT_GRID_SPEC,
    "synwoodscape_pretrain": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
}
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")

# combined.png 3색 팔레트 (RGB). 사람이 500장을 빠르게 훑어볼 때 쓰는 시각화 전용이고,
# 학습에는 occupancy.npy(0/1) + visible.npy(bool)를 그대로 쓴다.
COLOR_UNKNOWN = (128, 128, 128)
COLOR_DRIVABLE = (76, 175, 80)
COLOR_OBSTACLE = (192, 57, 43)


def load_cameras() -> dict:
    return {
        name: load_camera(DATASET_ROOT / "calibration_data" / f"{name}.json")
        for name in CAMERA_NAMES
    }


def discover_all_sample_ids() -> list:
    return sorted(p.stem[: -len("_BEV")] for p in DATASET_ROOT.glob("rgb_images/*_BEV.png"))


def combined_visualization(occupancy: np.ndarray, observed: np.ndarray) -> np.ndarray:
    image = np.full(occupancy.shape + (3,), COLOR_UNKNOWN, dtype=np.uint8)
    image[observed & (occupancy == 1)] = COLOR_DRIVABLE
    image[observed & (occupancy == 0)] = COLOR_OBSTACLE
    return image


def build_sample(
    sample_idx: str,
    cameras: dict,
    stride: int,
    output_dir: Path,
    grid_specs: dict,
    review_only: bool,
) -> dict:
    depth_maps = {
        name: np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{name}.npy")
        for name in CAMERA_NAMES
    }
    semantic_maps = {
        name: cv2.imread(
            str(DATASET_ROOT / "semantic_annotations/gtLabels" / f"{sample_idx}_{name}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for name in CAMERA_NAMES
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for spec_name, grid_spec in grid_specs.items():
        occupancy, observed = build_raycast_occupancy(
            grid_spec, cameras, depth_maps, semantic_maps, stride=stride
        )
        results[spec_name] = (occupancy, observed)

        stem = f"{sample_idx}_{spec_name}"
        Image.fromarray(combined_visualization(occupancy, observed)).save(
            output_dir / f"{stem}_combined.png"
        )
        if not review_only:
            np.save(output_dir / f"{stem}_occupancy.npy", occupancy)
            np.save(output_dir / f"{stem}_visible.npy", observed)
            Image.fromarray((occupancy * 255).astype(np.uint8)).save(output_dir / f"{stem}_occupancy.png")
            Image.fromarray((observed * 255).astype(np.uint8)).save(output_dir / f"{stem}_visible.png")
    return results


def write_summary_csv(output_dir: Path, rows: list) -> Path:
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample", "spec", "drivable_fraction_all", "drivable_fraction_observed", "observed_fraction"]
        )
        writer.writerows(rows)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    parser.add_argument("--all", action="store_true", help="dataset에 있는 모든 샘플을 처리한다")
    parser.add_argument("--stride", type=int, default=1, help="픽셀 서브샘플링 간격 (속도용)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--specs", nargs="+", choices=list(GRID_SPECS), default=list(GRID_SPECS),
        help="처리할 grid spec (기본: 전부)",
    )
    parser.add_argument(
        "--review-only", action="store_true",
        help="combined.png만 생성한다 (npy/occupancy.png/visible.png는 만들지 않음)",
    )
    args = parser.parse_args()

    sample_ids = discover_all_sample_ids() if args.all else args.samples
    grid_specs = {name: GRID_SPECS[name] for name in args.specs}
    cameras = load_cameras()

    csv_rows = []
    for i, sample_idx in enumerate(sample_ids):
        for spec_name, (occupancy, observed) in build_sample(
            sample_idx, cameras, args.stride, args.output_dir, grid_specs, args.review_only
        ).items():
            observed_fraction = float(observed.mean())
            drivable_all = float(occupancy.mean())
            drivable_observed = float(occupancy[observed].mean()) if observed.any() else float("nan")
            csv_rows.append([sample_idx, spec_name, drivable_all, drivable_observed, observed_fraction])
        if (i + 1) % 25 == 0 or (i + 1) == len(sample_ids):
            print(f"[{i + 1}/{len(sample_ids)}] {sample_idx} done")

    csv_path = write_summary_csv(args.output_dir, csv_rows)
    print(f"\nwrote {len(sample_ids)} samples -> {args.output_dir}")
    print(f"summary csv -> {csv_path}")


if __name__ == "__main__":
    main()
