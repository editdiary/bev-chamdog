"""하이브리드 occupancy GT: class 값은 top-down BEV label, observed는 raycast.

`tools/build_raycast_occupancy.py`는 occupancy 값도 4-cam raycast(각 어안 카메라 자신의
semantic label)에서 만드는데, 이 방식은 어안 카메라의 semantic segmentation 자체가 가진
결함(예: FV 카메라가 자기 차 본네트를 "road"로 잘못 라벨링)을 occupancy에 그대로 물려받는다.
top-down `_BEV.png` gtLabel은 같은 영역에서 ego-vehicle/road를 깨끗하게 구분하므로(실측
확인, `docs/dataset_analysis/synwoodscape_geometry_findings.md` 참고 예정), occupancy 값은
`crop_bev_occupancy`(top-down)에서 가져오고 observed는 raycast(`compute_observed_mask`,
어느 cell에 광선이 실제로 도달했는지 판정)에서만 가져오는 조합을 시험한다.

Run: python tools/build_hybrid_occupancy.py --num-samples 30 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.bev_crop import crop_bev_occupancy  # noqa: E402
from projects.bev_gt.grid import ROBOT_GRID_SPEC, SYNWOODSCAPE_PRETRAIN_GRID_SPEC  # noqa: E402
from projects.bev_gt.raycast_occupancy import compute_observed_mask  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
# 원본 다운로드 트리(dataset/synwoodscape/)와 분리된, 우리가 생성한 GT 전용 폴더.
# 파일명에서 spec 이름을 생략하므로(아래 build_sample 참고) 한 폴더에는 spec 하나의
# 결과만 담긴다 — 여러 spec을 같이 만들고 싶으면 --output-dir을 spec별로 따로 지정할 것.
DEFAULT_OUTPUT_DIR = Path("dataset/synwoodscape_occupancy_gt")

GRID_SPECS = {
    "robot": ROBOT_GRID_SPEC,
    "synwoodscape_pretrain": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
}
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")

# build_raycast_occupancy.py와 동일한 팔레트 — 두 파이프라인의 combined.png를 나란히 비교하기 쉽게.
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
    semantic_bev = np.array(
        Image.open(DATASET_ROOT / "semantic_annotations/gtLabels" / f"{sample_idx}_BEV.png")
    )
    depth_maps = {
        name: np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{name}.npy")
        for name in CAMERA_NAMES
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for spec_name, grid_spec in grid_specs.items():
        occupancy = crop_bev_occupancy(semantic_bev, grid_spec)
        observed = compute_observed_mask(grid_spec, cameras, depth_maps, stride=stride)
        results[spec_name] = (occupancy, observed)

        # 파일명에 spec을 넣지 않는다(폴더 하나 = spec 하나로 --output-dir에서 이미 구분됨) —
        # main()의 가드가 grid_specs를 정확히 1개로 강제해 이름 충돌을 막는다.
        stem = sample_idx
        Image.fromarray(combined_visualization(occupancy, observed)).save(
            output_dir / f"{stem}_combined.png"
        )
        # occupancy(target)/visible(loss mask)는 combined.png(사람이 보는 시각화)와 역할이
        # 다르므로 별도 배열로 저장한다 — combined.png에서 색을 되돌려 복원하지 않는다.
        if not review_only:
            np.save(output_dir / f"{stem}_occupancy.npy", occupancy)
            np.save(output_dir / f"{stem}_visible.npy", observed)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=None, help="처리할 샘플 id 직접 지정")
    parser.add_argument("--num-samples", type=int, default=30, help="--samples 없을 때 무작위로 뽑을 개수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="픽셀 서브샘플링 간격 (속도용)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--specs", nargs="+", choices=list(GRID_SPECS), default=["synwoodscape_pretrain"],
        help="처리할 grid spec (기본: synwoodscape_pretrain 하나만 — 파일명에 spec을 안 넣으므로 한 번에 하나만 가능)",
    )
    parser.add_argument(
        "--review-only", action="store_true",
        help="combined.png만 생성한다 (npy/occupancy.png/visible.png는 만들지 않음)",
    )
    args = parser.parse_args()

    if args.samples is not None:
        sample_ids = args.samples
    else:
        all_ids = discover_all_sample_ids()
        rng = random.Random(args.seed)
        sample_ids = sorted(rng.sample(all_ids, min(args.num_samples, len(all_ids))))

    grid_specs = {name: GRID_SPECS[name] for name in args.specs}
    if len(grid_specs) != 1:
        raise ValueError(
            "파일명에 spec을 넣지 않으므로 한 번에 spec 하나만 처리할 수 있다 "
            f"(--specs로 {list(grid_specs)}를 줬음) — spec별로 --output-dir을 나눠 따로 실행할 것."
        )
    cameras = load_cameras()

    for i, sample_idx in enumerate(sample_ids):
        build_sample(sample_idx, cameras, args.stride, args.output_dir, grid_specs, args.review_only)
        if (i + 1) % 10 == 0 or (i + 1) == len(sample_ids):
            print(f"[{i + 1}/{len(sample_ids)}] {sample_idx} done")

    print(f"\nwrote {len(sample_ids)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
