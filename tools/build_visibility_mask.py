"""Phase 2-c: 4-cam 어안 depth map으로 BEV grid cell별 가시성(ignore) mask를 만든다.

`tools/build_occupancy_gt.py`가 만든 binary occupancy(*_occupancy.npy)는 그대로 두고,
같은 shape의 `*_visible.npy` bool mask(True=최소 한 대의 카메라에서 실제로 보임)를
추가로 저장한다. 학습 시 loss는 `visible == True`인 셀만 사용하면 된다
(`projects.bev_gt.visibility` 참고).

binary 버전과 비교해보기 위한 것이므로 occupancy 배열 자체는 건드리지 않는다 — 두 아티팩트를
나란히 두고 어느 쪽으로 학습할지는 이후에 판단한다.

Run: python tools/build_visibility_mask.py --samples 00000 00001 00002
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.grid import ROBOT_GRID_SPEC, SYNWOODSCAPE_PRETRAIN_GRID_SPEC  # noqa: E402
from projects.bev_gt.visibility import compute_visible_mask  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/occupancy_gt")

GRID_SPECS = {
    "robot": ROBOT_GRID_SPEC,
    "synwoodscape_pretrain": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
}
CAMERA_NAMES = ("FV", "MVL", "MVR", "RV")


def load_cameras() -> dict:
    return {
        name: load_camera(DATASET_ROOT / "calibration_data" / f"{name}.json")
        for name in CAMERA_NAMES
    }


def build_sample(sample_idx: str, cameras: dict) -> dict:
    depth_maps = {
        name: np.load(DATASET_ROOT / "depth_maps/raw_data" / f"{sample_idx}_{name}.npy")
        for name in CAMERA_NAMES
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visible_masks = {}
    for spec_name, grid_spec in GRID_SPECS.items():
        visible = compute_visible_mask(grid_spec, cameras, depth_maps)
        visible_masks[spec_name] = visible

        stem = f"{sample_idx}_{spec_name}_visible"
        np.save(OUTPUT_DIR / f"{stem}.npy", visible)
        Image.fromarray((visible * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{stem}.png")
        print(
            f"[{sample_idx}/{spec_name}] shape={visible.shape} "
            f"visible_fraction={visible.mean():.3f} -> {OUTPUT_DIR}/{stem}.npy"
        )
    return visible_masks


def report_visible_fraction(per_spec_stack: dict) -> None:
    """샘플 간 가시 영역 비율 요약 — ignore 처리로 loss 신호가 얼마나 줄어드는지 확인용."""
    print("\n=== 가시 영역(visible) 비율 ===")
    for spec_name, masks in per_spec_stack.items():
        stack = np.stack(masks)
        fractions = stack.reshape(len(stack), -1).mean(axis=1)
        print(
            f"{spec_name:>22}: visible_fraction mean={fractions.mean():.3f} "
            f"std={fractions.std():.3f} min={fractions.min():.3f} max={fractions.max():.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    cameras = load_cameras()
    per_spec_stack = {name: [] for name in GRID_SPECS}
    for sample_idx in args.samples:
        for spec_name, visible in build_sample(sample_idx, cameras).items():
            per_spec_stack[spec_name].append(visible)

    if len(args.samples) > 1:
        report_visible_fraction(per_spec_stack)


if __name__ == "__main__":
    main()
