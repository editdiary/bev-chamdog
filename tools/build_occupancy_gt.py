"""Phase 2 실행: semantic_annotations의 _BEV.png로부터 로봇 기준 occupancy GT를 생성한다.

Run: python tools/build_occupancy_gt.py --samples 00000 00001 00002
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as `python tools/build_occupancy_gt.py` from the repo root
# without needing PYTHONPATH set: when invoked as a script, sys.path[0] is
# this file's directory (tools/), not the repo root, so `projects` is not
# importable unless we add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.bev_gt.bev_crop import crop_bev_occupancy  # noqa: E402
from projects.bev_gt.grid import ROBOT_GRID_SPEC  # noqa: E402

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/occupancy_gt")


def build_sample(sample_idx: str) -> None:
    semantic_bev = np.array(
        Image.open(DATASET_ROOT / "semantic_annotations/gtLabels" / f"{sample_idx}_BEV.png")
    )
    occupancy = crop_bev_occupancy(semantic_bev, ROBOT_GRID_SPEC)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{sample_idx}_occupancy.npy", occupancy)

    visualization = (occupancy * 255).astype(np.uint8)
    Image.fromarray(visualization).save(OUTPUT_DIR / f"{sample_idx}_occupancy.png")
    print(f"[{sample_idx}] drivable_fraction={occupancy.mean():.3f} -> {OUTPUT_DIR}/{sample_idx}_occupancy.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    for sample_idx in args.samples:
        build_sample(sample_idx)


if __name__ == "__main__":
    main()
