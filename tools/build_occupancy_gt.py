"""Phase 2 실행: semantic_annotations의 _BEV.png로부터 BEV occupancy GT를 생성한다.

두 개의 그리드 스펙에 대해 각각 생성한다 (`projects/bev_gt/grid.py` 참고).

- `robot` (`ROBOT_GRID_SPEC`, 전방4m/후방2m/좌우±3m=6m×6m): 자체 로봇 fine-tuning 타깃 스펙.
  SynWoodScape에서는 ego(풀사이즈 승용차) 차체가 그리드 상당 부분을 덮어 GT 변별력이 낮다.
  비교/추적용으로만 함께 뽑는다.
- `synwoodscape_pretrain` (`SYNWOODSCAPE_PRETRAIN_GRID_SPEC`, 전방8m/후방4m/좌우±6m=12m×12m):
  SynWoodScape pretraining에 실제로 쓰는 스펙. 수동 검수를 마친 라벨
  (`dataset/synwoodscape_2head_roi_8_4_6_h08`)의 ROI와 같은 값이다.

출력 배열/이미지는 `crop_bev_occupancy`가 이미 표시용 방향(row 0=최전방, col 0=차량 좌측)으로
돌려주므로 flip 없이 그대로 저장한다.

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
from projects.bev_gt.grid import (  # noqa: E402
    ROBOT_GRID_SPEC,
    SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/occupancy_gt")

GRID_SPECS = {
    "robot": ROBOT_GRID_SPEC,
    "synwoodscape_pretrain": SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
}


def build_sample(sample_idx: str) -> dict:
    semantic_bev = np.array(
        Image.open(DATASET_ROOT / "semantic_annotations/gtLabels" / f"{sample_idx}_BEV.png")
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    occupancies = {}
    for spec_name, grid_spec in GRID_SPECS.items():
        occupancy = crop_bev_occupancy(semantic_bev, grid_spec)
        occupancies[spec_name] = occupancy

        stem = f"{sample_idx}_{spec_name}_occupancy"
        np.save(OUTPUT_DIR / f"{stem}.npy", occupancy)
        # 방향 보정 없이 그대로 저장한다 — 배열이 이미 표시용 방향이다.
        Image.fromarray((occupancy * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{stem}.png")
        print(
            f"[{sample_idx}/{spec_name}] shape={occupancy.shape} "
            f"drivable_fraction={occupancy.mean():.3f} -> {OUTPUT_DIR}/{stem}.npy"
        )
    return occupancies


def report_variability(per_spec_stack: dict) -> None:
    """샘플 간 변동성 요약 — 그리드가 학습 신호를 담고 있는지 확인용."""
    print("\n=== 샘플 간 변동성 ===")
    for spec_name, occupancies in per_spec_stack.items():
        stack = np.stack(occupancies)
        fractions = stack.reshape(len(stack), -1).mean(axis=1)
        varying = (stack.min(axis=0) != stack.max(axis=0)).mean()
        print(
            f"{spec_name:>22}: drivable_fraction mean={fractions.mean():.3f} "
            f"std={fractions.std():.3f} min={fractions.min():.3f} max={fractions.max():.3f} | "
            f"샘플 간 값이 바뀌는 셀 비율={varying:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    per_spec_stack = {name: [] for name in GRID_SPECS}
    for sample_idx in args.samples:
        for spec_name, occupancy in build_sample(sample_idx).items():
            per_spec_stack[spec_name].append(occupancy)

    if len(args.samples) > 1:
        report_variability(per_spec_stack)


if __name__ == "__main__":
    main()
