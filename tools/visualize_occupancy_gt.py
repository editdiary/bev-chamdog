"""occupancy(binary)와 visibility mask를 색으로 합쳐서 한눈에 확인할 수 있게 만든다.

`build_occupancy_gt.py`/`build_visibility_mask.py`가 저장한 `*_occupancy.npy`,
`*_visible.npy`를 읽어 다음 색으로 합성한다 (grid cell 하나가 원본 이미지에서 너무 작아
`*_occupancy.png`/`*_visible.png`만 보면 착시가 생기기 쉬워서, `scale`배로 확대해 저장한다):

- 흰색: visible & drivable
- 회색: visible & non-drivable
- 빨강: not visible (ignore 대상) — occupancy 값과 무관하게 덮어 그린다

Run: python tools/visualize_occupancy_gt.py --samples 00000 00120 00360 --spec synwoodscape_pretrain
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path("outputs/occupancy_gt")

DRIVABLE_COLOR = (255, 255, 255)
NON_DRIVABLE_COLOR = (60, 60, 60)
IGNORE_COLOR = (220, 30, 30)


def build_composite(sample_idx: str, spec_name: str) -> np.ndarray:
    occupancy = np.load(OUTPUT_DIR / f"{sample_idx}_{spec_name}_occupancy.npy")
    visible = np.load(OUTPUT_DIR / f"{sample_idx}_{spec_name}_visible.npy")

    rgb = np.where(occupancy[..., None].astype(bool), DRIVABLE_COLOR, NON_DRIVABLE_COLOR)
    rgb = np.where(visible[..., None], rgb, IGNORE_COLOR).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00120", "00360"])
    parser.add_argument("--spec", default="synwoodscape_pretrain")
    parser.add_argument("--scale", type=int, default=4, help="nearest-neighbor 확대 배율")
    args = parser.parse_args()

    for sample_idx in args.samples:
        composite = build_composite(sample_idx, args.spec)
        image = Image.fromarray(composite).resize(
            (composite.shape[1] * args.scale, composite.shape[0] * args.scale), Image.NEAREST
        )
        out_path = OUTPUT_DIR / f"{sample_idx}_{args.spec}_composite.png"
        image.save(out_path)
        print(f"[{sample_idx}] visible_fraction={np.load(OUTPUT_DIR / f'{sample_idx}_{args.spec}_visible.npy').mean():.3f} -> {out_path}")


if __name__ == "__main__":
    main()
