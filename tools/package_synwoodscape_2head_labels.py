"""Package finalized SynWoodScape 2-head labels into a training-ready root.

The source label root is the reviewed ROI 8/4/+-6 folder:

    dataset/annotated_roi_8-4-6_semantic_crop/

Expected inputs:

    final_rgb/<sample_id>.png
    visibility_h08/visible/<sample_id>.npy

Outputs match `SynWoodScapeSimpleBEVDataset` naming:

    <sample_id>_occupancy.npy  # uint8, drivable=1, non-drivable=0
    <sample_id>_visible.npy    # bool, training loss mask
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SOURCE_ROOT = Path("dataset/annotated_roi_8-4-6_semantic_crop")
DEFAULT_OUTPUT_ROOT = Path("dataset/synwoodscape_2head_roi_8_4_6_h08")
NON_DRIVABLE_RGB = np.array([61, 61, 245], dtype=np.uint8)


def rgb_to_occupancy(rgb: np.ndarray) -> np.ndarray:
    """Convert reviewed RGB occupancy labels to drivable=1/non-drivable=0."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {rgb.shape}")
    non_drivable = np.all(rgb == NON_DRIVABLE_RGB, axis=2)
    return (~non_drivable).astype(np.uint8)


def _write_metadata(output_root: Path, summary: dict) -> None:
    metadata = {
        "name": "synwoodscape_2head_roi_8_4_6_h08",
        "source_root": str(summary["source_root"]),
        "roi_m": {"front": 8.0, "rear": 4.0, "half_width": 6.0},
        "cell_size_m": 0.05,
        "shape": [240, 240],
        "occupancy": {
            "filename": "<sample_id>_occupancy.npy",
            "dtype": "uint8",
            "drivable": 1,
            "non_drivable": 0,
            "source": "final_rgb/*.png",
            "non_drivable_rgb": NON_DRIVABLE_RGB.tolist(),
        },
        "visibility": {
            "filename": "<sample_id>_visible.npy",
            "dtype": "bool",
            "rule": "gather column visibility, H=0.8m, ego excluded",
            "source": "visibility_h08/visible/*.npy",
        },
        "processed_samples": summary["processed_samples"],
        "mean_drivable_fraction": summary["mean_drivable_fraction"],
        "mean_visible_fraction": summary["mean_visible_fraction"],
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = """# SynWoodScape 2-Head ROI 8/4/+-6 H=0.8 Labels

Training-ready local labels for Simple-BEV pretraining.

- `*_occupancy.npy`: `uint8`, drivable `1`, non-drivable `0`
- `*_visible.npy`: `bool`, visibility/loss mask generated with H=0.8 and ego excluded
- Shape: `240 x 240`
- Grid: front `8m`, rear `4m`, lateral `+-6m`, cell `0.05m`

The reviewed RGB labels remain in `dataset/annotated_roi_8-4-6_semantic_crop/`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def package_labels(source_root: Path, output_root: Path, *, overwrite: bool = False) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    final_rgb_root = source_root / "final_rgb"
    visible_root = source_root / "visibility_h08" / "visible"

    if not final_rgb_root.is_dir():
        raise FileNotFoundError(f"missing final RGB label directory: {final_rgb_root}")
    if not visible_root.is_dir():
        raise FileNotFoundError(f"missing visibility directory: {visible_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    sample_paths = sorted(final_rgb_root.glob("*.png"))
    if not sample_paths:
        raise FileNotFoundError(f"no final RGB labels found under: {final_rgb_root}")

    drivable_fractions = []
    visible_fractions = []
    for rgb_path in sample_paths:
        sample_id = rgb_path.stem
        visible_path = visible_root / f"{sample_id}.npy"
        if not visible_path.exists():
            raise FileNotFoundError(f"missing visibility for sample {sample_id}: {visible_path}")

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        occupancy = rgb_to_occupancy(rgb)
        visible = np.load(visible_path).astype(bool, copy=False)
        if visible.shape != occupancy.shape:
            raise ValueError(
                f"shape mismatch for {sample_id}: occupancy {occupancy.shape}, visible {visible.shape}"
            )

        np.save(output_root / f"{sample_id}_occupancy.npy", occupancy)
        np.save(output_root / f"{sample_id}_visible.npy", visible)
        drivable_fractions.append(float(occupancy.mean()))
        visible_fractions.append(float(visible.mean()))

    summary = {
        "source_root": source_root,
        "output_root": output_root,
        "processed_samples": len(sample_paths),
        "mean_drivable_fraction": float(np.mean(drivable_fractions)),
        "mean_visible_fraction": float(np.mean(visible_fractions)),
    }
    _write_metadata(output_root, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = package_labels(args.source_root, args.output_root, overwrite=args.overwrite)
    print(
        f"wrote {summary['processed_samples']} samples -> {summary['output_root']} | "
        f"mean_drivable={summary['mean_drivable_fraction']:.6f} | "
        f"mean_visible={summary['mean_visible_fraction']:.6f}"
    )


if __name__ == "__main__":
    main()
