import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def load_packager():
    script_path = Path("tools/package_synwoodscape_2head_labels.py")
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location("package_synwoodscape_2head_labels", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packages_rgb_occupancy_and_visibility_for_training(tmp_path):
    module = load_packager()
    source_root = tmp_path / "annotated"
    final_rgb = source_root / "final_rgb"
    visible_root = source_root / "visibility_h08" / "visible"
    final_rgb.mkdir(parents=True)
    visible_root.mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 1] = [61, 61, 245]
    rgb[1, 2] = [61, 61, 245]
    Image.fromarray(rgb, mode="RGB").save(final_rgb / "00000.png")

    visible = np.array([[True, False, True], [False, True, True]])
    np.save(visible_root / "00000.npy", visible)

    output_root = tmp_path / "training"
    summary = module.package_labels(source_root, output_root)

    occupancy_out = np.load(output_root / "00000_occupancy.npy")
    visible_out = np.load(output_root / "00000_visible.npy")

    assert summary["processed_samples"] == 1
    assert occupancy_out.dtype == np.uint8
    assert occupancy_out.tolist() == [[1, 0, 1], [1, 1, 0]]
    assert visible_out.dtype == np.bool_
    assert visible_out.tolist() == visible.tolist()
    assert (output_root / "metadata.json").exists()
    assert (output_root / "README.md").exists()
