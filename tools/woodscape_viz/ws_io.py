"""I/O helpers for the WoodScape ICCV19 dataset.

Reads annotations from the extracted dataset tree (all zips have been
unpacked to disk) and returns decoded numpy arrays / python objects. Every
image is returned in OpenCV BGR channel order.

The dataset layout this expects (train set, 8,234 samples). Extraction left
each annotation type inside a doubled folder (`<type>/<type>/...`):
    <root>/rgb_images/rgb_images/<sample>.png
    <root>/box_2d_annotations/box_2d_annotations/<sample>.txt
    <root>/instance_annotations/instance_annotations/<sample>.json
    <root>/semantic_annotations/semantic_annotations/{rgbLabels,gtLabels}/<sample>.png
    <root>/motion_annotations/motion_annotations/{rgbLabels,gtLabels}/<sample>.png
    <root>/calibration_data/calibration/<sample>.json
A <sample> is "<NNNNN>_<CAM>", e.g. "00000_FV" (CAM in FV/RV/MVL/MVR).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ["FV", "RV", "MVL", "MVR"]


def rgb_to_bgr(color):
    """Convert an [R, G, B] list (as stored in the *_info.json files) to a
    cv2 BGR tuple."""
    r, g, b = color
    return int(b), int(g), int(r)


def tag_color(tag: str):
    """Deterministic, readable BGR color for an instance class tag."""
    h = hashlib.md5(tag.encode()).digest()
    # keep values reasonably bright so overlays stay visible
    return int(80 + h[0] % 176), int(80 + h[1] % 176), int(80 + h[2] % 176)


def blend_nonzero(base_bgr, color_mask_bgr, alpha=0.5):
    """Alpha-blend a color mask onto a base image only where the mask is
    non-zero (i.e. leave 'void'/background pixels untouched)."""
    mask = np.any(color_mask_bgr > 0, axis=2)
    out = base_bgr.copy()
    out[mask] = (alpha * color_mask_bgr[mask] + (1 - alpha) * base_bgr[mask]).astype(np.uint8)
    return out


def legend_from_color_mask(mask_bgr, names, colors_rgb):
    """Build legend items by matching the exact colors present in a color-coded
    label mask against the class palette from a *_info.json. This is robust to
    class-index convention mismatches — the rgbLabels colors are authoritative.

    Returns a list of (name, bgr) for every non-black palette color present.
    """
    uniq = {tuple(c) for c in np.unique(mask_bgr.reshape(-1, 3), axis=0)}
    items = []
    for name, c in zip(names, colors_rgb):
        bgr = rgb_to_bgr(c)
        if any(bgr) and tuple(bgr) in uniq:
            items.append((name, bgr))
    return items


def draw_legend(img, items, origin=(10, 10)):
    """Draw a small color-swatch legend. `items` is a list of (name, bgr)."""
    x0, y0 = origin
    sw = 16
    pad = 4
    for i, (name, bgr) in enumerate(items):
        y = y0 + i * (sw + pad)
        cv2.rectangle(img, (x0, y), (x0 + sw, y + sw), bgr, -1)
        cv2.rectangle(img, (x0, y), (x0 + sw, y + sw), (255, 255, 255), 1)
        cv2.putText(img, name, (x0 + sw + 6, y + sw - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class WoodScape:
    def __init__(self, root):
        self.root = Path(root)
        self.rgb_dir = self.root / "rgb_images" / "rgb_images"
        self.box_dir = self.root / "box_2d_annotations" / "box_2d_annotations"
        self.instance_dir = self.root / "instance_annotations" / "instance_annotations"
        self.semantic_dir = self.root / "semantic_annotations" / "semantic_annotations"
        self.motion_dir = self.root / "motion_annotations" / "motion_annotations"
        self.calib_dir = self.root / "calibration_data" / "calibration"

    # ---- generic helpers ----
    @staticmethod
    def _read_bytes(path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"annotation not found: {p}")
        return p.read_bytes()

    @staticmethod
    def _decode_img(data):
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    def list_samples(self):
        return sorted(p.stem for p in self.rgb_dir.glob("*.png"))

    # ---- images / annotations ----
    def rgb(self, sample):
        p = self.rgb_dir / f"{sample}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {p}")
        return img

    def box2d(self, sample):
        """Return list of (class_name, class_index, x_min, y_min, x_max, y_max)."""
        data = self._read_bytes(self.box_dir / f"{sample}.txt")
        boxes = []
        for line in data.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            boxes.append((parts[0], int(parts[1]),
                          int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])))
        return boxes

    def instance(self, sample):
        """Return the inner annotation dict (unwraps the filename-keyed root)."""
        data = self._read_bytes(self.instance_dir / f"{sample}.json")
        obj = json.loads(data.decode())
        return obj[next(iter(obj))]

    def semantic_rgb(self, sample):
        return self._decode_img(self._read_bytes(
            self.semantic_dir / "rgbLabels" / f"{sample}.png"))

    def semantic_gt(self, sample):
        return self._decode_img(self._read_bytes(
            self.semantic_dir / "gtLabels" / f"{sample}.png"))

    def motion_rgb(self, sample):
        return self._decode_img(self._read_bytes(
            self.motion_dir / "rgbLabels" / f"{sample}.png"))

    def motion_gt(self, sample):
        return self._decode_img(self._read_bytes(
            self.motion_dir / "gtLabels" / f"{sample}.png"))

    def calib_dict(self, sample):
        return json.loads(self._read_bytes(
            self.calib_dir / f"{sample}.json").decode())

    def calib_tempfile(self, sample):
        """Copy the calibration json to a temp file and return its path, so
        the official path-based read_cam_from_json() can be reused as-is. The
        caller deletes the temp file, so we must not hand back the real path."""
        data = self._read_bytes(self.calib_dir / f"{sample}.json")
        f = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
        f.write(data)
        f.close()
        return f.name

    # ---- class metadata (from *_info.json) ----
    def _info(self, rel):
        return json.loads((self.root / rel).read_text())

    def box_classes(self):
        info = self._info("box_2d_annotations/box_2d_annotation_info.json")
        return info["classes"], info["class_colors"]

    def semantic_classes(self):
        info = self._info("semantic_annotations/seg_annotation_info.json")
        return info["class_names"], info["class_colors"]

    def motion_classes(self):
        info = self._info("motion_annotations/motion_annotation_info.json")
        return info["class_names"], info["class_colors"]
