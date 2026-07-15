"""Visualize the fisheye calibration by reusing the official projection code.

Two panels are produced and stacked side by side:
  (left)  the fisheye image with a ground-plane grid (vehicle coordinates,
          ISO 8855, z=0) projected onto it via the 4th-order radial-poly model
          -- this shows how BEV/ground coordinates map into the fisheye image.
  (right) the same image undistorted to a centered cylindrical projection
          (fisheye -> cylindrical remap), which straightens the horizon.

Reuses third_party/datasets/WoodScape/scripts/calibration/projection.py:
  read_cam_from_json, Camera, RadialPolyCamProjection, CylindricalProjection,
  create_img_projection_maps.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import ws_io

# Make the official calibration module importable.
_CALIB_DIR = Path(__file__).resolve().parents[2] / "third_party" / "datasets" / "WoodScape" / "scripts" / "calibration"
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

from projection import (  # noqa: E402  (import after sys.path tweak)
    Camera, RadialPolyCamProjection, CylindricalProjection,
    read_cam_from_json, create_img_projection_maps,
)
from scipy.spatial.transform import Rotation as SciRot  # noqa: E402


def make_cylindrical_cam(cam):
    """Generate a cylindrical camera with a centered horizon.

    Adapted from third_party/datasets/WoodScape/scripts/calibration/example.py (Valeo, MIT).
    """
    assert isinstance(cam.lens, RadialPolyCamProjection)
    lens = CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)  # snap to 90 deg
    rot_zxz[1] = np.pi / 2  # center horizon
    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio,
    )


def _draw_polyline(img, uv, color, thickness=1):
    """Draw a polyline, breaking wherever a point is invalid (NaN)."""
    prev = None
    for p in uv:
        if not np.all(np.isfinite(p)):
            prev = None
            continue
        cur = (int(round(p[0])), int(round(p[1])))
        if prev is not None:
            cv2.line(img, prev, cur, color, thickness, cv2.LINE_AA)
        prev = cur


def _draw_ground_grid(img, cam, x_range=(-6, 15), y_range=(-8, 8), step=1.0, fine=0.2):
    """Project a z=0 ground grid (vehicle coords) into the fisheye image."""
    xs = np.arange(x_range[0], x_range[1] + 1e-6, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, step)

    # constant-y lines (run along x)
    x_fine = np.arange(x_range[0], x_range[1] + 1e-6, fine)
    for y in ys:
        pts = np.stack([x_fine, np.full_like(x_fine, y), np.zeros_like(x_fine)], axis=1)
        color = (0, 200, 0) if abs(y) > 1e-6 else (0, 255, 255)  # y=0 axis in yellow
        _draw_polyline(img, cam.project_3d_to_2d(pts, do_clip=True), color)

    # constant-x lines (run along y)
    y_fine = np.arange(y_range[0], y_range[1] + 1e-6, fine)
    for x in xs:
        pts = np.stack([np.full_like(y_fine, x), y_fine, np.zeros_like(y_fine)], axis=1)
        color = (0, 200, 0) if abs(x) > 1e-6 else (0, 255, 255)  # x=0 axis in yellow
        _draw_polyline(img, cam.project_3d_to_2d(pts, do_clip=True), color)


def _title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def render(ws: "ws_io.WoodScape", sample: str):
    calib_path = ws.calib_tempfile(sample)
    try:
        cam = read_cam_from_json(calib_path)
    finally:
        os.remove(calib_path)

    fisheye = ws.rgb(sample)
    grid_img = fisheye.copy()
    _draw_ground_grid(grid_img, cam)
    _title(grid_img, f"{sample}: fisheye + ground grid (1m, vehicle z=0)")

    cyl_cam = make_cylindrical_cam(cam)
    map1, map2 = create_img_projection_maps(cam, cyl_cam)
    cyl_img = cv2.remap(fisheye, map1, map2, cv2.INTER_CUBIC)
    _title(cyl_img, f"{sample}: cylindrical (undistorted)")

    return np.hstack([grid_img, cyl_img])
