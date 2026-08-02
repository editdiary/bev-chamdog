import pickle

import numpy as np
from PIL import Image


def load_instance_bev_image(path) -> np.ndarray:
    return np.array(Image.open(path))


def load_box_3d_annotations(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def instance_pixel_bbox(instance_image: np.ndarray, instance_id: int):
    """Return (x_min, x_max, y_min, y_max) pixel bbox for instance_id, or None if absent."""
    mask = instance_image == instance_id
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def box_3d_footprint_meters(corners: np.ndarray):
    """Axis-aligned (X, Y) extent, in meters, of an (8, 4) or (8, 3) 3D box corner array."""
    corners = np.asarray(corners)[:, :3]
    extent = corners.max(axis=0) - corners.min(axis=0)
    return float(extent[0]), float(extent[1])


def estimate_scale_m_per_px(instance_image: np.ndarray, box_3d_by_instance: dict, instance_id: int):
    """(scale_x, scale_y) in meters/pixel, derived from one instance's pixel bbox vs. its 3D box footprint."""
    bbox = instance_pixel_bbox(instance_image, instance_id)
    if bbox is None or instance_id not in box_3d_by_instance:
        return None
    x_min, x_max, y_min, y_max = bbox
    px_width = x_max - x_min + 1
    px_height = y_max - y_min + 1
    width_m, length_m = box_3d_footprint_meters(box_3d_by_instance[instance_id])
    return width_m / px_width, length_m / px_height


def ego_vehicle_bbox_center(semantic_bev_image: np.ndarray, ego_vehicle_class_id: int = 24):
    """(cx, cy) pixel center of the ego-vehicle class blob in a semantic BEV image, or None if absent."""
    bbox = instance_pixel_bbox(semantic_bev_image, ego_vehicle_class_id)
    if bbox is None:
        return None
    x_min, x_max, y_min, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2
