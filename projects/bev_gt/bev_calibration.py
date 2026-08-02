"""BEV 이미지 원점·스케일 보정 보조 함수 (1회성 확인용).

⚠️ 여기 있는 **실루엣 기반 스케일 추정은 파이프라인 상수의 출처가 아니다.** 파이프라인이 쓰는
`projects.bev_gt.bev_crop.BEV_METERS_PER_PIXEL`은 기하 도출(15/512)이고, 실루엣 추정은
BEV 핀홀의 높이 확대 때문에 구조적으로 낮게 나온다 (`silhouette_scale_m_per_px` docstring 참고).
이 모듈은 **원점 (511.5, 511.5) 확인**과 **그 편향을 수치로 보여주는 용도**로만 쓴다.
"""
import pickle

import numpy as np
from PIL import Image

from projects.geometry.frames import apply_4x4, carla_frame_to_ego, invert_4x4


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
    """Axis-aligned (X, Y) extent, in meters, of an (8, 4) or (8, 3) 3D box corner array.

    Only meaningful when `corners` are already in the frame whose axes you want to measure
    (i.e. the ego frame for the ego's own box) — an axis-aligned extent of a world-frame box
    is contaminated by the actor's world yaw.
    """
    corners = np.asarray(corners)[:, :3]
    extent = corners.max(axis=0) - corners.min(axis=0)
    return float(extent[0]), float(extent[1])


def box_corners_to_ego(box_3d_by_instance: dict, instance_id: int, world_T_ego: np.ndarray):
    """`_unuserd/box_3d_annotations`의 world(CARLA) 코너 → ego 프레임 코너 (N, 3).

    `world_T_ego`는 `projects.geometry.frames.parse_vehicle_transform`의 반환값이다.
    이 변환 없이 world 코너의 축정렬 extent를 재면 actor의 world yaw가 섞여 들어간다
    (예: ego 박스의 extent가 샘플에 따라 (1.90, 3.76) ↔ (3.72, 1.83)로 뒤바뀐다).
    """
    if instance_id not in box_3d_by_instance:
        return None
    corners_world = np.asarray(box_3d_by_instance[instance_id])[:, :3]
    return carla_frame_to_ego(apply_4x4(invert_4x4(world_T_ego), corners_world))


def silhouette_scale_m_per_px(instance_image: np.ndarray, corners_ego: np.ndarray, instance_id: int):
    """(lateral_m_per_px, forward_m_per_px) — 실루엣 픽셀 bbox ÷ ego 프레임 3D 박스 치수.

    ⚠️ **구조적으로 편향된 추정치다. 파이프라인의 `BEV_METERS_PER_PIXEL`로 쓰면 안 된다.**
    BEV 카메라는 ego (0,0,15)의 핀홀(FOV 90°)이므로 높이 h인 면은 15/(15−h)배로 확대되어
    찍힌다. 차량 실루엣은 지면이 아니라 차체(ego 박스 z∈[0.009, 1.556])의 윤곽이라
    지면 스케일보다 작게 나온다.

    실측(샘플 00000/00010/00100/00200/00300, ego instance 24, 픽셀 bbox 66×132px):
    lateral 0.02710, forward 0.02807 — 5개 샘플 전부 동일. 참값 15/512 = 0.029297을
    ego 박스 z 범위의 확대율 [1.0006, 1.1157]로 나눈 구간 [0.02625, 0.02928] 안에 들어간다.

    축 대응: ego x(전후) ↔ 이미지 row, ego y(좌우) ↔ 이미지 col
    (`bev_crop`의 `SIGN_FORWARD`/`SIGN_LATERAL`가 정의하는 대응과 같다).
    """
    bbox = instance_pixel_bbox(instance_image, instance_id)
    if bbox is None or corners_ego is None:
        return None
    col_min, col_max, row_min, row_max = bbox
    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    forward_m, lateral_m = box_3d_footprint_meters(corners_ego)
    return lateral_m / n_cols, forward_m / n_rows


def ego_vehicle_bbox_center(semantic_bev_image: np.ndarray, ego_vehicle_class_id: int = 24):
    """(cx, cy) pixel center of the ego-vehicle class blob in a semantic BEV image, or None if absent."""
    bbox = instance_pixel_bbox(semantic_bev_image, ego_vehicle_class_id)
    if bbox is None:
        return None
    x_min, x_max, y_min, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2
