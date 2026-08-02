"""ego 프레임 3D 포인트 → fisheye 픽셀 투영과 depth 기반 가시성 판정.

검증 스크립트·테스트·Phase 2 occupancy 파이프라인이 같은 로직을 쓰도록 한 곳에 모았다.
(예전에는 스크립트와 테스트 두 곳에 복사돼 있었고 이미 미묘하게 갈라져 있었다.)
"""
from dataclasses import dataclass

import numpy as np

# depth map에서 하늘은 1000.0으로 채워져 있다. 실제 지형 값도 500~1000 구간에 아주 드물게
# 존재하지만(측정: 전체 픽셀의 0.008% 이하) 그 거리대의 포인트는 어차피 가시성 판정에
# 의미가 없으므로 500 m로 잘라 하늘 채움값을 확실히 배제한다.
MAX_VALID_DEPTH_M = 500.0
VISIBILITY_REL_TOL = 0.05


@dataclass(frozen=True)
class ProjectedPoints:
    """카메라 이미지 안에 떨어진 포인트들의 픽셀 좌표와 거리.

    - `index`: 원본 포인트 배열에서의 인덱스 (라벨 등을 되짚어 오기 위함)
    - `col`, `row`: 정수 픽셀 좌표 (이미지 범위로 clip됨)
    - `range_m`: 카메라 중심으로부터의 거리 (depth map과 비교할 값)
    """

    index: np.ndarray
    col: np.ndarray
    row: np.ndarray
    range_m: np.ndarray

    def __len__(self) -> int:
        return int(self.index.size)


def project_points_to_image(camera, points_ego: np.ndarray) -> ProjectedPoints:
    """ego 프레임 포인트를 fisheye 이미지에 투영하고, 이미지 안에 든 것만 남긴다.

    `radial_poly`는 광축 뒤쪽(θ > 90°) 포인트도 유한한 픽셀로 접어 넣으므로,
    이미지 경계 검사만으로는 뒤쪽 포인트를 걸러낼 수 없다. 반드시 카메라 좌표계에서
    `z > 0`인지 함께 확인해야 한다.
    """
    points_ego = np.asarray(points_ego, dtype=np.float64)
    pixels = camera.project_3d_to_2d(points_ego)
    points_cam = (points_ego - camera.translation) @ camera.rotation

    in_image = (
        ~np.isnan(pixels[:, 0])
        & ~np.isnan(pixels[:, 1])
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
        & (points_cam[:, 2] > 0)
    )
    index = np.nonzero(in_image)[0]

    # 경계 바로 아래의 float 좌표(예: width=1280일 때 1279.6)는 round()에서 1280이 되어
    # 인덱싱 범위를 벗어난다. round 후 clip으로 막는다.
    col = np.clip(pixels[index, 0].round().astype(int), 0, camera.width - 1)
    row = np.clip(pixels[index, 1].round().astype(int), 0, camera.height - 1)
    return ProjectedPoints(
        index=index, col=col, row=row, range_m=np.linalg.norm(points_cam[index], axis=1)
    )


def visibility_mask(
    projected: ProjectedPoints,
    depth_map: np.ndarray,
    rel_tol: float = VISIBILITY_REL_TOL,
    max_depth_m: float = MAX_VALID_DEPTH_M,
) -> np.ndarray:
    """투영된 포인트가 그 카메라에서 **실제로 보이는지** 판정한다.

    LiDAR(ego z=2.0)는 카메라(z=0.9~1.0)보다 높이 있어서 카메라가 못 보는 곳까지 본다.
    포인트까지의 거리가 그 픽셀의 depth와 맞으면 카메라에도 보이는 것이고, 거리가 더 멀면
    앞의 무언가에 가려진 것이다.
    """
    depth_at_pixel = depth_map[projected.row, projected.col]
    finite = (depth_at_pixel > 0) & (depth_at_pixel < max_depth_m)
    relative_error = np.abs(projected.range_m - depth_at_pixel) / np.where(finite, depth_at_pixel, 1.0)
    return finite & (relative_error < rel_tol)


def unproject_depth_to_ego(camera, rows: np.ndarray, cols: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """depth map의 픽셀들을 ego 프레임 3D 포인트로 되돌린다.

    depth 값은 카메라 중심으로부터의 **방사 거리**다 (평면 z-depth가 아니다 — 노면 픽셀로
    평면을 맞춰 확인했다: 방사 거리로 해석하면 법선 오차 0.2° 이내의 평면이 나오고,
    z-depth로 해석하면 발산한다).
    """
    screen_points = np.stack([cols, rows], axis=1).astype(float)
    return camera.project_2d_to_3d(screen_points, depth_map[rows, cols])
