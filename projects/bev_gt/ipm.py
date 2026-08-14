"""카메라 이미지를 지면 평면에 역투영한 BEV RGB 캔버스(IPM) -- 자체 리그(Double Sphere)용.

시각화 전용이다. 학습 경로에는 쓰이지 않는다.

왜 필요한가: BEV 예측을 눈으로 판정할 때 GT 라벨 두 장(occupancy/visibility)만 보면
"이 셀이 왜 장애물인지"를 알 수 없다. 같은 좌표계에 실제 장면을 깔아주면 예측이 통로를
제대로 따라가는지 바로 보인다. 어노테이션 프로젝트의 `review_png`가 같은 목적으로 쓴 것과
같은 그림이며, 실제로 이 구현이 그쪽 출력을 재현하는지 확인해 extrinsic 체인을 검증했다.

평평한 지면은 정확히 펴지고 높이가 있는 물체(작물·벽)는 방사상으로 번진다 -- IPM의 본질이라
버그가 아니다.
"""
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, cell_centers_m
from projects.geometry.double_sphere import se3_inverse

GROUND_HEIGHT_M = 0.0  # ego 프레임 원점이 이미 지면 높이다


def render_ipm(images_by_camera, cameras, ego_T_cams, grid_spec: OccupancyGridSpec) -> np.ndarray:
    """{카메라 이름: (H, W, 3) uint8} -> (n_rows, n_cols, 3) uint8 BEV 캔버스. 빈 셀은 0.

    셀마다 그 지면점에 **가장 가까운 카메라 한 대**의 색만 쓴다(`ipm.py`의 nearest 규칙).
    겹치는 카메라 색을 평균하면 off-plane 물체가 유령상으로 겹쳐 텍스처가 뿌옇게 된다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    forward_grid, lateral_grid = np.meshgrid(forward_m, lateral_m, indexing="ij")
    points_ego = np.stack(
        [
            forward_grid.ravel(),
            lateral_grid.ravel(),
            np.full(forward_grid.size, GROUND_HEIGHT_M),
            np.ones(forward_grid.size),
        ],
        axis=1,
    )

    canvas = np.zeros((points_ego.shape[0], 3), dtype=np.uint8)
    nearest = np.full(points_ego.shape[0], np.inf)
    for name, image in images_by_camera.items():
        camera = cameras[name]
        ego_T_cam = ego_T_cams[name]
        points_cam = (se3_inverse(ego_T_cam) @ points_ego.T).T[:, :3]
        u, v, valid = camera.project(points_cam)
        inside = valid & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

        centre = ego_T_cam[:3, 3]
        distance = np.hypot(points_ego[:, 0] - centre[0], points_ego[:, 1] - centre[1])
        take = inside & (distance < nearest)
        if not take.any():
            continue
        rows = v[take].astype(int)
        cols = u[take].astype(int)
        scale_y = image.shape[0] / camera.height
        scale_x = image.shape[1] / camera.width
        canvas[take] = image[(rows * scale_y).astype(int), (cols * scale_x).astype(int)]
        nearest[take] = distance[take]

    return canvas.reshape(grid_spec.n_rows, grid_spec.n_cols, 3)
