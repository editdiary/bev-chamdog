"""depth+semantic map을 직접 언프로젝션해 BEV occupancy를 만든다 (raycasting 방식).

`visibility.py`(grid cell을 z=0으로 가정하고 카메라에 투영 → depth map과 5%/0.20m 오차
이내로 맞는지 가설검정)는 근거리·가장자리에서 잔여 캘리브레이션 오차에 민감해 오탐이 남고,
캘리브레이션 오차와 실제 가려짐을 완벽히 분리할 수 없다는 근본적 한계가 있었다.

이 모듈은 가설검정을 하지 않는다. 각 카메라의 모든 픽셀을 그 픽셀의 depth로 실제
언프로젝션해 ego 프레임 3D 점(= 그 카메라가 실제로 본, 광선이 처음 부딪힌 지점)을 얻고,
그 점이 떨어지는 grid cell에 그 픽셀의 semantic 라벨을 그대로 찍는다. 카메라가 쏜 광선이
한 번도 도달하지 못한 cell은 (가려졌든, 시야 밖이든 이유를 따지지 않고) 그냥 `observed=False`로
남는다 — "보였는가"를 근사가 아니라 정의 그대로 구현한 것이다.
"""
import numpy as np
from scipy import ndimage

from projects.bev_gt.grid import DRIVABLE_CLASS_IDS, OccupancyGridSpec
from projects.geometry.reprojection import MAX_VALID_DEPTH_M, unproject_pixels_to_ego_fast

# 4-neighborhood(십자형) — 대각선으로만 붙은 셀은 다른 영역으로 취급한다.
_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# 다수결을 거친 뒤에도 남는, 진짜 물체 경계/그림자 경계의 미세한 얼룩(1~6 cell)을 지운다.
# 00120 샘플(synwoodscape_pretrain)에서 클래스별 연결요소 크기를 실측하면 노이즈 덩어리는
# 1~6 cell에 뭉쳐 있고 실제 영역(차체, 물체, 도로, 그림자)은 최소 십수 cell 이상이라 — 크기
# 한 자릿수 차이가 나는 뚜렷한 gap이 있다. 6 cell(=0.05m 셀 기준 실제 15cm²)을 문턱으로 쓰면
# 그 gap 한가운데를 잘라 노이즈만 걷어내고 실제 구조는 건드리지 않는다.
DEFAULT_MIN_REGION_CELLS = 6


def clean_small_regions(
    occupancy: np.ndarray, observed: np.ndarray, min_region_cells: int = DEFAULT_MIN_REGION_CELLS
):
    """unknown/drivable/obstacle 3-class 연결요소 중 `min_region_cells`보다 작은 것들을
    가장 가까운 "충분히 큰" 이웃 영역의 라벨로 채워 없앤다.

    왜 필요한가: 물체 경계·그림자 경계에서는 4대 카메라의 개별 시야 경계가 서로 다른
    위치에서 겹치고(각 카메라의 "그림자"가 서로 다르다), 동시에 연속적인 실루엣을 5cm
    격자에 투영하는 과정 자체가 계단형 앨리어싱을 만든다 — 그 결과 진짜 물체/그림자
    경계 바로 옆에 1~수 cell짜리 반대 라벨 얼룩이 낀다(예: 그림자 한가운데 초록 몇 칸).
    다수결(다른 함수)은 "한 cell 안에서 어느 라벨이 우세한가"만 보므로 이 얼룩 자체를
    막지 못한다 — 이건 cell 하나가 아니라 cell *사이*의 패턴이라 별도 후처리가 필요하다.
    """
    label_grid = np.where(~observed, 0, np.where(occupancy == 1, 1, 2)).astype(np.int32)

    keep = np.ones_like(label_grid, dtype=bool)
    for cls in (0, 1, 2):
        mask = label_grid == cls
        components, n_components = ndimage.label(mask, structure=_CONNECTIVITY)
        if n_components == 0:
            continue
        sizes = ndimage.sum(mask, components, index=np.arange(1, n_components + 1))
        small_ids = np.nonzero(sizes < min_region_cells)[0] + 1
        if small_ids.size:
            keep &= ~np.isin(components, small_ids)

    if keep.all():
        cleaned = label_grid
    else:
        _, nearest_index = ndimage.distance_transform_edt(~keep, return_indices=True)
        filled = label_grid[tuple(nearest_index)]
        cleaned = np.where(keep, label_grid, filled)

    return (cleaned != 2).astype(np.uint8), cleaned != 0


def _ego_points_to_cell_indices(points_ego: np.ndarray, grid_spec: OccupancyGridSpec):
    """ego (x=forward, y=lateral) 미터 좌표 -> grid (row, col) 정수 인덱스.

    `cell_centers_m`의 역변환이다 — 같은 grid_spec을 쓰는 한 서로 어긋나지 않는다.
    """
    row = np.floor((grid_spec.front_m - points_ego[:, 0]) / grid_spec.cell_m).astype(np.int64)
    col = np.floor((grid_spec.half_width_m - points_ego[:, 1]) / grid_spec.cell_m).astype(np.int64)
    in_bounds = (row >= 0) & (row < grid_spec.n_rows) & (col >= 0) & (col < grid_spec.n_cols)
    return row, col, in_bounds


def _raycast_camera_cells(
    camera,
    depth_map: np.ndarray,
    grid_spec: OccupancyGridSpec,
    max_depth_m: float = MAX_VALID_DEPTH_M,
    stride: int = 1,
):
    """카메라 이미지의 모든(또는 stride 간격) 픽셀을 ego 프레임 3D 점으로 언프로젝션하고,
    grid 범위 안에 떨어지는 것만 (원본 픽셀 row/col, grid row/col)로 돌려준다.

    `raycast_camera_hits`(semantic label까지 필요)와 `compute_observed_mask`(어느 cell에
    광선이 닿았는지만 필요)가 픽셀 언프로젝션 로직을 공유하기 위한 내부 헬퍼다.
    """
    height, width = depth_map.shape
    rows_px, cols_px = np.mgrid[0:height:stride, 0:width:stride]
    rows_px = rows_px.ravel()
    cols_px = cols_px.ravel()

    depth = depth_map[rows_px, cols_px]
    valid = (depth > 0) & (depth < max_depth_m)
    rows_px, cols_px = rows_px[valid], cols_px[valid]

    points_ego = unproject_pixels_to_ego_fast(camera, rows_px, cols_px, depth_map)
    row, col, in_bounds = _ego_points_to_cell_indices(points_ego, grid_spec)
    return rows_px[in_bounds], cols_px[in_bounds], row[in_bounds], col[in_bounds]


def raycast_camera_hits(
    camera,
    depth_map: np.ndarray,
    semantic_map: np.ndarray,
    grid_spec: OccupancyGridSpec,
    max_depth_m: float = MAX_VALID_DEPTH_M,
    stride: int = 1,
):
    """카메라 이미지의 모든(또는 stride 간격) 픽셀을 ego 프레임 3D 점으로 언프로젝션하고,
    grid 범위 안에 떨어지는 것만 (row, col, semantic_label)로 돌려준다.
    """
    rows_px, cols_px, row, col = _raycast_camera_cells(camera, depth_map, grid_spec, max_depth_m, stride)
    return row, col, semantic_map[rows_px, cols_px]


def compute_observed_mask(
    grid_spec: OccupancyGridSpec,
    cameras: dict,
    depth_maps: dict,
    max_depth_m: float = MAX_VALID_DEPTH_M,
    stride: int = 1,
) -> np.ndarray:
    """(n_rows, n_cols) bool 배열. True = 4대 카메라 중 하나라도 이 cell에 실제로 광선이 도달.

    class label(semantic_map)이 전혀 필요 없다 — "무엇을 봤는가"가 아니라 "봤는가"만
    묻는다. class 값을 top-down BEV label에서 가져오는 하이브리드 파이프라인
    (`tools/build_hybrid_occupancy.py`)이 쓰는 진입점이다.
    """
    n_rows, n_cols = grid_spec.n_rows, grid_spec.n_cols
    n_cells = n_rows * n_cols
    total_hits = np.zeros(n_cells, dtype=np.int64)
    for name, camera in cameras.items():
        _, _, row, col = _raycast_camera_cells(camera, depth_maps[name], grid_spec, max_depth_m, stride)
        flat_index = row * n_cols + col
        total_hits += np.bincount(flat_index, minlength=n_cells)
    return total_hits.reshape(n_rows, n_cols) > 0


def build_raycast_occupancy(
    grid_spec: OccupancyGridSpec,
    cameras: dict,
    depth_maps: dict,
    semantic_maps: dict,
    drivable_class_ids=DRIVABLE_CLASS_IDS,
    stride: int = 1,
    min_region_cells: int = DEFAULT_MIN_REGION_CELLS,
):
    """4-cam raycasting으로 (occupancy, observed) 쌍을 만든다.

    - `observed[r,c]`: 그 cell에 어느 카메라든 광선이 실제로 도달했는가.
    - `occupancy[r,c]`: 1=drivable, 0=obstacle 다수결. 한 cell에는 보통 카메라 여러 대·
      픽셀 여러 개가 겹쳐 떨어지는데(근거리일수록 수십~수백 개), "hit 하나라도 obstacle이면
      obstacle 우선"으로 하면 semantic label의 경계 anti-aliasing(수백 개 중 1~3픽셀만
      다른 class)이 그대로 오탐 cell을 만든다 — 실측(00000 샘플)으로 17개 cell에서 확인했다.
      다수결로 이 노이즈를 걸러낸다.
    """
    n_rows, n_cols = grid_spec.n_rows, grid_spec.n_cols
    n_cells = n_rows * n_cols
    total_hits = np.zeros(n_cells, dtype=np.int64)
    obstacle_hits = np.zeros(n_cells, dtype=np.int64)

    for name, camera in cameras.items():
        row, col, labels = raycast_camera_hits(
            camera, depth_maps[name], semantic_maps[name], grid_spec, stride=stride
        )
        flat_index = row * n_cols + col
        is_obstacle = ~np.isin(labels, list(drivable_class_ids))
        total_hits += np.bincount(flat_index, minlength=n_cells)
        obstacle_hits += np.bincount(flat_index[is_obstacle], minlength=n_cells)

    observed = total_hits.reshape(n_rows, n_cols) > 0
    occupancy = (obstacle_hits * 2 <= total_hits).reshape(n_rows, n_cols).astype(np.uint8)
    if min_region_cells > 0:
        occupancy, observed = clean_small_regions(occupancy, observed, min_region_cells)
    return occupancy, observed
