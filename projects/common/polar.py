"""BEV 격자를 ego 원점 기준 방위각 광선으로 읽는다 -- M3(range error)의 기하 부분.

왜 필요한가: `occupied`는 면적이 아니라 **광선이 멈춘 표면**이라 IoU가 틀린 도구다.
방향별 "첫 장애물까지의 거리"로 재면 통로 폭·정지거리로 직결되고, planner가 실제로
쓰는 양과 같아진다.

격자가 고정이므로 광선별 셀 인덱스는 한 번만 만들어 캐시한다.
"""
from dataclasses import dataclass

import numpy as np

from projects.common.npsafe import bool_not

from projects.bev_gt.grid import OccupancyGridSpec, cell_centers_m

RAY_OK = 0          # 첫 free 이후 첫 non-free를 격자 안에서 만났다
RAY_NO_FREE = 1     # 광선 위에 free 셀이 하나도 없다 (정적 사각·후방 등) -> 지표에서 제외
RAY_CENSORED = 2    # 격자 끝까지 free -> 회귀 통계에 섞지 않고 따로 센다

# 스펙 §12(2026-08-17 재측정): `permanent_blind ∪ invalid`를 오차로 치지 않고 복원
# 충실도만 보면 360→0.9519, 720→0.9920으로 §2.5를 재현하고 그 차이(0.0401)는 360에서
# 오히려 더 크다. 720은 비용이 사실상 0이고 아직 이 기본값으로 채점된 run이 없어
# 지금이 바꾸기 가장 싼 시점이므로 기본값을 720으로 올린다.
DEFAULT_N_THETA = 720


@dataclass(frozen=True)
class RayIndex:
    rows: np.ndarray      # (n_theta, n_steps) int64
    cols: np.ndarray
    radii_m: np.ndarray   # (n_steps,) float64, 원점으로부터의 거리
    inside: np.ndarray    # (n_theta, n_steps) bool, 격자 안인가


def build_ray_index(grid_spec: OccupancyGridSpec, n_theta: int = DEFAULT_N_THETA,
                    step_cells: float = 0.5) -> RayIndex:
    """`n_theta`개 방위각 × 반지름 표본의 (row, col) 인덱스를 미리 계산한다.

    row는 전방이 **감소** 방향이다(`grid.cell_centers_m`과 같은 규약). 원점의 격자 좌표는
    `(front_m/cell - 0.5, half_width_m/cell - 0.5)`이다.
    """
    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    max_radius_cells = float(np.hypot(
        max(grid_spec.front_m, grid_spec.rear_m), grid_spec.half_width_m
    ) / grid_spec.cell_m)

    radii_cells = np.arange(0.0, max_radius_cells, step_cells)
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    # theta=0은 전방(+x). row는 전방이 감소하므로 부호가 뒤집힌다.
    d_row = -np.cos(thetas)[:, None]
    d_col = -np.sin(thetas)[:, None]

    rows = np.round(origin_row + radii_cells[None, :] * d_row).astype(np.int64)
    cols = np.round(origin_col + radii_cells[None, :] * d_col).astype(np.int64)
    inside = (
        (rows >= 0) & (rows < grid_spec.n_rows) & (cols >= 0) & (cols < grid_spec.n_cols)
    )
    # 격자 밖 인덱스는 fancy indexing이 wrap하지 않도록 0으로 눌러 두고 `inside`로만 판단한다.
    return RayIndex(
        rows=np.where(inside, rows, 0),
        cols=np.where(inside, cols, 0),
        radii_m=radii_cells * grid_spec.cell_m,
        inside=inside,
    )


def first_free_range(free: np.ndarray, rays: RayIndex):
    """광선마다 `(첫 free 셀 이후 첫 non-free 셀까지의 거리, 상태)`.

    첫 free부터 재는 이유: ego 원점 주변은 `permanent_blind`(반경 약 0.5 m 원반)라 원점에서
    바로 쏘면 즉시 non-free를 만난다. 그 정적 영역을 건너뛰어야 실제 자유거리가 나온다.

    **격자 안 구간만 본다.** 광선이 격자를 벗어난 뒤의 `False`는 장애물이 아니라 ROI 밖이다.
    그걸 hit으로 세면 모든 광선이 "격자 경계에서 막혔다"가 되어 censored를 구분할 수 없다.
    원점에서 직선으로 나가므로 격자(볼록) 안 구간은 항상 연속이다.
    """
    sampled = free[rays.rows, rays.cols] & rays.inside
    n_theta = sampled.shape[0]
    r_m = np.full(n_theta, np.nan)
    status = np.full(n_theta, RAY_NO_FREE, dtype=np.int8)

    for i in range(n_theta):
        inside_steps = np.nonzero(rays.inside[i])[0]
        if inside_steps.size == 0:
            continue
        lo, hi = int(inside_steps[0]), int(inside_steps[-1]) + 1
        segment = sampled[i, lo:hi]
        if not segment.any():
            continue
        start = int(np.argmax(segment))
        # **`~`를 직접 쓰지 않는다** -- `segment`는 `sampled`의 view라 numpy 임시 소거가
        # 오작동하면 원본 표본 배열이 통째로 뒤집힌다(`projects/common/npsafe.py`).
        blocked = np.nonzero(bool_not(segment[start:]))[0]
        if blocked.size == 0:
            status[i] = RAY_CENSORED     # 격자 끝까지 free -- 회귀 통계에 섞지 않는다
            continue
        status[i] = RAY_OK
        r_m[i] = rays.radii_m[lo + start + int(blocked[0])]
    return r_m, status


def frontier_cells(r_m, status, rays: RayIndex, shape) -> np.ndarray:
    """광선이 멈춘 셀들의 마스크 -- **free 경계에서 유도한 `occupied`**.

    `reconstruct_free`가 "free를 되돌리는" 짝이라면 이쪽은 "표면을 되돌리는" 짝이다.
    라벨 정의(`free_space.decompose`)에서 `occupied = ~occ & vis`이고 `vis`가 ego 원점
    raycast이므로, **보이는 장애물 셀은 정의상 어떤 광선의 첫 hit**이다. 즉 GT `occupied`는
    free 영역의 ego 기준 경계이고 이 함수가 그것을 복원한다. 그래서 free만 예측하는
    정식화에서도 `iou_occupied`/`f1@τ`를 계속 보고할 수 있다.

    `RAY_OK`인 광선만 셀을 낸다. `RAY_CENSORED`(격자 끝까지 free)와 `RAY_NO_FREE`
    (광선 위에 free가 없다)는 멈춘 셀이 없으므로 아무것도 찍지 않는다 -- 격자 경계를
    장애물로 세면 모든 프레임의 ROI 테두리가 occupied가 된다.

    복원 셀이 GT `occupied` 대신 `unknown`이나 `valid=0`으로 떨어질 수 있다(경계가 장애물이
    아니라 정적 사각·수집 아티팩트인 경우). 그 비율이 이 유도의 정확도를 결정하므로
    `tools/measure_derived_occupied.py`가 실측한다.
    """
    frontier = np.zeros(shape, bool)
    for i in np.nonzero(status == RAY_OK)[0]:
        # radii_m은 단조증가이고 r_m은 그중 한 값과 정확히 같다(`first_free_range`가
        # 표본 반지름을 그대로 돌려준다) -- test_polar.py가 쓰는 것과 같은 관용구다.
        step = int(np.searchsorted(rays.radii_m, r_m[i]))
        frontier[rays.rows[i, step], rays.cols[i, step]] = True
    return frontier


def reconstruct_free(r_m, status, rays: RayIndex, shape) -> np.ndarray:
    """`r(θ)`만으로 free 영역을 복원한다.

    **원점(반지름 0)부터 채운다.** 그래서 `first_free_range`가 (건너뛰고 측정하지 않은)
    `permanent_blind`/`invalid` 같은 원점 부근 정적 마스크까지 복원 결과에 다시 채워
    넣는다. `free` 자체는 그 마스크들을 이미 제외하고 있으므로(§7, `load_masked_labels`),
    이 함수가 만드는 영역과 비교하면 그 마스크 넓이만큼 항상 거짓양성이 생긴다. 즉
    **polar 표현 자체의 손실이 아니라 이 복원 방식의 성질이다** -- 표현 충실도를 재려면
    호출자가 `permanent_blind ∪ invalid`를 both side에서 제외하고 비교해야 한다
    (스펙 §12, `tools/measure_label_geometry.py`).
    """
    restored = np.zeros(shape, bool)
    sampled_inside = rays.inside
    for i in range(rays.rows.shape[0]):
        if status[i] == RAY_NO_FREE:
            continue
        limit = rays.radii_m[-1] + 1.0 if status[i] == RAY_CENSORED else r_m[i]
        take = sampled_inside[i] & (rays.radii_m < limit)
        restored[rays.rows[i][take], rays.cols[i][take]] = True
    return restored


@dataclass(frozen=True)
class CellRayMap:
    """셀 -> (방위각 광선 인덱스, ego로부터의 거리). 격자가 고정이므로 한 번만 만든다."""
    theta_index: np.ndarray   # (n_rows, n_cols) int64, `RayIndex`의 광선 번호
    radius_m: np.ndarray      # (n_rows, n_cols) float64


def build_cell_ray_map(grid_spec: OccupancyGridSpec,
                       n_theta: int = DEFAULT_N_THETA) -> CellRayMap:
    """셀마다 자기 방위각에 가장 가까운 광선과 반경을 준다.

    `build_ray_index`와 **같은 각도 규약**이어야 한다: `theta=0`이 전방(+x)이고 격자에서
    forward는 row 감소, lateral(좌측 양수)은 col 감소다(`grid.cell_centers_m`). 그래서
    `theta = atan2(lateral, forward)`다. 두 함수가 갈리면 `d_i`의 부호가 조용히 뒤집힌다.

    **최근접 할당이고 보간하지 않는다.** `n_theta=720`에서 광선 간격은 0.5°이므로 반경
    `r`에서 이웃 광선 사이 거리는 `r * 0.0087` m다 -- 격자 셀(0.05 m)보다 커지는 것은
    `r > 5.7 m`부터이고 ROI 최대 반경이 5.0 m라 격자 안에서는 항상 셀보다 촘촘하다.
    즉 어떤 셀도 광선을 못 받는 일은 없다.
    """
    forward_m, lateral_m = cell_centers_m(grid_spec)
    forward = forward_m[:, None]
    lateral = lateral_m[None, :]
    radius = np.hypot(forward, lateral)
    theta = np.mod(np.arctan2(lateral, forward), 2 * np.pi)
    index = np.mod(np.round(theta / (2 * np.pi / n_theta)).astype(np.int64), n_theta)
    return CellRayMap(theta_index=index, radius_m=np.broadcast_to(radius, index.shape).copy())


def signed_boundary_distance(free: np.ndarray, rays: RayIndex,
                             cell_rays: CellRayMap) -> np.ndarray:
    """`d_i = R_gt(theta_i) - r_i` -- 셀별 부호 있는 GT 경계 거리 [m].

    `d > 0`이면 경계보다 안쪽(drivable 쪽), `d < 0`이면 바깥쪽이다. soft-boundary loss가
    영역을 나누는 양이고 정의는 `docs/soft_boundary_loss_design.md` §2가 정본이다.

    **경계가 없는 광선은 무한으로 보낸다** -- `nan`으로 두면 비교 연산이 조용히 False가 되어
    그 셀들이 세 영역 어디에도 안 들어가고 사라진다.

    | 상태 | 뜻 | `d` |
    |---|---|---|
    | `RAY_OK` | 격자 안에서 경계를 만났다 | `R_gt - r` |
    | `RAY_CENSORED` | 격자 끝까지 free | `+inf` (전부 확실한 drivable) |
    | `RAY_NO_FREE` | 광선 위에 free가 없다 | `-inf` (전부 확실한 non-drivable) |

    `R_gt`는 `first_free_range`가 준다 -- **ego 아래 `permanent_blind` 원반을 건너뛰고 재는
    것이 그 함수의 계약**이므로 여기서 다시 다루지 않는다. 다만 그 원반의 셀들은 `r`이
    작아 `d`가 큰 양수로 나오므로, **호출부가 `permanent_blind ∪ invalid`를 반드시 따로
    제외해야 한다**(같은 문서 §4.2).
    """
    r_m, status = first_free_range(free, rays)
    per_ray = np.where(status == RAY_CENSORED, np.inf,
                       np.where(status == RAY_NO_FREE, -np.inf, r_m))
    return per_ray[cell_rays.theta_index] - cell_rays.radius_m
