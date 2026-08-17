"""BEV 격자를 ego 원점 기준 방위각 광선으로 읽는다 -- M3(range error)의 기하 부분.

왜 필요한가: `occupied`는 면적이 아니라 **광선이 멈춘 표면**이라 IoU가 틀린 도구다.
방향별 "첫 장애물까지의 거리"로 재면 통로 폭·정지거리로 직결되고, planner가 실제로
쓰는 양과 같아진다.

격자가 고정이므로 광선별 셀 인덱스는 한 번만 만들어 캐시한다.
"""
from dataclasses import dataclass

import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec

RAY_OK = 0          # 첫 free 이후 첫 non-free를 격자 안에서 만났다
RAY_NO_FREE = 1     # 광선 위에 free 셀이 하나도 없다 (정적 사각·후방 등) -> 지표에서 제외
RAY_CENSORED = 2    # 격자 끝까지 free -> 회귀 통계에 섞지 않고 따로 센다


@dataclass(frozen=True)
class RayIndex:
    rows: np.ndarray      # (n_theta, n_steps) int64
    cols: np.ndarray
    radii_m: np.ndarray   # (n_steps,) float64, 원점으로부터의 거리
    inside: np.ndarray    # (n_theta, n_steps) bool, 격자 안인가


def build_ray_index(grid_spec: OccupancyGridSpec, n_theta: int = 360,
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
        blocked = np.nonzero(~segment[start:])[0]
        if blocked.size == 0:
            status[i] = RAY_CENSORED     # 격자 끝까지 free -- 회귀 통계에 섞지 않는다
            continue
        status[i] = RAY_OK
        r_m[i] = rays.radii_m[lo + start + int(blocked[0])]
    return r_m, status


def reconstruct_free(r_m, status, rays: RayIndex, shape) -> np.ndarray:
    """`r(θ)`만으로 free 영역을 복원한다 -- polar 표현의 정보 손실을 재는 회귀 테스트용."""
    restored = np.zeros(shape, bool)
    sampled_inside = rays.inside
    for i in range(rays.rows.shape[0]):
        if status[i] == RAY_NO_FREE:
            continue
        limit = rays.radii_m[-1] + 1.0 if status[i] == RAY_CENSORED else r_m[i]
        take = sampled_inside[i] & (rays.radii_m < limit)
        restored[rays.rows[i][take], rays.cols[i][take]] = True
    return restored
