import numpy as np
import pytest

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.polar import (
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_ray_index,
    first_free_range,
    reconstruct_free,
)

# 원점이 정확히 격자 중앙에 오는 대칭 스펙 -- 손계산이 가능하다.
SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def test_ray_index_radii_are_monotone_and_start_at_zero():
    rays = build_ray_index(SPEC, n_theta=8)

    assert rays.rows.shape == rays.cols.shape == rays.inside.shape
    assert rays.rows.shape[0] == 8
    assert rays.radii_m[0] == pytest.approx(0.0)
    assert np.all(np.diff(rays.radii_m) > 0)


def test_first_free_range_measures_to_the_first_non_free_cell():
    """원점 주변 정적 사각(0.5 m 원반)을 건너뛴 뒤 첫 free부터 재야 한다.

    허용오차는 딱 붙여야 한다: `step_cells=0.5`라 표본 간격이 0.025 m인데,
    `abs=0.05`(두 스텝)를 쓰면 "첫 non-free 셀"이 아니라 "마지막 free 셀"까지
    재는 off-by-one 버그(0.600)가 정답(0.625)보다 오히려 더 가깝게 통과해
    버려서 두 구현을 구분하지 못한다.
    """
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    # 전방(row 감소 방향) 한 줄만 free로 만든다: ego z=0 행은 row 19/20 근처.
    free[8:19, 20] = True

    rays = build_ray_index(SPEC, n_theta=4)     # theta=0 이 정확히 전방
    r_m, status = first_free_range(free, rays)

    assert status[0] == RAY_OK
    # 손계산: origin_row = front_m/cell_m - 0.5 = 19.5. free는 row 8까지이므로
    # 첫 non-free는 row 7. forward_m(row=7) = front_m - (7+0.5)*cell_m
    #                                       = 1.0 - 0.375 = 0.625.
    assert r_m[0] == pytest.approx(0.625, abs=1e-9)

    # 숫자만이 아니라 의미론 자체를 못박는다: r_m이 가리키는 셀은 non-free여야
    # 하고, 원점 쪽으로 그 바로 앞 표본은 free여야 한다 -- "마지막 free 셀까지"가
    # 아니라 "첫 non-free 셀까지"의 거리라는 계약.
    step = int(np.searchsorted(rays.radii_m, r_m[0]))
    assert not free[rays.rows[0, step], rays.cols[0, step]]
    assert free[rays.rows[0, step - 1], rays.cols[0, step - 1]]


def test_ray_with_no_free_cell_is_reported_as_undefined():
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    rays = build_ray_index(SPEC, n_theta=4)

    _, status = first_free_range(free, rays)

    assert np.all(status == RAY_NO_FREE)


def test_ray_free_all_the_way_out_is_censored_not_a_range():
    """격자 끝까지 free인 광선을 r_max로 회귀에 섞으면 통계가 그 값에 눌린다."""
    free = np.ones((SPEC.n_rows, SPEC.n_cols), bool)
    rays = build_ray_index(SPEC, n_theta=4)

    _, status = first_free_range(free, rays)

    assert np.all(status == RAY_CENSORED)


def test_star_convex_region_survives_the_polar_roundtrip():
    """free가 단일 원점 raycast 결과라 star-convex라는 성질의 회귀 테스트 (스펙 §2.5)."""
    rows, cols = np.mgrid[0:SPEC.n_rows, 0:SPEC.n_cols]
    origin_r, origin_c = SPEC.front_m / SPEC.cell_m - 0.5, SPEC.half_width_m / SPEC.cell_m - 0.5
    radius = np.hypot(rows - origin_r, cols - origin_c)
    free = radius < 12                                    # 원점 중심 원반 = star-convex

    rays = build_ray_index(SPEC, n_theta=720)
    r_m, status = first_free_range(free, rays)
    restored = reconstruct_free(r_m, status, rays, free.shape)

    intersection = (restored & free).sum()
    union = (restored | free).sum()
    assert intersection / union >= 0.99


def test_first_free_range_is_directionally_correct_left_vs_right():
    """좌우가 뒤집힌 광선 부채꼴(예: `d_col` 부호 오류)을 잡아낸다.

    원점 중심 원반(star-convex 회귀 테스트의 fixture)은 좌우 대칭이라 뒤집혀도
    IoU가 그대로 1.0이 나온다 -- 좌우 비대칭 fixture가 아니면 이 종류의 버그는
    원리적으로 검출 불가능하다.

    방향 도출: `build_ray_index`는 `d_col = -sin(theta)`를 쓴다. theta=+90°(=π/2)
    에서 `d_col = -1`이므로 반지름이 커질수록 col이 **감소**한다. 한편
    `grid.cell_centers_m`은 `lateral_m = half_width_m - (col+0.5)*cell_m`로
    정의하므로 col이 감소하면 lateral_m은 **증가**한다(더 양의 값). 그 문서의
    "col 0 = vehicle's left" 규약과 합치면, col이 0 쪽으로 줄어드는 방향 =
    lateral_m이 커지는 방향 = 차량의 왼쪽이다. 즉 theta=+90°(n_theta=4일 때
    ray index 1)는 왼쪽을, theta=+270°(ray index 3)는 오른쪽을 가리켜야 한다.
    """
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    # row 20(원점 행) 중 왼쪽 절반(col 10..19, col 0에 더 가까운 쪽)만 free로 둔다.
    free[20, 10:20] = True

    rays = build_ray_index(SPEC, n_theta=4)
    _, status = first_free_range(free, rays)

    assert status[1] == RAY_OK        # 왼쪽(θ=+90°) -- free 복도를 만난다
    assert status[3] == RAY_NO_FREE   # 오른쪽(θ=+270°) -- 이쪽엔 free가 전혀 없다


def test_first_free_range_handles_a_diagonal_ray():
    """row/col이 동시에 변하는 45도 광선 -- 지금까지 유일한 커버리지 공백.

    n_theta=4로 도는 다른 모든 테스트는 θ∈{0,90,180,270}만 쓰므로 매 스텝 row/col
    중 하나만 바뀐다. 둘이 함께 바뀌면서 반올림이 상호작용하는 경로는 검증된 적이
    없다. n_theta=8, ray index 1(θ=45°)을 쓴다: `d_row=-cos45°=d_col=-sin45°`로
    두 계수가 같으므로, 이 광선은 대칭 격자(origin_row=origin_col=19.5)에서
    row==col인 순수 대각선을 그린다 -- 아래 값은 그 공식으로 손계산한 것이다.

        step k: radius_cells = 0.5k, row(k) = col(k) = round(19.5 - 0.70711*0.5*k)
        k=12,13,14 -> row=col=15   (radius_m = 0.30, 0.325, 0.35)
        k=15       -> row=col=14   (radius_m = 0.375)  <- 첫 non-free

    따라서 free 셀을 (15,15) 단 하나만 켜 두면, 광선은 k=12에서 처음 free를
    만나고 k=15에서 (14,14)로 비free가 되어 radius_m=0.375에서 멈춰야 한다.
    """
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    free[15, 15] = True

    rays = build_ray_index(SPEC, n_theta=8)
    r_m, status = first_free_range(free, rays)

    assert status[1] == RAY_OK
    assert r_m[1] == pytest.approx(0.375, abs=1e-9)
