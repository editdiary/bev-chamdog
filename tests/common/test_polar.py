import numpy as np
import pytest

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.polar import (
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_ray_index,
    first_free_range,
    frontier_cells,
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
    """free가 단일 원점 raycast 결과라 star-convex라는 성질의 회귀 테스트 (스펙 §2.5, §12).

    이전 버전은 원점 중심 **꽉 찬** 원반(`radius < 12`)을 썼다. 그 fixture는 안쪽에
    blind hole이 없어서 원점(반지름 0)부터 채우는 `reconstruct_free`가 손해 볼 것이
    없고, 그래서 실제 라벨에서 raw roundtrip IoU가 0.99는커녕 ~0.80까지 떨어지는
    원인(§12: `permanent_blind`/`invalid`처럼 원점 부근 정적 마스크를 원점-fill이
    거짓양성으로 되살리는 것)을 원리적으로 재현할 수 없었다 -- 이 테스트는 그 결함이
    있는 채로도 항상 통과했다.

    안쪽에 구멍이 뚫린 **고리(annulus)**로 바꿔 그 효과를 직접 pin한다: `permanent_blind`
    역할을 하는 원점 주변 hole을 만들고, (a) hole을 제외한 -- 즉 §12가 "의미 있는 기준"
    이라고 정한 -- roundtrip IoU는 여전히 ≥0.99임을, (b) hole을 포함한 raw 값(=
    `reconstruct_free`를 그대로 쓰는 호출자가 실제로 받는 값)은 실측상 뚜렷이 낮다는
    것을 같은 테스트 안에서 함께 확인한다.
    """
    rows, cols = np.mgrid[0:SPEC.n_rows, 0:SPEC.n_cols]
    origin_r, origin_c = SPEC.front_m / SPEC.cell_m - 0.5, SPEC.half_width_m / SPEC.cell_m - 0.5
    radius = np.hypot(rows - origin_r, cols - origin_c)
    hole = radius <= 5                                     # permanent_blind 역할의 원점 주변 hole
    free = (radius > 5) & (radius < 12)                     # 고리 -- star-convex하되 안쪽이 비어 있다

    rays = build_ray_index(SPEC, n_theta=720)
    r_m, status = first_free_range(free, rays)
    restored = reconstruct_free(r_m, status, rays, free.shape)

    # (a) hole 제외 -- polar 표현 자체의 손실만 남긴, §12가 의미 있다고 정한 정의.
    restored_excl, free_excl = restored & ~hole, free & ~hole
    union_excl = (restored_excl | free_excl).sum()
    assert (restored_excl & free_excl).sum() / union_excl >= 0.99

    # (b) hole 포함 raw 값 -- `reconstruct_free`는 원점부터 채우므로 hole을 거짓양성으로
    # 되살린다. 실측(2026-08-17): 0.8214. "0.99에 한참 못 미친다"는 사실 자체가 §12의
    # root cause(원점-fill이 정적 마스크를 되살린다)를 이 단위 테스트 안에서 재현한다.
    union = (restored | free).sum()
    assert (restored & free).sum() / union <= 0.85


def test_frontier_cells_marks_the_cell_the_range_points_at():
    """`r_m`이 가리키는 셀 자체를 찍는다 -- off-by-one이면 마지막 free 셀을 찍는다."""
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    free[8:19, 20] = True

    rays = build_ray_index(SPEC, n_theta=4)
    r_m, status = first_free_range(free, rays)
    frontier = frontier_cells(r_m, status, rays, free.shape)

    # 전방 광선은 row 7에서 멈춘다(`test_first_free_range_...`의 손계산과 같은 fixture).
    assert frontier[7, 20]
    assert not (frontier & free).any()      # free 셀은 절대 표면이 아니다


def test_frontier_cells_ignores_censored_and_no_free_rays():
    """격자 끝까지 free인 광선을 표면으로 세면 모든 프레임의 ROI 테두리가 occupied가 된다."""
    rays = build_ray_index(SPEC, n_theta=8)
    shape = (SPEC.n_rows, SPEC.n_cols)

    all_free = np.ones(shape, bool)                                  # 전부 RAY_CENSORED
    assert not frontier_cells(*first_free_range(all_free, rays), rays, shape).any()

    nothing_free = np.zeros(shape, bool)                             # 전부 RAY_NO_FREE
    assert not frontier_cells(*first_free_range(nothing_free, rays), rays, shape).any()


def test_frontier_cells_recovers_the_boundary_of_a_convex_free_region():
    """유도된 표면 = free 영역의 경계라는 계약. (D) 정식화가 occupied를 보고하는 근거다.

    원점을 포함하는 볼록 영역이면 모든 광선의 첫 non-free가 그 영역의 바로 밖 테두리이므로,
    유도 결과는 테두리의 **부분집합**이어야 하고(거짓양성 0) 테두리를 거의 다 덮어야 한다.
    720 광선에서 실측 커버리지는 1.0이지만 모서리에서 각도 표본이 성길 수 있어 0.9로 둔다.
    """
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    free[10:30, 10:30] = True                        # 원점(19.5, 19.5)을 포함하는 사각형
    border = np.zeros_like(free)
    border[9:31, 9:31] = True
    border &= ~free                                  # 두께 1셀 테두리

    rays = build_ray_index(SPEC, n_theta=720)
    frontier = frontier_cells(*first_free_range(free, rays), rays, free.shape)

    assert frontier.any()
    assert not (frontier & ~border).any()
    assert (frontier & border).sum() / border.sum() >= 0.9


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
