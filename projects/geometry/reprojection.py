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

# 상대오차만으로는 근거리(BEV grid는 카메라에서 2~4 m 이내)에서 잔여 캘리브레이션 오차
# (translation ~4 cm, rotation ~0.2~0.5°, 문서 §1~2)가 그대로 5% 문턱을 근소하게 넘겨
# "장애물 없는데 안 보임"으로 오탐되는 걸 봤다 — 3개 샘플 실측: 진짜 노이즈로 보이는
# 케이스(00025)는 절대오차 0.12~0.19 m에 몰려 있고, 진짜 가려짐으로 보이는 케이스(00000)는
# 0.45 m 이상이라 둘이 뚜렷이 갈린다. 0.20 m를 바닥값으로 두면 00025의 노이즈는 100%,
# 비슷한 케이스인 00004는 75%가 구제되면서 00000의 진짜(로 보이는) 가려짐은 7%만 영향받는다
# (`docs/dataset_analysis/synwoodscape_geometry_findings.md`에 재현 스크립트로 남길 것).
VISIBILITY_ABS_TOL_M = 0.20


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
    abs_tol_m: float = VISIBILITY_ABS_TOL_M,
    max_depth_m: float = MAX_VALID_DEPTH_M,
) -> np.ndarray:
    """투영된 포인트가 그 카메라에서 **실제로 보이는지** 판정한다.

    LiDAR(ego z=2.0)는 카메라(z=0.9~1.0)보다 높이 있어서 카메라가 못 보는 곳까지 본다.
    포인트까지의 거리가 그 픽셀의 depth와 맞으면 카메라에도 보이는 것이고, 거리가 더 멀면
    앞의 무언가에 가려진 것이다.

    허용오차는 `max(abs_tol_m, rel_tol * depth)`다 — 상대오차만 쓰면 근거리에서 잔여
    캘리브레이션 오차가 그대로 문턱을 넘어 오탐(허위 가려짐)을 만든다 (`VISIBILITY_ABS_TOL_M`
    참고). 원거리에서는 여전히 상대오차가 지배적이라 기존 LiDAR 검증(Phase 1)의 동작은
    그대로 유지된다.
    """
    depth_at_pixel = depth_map[projected.row, projected.col]
    finite = (depth_at_pixel > 0) & (depth_at_pixel < max_depth_m)
    absolute_error = np.abs(projected.range_m - depth_at_pixel)
    tolerance = np.maximum(abs_tol_m, rel_tol * np.where(finite, depth_at_pixel, 1.0))
    return finite & (absolute_error < tolerance)


def unproject_depth_to_ego(camera, rows: np.ndarray, cols: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """depth map의 픽셀들을 ego 프레임 3D 포인트로 되돌린다.

    depth 값은 카메라 중심으로부터의 **방사 거리**다 (평면 z-depth가 아니다 — 노면 픽셀로
    평면을 맞춰 확인했다: 방사 거리로 해석하면 법선 오차 0.2° 이내의 평면이 나오고,
    z-depth로 해석하면 발산한다).

    포인트 수가 많으면(수십만~수백만) `unproject_pixels_to_ego_fast`를 대신 쓴다 — 이 함수는
    third-party `RadialPolyCamProjection._rho_to_theta`(포인트당 `np.roots` 호출)를 그대로
    타서 포인트 하나하나가 느리다.
    """
    screen_points = np.stack([cols, rows], axis=1).astype(float)
    return camera.project_2d_to_3d(screen_points, depth_map[rows, cols])


def unproject_pixels_to_ego_fast(
    camera,
    rows: np.ndarray,
    cols: np.ndarray,
    depth_map: np.ndarray,
    theta_max_deg: float = 140.0,
    theta_samples: int = 20000,
) -> np.ndarray:
    """`unproject_depth_to_ego`와 수학적으로 동일하지만, 포인트 전체를 벡터화해서 처리한다.

    third-party `RadialPolyCamProjection.project_2d_to_3d`는 theta(입사각)를 rho(픽셀
    반경)로부터 되찾을 때 포인트마다 `np.roots`로 다항식을 푼다 — LiDAR 몇천 점엔 괜찮지만
    depth map 전체(수십만~수백만 픽셀 × 4대 카메라)엔 감당이 안 된다.

    이 다항식(`theta_to_rho`)은 이 데이터셋의 4개 카메라 모두 0~180°에서 단조증가함을
    실측으로 확인했다(계수가 fisheye 어안 모델의 통상적 형태라 rho가 감소로 꺾이지 않는다).
    단조증가라는 성질만 있으면 역함수를 미리 조밀한 테이블로 뽑아 `np.interp`로 구해도
    포인트당 `np.roots`를 새로 푸는 것과 결과가 사실상 같다 — 그래서 여기서만 그 성질에
    기대 훨씬 빠른 경로를 쓴다. `theta_max_deg`(기본 140°)는 이 카메라들의 실측 이미지 코너
    입사각(~110°)보다 넉넉한 여유를 둔 값이다.
    """
    theta_table = np.linspace(0.0, np.radians(theta_max_deg), theta_samples)
    rho_table = camera.lens._theta_to_rho(theta_table)

    screen_points = np.stack([cols, rows], axis=1).astype(np.float64)
    lens_points = (screen_points - camera._principle_point) / camera._aspect_ratio
    rho = np.linalg.norm(lens_points, axis=1)
    theta = np.interp(rho, rho_table, theta_table)

    norm = depth_map[rows, cols].astype(np.float64)
    chi = norm * np.sin(theta)
    z_cam = norm * np.cos(theta)
    xy_cam = np.divide(chi, rho, where=(rho != 0))[:, np.newaxis] * lens_points
    points_cam = np.concatenate([xy_cam, z_cam[:, np.newaxis]], axis=1)
    return points_cam @ camera.rotation.T + camera.translation
