"""`semantic_annotations/gtLabels/*_BEV.png` → ego 기준 occupancy 그리드 크롭.

--------------------------------------------------------------------------------
좌표 규약 (실측 확정, 2026-07-30)
--------------------------------------------------------------------------------
소스 `_BEV.png`는 1024×1024이고, ego 프레임(x=전방, y=좌측; `projects.geometry.frames`)
좌표를 다음 식으로 픽셀에 대응시킨다 (`ego_to_bev_pixel`):

    row = 511.5 + SIGN_FORWARD * x / mpp      (SIGN_FORWARD = -1)
    col = 511.5 + SIGN_LATERAL * y / mpp      (SIGN_LATERAL = -1)

즉 소스 이미지 자체가 이미 **일반적인 top-down 지도 방향**이다: 전방(+x)이 위(row 감소),
차량 좌측(+y)이 왼쪽(col 감소). 부호 두 개는 4개 후보를 실데이터로 전수 비교해 확정했다
(자세한 근거는 아래 "검증 근거").

--------------------------------------------------------------------------------
반환 배열의 방향 (그대로 이미지로 저장하면 소스 이미지와 같은 방향)
--------------------------------------------------------------------------------
`crop_bev_occupancy`가 돌려주는 (n_rows, n_cols) 배열은 **표시용 방향 그대로**다.
추가로 `np.flipud`/`np.fliplr`를 걸 필요가 없다.

- `row 0` = **가장 앞**(전방 front_m 경계) → 이미지로 보면 맨 위
- `row -1` = 가장 뒤(후방 rear_m 경계) → 맨 아래
- `col 0` = 차량 **좌측**(+y, half_width_m 경계) → 이미지로 보면 맨 왼쪽
- `col -1` = 차량 우측(−y) → 맨 오른쪽

이 방향은 소스 `_BEV.png`를 같은 ROI로 잘라낸 부분영상과 정확히 동일하다(뒤집힘 없음).
초기 구현은 두 축이 모두 반대였다 — 이미지로 저장하면 앞뒤/좌우가 함께 뒤집혀 보였다.

--------------------------------------------------------------------------------
검증 근거 (실데이터 실측 — 추론이 아니다)
--------------------------------------------------------------------------------
1. **스케일 15/512**: BEV 카메라는 ego (0,0,15)에 pitch=−90으로 달린 **핀홀**이며
   (readme.txt), CARLA 기본 FOV는 90°다. 폭 1024px의 절반 512px가 지면(z=0)에서
   15·tan45° = 15 m를 덮으므로 지면 스케일은 정확히 `15/512 = 0.029296875` m/px
   (전체 이미지 = 30.0 m). SynWoodScape 논문도 BEV 커버리지를 "~30 m"로 적고 있다.
2. **핀홀임을 데이터로 확인**: `depth_maps/raw_data/00000_BEV.npy`에서 노면(class 7)
   depth 중앙값 14.994 m, ego 차체(class 24) 13.678 m(최소 13.443). 즉 카메라 높이 15 m가
   맞고, 차체 지붕처럼 **높이 h인 면은 15/(15−h) 배로 확대**된다.
3. **Phase 1 파이프라인 교차검증**: 4대 어안 카메라의 depth map에서 지면 class(6/7/8/14/20)
   픽셀을 `projects.geometry.reprojection.unproject_depth_to_ego`로 ego 프레임에 되돌려
   |z|<0.15, 반경 2.5~13 m인 점만 남기고(5샘플, 218,859점), mpp를 0.0270~0.0320 구간에서
   훑어 BEV semantic class 일치율을 재면 0.0292~0.0294에서 정점이 나온다(카메라별로도 동일).
   `15/512`에서 0.9786, 옛 값 0.0284에서 0.9442. LiDAR 지면 포인트로 따로 재도 같다(0.9800).
4. **부호**: 4개 후보 전수 비교 결과 (sign_forward, sign_lateral) = (−1, −1)이
   네 카메라 각각에서 압도적이다 (FV 0.980 / MVL 0.983 / MVR 0.974 / RV 0.978,
   차선책은 0.87~0.94). MVL 포인트의 ego y 평균이 +3.26, MVR이 −2.88인 것도 y=좌측 규약과 맞는다.

⚠️ **옛 값 0.0284로 되돌리지 말 것.** 그 값은 ego 차량의 BEV 실루엣 픽셀 크기를 3D 박스
치수로 나눠 얻은 것인데, 위 (2)의 핀홀 확대 때문에 **구조적으로 과소추정**이다: 차체 실루엣은
지붕(h≈1.5 m) 높이의 윤곽이라 지면보다 15/13.5 ≈ 1.11배 크게 찍히고, 그만큼 m/px가 작게
나온다(0.0286 ≈ 0.0293/1.024, 실루엣 상단·하단이 섞이므로 배율은 1.11보다 작게 희석된다).
차량 실루엣 기반 추정은 원점(511.5, 511.5) 확인용으로만 쓴다.
"""
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, remap_semantic_to_occupancy

# 지면(z=0) 스케일: 카메라 높이 15 m, FOV 90°, 폭 1024px(half-width 512px)
# → 15·tan(45°)/512 = 15/512 m/px. 전체 1024px = 정확히 30.0 m.
BEV_METERS_PER_PIXEL = 15.0 / 512  # = 0.029296875
BEV_ORIGIN_PX = (511.5, 511.5)  # (col, row) — 여러 샘플에서 ego bbox 중심이 항상 이 값
SIGN_FORWARD = -1  # ego +x(전방) → row 감소 (이미지 위쪽)
SIGN_LATERAL = -1  # ego +y(좌측) → col 감소 (이미지 왼쪽)


def ego_to_bev_pixel(
    forward_m,
    lateral_m,
    meters_per_pixel: float = BEV_METERS_PER_PIXEL,
    origin_px=BEV_ORIGIN_PX,
    sign_forward: int = SIGN_FORWARD,
    sign_lateral: int = SIGN_LATERAL,
):
    """ego 프레임 (x=전방, y=좌측) 미터 좌표 → 소스 `_BEV.png`의 (row, col) 실수 픽셀 좌표.

    파이프라인(`crop_bev_occupancy`)과 보정 스크립트(`tools/calibrate_bev_scale.py`)가
    **같은 식**을 쓰도록 여기 한 곳에만 둔다. 예전에 두 곳에 복사돼 있어서 "보정한 코드"와
    "실제로 쓰는 코드"가 갈라질 수 있었다.

    지면(z≈0) 점에 대한 식이다. 높이 z인 점은 핀홀 확대 때문에 유효 스케일이
    `meters_per_pixel · (15−z)/15`로 줄어든다 (모듈 docstring 참고).
    """
    origin_col, origin_row = origin_px
    row = origin_row + sign_forward * np.asarray(forward_m, dtype=np.float64) / meters_per_pixel
    col = origin_col + sign_lateral * np.asarray(lateral_m, dtype=np.float64) / meters_per_pixel
    return row, col


def crop_bev_occupancy(
    semantic_bev_image: np.ndarray,
    grid_spec: OccupancyGridSpec,
    meters_per_pixel: float = BEV_METERS_PER_PIXEL,
    origin_px=BEV_ORIGIN_PX,
    sign_forward: int = SIGN_FORWARD,
    sign_lateral: int = SIGN_LATERAL,
) -> np.ndarray:
    """Crop and remap a semantic BEV image into an ego-relative occupancy grid.

    Returns a (grid_spec.n_rows, grid_spec.n_cols) uint8 array in **display-natural**
    orientation — save it as an image directly, no flips needed:

    - row 0    = front-most cell   (image top)
    - row -1   = rear-most cell    (image bottom)
    - col 0    = vehicle's left    (image left)
    - col -1   = vehicle's right   (image right)

    See this module's docstring for the derivation and the real-data evidence behind
    `BEV_METERS_PER_PIXEL` / `SIGN_FORWARD` / `SIGN_LATERAL`.
    """
    # row 0 이 가장 앞, col 0 이 가장 왼쪽이 되도록 미터 좌표를 **내림차순**으로 만든다.
    # (전방 +x가 row 0, 좌측 +y가 col 0 → 표시 방향 그대로.)
    forward_m = grid_spec.front_m - (np.arange(grid_spec.n_rows) + 0.5) * grid_spec.cell_m
    lateral_m = grid_spec.half_width_m - (np.arange(grid_spec.n_cols) + 0.5) * grid_spec.cell_m

    # 행/열이 서로 독립이므로 축별로 따로 매핑한다(다른 축은 0으로 두고 결과를 버린다).
    # 공유 헬퍼를 그대로 쓰기 위한 형태다 — 매핑 식이 여기서 다시 인라인되지 않게.
    rows, _ = ego_to_bev_pixel(
        forward_m, 0.0, meters_per_pixel, origin_px, sign_forward, sign_lateral
    )
    _, cols = ego_to_bev_pixel(
        0.0, lateral_m, meters_per_pixel, origin_px, sign_forward, sign_lateral
    )

    height, width = semantic_bev_image.shape
    rows = np.clip(np.round(rows).astype(int), 0, height - 1)
    cols = np.clip(np.round(cols).astype(int), 0, width - 1)

    sampled_labels = semantic_bev_image[np.ix_(rows, cols)]
    return remap_semantic_to_occupancy(sampled_labels)
