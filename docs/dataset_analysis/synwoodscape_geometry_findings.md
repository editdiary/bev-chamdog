# SynWoodScape 기하 검증 결과 노트 (Phase 1 · Phase 2 · Phase 2.5)

- 작성일: 2026-07-30 / **2026-08-02 visibility·raycast 설계(§3) 추가**
- 대상 스펙: [`docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md`](../superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md)
- 대상 코드: `projects/geometry/`, `projects/bev_gt/`(`grid.py`, `bev_crop.py`, `visibility.py`,
  `raycast_occupancy.py`), `tools/verify_fisheye_projection.py`, `tools/calibrate_bev_scale.py`,
  `tools/build_occupancy_gt.py`(§2, 구 파이프라인), `tools/build_visibility_mask.py`(§3.1, 폐기),
  `tools/build_raycast_occupancy.py`(§3.2, 폐기), **`tools/build_hybrid_occupancy.py`(§3.3, 현재 파이프라인)**

> 목적: SynWoodScape의 어안 투영(Phase 1)과 BEV occupancy GT 생성(Phase 2, Phase 2.5)을
> 구현·검증하면서 **실측으로 밝혀낸 것들**을 남긴다. 특히 **데이터셋 문서가 말해주지 않거나
> 잘못 말해주는** 항목(extrinsic 부호, LiDAR 좌표 규약, BEV 카메라 스케일, 어안 semantic
> segmentation 자체의 라벨 결함)에 집중한다.
> 모든 수치는 로컬 데이터(`dataset/synwoodscape/SynWoodScape_V0.1.0/`)를 직접 읽어 측정했다.
> §1·§2(기하·스케일)는 여전히 유효하다. §3(visibility 설계)은 이후 여러 차례 갈아엎은
> 과정이라 **결론만이 아니라 왜 앞선 두 시도가 버려졌는지**를 함께 남긴다 — 같은 실수를
> 반복하지 않기 위함이다.

---

## 0. 한눈에 보기 — 확정된 규약과 상수

| 항목 | 값 | 어디에 있나 |
|---|---|---|
| 작업 프레임(ego) | X=전방, Y=**좌측**, Z=상방, **오른손계**, 원점=차량 중심·지면 높이 | `projects/geometry/frames.py` |
| CARLA 프레임(world/vehicle) | X=전방, Y=**우측**, Z=상방, **왼손계** | 같음 |
| `lidar_data/*.pkl`의 `points` | **센서 로컬 좌표 + ego와 같은 오른손계** → ego 변환은 `+(0,0,2.0)`뿐 | `lidar_points_to_ego()` |
| 카메라 extrinsic 회전 | JSON quaternion의 **roll 부호를 뒤집어야** 맞음 | `projects/geometry/fisheye.py` §1 |
| 카메라 extrinsic translation | CARLA 규약(Y=우측) → Y 반전. **MVL/MVR은 y/z 성분이 전치**되어 저장됨 | 같은 파일 §2 |
| `BEV_METERS_PER_PIXEL` | **15/512 = 0.029296875** m/px (1024px = 정확히 30.0 m) | `projects/bev_gt/bev_crop.py` |
| `BEV_ORIGIN_PX` | (511.5, 511.5) = 이미지 정중앙, (col, row) | 같은 파일 |
| `SIGN_FORWARD`, `SIGN_LATERAL` | **−1, −1** (ego +x → row 감소, ego +y → col 감소) | 같은 파일 |
| occupancy 배열 방향 | row 0 = 최전방, col 0 = 차량 좌측 (**표시용 그대로**, flip 불필요) | 같은 파일 |
| drivable class | 6 (road line), 7 (road) | `projects/bev_gt/grid.py` |
| occupancy(class) 값의 출처 | **top-down `_BEV.png` gtLabel** (어안 카메라 자신의 semantic label은 안 씀 — §3.2 이유) | `bev_crop.crop_bev_occupancy` |
| observed/visible(관측 여부) 출처 | **4-cam 어안 depth raycast** — 어느 cell에 광선이 실제로 도달했는가 (semantic label 불필요) | `raycast_occupancy.compute_observed_mask` |
| 최종 GT 위치·형식 | `dataset/synwoodscape_occupancy_gt/{sample}_{occupancy.npy,visible.npy,combined.png}` (`outputs/`가 아니다) | `tools/build_hybrid_occupancy.py` |
| 실제로 생성하는 grid spec | `synwoodscape_pretrain`만 (`robot`은 자체 로봇 fine-tuning 단계용으로 보류, SynWoodScape에는 생성 안 함) | 같은 파일 |

---

## 1. Phase 1에서 찾아 고친 세 가지 버그

Phase 1의 목표는 `radial_poly` 어안 project/unproject를 SynWoodScape 자체 데이터로 검증하는
것이었다. 초기 구현의 **semantic consistency rate가 0.001~0.451**로 나왔고, 원인은 셋이었다.

### 1.1 LiDAR pkl의 `points`를 world 좌표로 오해했다

`lidar_data/*.pkl`은 `points`와 `transform`(=world_T_lidar)을 함께 담고 있어서, 처음에는
`points`가 world 좌표라고 보고 `inv(transform)`을 곱했다. **틀렸다** — `points`는 이미 센서
로컬 좌표다. 잘못된 곱셈은 포인트를 차량의 world yaw(샘플마다 88°/−131°/82° …)만큼 추가로
회전시키고 ~140 m 평행이동시켰고, 오차가 샘플마다 다른 회전이라 **어떤 축 부호/순서 조합으로도
복구되지 않았다**.

근거 (실측):

- `points`의 원점 기준 거리 중앙값 5.65 m, 평균 좌표 ≈(0.5, 1.7, 0.4) → 원점 주변에 모여 있다.
  world 좌표라면 차량 위치 ≈(−111, 81) 주변이어야 한다.
- `points`의 축 규약도 CARLA(왼손)가 아니라 **ego와 같은 오른손(Y=좌측)** 이다. Y를 뒤집어
  CARLA 프레임으로 보내면 vehicle 라벨(10/21) 포인트가 `_unuserd/box_3d_annotations`의 차량
  3D 박스 안에 들어간다 (샘플 00000: 29.4%). 뒤집지 않으면 0.1%다.

**교훈:** "센서 pkl에 world transform이 함께 있으면 points도 world"라는 가정을 검증 없이 쓰면
안 된다. 좌표가 원점 주변에 모여 있는지 보는 것만으로 1분 안에 판별된다.

### 1.2 카메라 extrinsic quaternion의 roll 부호가 뒤집혀 있다

`calibration_data/*.json`의 quaternion은 CARLA 장착 각도 (pitch, yaw, roll)를 **일괄 부호
반전**해 오른손 ZYX에 넣은 것에 해당한다. 올바른 CARLA(왼손)→ego(오른손) 변환은 yaw와 roll만
반전하고 pitch는 유지하므로, 차이는 **roll 성분의 부호 하나**다 (전형적인 handedness 실수).

근거 — 카메라 자신의 depth map에서 노면(class 6/7) 픽셀을 역투영해 평면을 맞추고, 그 법선을
extrinsic이 예측하는 법선과 비교한 각도 오차:

| cam | mount roll | 보정 전 | 보정 후 |
|-----|-----------:|--------:|--------:|
| FV  |     −2.37° |   4.46° |   0.46° |
| MVL |    −10.18° |   4.70° |   0.18° |
| MVR |     +8.38° |   3.99° |   0.17° |
| RV  |     −0.12° |   0.18° |   0.16° |

roll이 거의 0인 RV만 원래도 맞았다는 점, 그리고 오차 크기가 각 카메라의 `2·|roll|`을 법선의
면내 성분에 투영한 값과 정량적으로 일치한다는 점이 진단의 결정적 근거였다.

### 1.3 미러 카메라(MVL/MVR) translation의 y/z 성분이 전치되어 저장돼 있다

    MVL: JSON (0.8, 1.0, -0.9) → 전치 (0.8, -0.9, 1.0) → Y 반전 (0.8, +0.9, 1.0)
    MVR: JSON (0.8, 1.0,  0.9) → 전치 (0.8,  0.9, 1.0) → Y 반전 (0.8, -0.9, 1.0)

전치를 빼면 MVL이 **지면 아래(z=−0.9)이고 좌우도 반대**로 놓인다. FV/RV는 y=0이라 이 전치를
적용하면 오히려 틀리므로(카메라가 지면 높이에 놓인다) **미러 카메라에만** 적용한다.

근거 — LiDAR 포인트 투영의 depth 상대오차를 robust 손실로 두고 (R, t)를 6-DOF 자유 재추정하면
네 카메라 모두 위 규칙의 값과 **4 cm 이내**로 일치한다.

### 1.4 완료 기준도 함께 고쳤다

초판의 "semantic consistency rate ≥ 0.90"은 **달성 불가능한 기준**이었다. LiDAR는 ego z=2.0,
카메라는 z=0.9~1.0이라 **LiDAR에는 보이지만 카메라에는 가려지는** 포인트가 구조적으로 20~35%
존재하고, 그 포인트들은 가림 물체의 class 위에 떨어진다.

42개 샘플 × 4대 = **168쌍**을 실측해 기준을 다시 정했다.

| 지표 | 실측 (168쌍) | 채택 기준 |
|---|---|---|
| `depth_agreement` | min 0.433 / median 0.689 / max 0.864 | 카메라별 median ≥ 0.60 |
| `visible_consistency_rate` | min 0.897 / median 0.973 / max 0.996 | 카메라별 median ≥ 0.93, 쌍별 ≥ 0.85 |

`depth_agreement`는 카메라 **높이** 오차에는 민감하지만 **횡방향** 오차에는 판별력이 약하다
(미러 y를 뒤집어도 6샘플 median이 0.600으로 기준선에 걸린다). 그래서 좌우 배치는 전용 기준을
따로 둔다: depth map의 `ego-vehicle`(class 24) 픽셀을 역투영한 ego y 중앙값이
**MVL > +0.3, MVR < −0.3, FV·RV는 |y| < 0.25**. 실측 MVL +0.756 / MVR −0.757 /
FV −0.082 / RV −0.001이고 42샘플에 걸쳐 ±0.001로 안정적이다.

---

## 2. Phase 2 — BEV 카메라 스케일: 왜 0.0284가 아니라 15/512인가

`_BEV.png`를 만든 가상 BEV 카메라의 intrinsic은 데이터셋 어디에도 없다. `readme.txt`가 주는
정보는 위치뿐이다: *"The BEV camera is attached to the ego vehicle at the position
x=0.0, z=15, y=0.0 … with pitch=-90.0 to face down."*

### 2.1 처음엔 차량 실루엣으로 역산했고, 그게 틀렸다

`_unuserd/instance_annotations` + `_unuserd/box_3d_annotations`로 ego 차량의 픽셀 bbox
(66×132 px)를 3D 박스 치수로 나눠 **0.0284~0.0288 m/px**를 얻었다. 두 축이 서로 비슷해서
그럴듯해 보였지만, 이 방법에는 두 가지 문제가 있다.

**(a) 핀홀 확대 (본질적 편향).** BEV 카메라는 정사영이 아니라 **z=15 m의 진짜 핀홀**이다.
따라서 높이 h인 면은 지면보다 `15/(15−h)`배 크게 찍힌다. 차량 실루엣은 지면이 아니라 차체
(ego 박스 z∈[0.009, 1.556] m)의 윤곽이므로 **항상 실제보다 크게** 찍히고, m/px는 그만큼
**작게** 나온다.

데이터로 확인 — `depth_maps/raw_data/00000_BEV.npy` (BEV depth는 방사 거리가 아니라 평면
z-depth다: 노면 값이 이미지 전체에서 14.94~15.05로 거의 일정하다):

| class | depth 중앙값 | 해석 |
|---|---:|---|
| 7 (road) | 14.994 m | 카메라 높이 15 m 확인 |
| 6 (road line) | 14.989 m | 같음 |
| 24 (ego-vehicle) | 13.678 m (최소 13.443) | 차체 지붕 h ≈ 1.32~1.56 m → 확대율 1.10~1.12 |

**(b) yaw 오염 (구현 버그).** `tools/calibrate_bev_scale.py` 초판은 여러 instance의 **world
프레임** 3D 박스의 축정렬 extent를 그대로 썼다. actor의 world yaw가 섞이므로 같은 ego 박스가
샘플에 따라 (1.90, 3.76) ↔ (3.72, 1.83)으로 뒤바뀐다. 이 때문에 스크립트가
`scale_x ≈ 0.031 / scale_y ≈ 0.019`처럼 **파이프라인 상수와도, 서로와도 안 맞는** 값을
출력했다. 코너를 ego 프레임으로 되돌리면 extent가 (3.705, 1.789)로 모든 샘플에서 동일해진다.

### 2.2 올바른 값 — 기하 도출

CARLA 카메라의 기본 FOV는 **90°**다. 이미지 폭 1024 px의 절반 512 px가 지면(z=0)에서
`15·tan(45°) = 15 m`를 덮으므로:

    BEV_METERS_PER_PIXEL = 15.0 / 512 = 0.029296875 m/px
    → 이미지 전체 1024 px = 정확히 30.0 m

SynWoodScape 논문도 BEV 이미지 커버리지를 **"~30 m"** 로 적고 있어 일치한다.

### 2.3 교차검증 — Phase 1 파이프라인의 지면 포인트

Phase 1의 (이미 검증된) 어안 파이프라인으로 지면 점을 ego 프레임에 되돌려, m/px를 sweep하며
BEV semantic class 일치율을 재는 방식으로 독립 검증했다.

방법: 4대 카메라의 depth map에서 지면 class(6/7/8/14/20) 픽셀을 `unproject_depth_to_ego`로
ego 프레임에 되돌리고, `|z| < 0.15 m`·반경 2.5~13 m인 점만 남긴다(5샘플 218,859점). 각 후보
m/px로 BEV 픽셀을 계산해 `_BEV.png`의 class와 비교한다.

| m/px | 일치율 (4대 합산) |
|---|---:|
| 0.0284 (실루엣 추정) | 0.9442 |
| 0.0290 | 0.9724 |
| **0.0292** | **0.9792** |
| 0.0294 | 0.9743 |
| 0.0300 | 0.9469 |
| **15/512 = 0.029297** | **0.9786** |

카메라별로 따로 재도 전부 0.0292~0.0294에서 정점이다 (FV 0.980 / MVL 0.983 / MVR 0.974 /
RV 0.978 @ 15/512). LiDAR 지면 리턴으로 따로 재도 같다 (0.9800 @ 15/512, 정점 0.0292).

격자를 0.00005로 좁혀 7샘플(28쌍)로 다시 재면 정점이 **0.02925 (일치율 0.98355)** 이고,
`15/512 = 0.029297`에서 **0.98294**다. 두 값의 차이는 0.00005 m/px = **0.16%** — 격자 한 칸,
즉 이 측정 방법의 분해능 수준이다. 기하 도출값이 경험적 정점과 사실상 같다.

    0.02915 -> 0.98250
    0.02920 -> 0.98339
    0.02925 -> 0.98355   <- 경험적 정점
    0.02930 -> 0.98294   (15/512 = 0.029297 이 여기)
    0.02935 -> 0.98139

이 검증은 `tests/bev_gt/test_bev_crop.py::test_bev_scale_beats_the_biased_vehicle_silhouette_estimate_on_ground_points`
로 회귀 테스트에 고정했다.

### 2.4 축 부호와 배열 방향

ego → 소스 픽셀 매핑은 한 곳(`bev_crop.ego_to_bev_pixel`)에만 둔다:

    row = 511.5 + SIGN_FORWARD * x_ego / mpp      (SIGN_FORWARD = -1)
    col = 511.5 + SIGN_LATERAL * y_ego / mpp      (SIGN_LATERAL = -1)

4개 부호 후보를 지면 포인트 class 일치율로 전수 비교한 결과 (−1, −1)이 카메라별로 모두
압도적이다:

| (sign_forward, sign_lateral) | 어안 depth 지면점 | LiDAR 지면점 |
|---|---:|---:|
| **(−1, −1)** | **0.9786** | **0.9800** |
| (+1, −1) | 0.9174 | 0.9021 |
| (−1, +1) | 0.7415 | 0.6331 |
| (+1, +1) | 0.7362 | 0.6311 |

즉 소스 `_BEV.png` **자체가 이미 일반적인 top-down 지도 방향**이다 — 전방이 위, 차량 좌측이
왼쪽. (MVL 포인트의 ego y 평균이 +3.26, MVR이 −2.88인 것도 ego Y=좌측 규약과 맞는다.)

**초기 구현은 여기서 두 번째 버그를 만들었다.** 부호는 맞았지만 출력 배열의 행/열 **인덱스
순서**를 반대로 만들어(row 0 = 최후방, col 0 = 차량 우측), 그대로 이미지로 저장하면 앞뒤·좌우가
함께 뒤집혀 보였다. 좌우 뒤집힘은 생성 결과를 원본 `rgb_images/*_BEV.png`와 눈으로 비교하다
발견됐다.

수치 확인 (샘플 00300, 전방7/후방3/좌우±5 창):

| | 좌반 drivable | 우반 drivable | 전방반 | 후방반 |
|---|---:|---:|---:|---:|
| 소스 창 (`_BEV.png` 직접 슬라이스) | 0.939 | 0.702 | 0.880 | 0.761 |
| **수정 후** `crop_bev_occupancy` | 0.937 | 0.701 | 0.878 | 0.760 |
| 수정 전 (버그) | 0.702 ← 뒤바뀜 | 0.939 | 0.761 | 0.880 |

이제 반환 배열은 **표시용 방향 그대로**다 (row 0 = 최전방, col 0 = 차량 좌측). 저장 시
`np.flipud`/`np.fliplr`가 필요 없고, 소스 이미지를 같은 ROI로 자른 부분영상과 정확히 같다.

**교훈:** 부호 상수(ego→소스 픽셀)와 출력 배열의 인덱스 순서는 **별개의 결정**이다. 부호를
검증했다고 배열 방향이 검증된 게 아니다. 그리고 축을 하나만 테스트하는 합성 이미지(모든 열이
균일한 이미지로 행 축만 확인)로는 다른 축의 뒤집힘이 **원리적으로 보이지 않는다**.

---

## 3. Visibility(관측 여부) 설계 — 가설검정 → raycast 단독 → 하이브리드 (2026-08-02)

§2까지의 `crop_bev_occupancy`는 `_BEV.png`를 crop+remap만 할 뿐이라, 어안 카메라가 실제로
그 지점을 봤는지와 무관하게 **모든 cell에 항상 값이 있다**. 하지만 ego 차체 바로 아래나
다른 차량에 가려진 노면은 실제로는 "안 보이는" 영역이라 drivable/obstacle을 단정할 근거가
없다 — 이 영역을 구분해 학습 loss에서 제외(ignore)하는 게 이번 절의 목표였다. 두 번의
시도가 실패하거나 새 버그를 만들었고, 세 번째(하이브리드)로 정착했다.

### 3.1 시도 1 — grid cell을 z=0으로 투영해 가설검정 (`visibility.py`, 폐기)

`projects/bev_gt/visibility.py` + `tools/build_visibility_mask.py`: 각 grid cell을 지면
(z=0) 점으로 보고 4대 어안 카메라에 투영한 뒤, 그 픽셀의 depth map 값과 카메라-점 거리가
`max(0.20m, 5%·depth)` 오차 이내로 맞는지 가설검정한다(`projects/geometry/reprojection.py::
visibility_mask`). 맞으면 "보임", 벗어나면 "가려짐".

**폐기 이유:** BEV grid는 카메라에서 2~4 m 이내인 근거리인데, 이 거리대에서는 §1.2/§1.3에서
고친 뒤에도 남는 잔여 캘리브레이션 오차(translation ~4cm, rotation ~0.2~0.5°)가 5% 상대오차
문턱을 근소하게 넘나든다. 즉 **"장애물이 없는데 안 보인다"는 허위 오탐**과 **실제 가려짐**을
가설검정만으로는 구분할 수 없다 — 오차의 원인(캘리브레이션 잔차 vs 진짜 가려짐)이 근본적으로
분리되지 않는 문제였다(`VISIBILITY_ABS_TOL_M` 도입으로 일부 완화했으나 원인 자체는 해결 못함).

### 3.2 시도 2 — 어안 카메라 자신의 semantic label로 raycast (`build_raycast_occupancy`, 폐기)

`projects/bev_gt/raycast_occupancy.py::build_raycast_occupancy` + `tools/
build_raycast_occupancy.py`: 가설검정 대신, 4대 카메라 각각의 depth map **모든 픽셀**을 그
픽셀의 depth로 실제 언프로젝션해 ego 프레임 3D 점(카메라 광선이 실제로 부딪힌 지점)을 얻고,
그 점이 떨어지는 cell에 **그 픽셀 자신의 semantic label**을 찍는다. 여러 카메라·픽셀이 한
cell에 겹치면 다수결로 drivable/obstacle을 정하고, 광선이 한 번도 안 닿은 cell은
`observed=False`로 남긴다.

이 방식은 "보였는가"를 문턱 없이 **정의 그대로** 판정하므로 시도 1의 오탐 문제를 완전히
해결했다. 하지만 occupancy **값**도 같은 raycast에서 만드는 구조라 새로운 문제가 생겼다:

**FV 카메라의 semantic segmentation이 ego 차량 자신의 본네트(hood)를 "road"(class 7)로
잘못 라벨링한다.** 실측(샘플 00000): FV 이미지 하단(row 766~917)의 RGB에는 "CARLA
Simulator" 워터마크가 박힌 본네트가 선명히 찍혀 있는데, 같은 위치의 `gtLabels`는 class 7
(road, 보라색)로 표시돼 있다. depth도 0.9~1.3m로 이웃 픽셀과 매끄럽게 이어져(불연속 없음)
depth 자체는 정확하다 — **semantic label만 잘못됐다**. 같은 영역을 top-down `_BEV.png`
gtLabel로 보면 ego-vehicle(class 24, 7346px)과 road(class 7, 23095px)가 깨끗이 분리돼
있어, 이 결함이 FV 카메라 자신의 segmentation에만 있음을 확인했다.

이 결함이 그대로 occupancy grid에 스며들어, ego 차량 실루엣 안쪽에 drivable(초록) 얼룩이
생겼다(실측: ego 안쪽 8/128 cell, 인접한 다른 차량 안쪽 21/1103 cell). 차량 실루엣도 카메라
4대의 독립적인 다수결이 경계마다 어긋나 4~5조각으로 끊겨 보였다.

### 3.3 최종 — 하이브리드: class는 top-down label, observed는 raycast (`build_hybrid_occupancy.py`, 현재)

시도 1은 visibility가 노이즈였고, 시도 2는 visibility는 견고했지만 class 값이 노이즈였다.
두 축의 강점만 합친다:

- **occupancy(class) 값** — §2의 `crop_bev_occupancy`(top-down `_BEV.png` label crop+remap)
  그대로 사용. top-down 카메라는 §3.2의 FV 본네트 결함이 없다(위 검증).
- **observed(visible) 값** — `raycast_occupancy.compute_observed_mask`. 시도 2와 같은
  raycast를 쓰되, semantic label은 아예 읽지 않고 "어느 cell에 광선이 도달했는가"만 계산한다.

`tools/build_hybrid_occupancy.py`가 이 조합을 실행한다. 샘플 00000에서 재검증한 결과:

| | 시도 2 (raycast 단독) | 하이브리드 |
|---|---:|---:|
| ego 실루엣 내부 drivable 오염 | 8 / 128 cell | **0** |
| 인접 차량 실루엣 내부 drivable 오염 | 21 / 1103 cell | **0** |
| 인접 차량 실루엣 연결성 | 4~5조각으로 단절 | 단일 연결(685 cell) |

**부가 발견 — obstacle 영역이 "얇은 테두리"가 아니라 넓게 채워지는 이유(정상 동작):** 인접
차량 실루엣에서 top-down label 기준 진짜 풋프린트는 1760 cell인데 그중 1486개(84%)가 실제로
raycast에 "관측됨"으로 잡히고 진짜 안 보이는(차체 바로 밑) 부분은 274개(16%)뿐이다. 차량의
옆면·지붕·보닛까지 카메라에 넓게 잡히면 그 표면들의 언프로젝션 점이 대부분 차량 자신의 지면
풋프린트 안에 떨어지기 때문 — "얇은 경계선만 obstacle"이라는 직관과 달리, 외부에서 잘 보이는
물체는 풋프린트 대부분이 정당하게 obstacle로 채워지는 게 맞다.

### 3.4 남겨둔 잔여 노이즈 — 일부러 정리하지 않음

두 종류의 잔여 아티팩트를 발견했지만, 값이 틀린 게 아니라 판단 여지가 있는 경우라 **정리
후처리를 넣지 않기로 결정**했다(둘 다 재검토 여지는 남겨둠).

1. **1~2 cell짜리 고립된 obstacle 조각** (이미지당 평균 7.6개, 전체 obstacle cell의 약 1%).
   500샘플 중 12개를 무작위로 뽑아 top-down label class를 역추적하면 vegetation(class 9,
   57개) > pole(class 5, 25개) > 차량 경계 조각(class 10, 2개) 순이었다. **크기만으로는
   "무시해도 될 잡초"와 "진짜 지켜야 할 기둥"을 구분할 수 없다** — 크기 기반 정리(작은
   연결요소 제거)는 잡초와 함께 진짜 장애물(기둥)까지 지워버릴 위험이 있어 채택하지 않았다.
2. **ego 차량 자신의 실루엣 경계에서만 두드러지는 salt-and-pepper 잡음**(observed/unknown
   경계). ego 차량은 자기 몸에 붙은 카메라로 자신을 거의 스치듯(grazing angle) 보기 때문에,
   픽셀당 depth 잡음이 5cm 격자 단위로 그대로 드러난다. `observed` mask에 3×3
   `scipy.ndimage.binary_closing`을 시험했고 — 그 과정에서 `border_value` 기본값(0)이
   이미지 최외곽 테두리 전체를 잘못 바꾸는 부작용을 발견해 `border_value=1`로 고쳐야
   한다는 것도 확인했다 — 고친 뒤에는 ego bbox 안쪽 51/25600 cell만 정확히, 다른 물체는
   건드리지 않고 매끈해지는 걸 확인했다. 그럼에도 **적용하지 않기로 결정**했다: 이 cell들은
   observed 여부와 무관하게 어차피 obstacle(ego 차체)이라 실제 학습 신호에는 영향이 거의
   없고, closing은 "여기는 관측이 애매하다"는 사실 자체를 지우는 셈이라 GT의 실측 충실도를
   낮춘다고 판단했다. 학습을 실제로 돌려봐서 이 잡음이 문제가 된다는 증거가 나오면 그때
   재검토한다.

---

## 4. 그리드 스펙이 두 개인 이유

`projects/bev_gt/grid.py`에 두 개의 `OccupancyGridSpec`이 있다. 둘 다 cell 5 cm.

| 스펙 | 범위 | 그리드 | 용도 |
|---|---|---|---|
| `ROBOT_GRID_SPEC` | 전방 4 m / 후방 2 m / 좌우 ±3 m (6m×6m) | 120×120 | 자체 로봇 **fine-tuning** 타깃 |
| `SYNWOODSCAPE_PRETRAIN_GRID_SPEC` | 전방 5 m / 후방 3 m / 좌우 ±4 m (8m×8m) | 160×160 | SynWoodScape **pretraining** |

`ROBOT_GRID_SPEC`은 자체 로봇(소형 플랫폼)의 실제 관심 영역이고 최종 타깃이므로 그대로 둔다.
(구 값 전방3m/후방1m/좌우±1m=4m×2m/80×40은 2026-07-29 최초 설계 초안의 placeholder였고,
아래 6m×6m 결정 이후에도 코드에 반영되지 않은 채 남아 있던 것을 뒤늦게 바로잡았다 — 로봇의
물리적 크기가 아니라 학습 label의 커버리지/해상도를 정하는 값이라, 로봇 자체를 바꾼 게 아니다.)
`SynWoodScape의 ego는 풀사이즈 승용차`(ego 프레임 박스 3.705 m × 1.789 m)라서, 이 그리드를
그대로 SynWoodScape에 적용해 GT를 뽑아도(비교/추적용) 차체가 그리드 상당 부분을 덮는다.

`SYNWOODSCAPE_PRETRAIN_GRID_SPEC`은 원래 전방7m/후방3m/좌우±5m(10m×10m, 200×200)였으나,
로봇 쪽 최종 fine-tuning 타깃이 0.05m/cell × 120×120(=6m×6m, 전방4/후방2/좌우±3)으로 정해지면서
8m×8m(160×160)로 좁혔다. 로봇이 실제로 마주할 근~중거리 스케일에 pretrain 거리 분포를
맞추면서, ego 차체가 그리드를 덮는 비율은 ~10% 선(6m×6m로 그대로 맞췄을 때의 ~18%보다 낮음)으로
유지하기 위한 절충점이다.

실측 (500개 샘플 전부, `tools/build_occupancy_gt.py`의 변동성 요약):

| 스펙 | drivable_fraction mean | std | min | max | 샘플 간 값이 바뀌는 셀 |
|---|---:|---:|---:|---:|---:|
| `robot` (6m×6m, 전방 편향 비대칭) | 0.814 | 0.029 | 0.547 | 0.826 | **65.7%** |
| `synwoodscape_pretrain` (8m×8m) | 0.860 | 0.056 | 0.550 | 0.902 | **80.3%** |

참고로 이전 10m×10m 스펙에서는 87.4%였다 — coverage를 8m×8m로 좁혀도 변동 셀 비율은
크게 줄지 않고(87.4% → 80.3%) 여전히 학습 신호로 충분한 수준을 유지한다.
`robot`도 구 값(4m×2m, 17.8%)일 때는 표준편차가 평균의 0.5%로 사실상 상수 GT였으나, 실제
결정값인 6m×6m로 바로잡은 뒤에는 변동 셀 비율이 65.7%로 크게 늘었다 — ego 차체가 차지하는
면적 비율이 half_width 1m(그리드 폭 2m)에서 3m(그리드 폭 6m)로 넓어지며 상대적으로 작아졌기
때문이다.

**주의 — 이후 학습 단계에서 다룰 것:** 두 그리드의 크기(120×120 vs 160×160)가 다르므로,
pretrain → fine-tune 전이 시 head의 출력 해상도/좌표 정규화를 어떻게 맞출지는 다음 스펙
(Simple-BEV 통합)에서 결정해야 한다. 이 노트는 GT 생성까지만 다룬다.

---

## 5. 재현 방법

```bash
conda activate bev-chamdog

# Phase 1: 어안 투영 검증 (오버레이 + 3개 지표)
python tools/verify_fisheye_projection.py --samples 00000 00001 00002

# Phase 2-a: BEV 상수 검증 (원점 / 부호 / 스케일 sweep / 실루엣 편향 시연)
python tools/calibrate_bev_scale.py

# Phase 2.5: 현재 파이프라인 — occupancy(top-down label) + visible(raycast) 하이브리드 GT
# 학습용 .npy까지 만들려면 --review-only를 빼고(기본이 이미 만듦), 500장 전부는 --num-samples 500
python tools/build_hybrid_occupancy.py --samples 00000 00001 00002

# 회귀 테스트 (데이터셋이 없으면 실데이터 테스트는 skip된다)
pytest -q
```

`tools/build_occupancy_gt.py`(§2, top-down label만·visibility 없음)와 `tools/
build_visibility_mask.py`(§3.1)/`tools/build_raycast_occupancy.py`(§3.2)는 폐기된
중간 단계 스크립트로 코드에는 남아 있지만 **더 이상 최종 GT를 만드는 데 쓰지 않는다** —
재현·비교 목적이 아니면 `build_hybrid_occupancy.py`만 실행하면 된다.

---

## 6. 남은 사실 관계 메모

- **`_BEV.png` 카메라는 ego와 함께 회전한다** (world 고정이 아니다). world yaw가 0°/88°/91°/147°인
  샘플들에서 ego 차량 bbox가 항상 같은 크기·같은 위치(정중앙 66×132 px)로 나온다 → world yaw
  보정이 불필요하고, 이미지 축이 곧바로 ego의 전후/좌우 축이다.
- **BEV depth는 평면 z-depth**, 어안 depth는 **방사 거리**다. 서로 다르다.
  (어안: 노면 픽셀로 평면을 맞추면 방사 거리 해석에서 법선 오차 0.2° 이내, z-depth 해석에서는 발산.
  BEV: 노면 depth가 이미지 전체에서 14.94~15.05로 거의 일정 → 평면 z-depth.)
- **3 m 이내 LiDAR 포인트 ~13,000개 중 ~12,700개는 ego 차체 self-hit**이고, CARLA semantic
  lidar는 이를 class 24가 아니라 **class 10**으로 라벨링한다. 카메라가 차체에 달려 있어
  `z_cam > 0` 검사에서 전부 걸러지므로 투영 지표에는 영향이 없다. 실제 지면(class 7) 리턴은
  5 m 이내 약 15,000개, 10 m 이내 약 25,000개로 충분하다.
- **occupancy GT에는 LiDAR를 쓰지 않는다.** 불가능해서가 아니라(근접 리턴은 충분하다) BEV
  semantic 이미지가 같은 목적에 **더 dense하고 단순**해서다. LiDAR 래스터화는 링 패턴 때문에
  5 cm 셀에 구멍이 생기고 보간 로직이 추가로 필요하다. 높이 정보나 occlusion 추론이
  필요해지면 재검토한다.
