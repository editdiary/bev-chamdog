# SynWoodScape 기하 검증 결과 노트 (Phase 1 · Phase 2)

- 작성일: 2026-07-30
- 대상 스펙: [`docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md`](../superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md)
- 대상 코드: `projects/geometry/`, `projects/bev_gt/`, `tools/verify_fisheye_projection.py`,
  `tools/calibrate_bev_scale.py`, `tools/build_occupancy_gt.py`

> 목적: SynWoodScape의 어안 투영(Phase 1)과 BEV occupancy GT 생성(Phase 2)을 구현·검증하면서
> **실측으로 밝혀낸 것들**을 남긴다. 특히 **데이터셋 문서가 말해주지 않거나 잘못 말해주는**
> 항목(extrinsic 부호, LiDAR 좌표 규약, BEV 카메라 스케일)에 집중한다.
> 모든 수치는 로컬 데이터(`dataset/synwoodscape/SynWoodScape_V0.1.0/`)를 직접 읽어 측정했다.

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

## 3. 그리드 스펙이 두 개인 이유

`projects/bev_gt/grid.py`에 두 개의 `OccupancyGridSpec`이 있다. 둘 다 cell 5 cm.

| 스펙 | 범위 | 그리드 | 용도 |
|---|---|---|---|
| `ROBOT_GRID_SPEC` | 전방 3 m / 후방 1 m / 좌우 ±1 m | 80×40 | 자체 로봇 **fine-tuning** 타깃 |
| `SYNWOODSCAPE_PRETRAIN_GRID_SPEC` | 전방 7 m / 후방 3 m / 좌우 ±5 m | 200×200 | SynWoodScape **pretraining** |

`ROBOT_GRID_SPEC`은 자체 로봇(소형 플랫폼)의 실제 관심 영역이고 최종 타깃이므로 그대로 둔다.
그러나 **SynWoodScape의 ego는 풀사이즈 승용차**(ego 프레임 박스 3.705 m × 1.789 m)라서
4 m × 2 m 그리드의 절반 이상이 ego 자신의 차체에 덮인다. 그 결과 GT가 거의 상수가 된다.

실측 (500개 샘플 전부, `tools/build_occupancy_gt.py`의 변동성 요약):

| 스펙 | drivable_fraction mean | std | min | max | 샘플 간 값이 바뀌는 셀 |
|---|---:|---:|---:|---:|---:|
| `robot` | 0.411 | 0.002 | 0.389 | 0.412 | 17.8% |
| `synwoodscape_pretrain` | 0.843 | 0.084 | 0.575 | 0.937 | **87.4%** |

`robot`은 표준편차가 평균의 0.5%로, 500샘플 전체에서 사실상 같은 그림이다. 즉 이 그리드로
SynWoodScape pretraining을 하면 모델이 "항상 이 모양"을 외우는 것으로 수렴한다.
그래서 pretraining에는 차량 스케일에 맞는 넓은 그리드를 쓰고, `ROBOT_GRID_SPEC`은 자체 로봇
데이터 fine-tuning 단계로 미룬다.

**주의 — 이후 학습 단계에서 다룰 것:** 두 그리드의 크기(80×40 vs 200×200)가 다르므로,
pretrain → fine-tune 전이 시 head의 출력 해상도/좌표 정규화를 어떻게 맞출지는 다음 스펙
(Simple-BEV 통합)에서 결정해야 한다. 이 노트는 GT 생성까지만 다룬다.

---

## 4. 재현 방법

```bash
conda activate bev-chamdog

# Phase 1: 어안 투영 검증 (오버레이 + 3개 지표)
python tools/verify_fisheye_projection.py --samples 00000 00001 00002

# Phase 2-a: BEV 상수 검증 (원점 / 부호 / 스케일 sweep / 실루엣 편향 시연)
python tools/calibrate_bev_scale.py

# Phase 2-b: occupancy GT 생성 (두 그리드 스펙 + 변동성 요약)
python tools/build_occupancy_gt.py --samples 00000 00001 00002

# 회귀 테스트 (데이터셋이 없으면 실데이터 테스트는 skip된다)
pytest -q
```

---

## 5. 남은 사실 관계 메모

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
