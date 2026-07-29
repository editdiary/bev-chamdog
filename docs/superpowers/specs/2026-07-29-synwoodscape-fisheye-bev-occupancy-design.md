# SynWoodScape Fisheye 투영 검증 및 BEV Occupancy GT 설계

- 작성일: 2026-07-29
- 관련 프로젝트 단계: README.md의 "3. SynWoodScape로 학습" 이전 준비 단계
- 다음 스펙과의 관계: 이 스펙의 산출물(검증된 fisheye project/unproject 함수, occupancy GT 생성 파이프라인)은 이후 "Simple-BEV 모델 통합 및 학습" 스펙의 입력이 된다.

## 1. 배경

프로젝트의 목표는 자체 구축한 fisheye 4-cam 데이터셋으로 **BEV occupancy map(drivable/non-drivable)** 을 예측하는 모델을 학습하는 것이며, baseline은 Simple-BEV다. 학습은 SynWoodScape로 먼저 검증한 뒤 자체 데이터셋으로 fine-tuning하는 순서를 따른다.

Simple-BEV(`third_party/models/simple_bev`)는 핀홀 카메라 모델을 전제로 구현되어 있다. 구체적으로 `utils/vox.py`의 `unproject_image_to_mem()`은 3D voxel 좌표를 이미지 픽셀로 투영할 때 `apply_4x4(pixB_T_camA, xyz_camA)`(4x4 행렬곱) 후 `xy = xyz[:2] / z`(원근분할)를 사용한다. 이는 핀홀 투영 공식(`u = fx·X/Z + cx`) 그 자체이며, `utils/geom.py`의 `camera2pixels`/`pixels2camera`도 동일한 가정 위에 있다.

반면 SynWoodScape의 4대 어안 카메라(FV/MVL/MVR/RV) 캘리브레이션(`calibration_data/*.json`)은 **`radial_poly`** 모델(`r(θ) = k1·θ + k2·θ² + k3·θ³ + k4·θ⁴`, 4차, θ=입사각)을 사용한다. 이는 비선형 r=f(θ) 계열로, 핀홀 행렬곱+원근분할로는 표현할 수 없다.

따라서 "모델만 어안으로 바꾸면 된다"는 가정은 정확하지 않다. Simple-BEV의 핵심 lifting 연산(`vox.py`/`geom.py`의 투영 함수)을 `radial_poly` 기반 project/unproject로 교체해야 하며, 그 전에 이 새 투영 함수 자체가 올바른지 검증이 필요하다.

## 2. 범위

이번 스펙은 다음 두 단계까지만 다룬다.

- **Phase 1**: `radial_poly` project/unproject 함수 구현 및 검증 (LiDAR 재투영 이용)
- **Phase 2**: SynWoodScape로부터 BEV occupancy GT(drivable/non-drivable) 생성 (BEV 이미지 스케일 보정 + 크롭)

**범위 밖** (다음 스펙에서 다룸):
- Simple-BEV `vox.py`/`geom.py`에 위 투영 함수를 실제로 통합하는 모델 래핑 작업
- 학습 데이터로더·학습 스크립트 구현
- 자체 데이터셋 fine-tuning

## 3. Phase 1 — Fisheye 투영/역투영 검증

### 3.1 목표

`projects/`에 `radial_poly` project(3D→2D)/unproject(2D→ray) 함수를 새로 구현하고, SynWoodScape의 실제 캘리브레이션·LiDAR 데이터로 정확성을 검증한다.

### 3.2 구성요소

- `projects/geometry/fisheye.py`
  - calibration JSON 파싱: extrinsic(quaternion + translation), intrinsic(`radial_poly`, k1~k4, cx_offset/cy_offset, aspect_ratio)
  - `project(xyz_cam) -> (u, v, valid_mask)`: 카메라 좌표계 3D 점 → 픽셀 좌표
  - `unproject(u, v) -> ray_cam`: 픽셀 좌표 → 카메라 좌표계 광선 방향
  - project→unproject 라운드트립에 대한 순수 수치 단위 테스트(실데이터 불필요, 함수 구현 자체의 정합성 검증용)
- `tools/verify_fisheye_projection.py`
  - 실데이터(캘리브레이션 + LiDAR)를 이용한 검증 스크립트

### 3.3 좌표계 규약 (확인 완료)

기존에는 LiDAR pkl(`lidar_data/*.pkl`)의 `transform`(4x4) 필드가 어떤 변환인지 불명확했으나, `vehicle_data/rgb_images/*.txt`와 대조해 확인했다.

- `vehicle_data/rgb_images/00000.txt`의 world pose: `Location(x=-111.032127, y=80.943199, z=-0.008363)`
- `lidar_data/00000.pkl`의 `transform` translation: `(-111.034286, 80.937408, 1.991628)`

x, y는 거의 일치하고 z만 약 2m 차이가 나며, 이는 readme.txt에 명시된 "LiDAR는 ego 기준 x=0, y=0, z=2.0에 장착"과 정확히 일치한다. 따라서:

- `transform` = **world_T_lidar** (LiDAR 센서 좌표계 → world 좌표계)
- LiDAR pkl의 `points`는 world 좌표계에 있음
- ego frame 변환 절차: `points_world` → `inverse(transform)` → `points_lidar` → readme의 lidar 장착 오프셋(ego 기준 x=0,y=0,z=2, 회전 없음 가정)을 보정 → `points_ego`

이 관계는 여러 샘플에서 일관되는지 재확인하는 것으로 검증을 대체하며(예: 5~10개 샘플에 대해 x/y 일치, z 차이가 ~2m로 일정한지 확인), 별도의 추가 조사는 필요하지 않다.

### 3.4 검증 절차

1. **좌표계 규약 재확인**: 여러 샘플에서 §3.3의 관계(x/y 일치, z 오프셋 일관성)를 재확인한다.
2. LiDAR 포인트(+ per-point semantic label)를 ego frame → 각 카메라(FV/MVL/MVR/RV)의 camera frame으로 변환(카메라 extrinsic: quaternion + translation).
3. `project()`로 픽셀 좌표를 계산하고, 실제 fisheye RGB 이미지 위에 라벨 색상으로 오버레이하여 도로/차선/보도 등 경계가 이미지 내용과 육안상 일치하는지 확인한다.
4. **정량 지표 — semantic consistency rate**: 재투영된 포인트 위치의 `semantic_annotations/gtLabels` 픽셀 class와 해당 LiDAR point의 label이 얼마나 일치하는지 계산한다. (대응 코너점 기반 재투영 오차 GT가 없으므로 이 방식으로 대체.)
5. 보조 정성 검증: WoodScape 툴킷 스타일로 지면(z=0)에 1m 간격 격자점을 넓은 범위(로봇의 실제 occupancy 그리드보다 넓게)로 만들어 fisheye 이미지에 투영해 시각적으로 확인한다. 좁은 범위(전방 3m/후방 1m)는 투영 오류가 시각적으로 잘 드러나지 않으므로, 이 검증 단계에서는 의도적으로 더 넓은 범위를 쓴다.
6. **unproject 검증**: `project()`로 얻은 픽셀을 다시 `unproject()`하여 원래 3D 방향(광선)과 일치하는지 수치 라운드트립 테스트를 수행한다.
7. **경계 처리**: θ가 다항식 발산 영역(≈90도 근방)에 가까운 포인트, FOV 밖으로 나가는 포인트는 invalid로 마스킹하고 별도 로그로 남긴다.

### 3.5 완료 기준 (제안, 조정 가능)

- 여러 샘플(예: 10개 이상)에서 오버레이 결과가 육안상 정상
- semantic consistency rate 90% 이상
- project→unproject 라운드트립 오차가 허용 오차(예: 서브픽셀) 이내

### 3.6 알아둘 점 — LiDAR 근접 커버리지

실제 데이터 확인 결과, SynWoodScape LiDAR는 **차량 근접(0~5m) 포인트가 여러 샘플에서 일관되게 0개**다(예: 5개 샘플 모두 5m 이내 0개, 20m 이내도 수~수백 개 수준). world 좌표계에서 차량 위치까지의 순수 거리로 확인했으므로 좌표계 변환 버그가 아니라 센서 스펙(수직 FOV/range gate)에 의한 특성으로 보인다.

이는 Phase 1(§3.4)의 검증 방식에는 영향이 없다 — 투영 검증은 카메라 FOV 안에 들어오는 어떤 거리의 포인트든(원거리 포함) 유효하며, 실제로 22~536m 범위의 포인트로도 건물/기둥 등 다양한 class에 대해 검증 가능함을 확인했다. 다만 **Phase 2에서 LiDAR를 근접 occupancy 그리드의 직접적인 래스터화 소스로 쓸 수 없다는 뜻이며, 이는 §4의 설계에 반영되어 있다.**

## 4. Phase 2 — BEV Occupancy GT 설계

### 4.1 그리드 정의

- 종방향: 전방 3m + 후방 1m = 4m
- 횡방향: 좌우 ±1m = 2m
- cell size: 5cm × 5cm
- 그리드 크기: 80(종방향) × 40(횡방향), ego 원점 기준 비대칭(전방 편향) 그리드

이 범위는 이후 자체 로봇 데이터셋과 스케일을 맞추기 위한 값이며(README의 pretrain→fine-tune 전략), SynWoodScape(차량 스케일 도로 장면) 위에서의 투영 **검증**에는 §3.4의 5단계처럼 별도의 넓은 범위를 쓴다. 즉 검증용 범위와 최종 occupancy GT용 그리드 범위는 분리한다.

### 4.2 Drivable class 정의

SynWoodScape semantic palette 기준:

| class id | 이름 | occupancy |
|---|---|---|
| 6 | road line | drivable |
| 7 | road | drivable |
| 그 외 전부 | - | non-drivable |

이 매핑은 `projects/bev_gt/`의 상수/설정으로 분리해, 추후 자체 로봇(보도 포함 등)에 맞게 바꾸기 쉽게 한다.

### 4.3 BEV 이미지 스케일/원점 보정 (1회성)

**문제**: `_BEV.png`(top-down semantic 라벨)를 만든 가상 BEV 카메라의 intrinsic(화각/스케일)이 `calibration_data`에도 `vehicle_data`에도 없다. readme.txt에는 위치(z=15, pitch=-90)만 명시돼 있다.

**해결책 (실측으로 확인됨)**: LiDAR가 아니라, 데이터셋에 이미 있는 **instance-level BEV 라벨 + 3D 박스**로 직접 역산한다.

- `_unuserd/instance_annotations/gtLabels/*_BEV.png` (1024×1024, grayscale)의 픽셀 값은 **instance id**다.
- `_unuserd/box_3d_annotations/*.pkl`은 `{instance_id: (8,4) 3D 코너 좌표}`를 담고 있다.
- 샘플 00000에서 instance id `24`가 **ego-vehicle 자신**이었고(semantic class 24=ego-vehicle과 픽셀 bbox 완전 일치, 66×132px, 중심 (511.5,511.5) = 이미지 정중앙), 그 3D 박스 치수(X=1.9m, Y=3.76m)로 스케일을 역산하면 **폭 기준 0.0288 m/px, 길이 기준 0.0285 m/px로 두 축이 거의 일치**한다 → 등방 스케일의 top-down 투영, 약 0.0286 m/px (1024px ≈ 29m 정사각형 커버리지).
- 4개 샘플(world yaw 0°/88°/91°/147°, 서로 다른 헤딩)에서 ego-vehicle bbox가 **항상 정확히 같은 크기·같은 위치(이미지 정중앙)** 로 나옴을 확인했다 → 이 BEV 카메라는 world 고정이 아니라 **ego 차량과 함께 회전하는(ego-relative) 카메라**다. 따라서 world yaw 보정이 불필요하고, 이미지 축이 곧바로 ego의 전후/좌우 축이다.

**절차**:
1. 여러 샘플 × 여러 instance(3D 박스가 있는 것)에서 BEV 픽셀 bbox 크기와 3D 박스 치수를 짝지어 스케일(m/px)을 계산하고 평균·표준편차로 신뢰도를 확인한다.
2. ego-vehicle(semantic class 24) bbox 중심이 여러 샘플에서 일관되게 이미지 정중앙인지 재확인해 원점을 확정한다.
3. 이미지 축(가로=폭/좌우, 세로=길이/전후) 중 어느 쪽이 전방(+)인지는 `rgb_images/*_BEV.png`(컬러) 육안 확인 또는 `vehicle_data`의 velocity 벡터 방향과 대조해 부호를 정한다.
4. 확정된 스케일·원점·축 매핑을 `projects/bev_gt/`의 상수로 고정한다(샘플마다 다시 계산하지 않음 — 동일 가상 카메라이므로 데이터셋 전체에 대해 한 번만 구하면 된다).

### 4.4 근접 occupancy GT 생성

1. §4.3에서 확정한 스케일/원점/축으로 `semantic_annotations/gtLabels/*_BEV.png`에서 §4.1 ROI(전방3m/후방1m/좌우±1m)에 해당하는 픽셀 영역을 잘라낸다.
2. §4.2의 class 매핑(road/road line → drivable, 나머지 → non-drivable)을 그대로 적용해 5cm 셀 그리드로 리샘플링한다.
3. LiDAR는 이 근접 그리드 생성에 관여하지 않는다(§3.6 — 근접 LiDAR 포인트가 사실상 없어 래스터화 소스로 부적합하다고 확인됨). LiDAR의 역할은 Phase 1(§3)의 fisheye project/unproject 검증으로 한정한다.

### 4.5 완료 기준 (제안, 조정 가능)

- §4.3의 스케일 추정이 서로 다른 샘플/instance에서 일관됨(예: 표준편차가 평균의 5% 이내)
- ego-vehicle bbox 중심이 여러 샘플에서 이미지 중앙 ±1px 이내로 일관됨
- 최종 occupancy 그리드를 몇 개 샘플에 대해 시각화했을 때 도로 형태가 육안상 타당함

## 5. 산출물

- 코드
  - `projects/geometry/frames.py` — 동차좌표 변환 헬퍼, LiDAR world→ego 변환, 좌표계 규약 회귀 테스트
  - `projects/geometry/fisheye.py` — radial_poly project/unproject (WoodScape 공식 구현 wrapping)
  - `projects/bev_gt/` — occupancy 변환(class remap, 그리드 스펙, BEV 이미지 스케일/원점 보정, ROI crop)
  - `tools/verify_fisheye_projection.py` — Phase 1 검증 스크립트 (LiDAR 재투영 오버레이 + semantic consistency rate)
  - `tools/calibrate_bev_scale.py` — §4.3의 스케일/원점 보정을 여러 샘플에 대해 수행하고 상수를 도출하는 스크립트
  - `tools/build_occupancy_gt.py` — Phase 2 occupancy GT 배치 생성 + 시각화
- 문서
  - 이 design doc
  - Phase 1/2 실행 결과를 기록할 짧은 findings 노트(`docs/dataset_analysis/` 또는 `docs/study/`에 추가 — 실행 후 작성)

## 6. 열린 질문 / 향후 확인 사항

- §4.3의 스케일은 샘플 00000의 instance 1개로 처음 확인했다. 더 많은 샘플/instance로 평균을 내 견고성을 확인해야 한다.
- 이미지 축의 전방(+) 방향 부호는 아직 육안/velocity 대조로 확정하지 않았다 — `tools/calibrate_bev_scale.py` 실행 시 확정한다.
- `_unuserd/instance_annotations`, `_unuserd/box_3d_annotations`는 이 보정 단계에서만 참조하고, 최종 occupancy GT 파이프라인(`tools/build_occupancy_gt.py`)은 `semantic_annotations`만 사용한다.
