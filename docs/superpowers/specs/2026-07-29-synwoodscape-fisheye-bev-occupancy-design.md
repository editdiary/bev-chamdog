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

### 3.3 좌표계 규약 (2026-07-30 전면 수정)

> ⚠️ **이 절의 초판 내용은 틀렸다.** "LiDAR pkl의 `points`가 world 좌표계에 있다"고 적었으나
> 실제로는 **이미 센서 로컬 좌표**다. 이 오해 때문에 초기 구현이 `inverse(transform)`을
> 불필요하게 곱해 semantic consistency rate가 0.001~0.451까지 떨어졌다.
> 자세한 조사 기록: `.superpowers/sdd/2026-07-29-synwoodscape-fisheye-bev-occupancy/task-5-investigation-report.md`

확정된 규약 (모두 SynWoodScape 자체 파일로 실측 확인):

- **CARLA 프레임**(world / vehicle): X=전방, Y=**우측**, Z=상방, **왼손계**.
  `vehicle_data/rgb_images/*.txt`의 `Transform`, `lidar_data/*.pkl`의 `transform`,
  `_unuserd/box_3d_annotations/*.pkl`의 코너가 모두 이 프레임이다.
- **ego 프레임**(작업 프레임): X=전방, Y=**좌측**, Z=상방, **오른손계**.
  원점은 CARLA vehicle actor 원점(차량 중심, 지면 높이)과 같다.
  `calibration_data/*.json`의 extrinsic이 카메라를 이 프레임으로 보내므로 여기에 맞춘다.
- `transform` = **world_T_lidar**. 회전부는 같은 샘플 `vehicle_data`가 보고하는
  `Rotation(pitch,yaw,roll)`의 CARLA `get_matrix()`와 오차 1e-6 이내로 일치하고,
  translation은 vehicle Location + (0,0,2.0)이다 → LiDAR는 차량과 같은 방향으로
  ego 기준 (0,0,2.0)에 장착.
- **`points`는 센서 로컬 좌표이고, 축 규약도 CARLA(왼손)가 아니라 ego와 같은 오른손(Y=좌측)** 이다.
  따라서 ego frame 변환은 **장착 오프셋 덧셈뿐**이다: `points_ego = points + (0,0,2.0)`.
  (`transform`은 world 좌표가 필요할 때만 쓴다.)
- **`calibration_data/*.json`의 extrinsic은 그대로 쓰면 안 된다** — 회전은 roll 부호가
  뒤집혀 있고, 미러 카메라(MVL/MVR) translation은 y/z 성분이 전치되어 있다.
  보정 규칙과 근거는 `projects/geometry/fisheye.py` 모듈 docstring에 정리했다.

### 3.4 검증 절차

1. **좌표계 규약 재확인**: 여러 샘플에서 §3.3의 관계(x/y 일치, z 오프셋 일관성)를 재확인한다.
2. LiDAR 포인트(+ per-point semantic label)를 ego frame → 각 카메라(FV/MVL/MVR/RV)의 camera frame으로 변환(카메라 extrinsic: quaternion + translation).
3. `project()`로 픽셀 좌표를 계산하고, 실제 fisheye RGB 이미지 위에 라벨 색상으로 오버레이하여 도로/차선/보도 등 경계가 이미지 내용과 육안상 일치하는지 확인한다.
4. **정량 지표 — semantic consistency rate**: 재투영된 포인트 위치의 `semantic_annotations/gtLabels` 픽셀 class와 해당 LiDAR point의 label이 얼마나 일치하는지 계산한다. (대응 코너점 기반 재투영 오차 GT가 없으므로 이 방식으로 대체.)
5. 보조 정성 검증: WoodScape 툴킷 스타일로 지면(z=0)에 1m 간격 격자점을 넓은 범위(로봇의 실제 occupancy 그리드보다 넓게)로 만들어 fisheye 이미지에 투영해 시각적으로 확인한다. 좁은 범위(전방 3m/후방 1m)는 투영 오류가 시각적으로 잘 드러나지 않으므로, 이 검증 단계에서는 의도적으로 더 넓은 범위를 쓴다.
6. **unproject 검증**: `project()`로 얻은 픽셀을 다시 `unproject()`하여 원래 3D 방향(광선)과 일치하는지 수치 라운드트립 테스트를 수행한다.
7. **경계 처리**: θ가 다항식 발산 영역(≈90도 근방)에 가까운 포인트, FOV 밖으로 나가는 포인트는 invalid로 마스킹하고 별도 로그로 남긴다.

### 3.5 완료 기준 (2026-07-30 수정)

- 여러 샘플(예: 10개 이상)에서 오버레이 결과가 육안상 정상
- ~~semantic consistency rate 90% 이상~~ → **달성 불가능한 기준이었다.**
  LiDAR는 ego z=2.0, 카메라는 z=0.9~1.0에 있어서 **LiDAR에는 보이지만 카메라에는 가려지는**
  포인트가 구조적으로 20~35% 존재한다. 그 포인트들은 가림 물체의 class 위에 떨어지므로
  투영이 완벽해도 이 지표는 0.9에 닿지 않는다.

  대신 아래 두 지표를 쓴다. **임계값은 42개 샘플 × 4대 = 168쌍**을 실측해 정했다
  (`range(0, 500, 12)`). 초판에서 제안한 `depth_agreement ≥ 0.65`는 샘플 3개에만 맞춘
  값이라 **168쌍 중 22%가 미달**해 폐기했다.

  | 지표 | 실측 (168쌍) | 카메라별 median 실측 | 채택 기준 |
  |---|---|---|---|
  | `depth_agreement` | min 0.433 / p5 0.596 / median 0.689 / max 0.864 | FV 0.663, MVL 0.702, MVR 0.666, RV 0.773 | **카메라별 median ≥ 0.60** |
  | `visible_consistency_rate` | min 0.897 / p5 0.926 / median 0.973 / max 0.996 | FV 0.946, MVL 0.981, MVR 0.981, RV 0.974 | **카메라별 median ≥ 0.93** 이고 **쌍별 ≥ 0.85** |

  **왜 쌍별(per-pair) floor이 아니라 median인가**: `depth_agreement`는 캘리브레이션 품질만이
  아니라 **장면 내용**(그 시점에 카메라가 얼마나 가려져 있는지)에 좌우된다. 교차로에서 옆 차량이
  시야를 막으면 투영이 완벽해도 값이 내려간다. 따라서 쌍별 하한은 장면 난이도에 대한 기준이
  되어버리고, 캘리브레이션 회귀를 잡는 데는 여러 샘플의 median이 적합하다.
  `visible_consistency_rate`의 기준을 median 0.93 / 쌍별 0.85로 잡은 것도 같은 이유다
  (FV의 168쌍 median이 0.946, 쌍별 최소가 0.897이라 0.94/0.89로는 여유가 1%p 미만이었다).

  **median 기준의 판별력은 부분적이다** — 미러 카메라의 y 부호를 뒤집어(카메라를 반대쪽에 놓아)
  측정하면 42샘플 median이 MVL 0.584 / MVR 0.417로 떨어진다. 그런데 회귀 테스트가 쓰는
  6개 샘플에서는 MVL이 **0.600으로 기준선에 딱 걸린다**. 즉 `depth_agreement`는 카메라
  **높이** 오차에는 민감하지만 **횡방향** 오차에는 신뢰할 만한 판별력이 없다. 그래서 좌우 배치는
  아래의 전용 기준으로 따로 검사한다.

- **미러 카메라 좌우(y) 배치**는 위 지표로 충분히 걸러지지 않으므로 별도 기준을 둔다:
  각 카메라의 depth map에서 `ego-vehicle`(class 24) 픽셀을 역투영했을 때의 ego 프레임 y 중앙값이
  **MVL > +0.3, MVR < −0.3, FV·RV는 |y| < 0.25**여야 한다. 실측값은 MVL +0.756, MVR −0.757,
  FV −0.082, RV −0.001이고 42개 샘플에 걸쳐 ±0.001 이내로 안정적이다. 좌우를 뒤집으면
  MVL −1.044 / MVR +1.043이 되어 즉시 탈락한다.
- project→unproject 라운드트립 오차가 허용 오차(예: 서브픽셀) 이내

### 3.6 알아둘 점 — LiDAR 근접 커버리지

> ⚠️ **이 절의 초판 관찰도 §3.3의 오해에서 나온 것이라 틀렸다.** "차량 근접 포인트가 0개"라는
> 결론은 `points`를 world 좌표로 착각한 채 거리를 재서 나온 것이다. `points`는 센서 로컬
> 좌표이므로 실제 센서 거리는 **원점 기준**으로 재야 하고, 그렇게 재면 거리 중앙값이 약 5.7 m,
> 최소값이 0.63 m다. 즉 **근접 포인트는 충분히 존재한다.**

정정된 내용 (샘플 00000/00001/00002 실측):

- 3 m 이내 포인트는 약 13,000개지만 그중 ~12,700개는 **ego 차량 자신의 차체 self-hit**이다
  (샘플마다 개수가 거의 일정하고, ego 3D 박스 안에 들어가며, CARLA semantic lidar가 ego를
  class 24가 아니라 class 10으로 라벨링한다). 카메라가 차체에 달려 있어 이 포인트들은
  `z_cam > 0` 검사에서 전부 걸러지므로 투영 지표에는 영향이 없다.
- 실제 지면(class 7) 리턴은 3 m 이내 ~400개, **5 m 이내 약 15,000개, 10 m 이내 약 25,000개**다.
  즉 근접 지면 커버리지는 충분하다.

따라서 §4.4에서 "근접 LiDAR 포인트가 없어서 래스터화 소스로 쓸 수 없다"고 한 근거는 무효다.
다만 §4의 최종 설계(BEV semantic 이미지에서 occupancy GT를 만드는 방식)는 그 자체로 더 단순하고
dense하므로 **설계 변경 없이 유지**한다 — LiDAR 래스터화는 필요해지면 그때 재검토한다.

Phase 1(§3.4)의 검증 방식에는 영향이 없다 — 투영 검증은 카메라 FOV 안에 들어오는 어떤 거리의 포인트든 유효하고, 정정 후 실제로 근거리 노면부터 원거리 건물/기둥까지 다양한 class로 검증했다(§3.5의 지표 참고).

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
3. LiDAR는 이 근접 그리드 생성에 관여하지 않는다. LiDAR의 역할은 Phase 1(§3)의 fisheye project/unproject 검증으로 한정한다.

   **이유 (2026-07-30 정정).** 초판의 근거였던 "근접 LiDAR 포인트가 사실상 없다"는 **틀렸다**
   (§3.6 — 실제로는 5 m 이내 노면 리턴이 약 15,000개 있다). LiDAR를 쓰지 않는 것은 여전히
   타당하지만 근거가 다르다:
   - BEV semantic 이미지는 ROI 전체가 **빽빽하게 채워진** 라벨이다. LiDAR 래스터화는 링 패턴
     때문에 셀 사이에 빈 곳이 생기고, 5 cm 셀에서는 그 구멍을 메우는 보간 로직이 추가로 필요하다.
   - 3 m 이내 포인트의 대부분(~12,700개)은 **ego 차체 self-hit**이라 별도 필터링이 필요하다.
   - drivable/non-drivable 판정에 필요한 것은 표면 class이고, 그건 BEV 라벨 이미지에 이미 있다.
     LiDAR per-point label을 쓰면 같은 정보를 더 sparse하게 얻는 셈이다.

   즉 "불가능해서"가 아니라 **"BEV 이미지가 같은 목적에 더 dense하고 단순해서"** 쓰지 않는다.
   향후 높이 정보나 occlusion 추론이 필요해지면 LiDAR를 다시 검토할 수 있다.

### 4.5 완료 기준 (제안, 조정 가능)

- §4.3의 스케일 추정이 서로 다른 샘플/instance에서 일관됨(예: 표준편차가 평균의 5% 이내)
- ego-vehicle bbox 중심이 여러 샘플에서 이미지 중앙 ±1px 이내로 일관됨
- 최종 occupancy 그리드를 몇 개 샘플에 대해 시각화했을 때 도로 형태가 육안상 타당함

## 5. 산출물

- 코드
  - `projects/geometry/frames.py` — 동차좌표 변환 헬퍼, LiDAR 센서로컬→ego 변환(`lidar_points_to_ego`), CARLA↔ego 규약 변환, 좌표계 규약 회귀 테스트
  - `projects/geometry/reprojection.py` — ego 포인트 → fisheye 픽셀 투영, depth 기반 가시성 마스크
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
