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

### 4.1 그리드 정의 (2026-07-30 수정: 그리드 스펙이 **두 개**다)

> ⚠️ **이 절의 수치도 그 뒤 다시 바뀌었다 (2026-07-30 재수정, `docs/dataset_analysis/
> synwoodscape_geometry_findings.md` §4).** 아래 표의 80×40 / 200×200은 이 문서 최초
> 수정 시점의 값이고, **현재 코드(`projects/bev_gt/grid.py`)의 확정값은 다르다**:
> `ROBOT_GRID_SPEC` = 전방4m/후방2m/좌우±3m(6m×6m) **120×120**,
> `SYNWOODSCAPE_PRETRAIN_GRID_SPEC` = 전방5m/후방3m/좌우±4m(8m×8m) **160×160**.
> 또한 실제로 GT를 생성해 `dataset/synwoodscape_occupancy_gt/`에 저장하는 건 현재
> **`synwoodscape_pretrain` 하나뿐**이다 — `robot`은 자체 로봇 fine-tuning 단계용으로
> 코드에는 남아 있지만 SynWoodScape에 대해 생성하지 않는다. 최신 수치·이유는 findings
> 노트를 참고할 것 — 아래는 이 문서가 처음 이 결정을 내렸을 때의 기록이라 그대로 둔다.

초판은 로봇 스케일 그리드 하나만 두었는데, 실제로 생성해 보니 SynWoodScape에서는 그 그리드가
**거의 상수 GT**를 만든다. 그래서 스펙을 둘로 나눴다 (`projects/bev_gt/grid.py`).

| 상수 | 범위 (이 문서 작성 당시) | 그리드 | 용도 |
|---|---|---|---|
| `ROBOT_GRID_SPEC` | 전방 3m / 후방 1m / 좌우 ±1m | 80×40 | 자체 로봇 **fine-tuning** 타깃 (초판의 값, 그대로 유지) |
| `SYNWOODSCAPE_PRETRAIN_GRID_SPEC` | 전방 7m / 후방 3m / 좌우 ±5m | 200×200 | SynWoodScape **pretraining** GT |

둘 다 cell size 5cm × 5cm, ego 원점 기준 비대칭(전방 편향) 그리드다.

**왜 나눴나** — SynWoodScape의 ego는 **풀사이즈 승용차**(ego 프레임 3D 박스 3.705 m × 1.789 m)라서
4m × 2m 그리드의 절반 이상이 ego 자신의 차체에 덮인다. 500개 샘플 전부에 대한 실측:

| 스펙 | drivable_fraction mean | std | min | max | 샘플 간 값이 바뀌는 셀 |
|---|---:|---:|---:|---:|---:|
| `robot` | 0.411 | 0.002 | 0.389 | 0.412 | 17.8% |
| `synwoodscape_pretrain` | 0.843 | 0.084 | 0.575 | 0.937 | **87.4%** |

`robot`은 표준편차가 평균의 0.5%로 사실상 같은 그림이 500번 반복된다 → pretraining 신호가 없다.
`ROBOT_GRID_SPEC`은 로봇의 실제 관심 영역이자 최종 타깃이므로 **삭제하지 않고** fine-tuning
단계용으로 남긴다. `tools/build_occupancy_gt.py`는 두 스펙에 대해 각각 GT를 생성한다.

두 그리드의 크기가 다르므로(80×40 vs 200×200) pretrain → fine-tune 전이 시 head 출력
해상도/좌표 정규화를 어떻게 맞출지는 다음 스펙(Simple-BEV 통합)에서 결정한다.

이 범위들은 이후 자체 로봇 데이터셋과 스케일을 맞추기 위한 값이며(README의 pretrain→fine-tune 전략), SynWoodScape(차량 스케일 도로 장면) 위에서의 투영 **검증**에는 §3.4의 5단계처럼 별도의 넓은 범위를 쓴다. 즉 검증용 범위와 최종 occupancy GT용 그리드 범위는 분리한다.

### 4.2 Drivable class 정의

SynWoodScape semantic palette 기준:

| class id | 이름 | occupancy |
|---|---|---|
| 6 | road line | drivable |
| 7 | road | drivable |
| 그 외 전부 | - | non-drivable |

이 매핑은 `projects/bev_gt/`의 상수/설정으로 분리해, 추후 자체 로봇(보도 포함 등)에 맞게 바꾸기 쉽게 한다.

### 4.3 BEV 이미지 스케일/원점 보정 (1회성) — 2026-07-30 전면 수정

> ⚠️ **이 절의 초판도 틀렸다.** "차량 실루엣 픽셀 bbox ÷ 3D 박스 치수"로 스케일을 역산하라고
> 했고 그렇게 0.0284~0.0288을 얻었으나, 그 방법은 **구조적으로 과소추정**한다. 확정된 값은
> **`15/512 = 0.029296875` m/px**다. 아래 "왜 실루엣 방법이 틀렸나"를 반드시 읽을 것.
> 상세 기록: [`docs/dataset_analysis/synwoodscape_geometry_findings.md`](../../dataset_analysis/synwoodscape_geometry_findings.md) §2

**문제**: `_BEV.png`(top-down semantic 라벨)를 만든 가상 BEV 카메라의 intrinsic(화각/스케일)이 `calibration_data`에도 `vehicle_data`에도 없다. readme.txt에는 위치(z=15, pitch=-90)만 명시돼 있다.

#### 확정된 상수 (`projects/bev_gt/bev_crop.py`)

| 상수 | 값 | 도출 근거 |
|---|---|---|
| `BEV_METERS_PER_PIXEL` | **15/512 = 0.029296875** | 기하 도출: 카메라 높이 15 m, CARLA 기본 FOV 90°, 폭 1024px의 절반 512px → 15·tan45°/512. 전체 이미지 = 정확히 **30.0 m**. SynWoodScape 논문의 "~30 m" 기술과 일치. |
| `BEV_ORIGIN_PX` | (511.5, 511.5) (col, row) | ego(class 24) bbox 중심이 여러 샘플에서 이미지 정중앙 |
| `SIGN_FORWARD` | **−1** | ego +x(전방) → row 감소 (이미지 위쪽) |
| `SIGN_LATERAL` | **−1** | ego +y(좌측) → col 감소 (이미지 왼쪽) |
| 반환 배열 방향 | row 0 = 최전방, col 0 = 차량 좌측 | **표시용 그대로** — 저장 시 flip 불필요 |

즉 소스 `_BEV.png` 자체가 이미 일반적인 top-down 지도 방향(전방=위, 차량 좌측=왼쪽)이다.
ego → 픽셀 매핑 식은 `bev_crop.ego_to_bev_pixel()` **한 곳에만** 두고, 파이프라인과 보정
스크립트가 같은 함수를 쓴다 (Phase 1에서 같은 종류의 중복이 실제로 문제를 일으켰다).

#### 왜 실루엣 방법이 틀렸나 (되돌리지 말 것)

BEV 카메라는 정사영이 아니라 **z=15 m의 진짜 핀홀**이다(readme.txt: z=15, pitch=−90).
따라서 높이 h인 면은 지면보다 `15/(15−h)`배 확대되어 찍힌다. 차량 실루엣은 지면이 아니라
차체(ego 박스 z∈[0.009, 1.556] m)의 윤곽이므로 항상 크게 찍히고, m/px는 그만큼 **작게** 나온다.

데이터 확인 — `depth_maps/raw_data/00000_BEV.npy`(BEV depth는 평면 z-depth다):
노면(class 7) 중앙값 **14.994 m**, ego 차체(class 24) **13.678 m**(최소 13.443)
→ 카메라 높이 15 m 확인 + 지붕 h ≈ 1.32~1.56 m에서 확대율 1.10~1.12배.

실루엣 추정치를 ego 프레임 박스로 올바르게 재면 lateral 0.02710 / forward 0.02807이고,
이는 참값 0.029297을 위 확대율 범위로 나눈 구간 [0.02626, 0.02930] 안에 정확히 들어간다.
즉 편향의 방향과 크기가 핀홀 모델로 정량적으로 설명된다.

추가로, 초판 절차 1(여러 instance에 걸쳐 평균)에는 별도의 구현 함정이 있었다:
`_unuserd/box_3d_annotations`의 코너는 **world(CARLA) 좌표**이므로 축정렬 extent가 actor의
world yaw에 오염된다. 같은 ego 박스가 샘플에 따라 (1.90, 3.76) ↔ (3.72, 1.83)로 뒤바뀌어,
스크립트가 `scale_x ≈ 0.031 / scale_y ≈ 0.019`처럼 서로도 안 맞는 값을 출력했다.
코너를 ego 프레임으로 되돌리면 (3.705, 1.789)로 모든 샘플에서 동일해진다
(`bev_calibration.box_corners_to_ego`).

#### 교차검증 절차 (실제로 수행한 것)

Phase 1의 검증된 어안 파이프라인을 스케일 검증에 재사용한다.

1. 4대 카메라 depth map에서 지면 class(6/7/8/14/20) 픽셀을 `unproject_depth_to_ego`로 ego
   프레임에 되돌린다. `|z| < 0.15 m`, 반경 2.5~13 m인 점만 남긴다.
2. m/px 후보를 0.0270~0.0320에서 sweep하며 `_BEV.png`의 class와 일치율을 잰다.
3. 정점이 기하 도출값과 맞는지 확인한다.

결과 (5샘플 218,859점): 0.0284 → **0.9442**, 0.0292 → **0.9792**, 15/512 → **0.9786**,
0.0300 → 0.9469. 카메라별로도 전부 0.0292~0.0294에서 정점(FV 0.980 / MVL 0.983 / MVR 0.974 /
RV 0.978 @ 15/512). LiDAR 지면 리턴으로 따로 재도 같다(0.9800 @ 15/512).
격자를 0.00005로 좁혀 7샘플로 재면 경험적 정점이 **0.02925**(0.98355)이고 15/512에서 0.98294 —
차이 0.16%로 격자 한 칸 수준이다. 즉 기하 도출값과 경험적 정점이 사실상 일치한다.

부호는 4개 후보 전수 비교로 확정했다: (−1,−1) 0.9786 / (+1,−1) 0.9174 / (−1,+1) 0.7415 /
(+1,+1) 0.7362 (어안 지면점 기준. LiDAR 기준도 순서 동일).

원점은 유지된다 — 4개 이상 샘플(world yaw 0°/88°/91°/147°)에서 ego-vehicle bbox가 **항상
정확히 같은 크기(66×132px)·같은 위치(정중앙)** 로 나온다 → 이 BEV 카메라는 world 고정이 아니라
**ego 차량과 함께 회전하는(ego-relative) 카메라**다. 따라서 world yaw 보정이 불필요하고,
이미지 축이 곧바로 ego의 전후/좌우 축이다.

#### 도구의 역할

`tools/calibrate_bev_scale.py`는 이제 상수를 **제안하지 않고 검증한다**: 원점 일관성, 4개 부호
후보 랭킹, 지면 포인트 스케일 sweep, 그리고 실루엣 추정치를 **"쓰면 안 되는 반례"로만** 출력한다.
회귀 테스트는 `tests/bev_gt/test_bev_crop.py`와 `tests/bev_gt/test_bev_calibration.py`에 있다.

### 4.4 근접 occupancy GT 생성

> ⚠️ **2026-08-02 추가.** 아래 3단계는 occupancy **class 값**을 top-down `_BEV.png` label에서
> 뽑는 부분만 다룬다 — 이 문서를 쓸 당시엔 "관측됐는지(visible/observed)"를 판정하는 축이
> 아예 없었다. 그 축은 이후 별도로 설계했고 시행착오를 거쳤다 (§4.6, `docs/dataset_analysis/
> synwoodscape_geometry_findings.md` §3). **최종 파이프라인은 `tools/build_occupancy_gt.py`가
> 아니라 `tools/build_hybrid_occupancy.py`다** — 아래 3단계(class 값 crop)는 그대로 재사용되고
> (`bev_crop.crop_bev_occupancy`), 여기에 §4.6의 raycast 기반 observed mask가 추가된다.

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

### 4.5 완료 기준 (2026-07-30 수정)

> ⚠️ 초판의 "스케일 추정의 표준편차가 평균의 5% 이내"는 **폐기했다.** 실측 표준편차는 평균의
> 45~59%였고(여러 instance × world yaw 오염 때문, §4.3 참고), 애초에 스케일을 실루엣 추정의
> **분산**으로 검증한다는 발상 자체가 틀렸다 — 그 방법은 분산이 작아도 **편향**되어 있다.

실제로 채택·달성한 기준:

- **스케일**: 기하 도출값(15/512)이 **지면 포인트 교차검증의 정점**과 일치할 것.
  기준: Phase 1 파이프라인으로 복원한 지면 점(4대 카메라, `|z|<0.15 m`, 반경 2.5~13 m)에 대해
  (a) `15/512`에서 BEV semantic class 일치율 **≥ 0.95**이고,
  (b) sweep 정점이 `15/512`와 **0.0003 m/px 이내**이고,
  (c) 실루엣 추정치(0.0284)보다 일치율이 **0.02 이상** 높을 것.
  실측: 일치율 0.9786(0.0284는 0.9442), 정점 0.0292. → 충족.
  회귀: `tests/bev_gt/test_bev_crop.py::test_bev_scale_beats_the_biased_vehicle_silhouette_estimate_on_ground_points`
- **축 부호**: 4개 후보 중 `(SIGN_FORWARD, SIGN_LATERAL) = (−1, −1)`이 최고이고 차순위와
  **0.02 이상** 벌어질 것. 실측 0.9786 vs 0.9174. → 충족.
  회귀: `..._test_bev_sign_convention_beats_all_three_alternatives_on_ground_points`
- **배열 방향**: `crop_bev_occupancy`의 출력이 소스 `_BEV.png`를 같은 ROI로 자른 부분영상과
  **뒤집힘 없이 일치**할 것(전/후, 좌/우 모두). 합성 이미지 테스트는 **두 축을 각각 따로**
  깨뜨릴 수 있어야 한다 — 초판 테스트는 모든 열이 균일한 이미지로 행 축만 확인해서 좌우 뒤집힘을
  원리적으로 잡을 수 없었다(실제로 놓쳤다).
  회귀: `..._test_crop_bev_occupancy_col0_is_vehicle_left`,
  `..._test_crop_bev_occupancy_matches_a_plain_slice_of_the_real_source_image`
- **원점**: ego-vehicle bbox 중심이 여러 샘플에서 이미지 중앙 ±1px 이내로 일관됨. 실측 ±0.5px.
- **GT가 학습 신호를 담고 있을 것**: 그리드가 샘플 간에 실제로 변해야 한다. 실측: `robot`은
  std/mean = 0.5%(사실상 상수) → pretraining 부적합, `synwoodscape_pretrain`은 std/mean = 10%,
  셀 87.4%가 변동 → 적합 (§4.1).
- 최종 occupancy 그리드를 몇 개 샘플에 대해 시각화했을 때 도로 형태가 육안상 타당하고,
  같은 ROI로 자른 `rgb_images/*_BEV.png`와 방향이 일치함 (샘플 00300으로 확인).

### 4.6 Visibility(관측 여부) 설계 (2026-08-02 추가)

§4.1~4.5는 occupancy **class 값**만 다루고, "어안 카메라가 실제로 그 지점을 봤는가"는
전혀 판정하지 않는다 — `crop_bev_occupancy`는 ROI 안 모든 cell에 항상 값을 채운다. ego
차체 바로 아래나 다른 물체에 가려진 곳은 drivable/obstacle을 단정할 근거가 없으므로, 이를
구분해 loss에서 제외(ignore)하는 축을 나중에 추가로 설계했다. 두 번 갈아엎었다:

1. **가설검정** (`projects/bev_gt/visibility.py`) — grid cell을 z=0 점으로 각 카메라에
   투영해 depth map과 5%/0.20m 오차 이내인지 검정. **폐기**: 근거리(2~4m)에서 잔여
   캘리브레이션 오차(§3.3 roll 보정 후에도 남는 것)가 문턱을 근소하게 넘나들어, 진짜
   가려짐과 캘리브레이션 오차를 구분하지 못했다.
2. **어안 카메라 자신의 semantic label로 raycast** (`raycast_occupancy.
   build_raycast_occupancy`) — 4대 카메라 depth map의 모든 픽셀을 실제 언프로젝션해 ego
   3D 점을 얻고, 그 픽셀 자신의 label을 그 cell에 찍는다. visibility 판정은 문턱 없이
   견고해졌지만, **occupancy class 값도 같은 소스에서 나오게 되어 새 버그가 생겼다**:
   SynWoodScape FV 카메라의 semantic segmentation이 ego 차량 자신의 본네트를 "road"로
   잘못 라벨링한다(§4.2 class 매핑은 원본 라벨이 맞다고 전제하는데, 이 전제가 깨진 사례).
   **폐기.**
3. **최종: 하이브리드** — class 값은 §4.4 그대로(top-down `_BEV.png`, FV 본네트 결함 없음
   확인됨), observed 값만 라벨 없이 raycast(`raycast_occupancy.compute_observed_mask`)로
   판정. `tools/build_hybrid_occupancy.py`가 현재 파이프라인이다.

상세 근거(실측 수치, 왜 시도 1·2가 실패했는지, 남겨둔 잔여 노이즈와 그 이유)는
`docs/dataset_analysis/synwoodscape_geometry_findings.md` §3에 있다 — 이 문서를 갱신하는
대신 그쪽에 자세히 적었다(실험 성격이 강해 findings 노트 쪽이 더 맞는 자리라 판단).

## 5. 산출물

- 코드
  - `projects/geometry/frames.py` — 동차좌표 변환 헬퍼, LiDAR 센서로컬→ego 변환(`lidar_points_to_ego`), CARLA↔ego 규약 변환, 좌표계 규약 회귀 테스트
  - `projects/geometry/reprojection.py` — ego 포인트 → fisheye 픽셀 투영, depth 기반 가시성 마스크
  - `projects/geometry/fisheye.py` — radial_poly project/unproject (WoodScape 공식 구현 wrapping)
  - `projects/bev_gt/` — `grid.py`(class remap, 그리드 스펙), `bev_crop.py`(ego→BEV 픽셀
    매핑 `ego_to_bev_pixel`, BEV 이미지 스케일/원점 보정, ROI crop, occupancy class 값),
    `visibility.py`(§4.6 시도 1, 폐기), `raycast_occupancy.py`(§4.6 시도 2·최종 —
    `build_raycast_occupancy`는 폐기, `compute_observed_mask`가 현재 observed 판정에 쓰임)
  - `tools/verify_fisheye_projection.py` — Phase 1 검증 스크립트 (LiDAR 재투영 오버레이 + semantic consistency rate)
  - `tools/calibrate_bev_scale.py` — §4.3의 스케일/원점 보정을 여러 샘플에 대해 수행하고 상수를 도출하는 스크립트
  - `tools/build_occupancy_gt.py` — Phase 2 초판 occupancy GT 배치 생성 + 시각화 (§4.6 이후 **폐기**, class crop 로직만 `bev_crop.py`로 남아 재사용됨)
  - `tools/build_visibility_mask.py` — §4.6 시도 1 (폐기)
  - `tools/build_raycast_occupancy.py` — §4.6 시도 2 (폐기)
  - **`tools/build_hybrid_occupancy.py`** — §4.6 최종, 현재 GT 생성 파이프라인. 출력은
    `dataset/synwoodscape_occupancy_gt/{sample}_{occupancy.npy,visible.npy,combined.png}`
- 문서
  - 이 design doc
  - **[`docs/dataset_analysis/synwoodscape_geometry_findings.md`](../../dataset_analysis/synwoodscape_geometry_findings.md)**
    — Phase 1/2/2.5 실행 결과 findings 노트 (2026-07-30 작성, 2026-08-02 visibility 설계
    추가): Phase 1에서 찾아 고친 3개 버그, 확정된 캘리브레이션 상수와 도출/교차검증 방법,
    그리드 스펙 2종의 존재 이유, visibility 설계 3단계 변천사와 남겨둔 잔여 노이즈, 재현 명령.

## 6. 열린 질문 / 향후 확인 사항

**해결됨** (2026-07-30):

- ~~§4.3의 스케일은 샘플 00000의 instance 1개로 처음 확인했다. 더 많은 샘플/instance로 평균을 내 견고성을 확인해야 한다.~~
  → **평균을 내는 접근 자체가 틀렸다.** 실루엣 방법은 분산이 아니라 **편향**이 문제였다.
  확정값은 기하 도출 `15/512`이고, 지면 포인트 교차검증으로 확인했다 (§4.3, §4.5).
- ~~이미지 축의 전방(+) 방향 부호는 아직 육안/velocity 대조로 확정하지 않았다.~~
  → **`(SIGN_FORWARD, SIGN_LATERAL) = (−1, −1)` 확정.** 4개 후보를 어안 지면점과 LiDAR
  지면점 두 소스로 각각 전수 비교했고, 두 소스·4대 카메라 모두에서 동일한 결론이다 (§4.3).
  덧붙여 **출력 배열의 인덱스 순서는 부호와 별개 결정**이라는 점이 뒤늦게 드러났다(§4.5의
  "배열 방향" 항목). 부호가 맞아도 배열이 뒤집힐 수 있다.

**해결됨** (2026-08-02):

- ~~occupancy GT에 "관측됐는지(visible)" 축이 없다 — ego 차체 아래·다른 물체에 가려진 곳도
  drivable/obstacle 값이 항상 채워진다.~~ → **하이브리드 설계로 해결** (§4.6): class 값은
  top-down label(기존과 동일), observed는 4-cam raycast. 시행착오(가설검정 → raycast
  단독의 FV 본네트 버그 → 하이브리드)와 남겨둔 잔여 노이즈(작은 vegetation/pole 조각,
  ego 실루엣 salt-and-pepper — 둘 다 의도적으로 정리하지 않음)는 findings 노트 §3에 기록.
- ~~그리드 스펙 수치(80×40 / 200×200)가 최종 확정인지 불명확.~~ → §4.1 상단 배너 참고 —
  **120×120 / 160×160으로 다시 바뀌었고**, 실제 SynWoodScape GT 생성에는
  `synwoodscape_pretrain`(160×160)만 쓴다. `robot`은 자체 로봇 fine-tuning용으로 보류.

**유효**:

- `_unuserd/instance_annotations`, `_unuserd/box_3d_annotations`는 보정 검증 단계에서만
  참조하고, 최종 occupancy GT 파이프라인(`tools/build_hybrid_occupancy.py`)은
  `semantic_annotations`(top-down label)와 `depth_maps`(어안 raycast)만 사용한다.
- 두 그리드 스펙의 크기가 다르므로(120×120 vs 160×160) pretrain → fine-tune 전이 시 head 출력
  해상도/좌표 정규화를 맞추는 방법은 다음 스펙(Simple-BEV 통합)에서 결정한다 (§4.1).
