# Phase 1: Fisheye 투영 함수 검증 + Simple-BEV native 통합 Plan

## 0. 문서의 목적

코드 작성 이전 단계의 설계 문서다. SynWoodScape(어안 4-cam)를 Simple-BEV에 **native로
(undistort 없이)** 투입하기 위해:

1. radial_poly 투영 함수(`project`/`unproject`)를 구현하고 **객관적 수치로 검증**한다.
2. 검증된 `project()`를 Simple-BEV의 lifting(`unproject_image_to_mem`)에 **통합**하고,
   교체된 투영이 옳게 동작하는지 확인한다.

핵심 원칙: **함수를 먼저 검증하고(층 A), 그 함수를 모델에 통합해 검증한다(층 C).**
(구 계획의 층 B = undistort remap 검증은 **폐기**. → `00_overview.md` 참조.)

### 0.1 전제 (전체 프로젝트)
- Task: **BEV semantic segmentation**, RGB-only, 초기 클래스 = drivable / occupied.
- 첫 모델: **Simple-BEV** (`third_party/models/simple_bev`, `nets/segnet.py`,
  lifting = `utils/vox.py: unproject_image_to_mem`).
- 어안은 **네이티브 입력**. Simple-BEV 투영 함수를 radial_poly로 교체한다.

---

## 1. 배경 — 왜 native이고, 무엇을 교체하나

### 1.1 데이터셋
- **SynWoodScape V0.1.0**, 500 samples, 10 FPS.
- 어안 4대: FV(전방), RV(후방), MVL(좌측미러), MVR(우측미러).
- BEV GT는 `semantic_annotations/gtLabels/XXXXX_BEV.png`로 **이미 제공**(z=15m 상공,
  pitch=-90 top-down 가상 카메라). → 별도 GT 구축 불필요.

### 1.2 교체 지점 (코드 확인 완료)
Simple-BEV lifting은 3D 격자점을 **정방향(project)** 으로 이미지에 쏴서 특징을 샘플한다.
핀홀 가정은 `unproject_image_to_mem` 내부 두 줄:
```python
xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)          # K 곱
xy_pixB  = xyz_pixB[:,:,:2] / torch.clamp(normalizer, min=EPS)  # perspective divide
```
이를 아래로 교체:
```python
xyz_camB = apply_4x4(camB_T_camA, xyz_camA)       # ego→카메라 (이미 있음)
uv = radial_poly_project(xyz_camB, fisheye_calib)  # 층 A에서 검증한 project()
```
→ **lifting은 project() 정방향만 필요.** (unproject는 GT 대조·검증용으로만 구현.)

---

## 2. Calibration 형식 (FV.json 기준)

### 2.1 intrinsic — radial_poly 어안 모델
```
model: "radial_poly", poly_order: 4
k1=341.725, k2=-26.4448, k3=32.7864, k4=0.50499
cx_offset=0.329055, cy_offset=-3.5506
aspect_ratio=1.00021
width=1280, height=966
```
- 핀홀 K가 아님. **입사각 θ(광선과 광축 사이 각) → 이미지 반경 r** 매핑:
  `r(θ) = k1·θ + k2·θ² + k3·θ³ + k4·θ⁴`
- 이후 cx_offset/cy_offset/aspect_ratio로 최종 픽셀 좌표 결정.
- 역투영(픽셀→θ)은 이 다항식의 수치적 역함수(뉴턴법 또는 LUT).
- **★ 참고 코드 존재:** WoodScape omnidet `data/generate_luts.py`의 `inverse_poly_lut()`가
  "radial_poly는 해석적 역함수가 없어 LUT로 역투영을 만든다"는 레퍼런스
  (`third_party/datasets/WoodScape/omnidet/data/`). project/unproject를 처음부터 짜지 말고
  이를 참고. (lifting은 정방향 project만 쓰므로 역투영 LUT는 검증·GT 대조용.)

### 2.2 extrinsic — quaternion + translation
```
quaternion: [w,x,y,z] 형태 추정 (순서 확인 필요)
translation (CARLA reference): [1.92, 0.0, 0.9]
```
- **주의: "CARLA reference"** → CARLA 축 규약(left-handed, x-forward 등) 확인 필수.
- quaternion 성분 순서(w-first vs w-last) 반드시 확정.
- Simple-BEV 규약과의 대응: Simple-BEV는 `camB_T_camA`(ego→camera 4×4)를 쓴다.
  SynWoodScape extrinsic을 이 형식으로 변환·부호 정합해야 함.

### 2.3 자체 카메라 캘리브레이션 (추후 — sim과 별개 모델)

**중요 원칙: sim과 real의 투영 모델을 통일할 필요가 없다.** 투영은 카메라별 고정 기하
함수일 뿐 학습 가중치가 아니므로(백본·BEV 디코더는 투영 계수를 배우지 않음), sim은
radial_poly, real은 자체 캘리 모델로 **각자 올바르기만 하면** pretrain→fine-tune이 성립.
→ **SynWoodScape를 UCM/eUCM 등으로 변환하지 않는다.**

- **직접 캘리브레이션 최적화 코드를 짜지 않는다.** 표준 툴(Kalibr / OpenCV / Basalt)로
  캘리하고, 나온 모델의 project/unproject만 구현한다.
- radial_poly는 표준 툴이 지원하지 않는 형태(연속 차수). 하지만 KB(OpenCV `cv2.fisheye`,
  홀수 차수)와 같은 "θ→r 다항식" 가족이라 개념적으로 특별하지 않음.
- **자체 카메라 모델 후보 (강어안 대상):**
  - **1순위: Double Sphere (DS).** 역투영이 닫힌 형태(closed-form) → 모델 코드 깔끔·빠름,
    강어안에 정확. Kalibr/Basalt 지원.
  - **후보: eUCM (Enhanced UCM).** 파라미터 적고 강어안 커버, Kalibr 지원. DS와 비교 검토용.
  - (참고: KB도 가능하나 강어안에서 DS/eUCM가 유리한 편.)
- 최종 선택은 자체 렌즈로 캘리 후 **재투영 오차(reprojection error)로 비교**해 확정.
  개념/원리 학습은 `docs/study/camera_models_and_calibration.md` 참조.

---

## 3. 검증 대상의 두 층 (native 버전)

```
[radial_poly 함수]  ──(이 함수를 넣는다)──▶  [Simple-BEV lifting(native 투영)]
     │                                                │
   층 A 검증                                        층 C 검증
 (함수 자체의 정확성)                          (모델 통합 후 투영 정합)
```

- **radial_poly 함수:** 카메라당 하나(렌즈 물리) → **4개**(FV/RV/MVL/MVR). 고정.
- **층 B(undistort remap)는 없음.** native이므로 remap 자체가 불필요.

**검증은 층 A → 층 C 순서.** 함수가 틀리면 통합도 틀리므로 함수부터 확정.

---

## 4. 층 A — radial_poly 함수 검증 (모델과 무관)

목표: "이 함수가 3D ↔ 픽셀 변환을 옳게 하는가" 확정.

### 4.1 구현할 함수
1. **project(P_cam) → (u, v)**: 카메라 좌표 3D 점 → 어안 픽셀.
   (광선 → θ → r(θ) 다항식 → offset/aspect 적용) — **lifting에서 실제 사용.**
2. **unproject(u, v) → ray_dir**: 어안 픽셀 → 카메라 좌표 광선 방향(단위벡터).
   (offset 역적용 → r → θ 역다항식(수치해) → 광선 복원) — **검증·GT 대조용.**

### 4.2 검증 1 — 왕복 재투영 오차 (자기일관성)
1. 어안 픽셀 `(u,v)` 다수 샘플.
2. `depth_maps/raw_data/XXXXX_FV.npy`에서 depth `d`.
3. `unproject(u,v)` → 광선 → `× d` → 3D 점 `P`.
4. `project(P)` → `(u', v')`.
5. 오차 `‖(u,v)-(u',v')‖` 측정.
- **합격: 평균 재투영 오차 < 1 px** (평균/최대 리포트).
- 한계: 정투영·역투영이 "둘 다 똑같이 틀린" 경우 미검출 → 검증 2 필요.

### 4.3 검증 2 — depth 역투영 vs LiDAR 대조 (물리 정확성)
1. FV depth 전체를 `unproject`+depth로 point cloud화(카메라 좌표).
2. extrinsic으로 ego 좌표 변환.
3. `lidar_data/XXXXX.pkl`(ego 기준 3D 정답)과 오버레이.
4. 두 point cloud 정합(최근접점 거리 분포) 측정.
- **합격: 평균 nearest-neighbor 거리가 센서 노이즈 수준 이내.**
- extrinsic/좌표계(CARLA 규약, quaternion 순서)의 정확성도 함께 검증됨.

### 4.4 층 A 디버깅 규칙
- 검증 1 실패 → radial_poly 파라미터 해석 또는 project/unproject 구현 오류.
- 검증 1 통과·검증 2 실패 → 함수는 맞고 **extrinsic/좌표계 규약** 오류.

---

## 5. 층 C — Simple-BEV native 투영 통합·검증

전제: 층 A 통과(project 확정).

### 5.1 통합 작업
- `utils/vox.py: unproject_image_to_mem`의 핀홀 투영 두 줄을 `project()` 호출로 교체.
  - 서브모듈 불가침 원칙: **직접 수정 대신** `projects/`에 `Vox_util` 서브클래스 또는
    투영을 주입할 수 있는 래퍼를 두는 방식을 우선 검토. (불가하면 최소 패치 + 문서화.)
- `camB_T_camA`(ego→camera)로 `xyz_camB` 계산은 그대로 사용.
- **valid 마스크 교체:** 기존 `z>0 & 픽셀 in-bounds` → **`θ<θ_max`(화각) & 픽셀
  in-bounds**. 어안은 θ가 클수록 유효, 그러나 렌즈 화각(≈95°) 밖·behind-camera는 배제.
- **★ feature-map stride 반드시 반영 (핀홀 경로와 동일 원리):** Simple-BEV는 원본이 아니라
  **다운샘플된 feature map**(Hf×Wf)을 샘플한다. `Segnet.forward`는 핀홀에서
  `sy=Hf/H, sx=Wf/W`로 `scale_intrinsics(pix_T_cams, sx, sy)`를 적용해 K를 feature 해상도로
  내린다. native에서는 `project()`가 **full-res 어안 픽셀**을 뱉으므로, 그 출력 `(u,v)`에
  **`(sx, sy)`를 곱해 feature-map 좌표로 변환**해야 한다. (누락 시 약 8배 어긋난 위치를
  샘플 → 학습이 조용히 실패.) valid 마스크의 in-bounds 판정도 feature 해상도 기준으로.

### 5.2 검증 3 — 격자→이미지 투영 오버레이 (native 정합)
- BEV 격자점(또는 알려진 3D 점: LiDAR/3D bbox 대표점)을 교체된 투영으로 어안 이미지에
  투영해 오버레이.
- **합격 기준:**
  - 전방 물체가 FV의 옳은 위치에, 좌측 물체가 MVL의 옳은 위치에 찍힌다.
  - 어안 왜곡을 따라 격자선이 휘어 투영된다(핀홀처럼 직선이면 오히려 오류).
  - 화각 밖 점이 valid 마스크에서 제외된다.
- 실패 시: project() 통합부 좌표 부호 / extrinsic 변환 / θ 마스크 오류.

### 5.3 검증 4 — 특징 샘플링 sanity (선택)
- 교체 후 `unproject_image_to_mem` 출력(voxel feature)이 NaN/전영역 0이 아닌지,
  valid_mem 비율이 물리적으로 타당한지 확인.

---

## 6. 데이터 요구사항 (SynWoodScape 폴더)

### 검증 단계
- `rgb_images/XXXXX_{FV,RV,MVL,MVR}.png` — 어안 원본.
- `calibration_data/{FV,RV,MVL,MVR}.json` — radial_poly + extrinsic.
- `depth_maps/raw_data/XXXXX_FV.npy` — 검증 1·2용.
- `lidar_data/XXXXX.pkl` — 검증 2용.

### 학습 단계
- 어안 4뷰 RGB + calibration.
- `semantic_annotations/gtLabels/XXXXX_BEV.png` — BEV 학습 GT.
- (temporal 확장 시) `previous_images/` + `vehicle_data/` — 초기엔 불필요.

### 사용하지 않음 (초기)
- `box_2d_annotations`, `box_3d_annotations`(detection), `dvs_signals`,
  `optical_flow`, `instance_annotations`, `motion_annotations`, `distances_traveled`.
- `depth_maps`/`lidar_data`는 **검증에만** 사용, 학습 입력엔 미사용.

---

## 7. 클래스 remap — v0 = 주행성(traversability) 중심 소수 클래스 (확정)

**[결정] task 범위:** v0 타깃은 **주행성 중심 소수 클래스**(semantic 세분이 아님).
세밀한 semantic 클래스(wall/fence/pole 구분)와 이산 개체 탐지(사람 등)는 **확장 task로 분리**
(→ phase3 §10, overview §5). 근거: 코드 차이는 채널 수뿐이나, **라벨이 곧 프로젝트 비용**이며
자체 데이터에서 세분 클래스는 비싸고 모호 + 소량 데이터에서 희귀 클래스 학습난. 로봇이
실제 쓰는 축(주행 가능/주의/막힘)으로 정의하는 것이 타당.

**v0 클래스(초기 이분, 필요 시 caution 추가):**
- **drivable** ← road(7), road line(6) [+선택: sidewalk(8), ground(14)]
- **occupied(blocked)** ← building(1), fence(2), wall(11), pole(5), pedestrian(4),
  four-wheeler(10), two-wheeler(21), vegetation(9), traffic sign(12), guard rail(17),
  static(22), dynamic(23) 등 통과 불가 대상.
- **background/ignore** ← unlabeled(0), sky(13), ego-vehicle(24).
- (선택) **caution** ← 넘어갈 수 있으나 주의(턱·융기 등). 주행에 실제로 다를 때만 도입.

**[결정] 계층적 라벨 전략:**
- **sim(SynWoodScape)은 세분 클래스로 학습**(25클래스는 공짜) → **평가/타깃은 coarse로
  remap.** 풍부한 감독을 받으면서 v0 타깃은 강건·디버깅 선명하게. 잃을 것 없음.
- **원본 gtLabels 보존, 학습 시 remap.** 자체 데이터도 "필요해질 때 세분"할 수 있게 라벨
  구조를 계층적으로 열어둔다(재라벨 없이 확장).
- remap 테이블(원 클래스 → v0 클래스)을 코드가 아닌 설정으로 관리해 클래스 구성 변경에 재활용.

---

## 8. 작업 순서 체크리스트

- [ ] **8.1** `calibration_data/*.json` 파서: intrinsic/extrinsic 로드, quaternion 순서·
      CARLA 규약 확정, Simple-BEV `camB_T_camA` 형식으로 변환.
- [ ] **8.2** `project()`/`unproject()` 구현 (radial_poly 정/역투영).
- [ ] **8.3** 검증 1 (왕복 재투영 < 1px) — FV 먼저.
- [ ] **8.4** 검증 2 (depth point cloud vs LiDAR 정합).
- [ ] **8.5** (층 A 통과 후) 나머지 3개 카메라 함수 검증.
- [ ] **8.6** Simple-BEV lifting에 `project()` 통합(`projects/` 래퍼 우선), θ valid 마스크.
- [ ] **8.7** 검증 3 (격자→이미지 오버레이 정합) — 뷰마다.
- [ ] **8.8** 검증 4 (특징 샘플링 sanity, 선택).
- [ ] **8.9** 클래스 remap 전처리 (25 → drivable/occupied).
- [ ] **8.10** → Phase 2로.

---

## 9. 미해결 / 확인 필요

- Simple-BEV 투영 통합을 서브모듈 불가침으로 하는 최선의 방법(서브클래스 vs 주입 vs 패치).
- quaternion 성분 순서(w-first vs w-last) — 실측 검증으로 확정.
- CARLA 축 규약 ↔ ego 좌표계 ↔ Simple-BEV 좌표계 대응.
- θ_max(화각) 값, behind-camera·θ>90° 처리.
- radial_poly 역함수 수치 안정성(넓은 θ에서 다항식 단조성).
- BEV GT(z=15 top-down) 물리 커버 범위·해상도 (Phase 2에서 확정).
