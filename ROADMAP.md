# 로드맵 (ROADMAP)

> 프로젝트의 큰 흐름을 잡기 위한 가이드입니다. 반드시 이대로 지켜야 하는 것은 아니며, 진행하면서 **유동적으로 변경**될 수 있습니다.

**최종 목표:** 자체 구축한 Fisheye 4-cam 데이터셋으로 **BEV occupancy map**(각 BEV 격자 셀이 주행 가능한 영역인지 여부)을 예측하는 모델을 학습·평가한다. baseline은 **Simple-BEV**.

---

## Phase 0 — 개발 환경 세팅 & 데모 검증 ✅ 완료

- conda 환경 및 버전 고정 (→ `docs/setup_guide_pro6000.md`)
- GPU 인식, sm_120 커널 실행, 사전학습 encoder, 로깅 스택, simple_bev import 정상 동작 확인
- 최초 구축(RTX 3080) 당시에는 mmdet3d 스택까지 함께 깔았으나, 현재 학습 경로(Simple-BEV)에는 PyTorch·numpy만 필요하다. **RTX PRO 6000 서버로 이전하면서 mmcv 계열은 설치하지 않는다** (prebuilt wheel이 torch 2.1까지만 존재). 3080 시절 이력은 `docs/setup_guide.md`에 보존

## Phase 1 — 벤치마크 데이터셋 분석 ✅ 완료

- **WoodScape** 구조·annotation·fisheye 특성 분석 (→ `docs/dataset_analysis/woodscape_analysis.md`, 탐색 스크립트는 `tools/woodscape_viz/`)
  - 공개본은 2D 라벨만 제공하며 3D·depth GT가 없다
- **SynWoodScape** 구조 파악 — 500 samples, 어안 4-cam(FV/RV/MVL/MVR) + BEV 1대, `semantic_annotations`(BEV 포함) · `depth_maps` · `lidar_data` · `box_3d_annotations` · `calibration_data`
- **캘리브레이션 규약 확정** — WoodScape 공식 `calibration_readme.txt` 기준: vehicle 프레임은 **ISO 8855**(원점 = 뒷축 중점 아래 지면, X 전방 / Y 좌측 / Z 상방), camera 프레임은 **OpenCV**, extrinsic(translation + quaternion)은 **camera → vehicle** 변환, 렌즈는 `radial_poly`
- **SynWoodScape 기하 실측 검증** (→ `docs/dataset_analysis/synwoodscape_geometry_findings.md`) — 어안 project/unproject 검증, extrinsic 보정(roll 부호·미러 카메라 translation 전치), LiDAR 좌표 규약, BEV 카메라 스케일(`15/512` m/px)과 occupancy GT 그리드 스펙 2종

## Phase 2 — 자체 BEV 데이터셋 구축 ✅ 완료 (별도 프로젝트)

- 수집 → 카메라 캘리브레이션 → 라벨링을 거쳐 **BEV occupancy GT까지 완성**된 상태
- 이 저장소에서는 Phase 4에서 포맷만 맞춰 가져다 쓴다

## Phase 3 — SynWoodScape로 Simple-BEV 학습 ⬅️ 현재 단계

목적은 성능 최적화가 아니라, **어안 4-cam 입력 → BEV occupancy 예측 파이프라인이 실제로 동작함을 증명**하는 것이다.

### 3.1 데이터 파이프라인

- SynWoodScape를 Simple-BEV 입력 형태로 읽는 로더 작성 (`projects/`)
- **BEV GT를 binary occupancy로 remap** — `road`(7) + `road line`(6) → **drivable**, 그 외는 non-drivable. `unlabeled`(0)과 `ego-vehicle`(24)은 loss에서 제외(ignore)할지 결정한다
- **BEV GT의 미터/픽셀 스케일 확정** — `calibration_data/`에 BEV 카메라 json이 없다. readme에는 "높이 15 m, pitch −90°"만 적혀 있고 **FOV가 없어서 스케일이 문서로 확정되지 않는다.** LiDAR 또는 어안 depth로 실제 스케일을 역산해 확정한다
- 학습에 쓸 BEV 범위(예: 자차 주변 ±m)와 격자 해상도 결정
- 500 samples의 train / val split 정의 (시퀀스라 프레임이 인접하므로 무작위 분할은 누수 위험 — 구간 분할 검토)

### 3.2 어안 투영 주입

- Simple-BEV의 lifting은 **BEV 격자 → 픽셀 정방향 project만** 사용하므로, 핀홀 project를 `radial_poly` project로 교체하면 어안을 그대로 입력할 수 있다 (undistort 불필요)
- **submodule은 수정하지 않는다** — `projects/`에서 래핑해 투영 함수를 주입한다
- 투영 정합성 검증. **단, 규약은 위 Phase 1의 공식 문서 내용으로 확정하고, 검증 지표를 세우기 전에 known-good 대조군으로 그 지표가 무엇에 민감/둔감한지 먼저 확인한다** (지표를 먼저 만들고 결론을 낸 탓에 잘못된 결론을 반복한 전례가 있다)

### 3.3 학습 & 평가

- 하드웨어: RTX PRO 6000 96GB. VRAM 제약이 없으므로 batch size 축소·gradient accumulation은 불필요하고, **데이터 로딩(`num_workers`)과 GPU 활용률**을 먼저 살핀다. bf16 AMP는 검토 대상
- 지표: **drivable IoU**를 주지표로 사용
- 성공 기준: 학습이 수렴하고, 예측 BEV가 GT와 육안으로도 정합한다

## Phase 4 — 자체 데이터셋 fine-tuning

- Phase 3의 체크포인트를 **pre-trained weight**로 사용
- 자체 데이터셋의 BEV occupancy GT를 Phase 3 로더와 **같은 인터페이스**로 변환
- 자체 카메라의 어안 모델(캘리브레이션 결과)을 투영 모듈에 연결. `radial_poly`가 아닐 수 있으므로 투영 함수를 교체 가능한 형태로 둔다
- drivable 정의는 **온실 환경 기준으로 새로 정의**한다 (Phase 3의 road/road line 기준은 CARLA 도시 장면용)
- sim → real 도메인 갭(CARLA 도시 → 온실) 대응

## Phase 5 — 분석 · 개선 · 문서화

- 결과 분석, 실패 사례 정리, 개선 반복
- 재현 가능한 형태로 문서/코드 정리

---

## 이후 확장 (현재 범위 밖)

- **모델:** Simple-BEV → LSS → BEVFormer (`third_party/models/`에 참고용 submodule로 확보해 둠)
- **task:** 세밀한 BEV semantic 구분, 3D bounding box 검출. 3D 검출로 확장할 때 `mmdetection3d/` submodule을 다시 쓴다
