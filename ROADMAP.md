# 로드맵 (ROADMAP)

> 프로젝트의 큰 흐름을 잡기 위한 가이드입니다. 반드시 이대로 지켜야 하는 것은 아니며, 진행하면서 **유동적으로 변경**될 수 있습니다.

**최종 목표:** 온실 로봇의 전방 어안 카메라 3대로 **BEV free space**(각 BEV 격자 칸이 지금 지나갈 수 있는지)를 예측하는 모델을 학습·평가하고, Jetson AGX Orin에서 돌아감을 보인다. baseline은 **Simple-BEV**.

---

## Phase 0 — 개발 환경 세팅 & 데모 검증 ✅ 완료

- conda 환경 및 버전 고정 (→ `docs/setup_guide_pro6000.md`)
- GPU 인식, sm_120 커널 실행, 사전학습 encoder, 로깅 스택, simple_bev import 정상 동작 확인
- 최초 구축(RTX 3080) 당시에는 mmdet3d 스택까지 함께 깔았으나, 현재 학습 경로(Simple-BEV)에는 PyTorch·numpy만 필요하다. **RTX PRO 6000 서버로 이전하면서 mmcv 계열은 설치하지 않는다** (prebuilt wheel이 torch 2.1까지만 존재). 3080 시절 이력은 `docs/archive/setup_guide.md`에 보존

## Phase 1 — 벤치마크 데이터셋 분석 ✅ 완료

- **WoodScape** 구조·annotation·fisheye 특성 분석 (→ `docs/dataset_analysis/woodscape_analysis.md`, 탐색 스크립트는 `tools/woodscape_viz/`)
  - 공개본은 2D 라벨만 제공하며 3D·depth GT가 없다
- **SynWoodScape** 구조 파악 — 500 samples, 어안 4-cam(FV/RV/MVL/MVR) + BEV 1대, `semantic_annotations`(BEV 포함) · `depth_maps` · `lidar_data` · `box_3d_annotations` · `calibration_data`
- **캘리브레이션 규약 확정** — WoodScape 공식 `calibration_readme.txt` 기준: vehicle 프레임은 **ISO 8855**(원점 = 뒷축 중점 아래 지면, X 전방 / Y 좌측 / Z 상방), camera 프레임은 **OpenCV**, extrinsic(translation + quaternion)은 **camera → vehicle** 변환, 렌즈는 `radial_poly`
- **SynWoodScape 기하 실측 검증** (→ `docs/dataset_analysis/synwoodscape_geometry_findings.md`) — 어안 project/unproject 검증, extrinsic 보정(roll 부호·미러 카메라 translation 전치), LiDAR 좌표 규약, BEV 카메라 스케일(`15/512` m/px)과 occupancy GT 그리드 스펙 2종

## Phase 2 — 자체 BEV 데이터셋 구축 ✅ 완료 (별도 프로젝트)

- 수집 → 카메라 캘리브레이션 → 라벨링을 거쳐 **BEV occupancy GT까지 완성**된 상태
- 이 저장소에서는 Phase 4에서 포맷만 맞춰 가져다 쓴다

## Phase 3 — SynWoodScape로 파이프라인 검증 ✅ 완료

어안 4-cam 입력 → BEV 예측 파이프라인이 실제로 동작함을 증명했다. occupancy GT 생성(하이브리드 방식),
Simple-BEV 데이터로더, 어안 `radial_poly` 투영 주입(`projects/models/fisheye_vox.py`, submodule 무수정),
버스트 단위 train/val 분할, 학습 스크립트(`tools/train_synwoodscape.py`). 기록은
`docs/archive/synwoodscape_pretrain_experiment_log.md`, 기하 조사는 `docs/dataset_analysis/synwoodscape_geometry_findings.md`.

## Phase 4 — 자체 데이터셋 학습과 설계 확정 ✅ 완료 (2026-08 ~ 09)

- 자체 리그: **3-cam(front/left/right), Double Sphere**, 120×120 격자 5 cm. 로더 `projects/datasets/robot_simplebev.py`
- 정식화: 2-head → 3-class → **이진 free-space**로 두 번 바꿨다(옛 지표가 트리비얼 예측기에 졌다)
- 확정 설정: 어안 직접 투영, **`Y=4` lifting**(단일 변경으로 가장 큰 개선), **soft-boundary 손실**(δ 0.30 m, σ 0.10 m)
- 기각: stride 4, 정규화 손잡이들, ROI crop, BEV 격자 변경 등
- 연구 기록 전체: `docs/archive/research/`(서사는 `experiment_history.md`, 근거는 `finetune_overfitting_diagnosis.md`)

## Phase 5 — 논문 실험 캠페인 ✅ 완료 (v3, 2026-10-03)

- 지표 개편(Precision·Recall·BF@τ, 프레임 macro), 광선 보조항 제거, 100 epoch, 짝지은 t-검정 95 % CI
- 90런: 01 대표 성능 + LOSO, 02 투영·사전학습, 02b 원래 기하 사전학습, 03 손실 사다리
- **결과·표·그림·해석: `docs/99_paper_results/`** (실행 원장 `docs/paper_final_experiments.md`, 절차 `docs/paper_campaign_protocol.md`)

## Phase 6 — Jetson AGX Orin 배포 측정 ⬅️ 현재 단계

- 옛 측정 20.2 FPS(512×288 fp16, PyTorch)는 `Y=1` 모델 값 → **`Y=4` 재측정 필요**
- 정할 것: 보고할 지표(지연·FPS·전력·메모리), TensorRT 변환 여부, 목표 FPS
- 옛 측정 절차·환경 함정: `docs/archive/research/finetune_overfitting_diagnosis.md` §32

---

## 이후 확장 (현재 범위 밖)

- **모델:** Simple-BEV → LSS → BEVFormer (`third_party/models/`에 참고용 submodule로 확보해 둠)
- **task:** 세밀한 BEV semantic 구분, 3D bounding box 검출. 3D 검출로 확장할 때 `mmdetection3d/` submodule을 다시 쓴다
