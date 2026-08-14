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

## Phase 3 — SynWoodScape로 Simple-BEV 학습 ✅ 완료

목적은 성능 최적화가 아니라, **어안 4-cam 입력 → BEV occupancy 예측 파이프라인이 실제로 동작함을 증명**하는 것이다.

### 3.1 데이터 파이프라인

- **occupancy GT 생성 ✅ 완료** — `road`(7) + `road line`(6) → drivable, 그 외는 obstacle로
  remap. `unlabeled`(0)·`ego-vehicle`(24) 등은 loss에서 제외(ignore)하기로 결정 — top-down
  `_BEV.png` label(class 값) + 4-cam 어안 depth raycast(observed/visible mask)를 합친
  하이브리드 방식(`tools/build_hybrid_occupancy.py`)으로 확정. BEV GT의 미터/픽셀 스케일도
  `15/512 m/px`로 확정. 학습 범위·격자 해상도는 `SYNWOODSCAPE_PRETRAIN_GRID_SPEC`(전방5/후방3/
  좌우±4m=8m×8m, 0.05m/cell, 160×160)로 확정. 500 samples 전체를 생성해
  `dataset/synwoodscape_occupancy_gt/`에 `{sample}_occupancy.npy`/`_visible.npy`/
  `_combined.png`로 저장 완료. 상세 근거·설계 변천사는 `docs/dataset_analysis/
  synwoodscape_geometry_findings.md` §3~§4 참고.
- **SynWoodScape → Simple-BEV 데이터로더 ✅ 완료** — `projects/datasets/`. 캘리브레이션에서
  임시 핀홀 `pix_T_cams`(k1을 focal length로 근사, 실제 `radial_poly` 교체는 3.2), 검증된
  ego 카메라 pose를 Simple-BEV `Z/Y/X` 기준 프레임으로 회전시킨 `cam0_T_camXs`(`Y=1`, 지면
  높이 한 bin), `SynWoodScapeSimpleBEVDataset`을 구현. 4-cam(FV/MVL/MVR/RV) 그대로 사용 —
  Simple-BEV는 카메라별 전용 파라미터가 없어(encoder 공유 + masked-mean fusion) 이후 자체
  로봇 3-cam(후면 제외) fine-tuning으로 전환해도 구조적으로 문제 없음.
  `tools/smoke_test_simplebev_dataloader.py`로 `Segnet.forward()`+`backward()`가 실제
  데이터에서 shape 에러 없이 도는 것과, 카메라별 BEV grid 커버리지가 실제 장착 방향과
  맞는지(기하 정합성)를 확인 완료.
- **500 samples train/val split ✅ 완료** — `projects/datasets/synwoodscape_split.py`.
  `vehicle_data`의 ego 위치를 실측한 결과 500장은 **하나의 연속 시퀀스가 아니라** 맵 전역에
  흩어진 길이 1~5의 짧은 버스트(64개, 인접 인덱스 간 거리 중앙값 12.7m)였다. 버스트를
  통째로 train/val 중 한쪽에만 배정하는 근접도 기반 클러스터 분할로 근접 프레임 누수를 막음.

### 3.2 어안 투영 주입 ✅ 완료

- **`projects/models/fisheye_vox.py`** — Simple-BEV `Vox_util`을 서브클래싱해 `unproject_image_to_mem`
  하나만 오버라이드(`FisheyeVoxUtil`). 원본은 외파라미터+내파라미터를 합친 4x4 행렬에
  원근분할을 적용해 픽셀 좌표를 얻는데(핀홀 전용), 별도 인자로 들어오는 `camB_T_camA`(외파라미터만,
  강체변환)는 그대로 재사용하고 그 카메라좌표계 3D 점에 실제 WoodScape `radial_poly`(theta/rho
  다항식)를 적용해 픽셀을 계산하도록 바꿨다. **submodule(`nets/segnet.py`, `utils/vox.py`)은
  한 줄도 수정하지 않음** — `Segnet.forward()`가 이 `FisheyeVoxUtil` 인스턴스를 그대로 받아 쓴다.
- **검증**: torch 재구현이 WoodScape 공식 numpy `RadialPolyCamProjection`과 1e-6 오차로 일치
  (`tests/models/test_fisheye_vox.py`). 3.1의 핀홀 근사 대비 BEV grid 이미지-경계 커버리지가
  기대대로 크게 늘어남을 확인(FV 0.29→0.45, MVL/MVR 0.24→0.77, RV 0.07→0.25 — 광각 미러
  카메라일수록 핀홀이 화각을 심하게 과소평가했던 만큼 개선폭이 큼).
  `tools/smoke_test_fisheye_segnet.py`로 실제 데이터에서 `Segnet.forward()`+`backward()`가
  shape 에러 없이 도는 것을 batch_size=1(peak GPU 메모리 2.73 GiB)로 확인 — 다른 연구실이
  같은 GPU를 쓰고 있어 메모리를 가볍게 유지함.

### 3.3 학습 & 평가 — 스크립트 준비 ✅, 본 학습 실행은 다음 단계

- **`tools/train_synwoodscape.py` ✅ 완료** — Simple-BEV 원본(`train_nuscenes.py`) 관례(Fire
  키워드 인자 + `configs/train_synwoodscape_baseline.sh` 셸 스크립트)를 따름. AdamW +
  OneCycleLR, BCE(자동 계산 `pos_weight`) + `saverloader` 체크포인트 + tensorboard 로깅.
  `--use_fisheye`로 Phase 3.2 실제 어안 투영(`FisheyeVoxUtil`)과 3.1 핀홀 근사를 전환 가능.
- **지표를 drivable IoU 하나에서 drivable+obstacle IoU 둘로 확장.** 실측 결과 500 samples
  전체 기준 drivable 비율이 93%대라 "항상 drivable로 예측"만 해도 drivable IoU가 0.93을
  넘는다(trivial baseline) — obstacle IoU를 함께 보지 않으면 trivial 해와 실제 학습을
  구분할 수 없다. 체크포인트 선정도 `(drivable_iou+obstacle_iou)/2` 최고 시점 기준으로 변경.
  상세 근거·튜닝 가이드는 **`docs/training_guide.md`**.
- 학습 루프(옵티마이저 스텝·스케줄러·val 루프·체크포인트 저장·tensorboard 기록)를
  8-sample/2-epoch smoke run으로 GPU에서 실제 검증 완료. **500 samples 전체로 실제 학습을
  돌리고 결과(수렴 여부, 최종 IoU, 예측 BEV 육안 확인)를 보는 것은 아직 안 함 — 다음 단계.**
- `rand_flip`은 껐다(고정) — `SYNWOODSCAPE_PRETRAIN_GRID_SPEC`이 전후 비대칭이라 Simple-BEV의
  Z축 flip 증강이 물리적으로 안 맞음(`docs/training_guide.md` §6).
- 하드웨어: RTX PRO 6000 96GB, 단 현재 다른 연구실 job과 공유 중(여유 ~25GB) — 배치
  크기·encoder 선택 시 이 여유를 고려(`docs/training_guide.md` §5). VRAM이 완전히 여유로워지면
  batch size 확대·gradient accumulation 불필요 원칙(CLAUDE.md)으로 돌아간다.
- 성공 기준: 학습이 수렴하고(특히 obstacle IoU가 trivial baseline인 0을 유의미하게 넘김),
  예측 BEV가 GT와 육안으로도 정합한다

## Phase 4 — 자체 데이터셋 fine-tuning ⬅️ 현재 단계

**파이프라인 ✅ 완료 (2026-08-14).** 실행 방법·지표 해석·문제 대응은
[`docs/finetuning_guide.md`](docs/finetuning_guide.md)를 정본으로 본다.
설계 근거는 [`docs/finetuning_preparation.md`](docs/finetuning_preparation.md).

- Phase 3의 체크포인트를 **pre-trained weight**로 사용 ✅ — 240×240 4-cam → 120×120 3-cam으로
  0 missing / 0 unexpected 로드 확인
- 자체 데이터셋 GT를 Phase 3 로더와 **같은 인터페이스**로 변환 ✅ —
  `projects/datasets/robot_simplebev.py`. 시퀀스 단위 split
- 자체 카메라의 어안 모델을 투영 모듈에 연결 ✅ — **Double Sphere**였다.
  `projects/geometry/double_sphere.py` + `projects/models/double_sphere_vox.py`.
  extrinsic 체인은 IPM을 어노테이션 프로젝트 출력과 대조해 검증
- drivable 정의는 **온실 환경 기준으로 새로 정의** ✅ — 수동 어노테이션 기준
  (`0=obstacle / 1=drivable`)
- 관측 불가 영역 마스킹 ✅ — 배포에도 남는 가림은 `vis=0`, 수집 아티팩트는 `valid=0`으로
  분리 (가이드 §3)
- sim → real 도메인 갭 대응 ⬅️ **남은 작업.** 현재 병목은 코드가 아니라 어노테이션 물량이다.
  raws1 38장에서는 best checkpoint가 pretrain 초기값 그 자체였다

## Phase 5 — 분석 · 개선 · 문서화

- 결과 분석, 실패 사례 정리, 개선 반복
- 재현 가능한 형태로 문서/코드 정리

---

## 이후 확장 (현재 범위 밖)

- **모델:** Simple-BEV → LSS → BEVFormer (`third_party/models/`에 참고용 submodule로 확보해 둠)
- **task:** 세밀한 BEV semantic 구분, 3D bounding box 검출. 3D 검출로 확장할 때 `mmdetection3d/` submodule을 다시 쓴다
