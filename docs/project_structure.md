# 프로젝트 폴더 구조

## 전체 트리

```
bev-chamdog/
├── README.md              # 프로젝트 개요
├── AGENTS.md              # AI agent 협업 핵심 지침
├── CLAUDE.md              # Claude 호환용 포인터 (정본은 AGENTS.md)
├── ROADMAP.md             # 단계별 로드맵
├── constraints.txt        # 패키지 버전 상한 고정 (numpy<2, opencv<5)
├── requirements.txt       # Simple-BEV 학습에 실제로 필요한 의존성
├── .gitignore
├── .gitmodules            # submodule 정의
├── docs/                  # 문서 모음
│   ├── setup_guide_pro6000.md # ★ 환경 세팅 가이드 (현재 정본)
│   ├── setup_guide.md         # 〃 RTX 3080 시절 이력 (참고용)
│   ├── env/                   # pip freeze 스냅샷
│   ├── git_workflow.md        # git 브랜치 전략
│   ├── project_structure.md   # (현재 문서)
│   ├── dataset_analysis/      # WoodScape/SynWoodScape 등 분석 노트
│   └── study/                 # 배경 지식 정리 (카메라 모델·캘리브레이션 등)
├── dataset/               # 데이터셋 (내용물은 git 미추적, /data 로 symlink)
├── third_party/           # 외부 레포 (git submodule 모음)
│   ├── datasets/
│   │   └── WoodScape/         # WoodScape 공식 레포 (캘리브레이션 규약·참고 코드)
│   └── models/
│       ├── simple_bev/        # ★ 현재 baseline 모델
│       ├── lift-splat-shoot/  # 참고용 논문 구현체
│       ├── bevformer/         # 〃
│       └── BEVDet/            # 〃
├── mmdetection3d/         # MMDetection3D (submodule, v1.4.0 — 현재 미사용)
├── configs/               # 자체 실험 설정
├── tools/                 # 커스텀 스크립트
└── projects/              # 커스텀 파이썬 모듈 (로더·투영·모델 래퍼 등)
```

## 폴더별 역할

| 폴더 | 역할 |
|---|---|
| `docs/` | 모든 문서. 환경/워크플로/구조, 데이터셋 분석 노트, 배경 지식 정리. |
| `dataset/` | 원본·가공 데이터셋. 용량이 크므로 **내용물은 git으로 추적하지 않음** (`dataset/README.md`만 유지). |
| `third_party/datasets/` | 데이터셋 공식 레포(submodule). 코드에서 직접 참조하는 것만 둔다 (예: WoodScape 캘리브레이션 스크립트·규약 문서). |
| `third_party/models/` | BEV 논문 구현체(submodule). **`simple_bev/`는 현재 baseline으로 직접 사용**하며, 나머지는 참고·확장용. |
| `mmdetection3d/` | 학습 프레임워크. **현재 task(BEV occupancy)에서는 사용하지 않음.** 이후 3D 검출로 확장할 때를 위해 남겨둠. 실행 경로가 루트 기준으로 고정돼 있어 루트에 유지. |
| `configs/` | 자체 실험용 설정(하이퍼파라미터·경로·BEV 범위 등). → `configs/README.md` |
| `tools/` | 데이터 변환, 분석, 시각화, 검증 등 보조 스크립트. |
| `projects/` | 이 프로젝트 고유의 파이썬 코드(데이터 로더, BEV GT 변환, 어안 투영, 모델 래퍼 등). |

## submodule 취급

`third_party/*`와 `mmdetection3d/`는 모두 git submodule이다.

- **직접 수정 금지.** baseline(`simple_bev`)의 동작을 바꿔야 하더라도 submodule을 고치지 않고 `projects/`에서 래핑한다. submodule을 고치면 업스트림 갱신이 불가능해지고 변경 이력이 부모 저장소에 남지 않는다.
- **클론:** `git clone --recursive ...` 또는 클론 후 `git submodule update --init --recursive`.
- **버전 변경(신중히):**
  ```bash
  git -C <submodule-경로> fetch --tags
  git -C <submodule-경로> checkout <새-태그-또는-커밋>
  git add <submodule-경로>          # 부모 repo에 새 커밋 포인터 기록
  # merge/push는 사용자가 직접 (docs/git_workflow.md 규칙 준수)
  ```
- `mmdetection3d/`는 `v1.4.0`에 고정되어 있다.

## `projects/` 사용 방식

Simple-BEV는 registry나 플러그인 체계가 없는 평범한 파이썬 코드다. 따라서 `projects/`도 특별한 등록 절차 없이 **일반 파이썬 패키지**로 쓴다.

1. `projects/` 아래에 모듈을 작성한다. 현재 구성:

   | 하위 패키지 | 역할 | 주요 모듈 |
   |---|---|---|
   | `datasets/` | Simple-BEV 입력 텐서로 변환 | `synwoodscape_simplebev.py`(pretrain), `robot_simplebev.py`(자체 데이터셋), `simplebev_vox.py`(그리드↔ref 프레임), `photometric.py` |
   | `geometry/` | 카메라 모델과 좌표 프레임 | `fisheye.py`(SynWoodScape `radial_poly`), `double_sphere.py`(자체 리그 DS + ego extrinsic 체인), `frames.py`, `reprojection.py` |
   | `models/` | submodule을 건드리지 않는 래퍼 | `fisheye_vox.py`·`double_sphere_vox.py`(`Vox_util` 서브클래싱), `simplebev_two_head.py` |
   | `bev_gt/` | BEV GT 생성·판정 | `grid.py`(ROI 스펙), `visibility.py`(depth 기반 가림), `camera_coverage.py`(화각 커버리지), `ipm.py`(시각화용) |
   | `common/` | 데이터셋 비의존 공용 | `two_head_metrics.py`(지표·로깅), `bev_panels.py`(시각화 패널), `metrics.py` |

   **두 데이터셋은 렌즈 모델과 캘리브레이션 포맷이 달라 로더·투영을 분리했다**
   (SynWoodScape는 `radial_poly`, 자체 리그는 Double Sphere). 대신 그 뒤 단계인 지표·로깅·
   시각화는 `common/`에서 공유해 pretrain과 fine-tune 숫자를 나란히 읽을 수 있게 했다.
   스크립트끼리 import하지 않는 이유도 같다 — 한쪽을 고칠 때 다른 쪽이 깨지지 않도록.
2. submodule 코드를 임포트할 때는 경로를 추가한다:
   ```python
   import sys
   sys.path.insert(0, "third_party/models/simple_bev")
   ```
   (`simple_bev`는 설치형 패키지가 아니라 스크립트 저장소라 이 방식이 필요하다.)
3. **실행은 저장소 루트에서** 수행한다 (`projects`가 임포트 경로로 잡히도록).
   ```bash
   python tools/<my_script>.py
   ```

---

## `tools/` 색인 (2026-08-27, 39개)

**각 도구의 맨 위 docstring이 정본이다** -- 무엇을 왜 재는지, 어떻게 읽는지가 거기 있다.
아래는 찾아가기용 목차이고, **근거 문서**는 그 도구가 만든 숫자가 실린 절이다.

### 학습·실행

| 도구 | 무엇 |
|---|---|
| `train_robot_bev.py` | **현행 학습 진입점.** 자체 데이터셋 fine-tuning. `configs/train_robot_bev_finetune.sh`가 감싼다 |
| `train_synwoodscape.py` | SynWoodScape pretrain. **현행은 from scratch라 쓰지 않는다**(진단 §11) |
| `prune_runs.py` | 산출물 정리. **`model_best`는 재학습 말고 복구 수단이 없다**(진단 §14) |

### 재채점·집계 (판정에 쓰는 것)

| 도구 | 무엇 | 근거 문서 |
|---|---|---|
| `rescore_checkpoints.py` | 체크포인트를 다시 채점. 시퀀스별 분해도 낸다 | 진단 §20.3 |
| `summarize_repeats.py` | 반복 실험 mean±sd. **`@fixed`와 `@best`를 나란히** | 진단 §24 |
| `report_ablation.py` | loss 사다리 4칸 -- 인접 계단 Δ / 대조군 Δ / **재현성** | 설계 §15 |
| `report_threshold_sweep.py` | **τ 스윕 -- 같은 `free_miss`에서 비교.** "안전 개선"을 철회시킨 도구 | 설계 §16 |
| `report_seed_jitter.py` | 시드 간 경계 흔들림 [cm]. σ_전역 / σ_ray / **상태 불일치** | 설계 §17 |
| `report_ensemble.py` | loss 계열 교차 앙상블 | 설계 §18 |
| `report_f1_by_range.py` | **거리별 `f1@τ`.** 링 무늬 판정의 주 도구 | 진단 §27.4·§28.10·§29.9 |
| `report_pixel_offset.py` | 표본 좌표 스윕 집계 | 진단 §18.3.5 |
| **`report_loso.py`** | **LOSO fold별 집계.** 기준선 마진 주 열 + 요인 라벨 + fold SE + `min` 편향 경고 | 진단 §25.6·§31 |
| `report_convergence.py` | 되올림·수렴 곡선 | 설계 §9.3 |

### 진단 측정 (학습 불필요)

| 도구 | 무엇 | 근거 문서 |
|---|---|---|
| `analyze_error_structure.py` | 프레임별 분포 + 방위각 프로파일 + 품질별 대조 | 진단 §22 |
| `measure_boundary_regions.py` | `Ω_F`/`Ω_N`/`Ω_B` 비율, 수직 대 반경 거리 대조 | 설계 §2.1 |
| `measure_derived_occupied.py` | `occupied`가 free의 잉여인가 (binary 전환의 전제) | 진단 §15 |
| `measure_image_dependence.py` | 카메라를 섞으면 성능이 떨어지나 (0.79 → 0.37) | 진단 §18.2 |
| `measure_label_geometry.py` | IPM 대 GT 라벨 정렬 | 진단 §18 |
| `measure_lifting_resolution.py` | **표본 간격·왜곡·특징 예산.** 캘리브레이션만 쓴다 | 진단 §27 |
| **`measure_height_bin_visibility.py`** | **높이 bin이 화각 안에 들어오나.** `Y>1` 실험의 사전 확인 -- 화각 밖이면 그 bin은 항상 0이라 "변화 없음"이 가설 기각이 아니게 된다 | 진단 §33.2, compendium §12.2 |
| `measure_range_gradient.py` | `λ_R` gradient 비 캘리브레이션 | 설계 §13.3 |
| **`measure_perturbation_stability.py`** | **same-frame perturbation consistency.** 모션 성분 0인 축 | 진단 §30.1 |
| **`measure_frame_gap_stability.py`** | **frame-gap consistency diagnostic.** pose·라벨 불필요 | 진단 §30.2 |
| **`benchmark_inference.py`** | **추론 지연·FPS·메모리.** 데이터셋·체크포인트 없이 돈다 -- **Orin에서 돌리는 것이 목적** | 진단 §32 |

### 라벨 생성·검증 (초기 단계, 지금은 거의 안 쓴다)

`build_occupancy_gt` · `build_raycast_occupancy` · `build_hybrid_occupancy` ·
`build_visibility_mask` · `calibrate_bev_scale` · `verify_fisheye_projection` ·
`package_synwoodscape_2head_labels` · `smoke_test_*`

### 시각화

| 도구 | 무엇 |
|---|---|
| `visualize_robot_predictions.py` | 예측 패널(현행). **`- 0.5` 정규화 누락 사고가 있었던 경로**(진단 §18.1) |
| `render_prediction_video.py` | 전체 시퀀스 영상. `cam0..3` 매핑은 `orientation.json`이 정한다(**`left=cam3`**). **`--compare_ckpt`로 체크포인트 둘을 나란히 + 차이 지도** -- `Y`가 달라도 각자 자기 `height.json`을 따라간다 |
| `visualize_predictions.py` | SynWoodScape 시절 경로 |
| `visualize_occupancy_gt` · `visualize_camera_visibility` · `visualize_depth_overlay` | 라벨·가시성·깊이 점검 |

> **[함정] 재채점 도구에 `--encoder_type`을 반드시 맞춰 넘긴다.** 기본값이 `res101`이라
> `res101_s4` 런을 채점하려면 명시해야 하고, 안 넘기면 `load_state_dict(strict=True)`가
> 즉시 실패한다(진단 §29.3). **표본 규약은 런의 `config.json`에서 자동으로 되찾는다**
> -- 한 표에 규약이 섞이면 경고가 아니라 `SystemExit`이다(진단 §28.6).
> **표본 높이(`Y`)도 마찬가지다** -- 체크포인트 옆 `height.json`에서 되찾고, 파일이 없으면
> `LEGACY_HEIGHT_BINS = 1`(2026-08-27 이전 런)로 읽는다. 섞이면 `SystemExit`이다(진단 §33.5).
