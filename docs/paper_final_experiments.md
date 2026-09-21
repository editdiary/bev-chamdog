# 논문 최종 실험 원장

이 문서는 논문에 사용할 **최종 설정 재실험 캠페인의 단일 진행 원장**이다. 실험 계획,
동결 설정, 실행 상태, 산출물 위치, 무결성 검사, 논문용 결과를 여기서 계속 갱신한다.
개별 숫자의 기계 판독 원본은 각 실험의 `runs/paper_final/*/analysis/RESULTS.json`이고,
이 문서는 그 출처를 링크하고 사람이 읽는 결론을 기록한다.

> **상태 기준일:** 2026-09-18  
> **작업 브랜치:** `exp/paper-final`  
> **실행 GPU:** `CUDA_VISIBLE_DEVICES=0`  
> **정본 환경:** conda `bev-chamdog` · Python 3.11 · torch 2.7.0+cu128  
> **대용량 산출물 루트:** `/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final`

---

## 1. 목적과 운영 원칙

과거 실험은 lifting 높이, loss 매개변수, range-loss 설정이 확정되기 전 결과와 확정 후
결과가 섞여 있다. 논문 표에 사용할 결과는 `runs/paper_final/`에서 다음 규약으로 다시
관리한다.

1. **설정을 기본값에 맡기지 않는다.** 실행 스크립트가 논문 설정을 전부 명시한다.
2. **주 결과는 고정 epoch 40이다.** validation 최고 epoch 결과는 진단용으로만 병기한다.
3. **시드별 원자료를 버리지 않는다.** 평균·표준편차만 남기지 않고 각 런의 값을 보존한다.
4. **threshold 적용 전 확률맵을 보존한다.** 새 threshold나 영역 지표는 재학습 없이 계산한다.
5. **학습 로그와 내보낸 확률맵을 독립 경로로 재채점한다.** 무결성 검사가 실패한 결과는
   논문 숫자로 사용하지 않는다.
6. **constant-map baseline을 항상 병기한다.** 온실의 규칙적 레이아웃을 외운 상수 지도와
   비교하지 않은 `iou_free`는 단독으로 보고하지 않는다.
7. **실행 중 계획을 바꾸지 않는다.** 변경이 필요하면 이 문서에 이유와 절단선을 먼저 남기고
   새 실험 ID를 만든다. 완료된 런을 새 설정과 섞지 않는다.
8. `runs/`는 Git 비추적 대용량 저장소다. 재현 가능한 스크립트·문서·검증 도구는 Git에,
   체크포인트·TensorBoard·예측값은 `runs/paper_final/`에 둔다.

---

## 2. 캠페인 구성

| ID | 논문 역할 | 평가 프로토콜 | 상태 |
|---|---|---|---|
| `01_overall` | 최종 모델의 대표 성능과 시퀀스 일반화 | 고정 split × 5 seeds + LOSO 7 folds × 5 seeds | **완료: 40/40, 무결성 통과, bundle 생성** |
| `02_sensor_task_adaptation` | 카메라 모델·source prior·센서 범위 불일치 | 고정 split × 5 seeds | 계획 |
| `03_boundary_uncertainty` | boundary-aware loss의 효과와 원인 | 기존 5조건 × 5 seeds 재사용 | 원자료 존재, 논문용 재정리 대기 |
| `04_edge_deployment` | Jetson AGX Orin 지연·FPS·전력·메모리 | Orin 반복 측정 | 사용자 장비 실행 대기 |

고정 split은 모든 통제 ablation의 공통 benchmark다. LOSO는 최종 모델 하나에 대해서만
“처음 보는 시퀀스” 일반화와 fold 변동을 측정한다. 두 프로토콜은 train 크기와 질문이 다르므로
숫자를 합치거나 직접 우열 비교하지 않는다.

---

## 3. 전 실험 공통 최종 설정

아래는 `01_overall`의 두 프로토콜과 이후 adaptation 대조군이 공유하는 설정이다.

| 축 | 동결값 |
|---|---|
| 정식화 | binary `free` / `not-free` |
| 모델 | Simple-BEV, ResNet-101, encoder stride 8, latent dim 128 |
| 초기화 | ImageNet encoder + 랜덤 BEV decoder (`INIT_CHECKPOINT=none`) |
| 입력 | 3-camera `front,left,right`, 512×288, Double Sphere native projection |
| BEV | 120×120, 5 cm, 전방 4 m / 후방 2 m / 좌우 ±3 m |
| lifting | `Y=4`, 높이 −0.25~1.75 m, 표본 0/0.5/1.0/1.5 m |
| 특징 표본 | `pixel_center`, offset 0 |
| loss | `soft_boundary`, gaussian target |
| boundary | `DELTA_M=0.30`, `SIGMA_M=0.10`, `LAMBDA_B=0.5`, `BAND_KAPPA=1.0`, `LABEL_EPS=0.0` |
| range 보조항 | `LAMBDA_R=0.3`, `DELTA_R_M=0.15`, 대칭, `HUBER_BETA_M=0.15` |
| 최적화 | AdamW, lr 1e-4, weight decay 1e-7, batch 8, 40 epochs |
| 증강 | photometric on, flip off |
| 주 체크포인트 | 고정 epoch 40 |
| 학습 시드 | 0, 1, 2, 3, 4 |

논문에서 `scratch`라고 축약할 경우에도 실제 의미는 완전 무작위 초기화가 아니라
**target-only training without BEV pretraining**임을 명시한다.

---

## 4. 실험 01 — Overall Performance

### 4.1 질문

1. 미리 정한 대표 split에서 최종 모델의 성능과 학습 시드 산포는 얼마인가?
2. 7개 각 시퀀스를 처음 보는 데이터로 두었을 때 성능과 fold 변동은 얼마인가?
3. 이미지를 보지 않는 constant-map baseline보다 각 fold에서 얼마나 나은가?

### 4.2 A — 고정 split, 시드 5개

- train: `raws2,raws3,rawos1,rawos2,rawos4` (192 frames)
- validation: `raws1,rawos3` (75 frames)
- seeds: `0,1,2,3,4`
- 런 수: 5
- 결과 루트: `runs/paper_final/01_overall/fixed_split/`

이 결과가 논문의 대표 benchmark 성능이며 이후 sensor/task adaptation 표의 공통 대조군이다.
모든 arm은 같은 split과 같은 시드를 사용하고, seed가 같은 런끼리 paired difference를 계산한다.

### 4.3 B — LOSO, 7 folds × 시드 5개

- folds: `raws1,raws2,raws3,rawos1,rawos2,rawos3,rawos4`
- 각 fold: held-out 시퀀스 하나를 validation으로, 나머지 6개를 train으로 사용
- seeds: `0,1,2,3,4`
- 런 수: 35
- 결과 루트: `runs/paper_final/01_overall/loso/`

LOSO에서는 held-out 시퀀스로 epoch를 고르면 test fold에 맞춘 선택 편향이 생긴다. 따라서
**epoch 40만 주 결과**로 쓴다. fold별로 먼저 seed 평균을 구하고, 본문에는 다음을 보고한다.

- 7-fold macro mean과 fold 간 표준편차
- 최저/최고 fold
- fold별 constant-map baseline 및 모델−baseline 마진
- 외삽 fold인 `raws3` 결과
- fold 내부 seed 표준편차의 요약

7개 fold는 train 데이터를 6/7씩 공유하므로 `SD/√7`을 독립 표본의 표준오차처럼 쓰지 않는다.
fold 표준편차는 기술통계이고, seed별·fold별 전체 행은 appendix용 CSV에 남긴다.

### 4.4 주 지표와 보조 지표

| 우선순위 | 지표 | 보고 규칙 |
|---|---|---|
| 주 | `iou_free` | constant-map baseline과 반드시 병기 |
| 안전 동반 | `fatal_rate` | `free_miss_rate`와 같은 동작점에서 함께 보고 |
| 보수성 | `free_miss_rate` | `fatal_rate`와 분리해 단독 개선 주장 금지 |
| 경계 일치 | `f1@10cm`, `f1@20cm`, `f1@40cm` | 라벨과의 일치도이며 물리 GT 정확도로 해석하지 않음 |
| 광선 | `range_mae`, `range_bias`, `missed_obstacle_rate` | 셀 지표와 다른 실패 양상 진단 |

### 4.5 산출물 계약

각 프로토콜의 `analysis/`에는 최소한 다음을 둔다.

| 산출물 | 역할 |
|---|---|
| `README.md` | 사람이 어디부터 읽을지와 핵심 결과 안내 |
| `RESULTS.json` | 환경·설정·무결성·런별 값·집계를 담은 자기 설명적 결과 |
| `RESULTS.csv` | 논문 표와 pandas/R 분석용 긴 형식 |
| `configs.json` | 모든 런의 해석된 전체 설정 |
| `manifest.json` | 완료 여부, epoch 수, split 크기, 체크포인트, 파일 hash |
| `scalars.csv` | 런 × epoch × TensorBoard 전 지표 |
| `predictions/` | threshold 전 `p(free)`, GT, valid mask, 거리장 |
| `verify_predictions.json` | 확률맵 재채점과 학습 로그 대조 |
| `report_*.txt` | 사람이 읽는 표의 콘솔 출력 전문 |

체크포인트는 고정 epoch 40을 반드시 보존한다. validation-best 체크포인트는 진단·대조를 위해
보존할 수 있지만, LOSO 논문 주 결과에는 사용하지 않는다.

### 4.6 실행 전 검증 게이트

- [x] 격리 worktree `exp/paper-final` 생성
- [x] 정본 conda/Python/torch/CUDA/numpy 확인
- [x] 데이터셋 로컬 연결 확인
- [x] baseline test: 409 passed, 7 skipped
- [x] 논문용 실행 스크립트와 분석 스크립트 구현
- [x] 1 epoch dry-run: 고정 split seed 0
- [x] 1 epoch dry-run: LOSO fold 2개, seed 0
- [x] dry-run config·split·`height.json`·확률맵 무결성 확인
- [x] 40런 본 실행 시작
- [x] 40런 본 실행 완료 (2026-09-18 17:35 KST)
- [x] 완료 산출물 게이트 (epoch 40, Y=4, 설정 균일성, split 구성)
- [x] 확률맵 무결성 8건 전부 통과
- [x] 결과 bundle 생성

---

## 5. 진행 기록

### 2026-09-18 — 캠페인 설계 동결

- 사용자가 `runs/paper_final/` 아래 실험별 구조를 승인했다.
- ablation은 고정 split × 5 seeds, 최종 모델은 추가로 LOSO를 수행하기로 했다.
- LOSO seed 수를 3에서 **5로 상향**했다.
- GPU 0을 사용한다.
- 기존 작업공간의 미커밋 변경과 분리하기 위해 `/tmp/bev-chamdog-paper-final`에
  `exp/paper-final` worktree를 만들었다.
- 새 worktree의 첫 test collection은 submodule 미초기화로 실패했다. 원 저장소와 비교해
  `simple_bev`와 `WoodScape` gitlink가 비어 있음을 확인하고 고정 커밋을 checkout했다.
  이후 전체 테스트가 **409 passed, 7 skipped**로 통과했다. 코드 결함은 없었다.

### 2026-09-18 — `01_overall` 1-epoch dry-run 통과

- 실행 위치: `runs/paper_final/01_overall_dryrun/`
- 실행 전 환경: GPU 0 `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`,
  torch `2.7.0+cu128`, CUDA build 12.8, numpy 1.26.4. 실제 CUDA tensor 연산까지 통과했다.
- 고정 split `final_s0`: train/val **192/75 frames**, epoch 1 체크포인트와 best
  체크포인트 생성. 해석된 config에서 확정 loss·ResNet-101·증강 설정을 확인했고,
  `height.json`은 `Y=4`, −0.25~1.75 m였다.
- LOSO `loso_raws1_s0`: train/val **229/38 frames**. train 6개 시퀀스에 held-out
  `raws1`이 없고 val에는 `raws1`만 있음을 확인했다.
- LOSO `loso_raws3_s0`: train/val **231/36 frames**. 같은 방식으로 `raws3`의
  train/val 분리와 Y=4 메타데이터를 확인했다.
- 세 런 모두 고정 epoch 예측을 fold별 `labels.npz`와 `prob_free` NPZ로 내보냈다.
  학습 로그와 독립 재채점한 8개 지표의 최대 절대 차이는 각각 고정 split
  **1.91e-4**, LOSO raws1 **2.40e-4**, LOSO raws3 **2.78e-4**였고, 허용치 1e-3
  안에서 **실패 0건**이었다. 행 순서·Y=4 기하·체크포인트 선택을 함께 검증했다.
- dry-run 후 전체 회귀 테스트는 **421 passed, 7 skipped**였다.

### 2026-09-18 — `01_overall` 40런 본 실행 시작

- 시작: 2026-09-18 13:41 KST, GPU 0, 결과 루트 `runs/paper_final/01_overall/`.
- 실행 Git commit: `9f7ced8945a7cb0e35ad67d4343cb038e84f427a`.
- 시작 manifest를 감사했다: 고정 split 5런 + LOSO 35런 = **40개 고유 런**, seeds
  0~4, 7 folds 각각 5런이며 split/런 식별자를 제외한 30개 환경 설정이 모두 같았다.
- 실행 manifest가 기록한 정본 환경은 Python 3.11.15, torch 2.7.0+cu128, CUDA 12.8,
  numpy 1.26.4, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition이다.
- 첫 런 `final_s0`는 epoch 40까지 정상 완료했고 고정 epoch `iou_free=0.811`이었다.
  이는 진행 상태 확인값이며, 논문 숫자는 5개 시드 완료 후 결과 bundle에서만 확정한다.

### 2026-09-18 — 사용자 요청에 따라 3/40에서 일시 중단

- 완료 런: `final_s0`, `final_s1`, `final_s2`. 세 런 모두
  `model-000000040.pth`와 Y=4 메타데이터가 있다. 다음 런 `final_s3`는 시작하지 않았다.
- 최초 대화 세션에 연결된 PTY 출력은 읽지 않을 때 버퍼가 차서 큐가 대기하는 문제가 있었다.
  장시간 실행은 `training_queue.log`로 표준출력을 보내는 방식으로 바꿨다.
- 출력 방식 변경 중의 `final_s2` 2-epoch 산출물은 삭제하지 않고
  `archive_interrupted_20260918_1353/`에 보존했다. 이후 `final_s2`를 처음부터 다시 실행해
  epoch 40까지 정상 완료했다.
- 중단 시 GPU 0은 사용률 0 %, 메모리 843 MiB로 유휴 상태였다.
- 재개 시 같은 전체 명령을 사용하되 출력을 파일로 보낸다. orchestrator가 완료된 세 런을
  자동으로 건너뛰고 `final_s3`부터 시작한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall \
PROTOCOL=all SEEDS=0,1,2,3,4 NUM_EPOCHS=40 \
    bash configs/paper_final_overall.sh \
    > /data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall/training_queue.log 2>&1
```

### 2026-09-18 — `01_overall` 40런 완료·검증·결과 확정

재개 실행 Git commit `78da7364f78221cd07f60c4744bf66de2d1a6379`. 재개 14:06:49 KST,
학습 종료 17:35:43 KST(37런 3시간 29분), 분석 17:37:22~17:41:16(약 4분).
출력은 `training_queue.log`(학습)와 `analysis_driver.log`(분석)에 있다.

**실행 결과.** `queue_status.json`은 `completed` 37건 + `skipped_complete` 3건 = 40건이고
`failed`·`blocked_incomplete`는 0건이다. 고정 split 5개, LOSO 35개 디렉터리 모두
`model-000000040.pth`를 갖는다.

**산출물 게이트(분석 전).** config 40개 전부 `num_epochs=40`, `height.json` 40개가
모두 `{Y=4, −0.25~1.75 m}` 단일 기하, 런 이름·seed·split을 제외한 설정 차이 **0건**,
LOSO 7 fold 각각 5런이며 held-out 시퀀스가 train에 섞이지 않았고 train은 항상 6시퀀스다.

**무결성.** 확률맵 재채점 JSON 8건(고정 split 1 + LOSO fold 7) 모두 `failures=0`.
최대 절대 차이는 고정 split **3.47e-4**(파일 10개), LOSO **5.16e-4**(파일 70개)로
허용치 1e-3 안이다. `analysis/RESULTS.json`의 `integrity.passed`는 `true`이고
`missing_files`는 비어 있다.

**A — 고정 split (epoch 40, n=5).** `raws2,raws3,rawos1,rawos2,rawos4` → `raws1,rawos3`,
192/75 frames.

| 지표 | 값 |
|---|---|
| `iou_free` | **0.8119 ± 0.0018** |
| constant-map baseline | **0.5180** |
| margin | **+0.2938** |
| `fatal_rate` | 0.1204 ± 0.0015 |
| `free_miss_rate` | 0.0844 ± 0.0010 |
| `f1@10cm` / `@20cm` / `@40cm` | 0.5981 ± 0.0058 / 0.8313 ± 0.0026 / 0.9290 ± 0.0019 |
| `range_mae` / `range_bias` | 0.2117 ± 0.0036 m / +0.0518 ± 0.0034 m |
| `range_missed_obstacle_rate` | 0.0377 ± 0.0019 |

진단용 validation-best는 `iou_free` 0.8126 ± 0.0013(best epoch 27/15/30/40/36)으로
고정 epoch 40과 0.0007 차이다. 주 결과는 §1-2에 따라 epoch 40이다.

고정 split의 constant-map baseline은 학습 스크립트가 콘솔에만 찍고 TensorBoard에 남기지
않아 bundle에서 빠져 있었다. `report_loso.fold_baseline`과 같은 계산(train 시퀀스의 셀별
다수결 free map을 val에 채점)으로 다시 구해
`fixed_split/analysis/constant_map_baseline.json`에 저장했다. §1-6이 baseline 병기를
요구하므로 이 값 없이는 `iou_free`를 단독 보고하지 않는다.

**B — LOSO (epoch 40, 7 folds × 5 seeds).**

| fold | 조명 | 통로폭 | 외삽 | n_val | `iou_free` | baseline | margin | `fatal_rate` | `free_miss_rate` |
|---|---|---|---|---|---|---|---|---|---|
| `raws1` | 햇빛 | 좁음 | | 38 | 0.8221 ± 0.0014 | 0.6304 | +0.1917 | 0.1012 ± 0.0022 | 0.0877 ± 0.0016 |
| `raws2` | 햇빛 | 좁음 | | 41 | 0.8723 ± 0.0013 | 0.6705 | +0.2018 | 0.0603 ± 0.0018 | 0.0869 ± 0.0050 |
| `raws3` | 햇빛 | 넓음 | **O** | 36 | 0.8220 ± 0.0035 | 0.3854 | **+0.4366** | 0.1068 ± 0.0053 | 0.0901 ± 0.0053 |
| `rawos1` | 가림막 | 좁음 | | 39 | 0.8485 ± 0.0016 | 0.6845 | **+0.1640** | 0.0890 ± 0.0020 | 0.0858 ± 0.0016 |
| `rawos2` | 가림막 | 좁음 | | 40 | 0.8401 ± 0.0036 | 0.6074 | +0.2327 | 0.0906 ± 0.0033 | 0.1033 ± 0.0009 |
| `rawos3` | 가림막 | 넓음 | | 37 | 0.8066 ± 0.0024 | 0.3912 | +0.4154 | 0.1260 ± 0.0030 | 0.0830 ± 0.0039 |
| `rawos4` | 가림막 | 넓음 | | 36 | **0.8012 ± 0.0031** | 0.3875 | +0.4137 | 0.1042 ± 0.0016 | 0.1186 ± 0.0041 |
| **macro** | | | | | **0.8304**, fold SD 0.0249 | 0.5367, fold SD 0.1413 | **+0.2937**, fold SD 0.1218 | 0.0969, fold SD 0.0203 | 0.0936, fold SD 0.0128 |

- fold 내부 seed SD 평균은 `iou_free` **0.0024**, `fatal_rate` 0.0027, `free_miss_rate` 0.0032이다.
- fold 간 SD는 fold 내부 seed SD의 **10.3배**다(0.0249 / 0.0024).
- `iou_free` 최고 `raws2` 0.8723, 최저 `rawos4` 0.8012, 차 0.0710.
- margin 최고는 외삽 fold `raws3` +0.4366, 최저는 `rawos1` +0.1640이다.
- 통로폭별 평균: 좁음 4 fold는 baseline 0.6482 / margin +0.1975, 넓음 3 fold는
  baseline 0.3880 / margin +0.4219다. 조명별 차이는 이보다 작다(햇빛 baseline 0.5621 /
  margin +0.2767, 가림막 0.5176 / +0.3064).

**해석(사실과 분리).** 위 표는 측정값이고, 아래는 읽는 방식이다.

- **raw `iou_free` 순위와 margin 순위가 뒤집힌다.** `rawos1`은 `iou_free` 0.8485로 상위지만
  margin은 +0.1640으로 7개 중 최저이고, `raws3`는 `iou_free` 0.8220으로 중하위지만 margin은
  +0.4366으로 최고다. baseline 없이 fold를 줄세우면 결론이 반대가 된다.
- **baseline 변동(fold SD 0.1413)이 모델 변동(0.0249)보다 5.7배 크다.** 즉 fold 간 raw
  `iou_free` 차이는 모델의 일반화 능력보다 "그 장면이 상수 지도로 얼마나 잘 맞춰지는가"를
  더 많이 반영한다. 넓은 통로 fold에서 baseline이 낮은 것이 그 구조다.
- **외삽 fold `raws3`**(햇빛 O · 넓은 통로 조합이 train에 없는 유일한 fold)는 `iou_free`
  0.8220으로 macro 평균 0.8304보다 0.0084 낮은 데 그쳤고 margin은 가장 컸다. 다만 fold
  하나의 값이므로 §4.3의 단서(표본 오차·라벨 품질을 분리할 수 없음)가 그대로 적용된다.
- **두 프로토콜의 margin이 거의 같다**(고정 split +0.2938, LOSO macro +0.2937). 우연의
  일치일 수 있고 train 크기(192 대 229~231 frames)와 질문이 다르므로 §2에 따라 합치거나
  직접 우열을 비교하지 않는다.
- fold 간 차이의 원인은 장면 난이도·표본 오차·시퀀스별 라벨 품질 셋이며 분리할 수 없다.
  `SD/√7`을 독립 표본 표준오차로 쓰지 않는다.

**산출물 경로.** 루트 `runs/paper_final/01_overall/`.

| 경로 | 내용 |
|---|---|
| `analysis/RESULTS.json` · `RESULTS.csv` · `README.md` | 상위 색인과 무결성 요약(두 프로토콜을 합치지 않음) |
| `fixed_split/analysis/RESULTS.json` · `RESULTS.csv` | 런별 값 + epoch 40 mean/SD + best-epoch 진단 |
| `fixed_split/analysis/constant_map_baseline.json` | 고정 split constant-map baseline |
| `loso/analysis/RESULTS.json` · `RESULTS.csv` · `report_loso.json` | fold별 seed 원자료·mean/SD·baseline·margin·macro |
| `*/analysis/scalars.csv` · `configs.json` · `manifest.json` | 런 × epoch 전 지표, 해석된 설정, 완료·hash |
| `fixed_split/analysis/predictions/` (10개) · `loso/analysis/predictions/{fold}/` (70개) | threshold 전 `p(free)`, GT, valid mask, 거리장 |
| `fixed_split/analysis/verify_predictions.json` · `loso/analysis/verify_predictions_{fold}.json` | 재채점 대조 8건, 전부 `failures=0` |
| `training_queue.log` · `analysis_driver.log` | 학습·분석 콘솔 전문 |

**논문 작성용 패키지.** 위 산출물 중 논문에 필요한 것만 추려 `docs/paper_package/`에
모았다(1 MB 미만, 가중치·확률맵 제외). 노트북에 통째로 받아 서버 접속 없이 논문을 쓸 수
있게 구성했다. 실험이 넷이므로 **실험별 폴더는 독립으로 두되 공통 사항은 `common/`에만**
둔다 — 지표 정의를 실험마다 복사하면 하나만 고쳤을 때 나머지가 조용히 틀려진다.

```
docs/paper_package/
├── README.md                 색인 + 실험 폴더 규약
├── common/setup.md           과제·시퀀스 7개·라벨 출처·동결 설정·환경·무결성 방식
├── common/metrics.md         지표 정의·range_bias 부호 규약·보고 규칙
└── 01_overall/               REPORT.md · README.md · make_package.py · data/ · figures/ · provenance/
```

`data/`의 CSV는 `make_package.py`가, Figure 1은 `figures/make_margin_inversion.py`가
각각 정본 생성 스크립트다 — **내보낸 CSV와 그림을 손으로 고치지 않는다.** 문서의 수치
57건을 생성된 CSV와 자동 대조해 전부 일치함을 확인했고, 그림 색은 colorblind 검사를
통과한 조합이다. 노트북 재생성 의존성은 matplotlib 하나다.

디스크는 `loso` 33 GB, `fixed_split` 4.7 GB다. 출력 방식 변경 중 생긴 `final_s2`의
2-epoch 산출물은 `archive_interrupted_20260918_1353/`(471 MB)에 그대로 두었고 논문
숫자에는 쓰지 않는다.

### 2026-09-21 — worktree를 `/tmp`에서 Desktop으로 옮김

사용자가 `docs/paper_package/`가 자기 폴더에서 보이지 않는다고 지적했다. 원인은 캠페인
worktree가 `/tmp/bev-chamdog-paper-final`에 있었기 때문이다. 실행 중에는 메인 체크아웃의
미커밋 변경과 분리하는 합리적인 격리였지만, 산출물을 전달하는 단계에서는 잘못된 위치였다.

`/tmp`의 정리 정책이 `D /tmp 1777 root root 30d`였다 — `D`는 **부팅 시 내용 삭제**,
`30d`는 30일 경과 파일의 주기적 삭제다. uptime이 6주여서 아직 남아 있었을 뿐이고,
재부팅 한 번이면 원장과 패키지가 사라질 상태였다.

`/data/home/dhlee/Desktop/bev-chamdog-paper-final`으로 옮겼다. `git worktree move`는
submodule이 있는 worktree를 거부하므로(`working trees containing submodules cannot be
moved`) 수동 `mv` 후 `git worktree repair`를 썼다. 그 뒤 submodule 두 개
(`simple_bev`, `WoodScape`)가 깨졌는데, `.git` 파일의 gitdir이 **루트 기준 상대경로**
(`../` 5개)라 디렉터리 깊이가 달라지며 어긋난 것이었다. 양방향 포인터를 절대경로로 고쳤다.

- `<checkout>/.git` 의 `gitdir:` → `<main>/.git/worktrees/bev-chamdog-paper-final/modules/<name>`
- 그 모듈 `config` 의 `core.worktree` → 새 체크아웃 경로

이동 후 `git status`·`git submodule status` 정상, submodule 고정 커밋 동일, 전체 테스트
**421 passed, 7 skipped**로 이동 전과 같았다. 패키지 데이터·그림 재생성과 수치 대조도
새 경로에서 다시 통과했다.

전달 방식은 사용자가 "서버에 두면 직접 받아가겠다"로 정했다. 따라서 tar 묶음은 만들지
않는다(실험이 추가되면 즉시 낡기 때문이다). 폴더를 그대로 가져간다.

### 다음 작업

`01_overall`은 종료됐다. 남은 것은 `02_sensor_task_adaptation`(§6이 정리한 미동결 사항을
먼저 확정해야 한다), `03_boundary_uncertainty` 재정리, `04_edge_deployment`의 Orin 측정이다.
`04`는 배포 메모의 20.2 FPS가 `Y=1` 값이므로 `Y=4` 재실측이 필요하다.

재현 진입점:

```bash
# 학습 (완료된 런은 자동으로 건너뛴다)
CUDA_VISIBLE_DEVICES=0 \
OUT_ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall \
PROTOCOL=all SEEDS=0,1,2,3,4 NUM_EPOCHS=40 \
    bash configs/paper_final_overall.sh

# 분석·무결성·bundle
CUDA_VISIBLE_DEVICES=0 \
ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall \
    bash configs/paper_final_overall_analysis.sh
```

---

## 6. 아직 동결하지 않은 후속 실험 사항

아래는 `01_overall` 실행을 막지 않으며, 해당 실험을 시작하기 전에 별도로 확정한다.

- `02` DS-native 대 undistort+pinhole의 virtual pinhole FOV와 리샘플링 규약
- `02` SynWoodScape pretraining의 source 정식화·source seed 처리·전이 범위
- `02` front-only의 고정 3-camera task 평가와 front 공통 가시영역 보조 평가
- `04` Orin warm-up, 반복 횟수, latency percentile, 전력 측정 명령
