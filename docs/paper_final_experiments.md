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
| `01_overall` | 최종 모델의 대표 성능과 시퀀스 일반화 | 고정 split × 5 seeds + LOSO 7 folds × 5 seeds | **dry-run 통과, 40런 실행 대기** |
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
- [ ] 40런 본 실행 시작

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

### 다음 작업

`01_overall` 실행 orchestrator는 40개 런을 완전한 명시 설정으로 생성하며, plan-only
manifest에서 고정 split 5개와 LOSO 35개, 시드 0~4, 7 folds, 고유 런 이름을 확인했다.
불완전 런은 덮어쓰거나 삭제하지 않고 `blocked_incomplete`로 남긴다. 분석 driver와 결과
bundle 구현도 완료했고 두 종류의 1-epoch dry-run도 통과했다. 다음은 40런 본 실행이다.
LOSO 집계기는 fold별
seed 원자료·mean/SD·constant-map margin을 JSON으로도 보존한다. 과거 `Y=1`에서 얻은 프레임
표준편차 상수 기반 `fold SE`는 새 구조화 결과에서 제외해 절단선 이전 숫자가 섞이지 않게 했다.
확률맵은 LOSO fold마다 별도 `labels.npz`와 함께 저장하며, fold별 무결성 JSON이 모두 통과해야
결과 bundle 생성 단계로 진행한다. Bundle은 프로토콜별 `RESULTS.json`/CSV/README와 상위
색인을 만들되, 고정 split과 LOSO를 하나의 평균으로 합치지 않는다.

실행 진입점:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall \
PROTOCOL=all SEEDS=0,1,2,3,4 NUM_EPOCHS=40 \
    bash configs/paper_final_overall.sh
```

---

## 6. 아직 동결하지 않은 후속 실험 사항

아래는 `01_overall` 실행을 막지 않으며, 해당 실험을 시작하기 전에 별도로 확정한다.

- `02` DS-native 대 undistort+pinhole의 virtual pinhole FOV와 리샘플링 규약
- `02` SynWoodScape pretraining의 source 정식화·source seed 처리·전이 범위
- `02` front-only의 고정 3-camera task 평가와 front 공통 가시영역 보조 평가
- `04` Orin warm-up, 반복 횟수, latency percentile, 전력 측정 명령
