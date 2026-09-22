# 논문 최종 실험 원장

이 문서는 논문에 사용할 **최종 설정 재실험 캠페인의 단일 진행 원장**이다. 실험 계획,
동결 설정, 실행 상태, 산출물 위치, 무결성 검사, 논문용 결과를 여기서 계속 갱신한다.
개별 숫자의 기계 판독 원본은 각 실험의 `runs/paper_final/*/analysis/RESULTS.json`이고,
이 문서는 그 출처를 링크하고 사람이 읽는 결론을 기록한다.

> **상태 기준일:** 2026-09-22  
> **작업 브랜치:** `exp/paper-campaign`  
> &nbsp;&nbsp;(2026-09-22에 `exp/cumulative-ray-loss`에서 이름만 바꿨다 -- 커밋은 그대로이고
> 그 브랜치에 담긴 일이 range-loss 프로브를 넘어 캠페인 전체가 됐기 때문이다.
> `develop`과 같은 커밋에서 출발한다. **`main`은 실험 정리가 끝난 뒤에 올린다.**)  
> &nbsp;&nbsp;(2026-09-18~19의 `exp/paper-final` worktree는 **폐기했다** -- submodule 때문에
> `git worktree move`/`remove`가 둘 다 거부되고 산출물을 못 찾는 사고가 났다. §4.6의 체크리스트는
> 그 시점의 기록이다. **worktree를 쓰지 않는다** -- `AGENTS.md`의 같은 경고를 본다.)  
> **실행 GPU:** `CUDA_VISIBLE_DEVICES=0`  
> **정본 환경:** conda `bev-chamdog` · Python 3.11 · torch 2.7.0+cu128  
> **대용량 산출물 루트:** `/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final`

---

> ## ⚠ 캠페인의 loss 설정 — **[2026-09-22] 재학습으로 해소하기로 했다**
>
> **[2026-09-21] 무엇이 어긋났나.**
>
> 같은 날 `runs/loss_effect`에 `E_cumulative`(`RANGE_LOSS_MODE=cumulative_l1`, `λ_R=0.15`)를
> 넣었고, **사용자 결정으로 논문의 loss ablation 표에서는 `D_range`(`arc_huber`, `λ_R=0.3`)를
> 빼고 `E`만 쓴다**([`loss_effect_results.md`](loss_effect_results.md) **§16.7**).
>
> **그런데 이 캠페인의 런은 전부 `arc_huber`, `λ_R=0.3`으로 학습돼 있다** — 완료된
> `01_overall` 40런과 `02_projection_and_prior`가 모두 그렇다(각 런의 `config.json`에서 확인).
> 지금 상태로 쓰면 **"방법 절은 `E`인데 주 결과표의 모델은 `D`"**가 된다.
>
> ### ⚠⚠ [2026-09-22 오후] 결정 — **캠페인을 다시 돌린다**
>
> 사용자가 **`arc_huber`를 논문에서 완전히 걷어내기로** 했다. 아래 "(a) 다시 돌리지
> 않는다"는 **철회**이고, 기록으로만 남긴다.
>
> **연휴(2026-09-23~27) 동안 `cumulative_l1`(`λ_R=0.15`)로 70런을 다시 돌린다.**
>
> | 대상 | 런 수 | 스크립트 |
> |---|---:|---|
> | SynWoodScape 사전학습 | 5 | `configs/train_synwoodscape_binary_pretrain.sh` (**`--lambda_r=0.3`을 고쳐야 한다**) |
> | `01_overall` 고정 split | 5 | |
> | `01_overall` LOSO (7 fold × 5 시드) | 35 | |
> | `02_projection_and_prior` (4팔 × 5 시드) | 20 | |
>
> **사전학습을 빠뜨리면 안 된다** — 그 스크립트도 `arc_huber`라서, 그대로 두면 실험 2의
> 사전학습 팔이 `arc_huber`로 학습된 가중치를 물려받는다. **전부 아니면 전무다.**
>
> **`03`의 사다리 넷은 재학습 대상이 아니다.** `A_ce`~`C_soft`는 `λ_R = 0.0`이고 저장소가
> 그때 보조항을 **계산조차 하지 않는다**(`tools/train_robot_bev.py:322`, "대조군 보호").
> 로그에 `train/loss_range_epoch` 태그가 아예 없는 것이 그 증거다.
>
> **그때까지는 현재 숫자로 논문을 쓰고 자릿수만 갈아끼운다.** 실험 1의 헤드라인은 약
> 1 σ_seed 움직이고(`D`→`E`의 `iou_free` 차이 +0.002 대 σ_seed 0.0018), 실험 2의 결론은
> 팔 사이의 **짝지은 차이**라 공통 이동이 상쇄된다. **문장은 안 바뀌고 자릿수만 바뀐다.**
> 패키지 CSV는 전부 생성 스크립트가 만드므로 재생성 후 diff하면 바뀐 칸만 뜬다.

---

> <details><summary>철회된 결정 (2026-09-22 오전) — 기록용</summary>
>
> **[2026-09-22 사용자 결정] (a)를 채택한다 — 캠페인을 다시 돌리지 않는다.**
>
> loss ablation 표에서만 `E`를 쓰고, **시스템 결과표에는 그 모델이 `arc_huber`, `λ_R=0.3`
> 설정임을 명시**한다. 재실행하지 않는 근거는 얻는 것이 작다는 점이다 — `E`가 `D`보다
> 낫다고 쓸 수 있는 것은 `iou_free` **+0.0020**과 `range_mae` **−0.0033 m**뿐이고
> `missed_obstacle`은 **+0.0010으로 나빠진다.** 안전·재현성·되올림은 개선되지 않는다.
> 60런을 다시 돌릴 만한 차이가 아니다.
>
> **그래서 논문에 이렇게 쓴다.**
>
> 1. 시스템 결과표(`01`·`02`)의 캡션 또는 Training Details에 **보조항이 `arc_huber`
>    형태이고 `λ_R=0.3`임을 명시**한다. 캠페인 런에는 `range_loss_mode` 키 자체가 없는데,
>    그 플래그가 생기기 전이고 **저장소 기본값이 `arc_huber`**이기 때문이다
>    (`configs/train_robot_bev_finetune.sh:118`).
> 2. loss ablation 절에서 `E`(`cumulative_l1`, `λ_R=0.15`)를 권고 형태로 제시할 때,
>    **시스템 결과는 그 이전 형태로 학습됐다는 것을 한 줄로 밝힌다.** 숨기면 방법 절과
>    결과 절이 어긋난 채로 남는다.
> 3. 두 형태의 차이가 위 세 숫자뿐이라는 것을 같이 적으면, "왜 다시 안 돌렸나"에 대한
>    답이 본문 안에서 끝난다.
>
> 반영 위치: `docs/paper_package/common/training_details.md` §9.2와 [`setup.md`](paper_package/common/setup.md) §4.
>
> </details>

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
| `02_projection_and_prior` | 카메라 모델(DS-native 대 undistort+pinhole)과 source prior(SynWoodScape) | 고정 split × 5 seeds (대조군은 `01`의 5런 재사용, 시드 1:1) | **완료: 20/20, 무결성 통과, 논문 패키지 생성.** front-only는 기각(§6.1) |
| [`03_boundary_uncertainty`](paper_package/03_boundary_uncertainty/) | boundary-aware loss의 효과와 원인 | 고정 split, **6조건 × 5 seeds = 30런** (`runs/loss_effect`) | **완료** (2026-09-22). 되올림 47.6 → 0.5 % · 증가분의 96.5 %가 경계에서 · 목적함수-품질 정렬 22.6 → 2.2 epoch · **정확도 불변, `f1@10cm` −0.012**. 본문 사다리는 `A_ce→B_perset→C_hard→C_soft→E_cumulative` 다섯이고 `D_range`는 뺀다 |
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

### 2026-09-23 — **캠페인 전체 재학습 완료** (`cumulative_l1`), 결과 확정

연휴 계획(§8)대로 70런을 `runs/paper_final_cumulative/`에 다시 돌렸다. **아래 §4·§6의
숫자는 모두 옛 `arc_huber` 캠페인 것이고, 논문에 쓰는 정본은 패키지
(`docs/paper_package/`)에 있는 새 숫자다.** 이 절이 그 차이를 요약한다.

#### 바뀌지 않은 것 (논문 주장 그대로)

| 주장 | 옛 | 새 |
|---|---|---|
| 고정 split `iou_free` | 0.8119 ± 0.0018 | **0.8146 ± 0.0007** |
| LOSO macro | 0.8304 | **0.8302** |
| **σ_fold / σ_seed** | 10.3배 | **10.1배** |
| **raw IoU 순위와 margin 순위 역전** | 성립 | **성립** (`raws2` 최고 IoU·낮은 margin, `raws3` 낮은 IoU·최고 margin) |
| 어안 > 핀홀 120° | −0.0134 | **−0.0149** |
| 어안 > 핀홀 150° | −0.0078 | **−0.0094** |
| 커버리지 > 해상도 (150° > 120°) | +0.0055 | **+0.0055** |

#### 바뀐 결론 셋 — **그대로 적었다**

1. **`02` 합성 사전학습의 `iou_free` 판정이 선다.** −0.0020(|Δ|/SE 1.23, 4/5, "차이 없음")
   → **−0.0027(|Δ|/SE 3.00, 5/5, "target-only 우세")**. 보수적인 옛 눈금으로도 넘지만
   묶음 눈금 0.0026과는 **0.0001 차이라 아슬아슬하다** — 그 사실을 REPORT에 적었다.
   `f1@40cm`은 반대로 −0.0022 → **+0.0033**로 뒤집힌다: 경계가 부서진 게 아니라 흐려졌다.
2. **`02` §4.2 기전 논증의 근거가 한 갈래로 좁아졌다.** 옛 캠페인은 1에폭 `fatal_rate`와
   `free_miss_rate`가 **반대 방향으로** 벌어졌는데(0.190→0.357, 0.139→0.108), 지금은
   free-miss가 0.123→0.117로 **시드 산포(±0.027) 안에 들어간다.** 논증은 이제 fatal
   급등(0.196→0.344) 하나에 기댄다.
3. **`01` §5.6의 "두 프로토콜 마진이 소수 넷째 자리까지 같다"는 우연이 사라졌다**
   (+0.2966 대 +0.2935). 그때 "우연의 일치로 보는 게 맞다"고 적은 판단이 옳았다는
   사후 확인이다.

#### 검증

- **70/70 정상.** watchdog 발동 0회, Traceback·OOM 0건. 8시간 43분, 디스크 65 GB
- **무결성**: 다섯 팔(01 고정·01 LOSO·02 세 팔·02b) 전부 확률맵을 독립 경로로 재채점,
  검사 80건씩, **실패 0**, 최대 절대차 **6.9e-4 < 허용치 1e-3**
- **트리 순수성**: 산출물이 옛 루트를 가리키지 않는다(`tools/check_results_tree_purity.py`)
- **constant-map 기준선이 1e-9 이내로 같다** — 학습과 무관한 값이라 같아야 하고,
  실제로 같았다(라벨·split이 그대로라는 확인)
- **산문 숫자 감사**: 낡은 숫자 81개를 고쳤다. 최종 추적률 01 123/127 · 02 156/161 ·
  02b 86/89. 감사기가 **내 반올림 오차 둘도 잡았다**

#### 판정 눈금 — 낮추지 않았다

새 대조군의 고정 split σ_seed가 0.0007로 보였지만, **8군 묶음 추정(자유도 32)으로는
0.0025 → 0.0026, 변화 없다.** 5표본 sd의 흔들림이다(LOSO fold별 추정치 14개가
0.0010~0.0049에 흩어진다). 사전 등록값 **0.0018을 유지**하고, 아슬아슬한 칸은 묶음값
0.0026으로도 따져 둘 다 적었다.

#### 실험 03은 건드리지 않았다

30런 `config.json` 실측으로 확인했다 — 논문 사다리 다섯 칸(`A_ce`~`C_soft`는 λ_R=0,
`E_cumulative`는 이미 `cumulative_l1`) 중 **`arc_huber`로 학습된 칸이 없다.**

---


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

### 2026-09-21 — `02_projection_and_prior` 설계 동결·구현·실행 시작

설계는 §6이 정본이다. 여기에는 **구현하면서 내린 결정과 실행 상태**만 적는다.

**구현 결정 넷** (전부 "비교가 축 하나만 다르게" 만들기 위한 것이다).

1. **핀홀 lifting은 `VirtualPinholeVoxUtil`로 한다** -- upstream `Vox_util`이 아니다.
   upstream은 `legacy_index` 규약의 배율 오차(`x·W/(W−1) − 0.5`)를 갖고 있어서, 그것을
   쓰면 "DS + 옳은 표본 위치" 대 "핀홀 + 틀린 표본 위치"를 비교하게 된다. 그래서
   `DoubleSphereVoxUtil`에서 투영만 `_project_to_native`로 떼어내고 나머지 경로(해상도
   스케일·`pixel_offset`·`mirror_x`·유효 영역 판정·`grid_sample` 정규화)를 **상속으로**
   물려받는다.
2. **리샘플링은 512×288로 줄인 어안에서 한다.** native 1280×720에서 직접 펴면 핀홀 팔만
   원본 해상도를 더 쓰게 되고, 게다가 120° 핀홀의 중심 각해상도(2.58 px/deg)가
   native(≈6.4)보다 낮아 bilinear 탭 하나로는 **중심부에 에일리어싱**이 생긴다
   (PIL 축소는 안티에일리어싱을 하지만 `cv2.remap`은 하지 않는다). 줄인 뒤 펴면 중심이
   거의 1:1(2.55 → 2.58)이라 두 문제가 동시에 사라지고, 두 팔이 **글자 그대로 같은 입력
   이미지**를 본다.
3. **`permanent_blind` 마스크는 DS 기준 그대로 둔다.** 핀홀이 못 보는 셀도 과제에 남아야
   `01_overall`과 같은 과제가 되고 커버리지 손실이 결과에 정직하게 반영된다.
4. **셸에 가드를 넣었다.** `PROJECTION=pinhole`인데 화각을 빠뜨리면 조용히 확정 설정으로
   돌아가 "pinhole120"이라는 이름의 DS 런이 생긴다. 그 런은 겉보기에 정상이라 분석
   단계까지 아무도 알아채지 못한다.

**축 B의 전제를 실측으로 확인했다.** binary source 체크포인트를 target 모델에 붙이면
**668 텐서 전부, `skipped 0`**이다(3-class source면 head 2개가 빠진다). 격자가
240×240 → 120×120으로 달라도 상관없다 -- BEV decoder가 전부 convolution이고 셀이 둘 다
5 cm라 "필터 하나가 몇 미터를 보는가"가 보존된다.

**전이하는 체크포인트는 `model-000000040.pth`(고정 epoch)다.** `model_best`가 아니다 --
캠페인 규약이 val 기반 선택을 쓰지 않는다(§3).

**실행 상태.** 20런(사전학습 5 + `source_prior` 5 + `pinhole120` 5 + `pinhole150` 5),
GPU 0, 출력은 `runs/paper_final/02_projection_and_prior/training_queue.log`로 리다이렉트해
세션과 무관하게 돈다. 사전학습 한 런이 에폭 약 29초 × 40 = 20분이다.

### 2026-09-21 — `02_projection_and_prior` 20런 완료·검증·결과 확정

**실행.** 20런, GPU 0, 총 약 4시간 30분(사전학습 5런이 런당 20분으로 대부분을 차지한다).
`source_prior` 5런은 orchestrator 버그로 1차에서 건너뛰어져 재실행했다(아래 §사고 기록).

**게이트.**

| 확인 | 결과 |
|---|---|
| 런 | 20/20 완료 |
| 확률맵 재채점 무결성 | **240건 전부 통과**, 최대 절대차 **4.50e-4** (허용치 1e-3) |
| 투영 되찾기 | 팔마다 `config.json`에서 읽어 정확 (120° / 150° / ds_native) |
| 대조군 | `01_overall` 고정 split 5런, 시드 1:1 |

#### 결과 A — 카메라 모델 (축 A)

| 구성 | 화각 | BEV 커버리지 | 정면 px/deg | `iou_free` | 마진 | 짝지은 차이 | \|Δ\|/SE | 부호 |
|---|---|---|---|---|---|---|---|---|
| **어안 원본 (확정)** | — | 99.8 % | 2.55 | **0.8119 ± 0.0018** | **+0.2938** | — | — | — |
| 핀홀 150° | 150° | 96.7 % | 1.20 | 0.8040 ± 0.0017 | +0.2860 | −0.0078 | 27.5 | 5/5 |
| 핀홀 120° | 120° | 93.1 % | 2.58 | 0.7985 ± 0.0013 | +0.2805 | −0.0134 | 21.4 | 5/5 |

9개 지표 **전부** 어안 우세, **5시드 전부** 같은 방향이다. 안전 쪽이 특히 나빠진다 —
핀홀 120°에서 `range_missed_obstacle_rate` 0.0377 → 0.0497(**+32 %**),
`range_bias` +0.0176(자유공간 과대평가 = 위험한 방향), `fatal_rate` +0.0109.

**예상과 달랐던 것: 핀홀 150°가 120°보다 낫다.** 두 팔을 서로 짝지으면 `iou_free`
**+0.0055**(|Δ|/SE 11.5, 5/5), 놓친 장애물 **−0.0082**(11.1, 5/5)다. 150°는 정면
각해상도가 120°의 **0.46배**인데도 이긴다 — 커버리지만 93.1 % → 96.7 %로 넓다.

> **[해석]** 어안의 이득은 "선명함"이 아니라 **"시야"**로 보인다. 정면 해상도를 어안과
> 맞춘 팔이 지고, 해상도를 포기하고 시야를 넓힌 팔이 그보다 나았다. 설계할 때 150°는
> "핀홀에게 유리한 조건"으로만 의도했는데 **기전을 가르는 증거**가 됐다.
> 부수 효과로 "화각을 유리하게 골랐다"는 반박도 막힌다 — 핀홀에게 유리한 쪽이 실제로
> 더 나았기 때문이다.

#### 결과 B — SynWoodScape source prior (축 B)

| 지표 | target-only | 사전학습 | 짝지은 차이 | \|Δ\|/SE | 부호 | 판정 |
|---|---|---|---|---|---|---|
| `iou_free` ↑ | 0.8119 | 0.8099 | −0.0020 | 1.23 | 4/5 | **차이 없음** |
| `fatal_rate` ↓ | 0.1204 | 0.1213 | +0.0009 | 0.36 | 4/5 | 차이 없음 |
| `free_miss_rate` ↓ | 0.0844 | 0.0846 | +0.0001 | 0.18 | 3/5 | 차이 없음 |
| **`f1@10cm` ↑** | 0.5981 | 0.5812 | **−0.0169** | 2.74 | **5/5** | target-only 우세 |
| **`f1@20cm` ↑** | 0.8313 | 0.8158 | **−0.0155** | 3.61 | **5/5** | target-only 우세 |
| `f1@40cm` ↑ | 0.9290 | 0.9268 | −0.0022 | 0.87 | 3/5 | 차이 없음 |
| `range_mae` ↓ | 0.2117 | 0.2214 | +0.0096 | 2.96 | 4/5 | target-only 우세 |
| `range_missed_obstacle_rate` ↓ | 0.0377 | 0.0385 | +0.0008 | 0.53 | 4/5 | 차이 없음 |

**자유공간의 양과 안전성은 바뀌지 않고 경계 정밀도만 나빠진다.** 허용오차를 넓힐수록
차이가 줄어드는 것(10 cm −0.0169 → 20 cm −0.0155 → 40 cm 차이 없음)이 "경계가 흐려졌다"의
서명이다.

**기전(5시드 평균).** 1에폭에서 사전학습 모델은 `free_miss_rate`가 **낮고**(0.108 대 0.139)
`fatal_rate`가 **높다**(0.357 대 0.190). 둘이 동시에 일어나는 방식은 하나뿐이다 —
자유공간을 더 넓게 선언하는 것이다. **그리고 2에폭이면 사라진다**(0.134 대 0.130).

> **[해석]** §6.3에 적어 둔 예상("사전학습이 질 것이고 기전은 관측 가능성 prior")은
> **기전 쪽이 맞고 크기 쪽이 틀렸다.** prior의 서명은 1에폭에 분명히 보이지만 얕아서
> 3에폭이면 풀리고, 최종 성능에는 경계 정밀도만 남는다. `Y=1` 시절의 "9개 중 7개 패배"와
> 비교하면 **결론의 범위가 좁아졌다** — "전면 순손실"이 아니라 "경계에 국한된 손실"이다.
> 논문에 쓸 권고는 "쓰지 말라"가 아니라 **"쓰면 경계 정밀도를 확인하라"**다.

#### 판정 규칙을 고쳤다

처음 만든 표는 축 B의 `iou_free`를 "대조군이 낫다"로 **잘못 찍었다.** σ_seed(0.0018)와
비교했는데 그것은 *한 런*의 산포이지 *짝지은 차이*의 산포가 아니다. 그 차이는 시드마다
0.0036씩 흔들려 `|Δ|/SE`가 1.23뿐이다. 이제 **문턱 둘을 모두** 넘어야 우열로 쓴다:
`|Δ| > σ_seed`(실용) **그리고** `|Δ| ≥ 2·SE`(통계). 표에 `|Δ|/SE` 열을 넣었다.

#### 사고 기록 — `source_prior` 5런이 조용히 건너뛰어졌다

**증상.** 1차 실행에서 20런 중 15런만 돌고 `source_prior` 5런이 실행되지 않았다.
학습은 16:46에 끝났는데 18:23에야 알아챘다.

**원인 둘, 둘 다 orchestrator/감시 코드의 결함이다.**

1. 큐를 시작할 때 "짝지은 사전학습 체크포인트가 없다"로 `blocked_missing_input`을 찍었는데,
   **그 체크포인트는 같은 큐의 앞 팔이 만들어 주는 것**이라 그 시점에 없는 것이 정상이었다.
   실행 루프에서 다시 확인하도록 주석까지 써 두고는, 그 위의 `continue`가 막힌 런을 먼저
   걸러내고 있었다.
2. 분석 자동 실행의 대기 조건이 `pgrep -f "run_paper_final_projection_prior"`였는데
   **대기 프로세스 자신의 명령줄에 그 문자열이 있어서** 영원히 자기를 기다렸다. 그래서
   ①을 가려 주었다 — 분석이 제때 돌았다면 누락 때문에 즉시 실패해 바로 알았을 것이다.

**잃은 것은 시간(약 1시간 40분)뿐이다.** 틀린 결과는 만들어지지 않았다. 건너뛴 런은
로그에 기록이 남았고 완료된 15런은 정상이다.

**고친 것.** 실행 루프가 `blocked_missing_input`을 차례가 올 때 다시 판정한다.
`queue_status.json`과 `experiment_manifest.json`을 덮어쓰지 않고 **병합**한다(팔을 나눠
돌리면 앞 호출의 기록이 사라져 20런 manifest가 10런으로 줄어 있었다).
**회귀 테스트 8개**를 붙였고 첫 번째가 이 버그다.

#### 산출물

| 무엇 | 경로 |
|---|---|
| 짝지은 비교 (정본) | `runs/paper_final/02_projection_and_prior/analysis/paired_arms.json`, `report_paired_arms.txt` |
| 무결성 | `runs/paper_final/02_projection_and_prior/*/analysis/verify_predictions.json` |
| 런 목록·설정 | `runs/paper_final/02_projection_and_prior/experiment_manifest.json` |
| **논문 패키지** | **`docs/paper_package/02_projection_and_prior/`** (표 2개·그림 2개·CSV 10개) |
| 설계 근거 측정 | `tools/measure_projection_tradeoff.py`, `tools/measure_domain_prior.py` |

### 다음 작업

`01_overall`·`02_projection_and_prior`·`03_boundary_uncertainty`가 **셋 다 끝났다.**
결과·해석·논문 구성 제안은 각각 `docs/paper_package/<실험>/REPORT.md`에 있다.

**남은 것 둘.**

0. **[진행 중] `02b_native_source_prior` 10런** -- 원본 기하(30 m, 200×200) 사전학습과 그
   미세조정. §6b. **사용자 지시로 이것을 먼저 돌린다.** 약 2시간.

1. **연휴 재학습 70런** -- `cumulative_l1`(`λ_R=0.15`)로 `arc_huber`를 논문에서 걷어낸다.
   상세 계획은 **§8**에 있다. 대상은 01의 40 + 02의 20 + 02b의 10이다
   (`source_pretrain`이 02의 20에 이미 포함돼 있다 -- 예전에 65·75로 적은 것은 그것을
   두 번 센 오류였다).

2. **`04_edge_deployment`** -- Orin 실측. **사용자 장비가 필요하다.** 기존 20.2 FPS는
   `Y=1` 값이므로 `Y=4` 재실측이 필수다. 미동결 항목은 §6.5에 있다.

> **⚠ `03`을 다시 만질 때 철회를 유지한다.** **"안전 개선"은 동작점 이동으로 철회됐고**
> (같은 `free_miss`에서 곡선이 겹친다), **"정확도 개선"도 마찬가지다**(§17.1 -- 동작점을
> 맞추면 `iou_free` 차이가 사라지고 부호가 뒤집힌다). 되살리면 안 된다.
>
> 살아남은 주장은 **목적함수 수렴 · 경계 국소화 · 목적함수-품질 정렬 · 문턱 재현성** 넷이고,
> 그 인과 귀속은 `C_hard` ↔ `C_soft` 쌍(대역 target만 다르다)이 담당한다.
> **그리고 `f1@10cm` −0.012는 동작점을 맞춰도 남는 실재 비용이므로 반드시 함께 쓴다**(§17.2).

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

# 실험 02 (완료된 런은 자동으로 건너뛴다). --plan_only로 먼저 계획만 볼 수 있다
python tools/run_paper_final_projection_prior.py --arms=all --gpu=0

# 설계 근거가 된 기하·라벨 측정 (학습 불필요)
python tools/measure_projection_tradeoff.py
python tools/measure_domain_prior.py
```

---

## 6. 실험 02 — 카메라 모델과 source prior (`02_projection_and_prior`)

**상태: 2026-09-21 완료(20런).** 세 축을 검토해 **둘을 채택하고 하나를 기각**했다.
아래는 확정 설계이고, **결과는 §5의 2026-09-21 완료 항목**에 있다. 각 축의
"예상 결과와 그 해석"은 **실행 전에** 적어 둔 것이므로 지우지 않는다 -- 무엇을 맞히고
무엇을 틀렸는지가 결과를 읽는 데 쓰인다.

> **실행 전 예상과 실제.** 축 A는 예상대로 어안이 이겼고, **예상하지 못한 것은 핀홀 150°가
> 120°를 이긴 것**이다(커버리지 > 해상도). 축 B는 **기전은 맞고 크기는 틀렸다** --
> 관측 가능성 prior의 서명이 1에폭에 보이지만 얕아서 최종 `iou_free`는 차이가 없고
> 경계 정밀도만 남았다.

**이름을 바꿨다.** 원래 `02_sensor_task_adaptation`이었는데 "sensor" 부분(front-only)이
기각되어 이름이 내용과 어긋났다. 아직 이 ID로 실행한 런이 없으므로 지금 고친다.

### 6.0 무엇을 묻는 실험인가

`01_overall`은 "우리 시스템이 얼마나 잘하나"를 쟀다. `02`는 **"우리가 내린 두 개의
설계 결정이 옳았나"**를 묻는다. 둘 다 확정 설정(§3)에서 **한 축만** 바꾼다.

| 축 | 방어하는 결정 | 심사자가 물을 형태 |
|---|---|---|
| A. 카메라 모델 | 어안을 펴지 않고 Double Sphere 투영을 그대로 썼다 | "왜 undistort해서 통상적인 핀홀 파이프라인을 쓰지 않았나?" |
| B. source prior | 합성 데이터 사전학습 없이 target만으로 학습했다 | "SynWoodScape가 있는데 왜 안 썼나?" |

**대조군을 다시 돌리지 않는다.** `01_overall`의 고정 split 5런(`final_s0`~`s4`)이 두 축의
공통 대조군이고, **같은 시드끼리 짝지어 paired difference**를 본다. 시드가 짝지어지므로
런 간 산포(σ_seed ≈ 0.0018)가 차이에서 상쇄된다.

### 6.1 기각한 축 — front-only (카메라 1대)

**기각한다. 학습을 돌리지 않는다.**

이유는 **결과가 학습 전에 기하로 이미 정해져 있기** 때문이다. 캘리브레이션만으로 계산했다
(`tools/measure_projection_tradeoff.py` §1, GPU 미사용, `Y=4` 표본 높이 0/0.5/1.0/1.5 m,
3대 중 하나라도 보면 커버):

| 카메라 구성 | 보는 BEV 셀 | 비율 |
|---|---|---|
| front + left + right | 14,365 / 14,400 | **99.8 %** |
| front만 | 11,768 / 14,400 | **81.7 %** |

거리대별 front-only 커버리지: 0~1 m **64 %**, 1~2 m **68 %**, 2~3 m 84 %, 3~4 m 95 %.
**충돌이 일어나는 근거리가 가장 크게 빈다** — 측면 카메라가 보던 영역이다.

5런을 돌려서 얻는 것은 이 기하 사실을 `iou_free` 단위로 다시 쓴 것뿐이다. 그리고 축 A·B와
달리 **방어할 결정이 아니다** — 카메라 3대가 이미 달려 있는데 1대만 쓰자고 주장할 사람이 없다.

> **대신 할 것:** 위 표를 논문 setup 절에 **기하 사실 한 줄**로 넣는다("front 단독 커버리지
> 81.7 %, 1 m 이내 64 % — 그래서 3대를 쓴다"). 학습 0런으로 같은 질문에 답한다.
> 심사자가 더 요구하면 그때 돌려도 5런 35분이다.

### 6.2 축 A — Double Sphere native 대 undistort + pinhole

#### 무엇을 말하려는가

> 대부분의 BEV 인식 파이프라인은 핀홀 투영을 전제하므로 어안을 먼저 편다. 이 관행의 비용을
> **정면 각해상도를 동일하게 맞춘 조건**에서 측정한다. 비용은 두 갈래다 — 핀홀이 원리적으로
> 담을 수 없는 **화각 손실**과, 리샘플링이 들여오는 **품질 손실**. 어안 투영을 원본 그대로
> 쓰면 둘 다 들지 않는다.

이 시스템에서 가장 눈에 띄는 구현 선택이 DS-native이고, 어안 BEV 논문의 심사자가 반드시
요구할 ablation이다. 지금은 "그렇게 했다"만 쓸 수 있고, 이 실험이 있으면 **"그렇게 해서
이만큼 얻었다"**가 된다.

#### 화각을 정해야 하는 이유 (원리)

핀홀 이미지는 평면이고, 광축에서 θ 벗어난 광선은 중심에서 `f·tan(θ)`에 찍힌다.
**`tan(90°)`는 무한대**이므로 핀홀은 90° 이상의 광선을 **해상도와 무관하게** 담을 수 없다.
구현 한계가 아니라 원리적 불가능이다. 이 렌즈는 실측 ±96~101°를 보므로 절반 이상이
어떤 핀홀에도 안 들어간다. 그래서 자를 곳을 정해야 하고, 자르면 정보를 버린다.

#### 화각 선택의 근거 — 정면 각해상도를 맞춘다 (실측)

512 px 폭 출력 기준, 광축 방향 1도당 화소 수(`tools/measure_projection_tradeoff.py` §2, 캘리브레이션만 사용):

| | 정면 px/deg | 어안 대비 | BEV 커버리지 |
|---|---|---|---|
| **어안 원본 (DS)** | **2.55** | — | **99.8 %** |
| 핀홀 90° | 4.47 | 1.75× | 84.9 % |
| 핀홀 100° | 3.75 | 1.47× | 90.0 % |
| **핀홀 120°** | **2.58** | **1.01×** | **93.1 %** |
| 핀홀 140° | 1.63 | 0.64× | 95.6 % |
| **핀홀 150°** | 1.20 | 0.47× | **96.7 %** |
| 핀홀 160° | 0.79 | 0.31× | 97.7 % |

**두 팔을 쓴다.**

- **`pinhole120` (주 비교).** 정면 각해상도가 어안과 **정확히 같다**(2.58 대 2.55).
  그래서 "베이스라인을 굶겼다"는 반박이 원천 차단된다. 로봇이 가는 방향에서 두 방식이 같은
  화소를 쓰고, 남는 차이는 화각 6.9 %와 왜곡 처리 방식뿐이다.
- **`pinhole150` (유리한 조건).** 커버리지를 96.7 %까지 올려 준다. 대신 정면 해상도가
  0.47배로 떨어진다. **불리한 조건과 유리한 조건 양쪽에서 이겼다**가 되면 화각 반박이 끝난다.

부수적으로 어안의 장점이 같은 표에 드러난다 — DS는 정면 2.55 / 45° 2.49 / 90° 1.96으로
**거의 균일**한데, 핀홀은 넓힐수록 가장자리에 화소를 낭비하면서(150°에서 가장자리 17.9 px/deg — 정면의 15배)
정작 중요한 정면을 굶긴다.

#### 설계

| 항목 | 값 |
|---|---|
| 팔 | `pinhole120`, `pinhole150` (각 시드 0~4) |
| 대조군 | `01_overall/fixed_split/final_s0`~`s4` (재사용, 시드 1:1) |
| 리샘플링 | 가상 핀홀 광선 격자 → DS로 투영 → bilinear (`cv2.remap`), **미리 디스크에 저장** |
| 출력 해상도 | 512×288 — 두 팔과 대조군이 **같아야** 연산량 비교가 성립한다 |
| 광축 | 어안과 동일 (새 pose를 도입하지 않는다) |
| lifting | 핀홀 경로(`simplebev_vox`) + 가상 핀홀 intrinsic. SynWoodScape가 쓰는 기존 경로다 |
| 화각 밖 화소 | 검게 남으므로 **유효로 세지 않는다** — 마스크를 lifting에 넘긴다 |
| 증강 | photometric은 **편 뒤에** 건다 (순서가 바뀌면 왜곡 정도가 다른 화소에 같은 세기 잡음이 들어간다) |
| 나머지 | §3 확정 설정 그대로 |

10런, 약 70분.

#### 예상 결과와 그 해석

- **DS-native가 두 팔 모두 이긴다 (예상).** 그러면 주장은 "어안 원본 사용이 유효하다"이고,
  `pinhole120`과의 차이는 순수 품질(해상도가 같으므로), `pinhole150`과의 차이는
  품질−커버리지 절충으로 읽는다.
- **`pinhole150`이 `pinhole120`을 이긴다면** 커버리지가 해상도보다 중요하다는 뜻이고,
  그 자체가 보고할 만한 결과다(어안이 좋은 이유가 커버리지에 있다는 증거).
- **DS-native가 진다면** 그대로 보고한다. 확정 설정을 바꾸지는 않되(01이 이미 그 위에 있다)
  "투영 방식은 이 과제에서 지배적 요인이 아니다"를 논문에 쓴다.
- 차이가 σ_seed(0.0018)보다 작으면 **차이 없음**으로 보고한다. 우열을 만들어 내지 않는다.

### 6.3 축 B — SynWoodScape source prior

#### 무엇을 말하려는가 — 기록된 설명을 정정하면서

기존 문서(`docs/paper_experiment_compendium.md` §4)는 사전학습이 해로운 이유를
**"SynWoodScape는 격자의 83 %가 free인데 로봇은 20 %"**라고 적었다. 두 데이터셋 **전수**를
**같은 정의**로 다시 재 보니 그 대조가 두 가지 다른 양을 비교한 것이었다
(`tools/measure_domain_prior.py`, 500 + 267샘플 전수):

| | free / 격자 전체 | free / **관측된 셀만** | 관측된 셀이 격자의 |
|---|---|---|---|
| SynWoodScape | 82.2 % | 87.6 % | **93.8 %** |
| 로봇 | 26.4 % | **95.1 %** | **27.8 %** |

**관측된 셀만 보면 로봇이 오히려 더 free하다**(95.1 % 대 87.6 %). 두 도메인의
free/장애물 균형은 어긋나 있지 않다. 82 대 26의 차이는 전부 **관측 비율**에서 온다 —
SynWoodScape는 시뮬레이터가 전지적이라 격자의 94 %에 정답이 있고, 로봇 라벨은 LiDAR가
실제로 훑은 **28 %**에만 있다.

> **그래서 불일치의 정체는 "free가 많은 세계"가 아니라 "어디든 답을 아는 세계"다.**
> 사전학습 모델이 물려받는 것은 free 편향이 아니라 **격자 전체에 대해 확신하는 습관**이고,
> 그것이 로봇 데이터에서는 감독되지 않는 4분의 3에 쏟아진다.

기존 문서의 관측(ep1 `fatal_rate` 0.663)은 사실이다. 바뀌는 것은 **원인 설명**이고,
그 문서도 해당 대목을 "가설이지 확정이 아니다"라고 유보해 두었다.
→ 정정 기록을 `paper_experiment_compendium.md` §4에 남긴다.

그래서 이 실험이 말하려는 것은:

> 합성 자율주행 데이터로 사전학습하면 **밀집 정답(격자의 94 %)에 맞춰진 관측 가능성 prior**가
> 함께 따라온다. LiDAR 기반 실측 라벨은 격자의 28 %만 감독하므로, 사전학습 모델은 감독되지
> 않는 영역까지 확신하는 상태로 출발하고 fine-tuning이 그것을 푸는 데 용량을 쓴다.
> 도메인 격차는 외형(합성 대 실사)이 아니라 **라벨 밀도**에서 왔다.

**왜 이 형태가 나은가.** "합성 데이터가 도움이 안 됐다"는 데이터셋 하나짜리 관찰이라
일반화가 안 된다. 라벨 밀도 논리는 **LiDAR로 라벨을 만드는 모든 실외 로봇 연구**에 적용된다.
그리고 **결과에 걸지 않아도 되는 실험**이다 — 도움이 되면 "저비용 합성 사전학습이 통한다",
안 되면 "이 함정을 조심하라"로 양쪽 다 쓸 수 있다.

#### 재실행이 필요한 이유

기존 체크포인트는 **없다.** 2026-09-01 정리 때 삭제됐고(`runs/archive/synwoodscape_threeclass/ckpt/`가
비어 있다), 어차피 `Y=1` 시절 3-class라 확정 설정에 붙지 않는다.

#### 설계 — "확정 설정 그대로, 데이터만 바꾼다" (사용자 결정)

사전학습도 **확정 설정과 같은 정식화·같은 loss·같은 `Y=4`**로 맞춘다. 그러면 출력 head까지
포함해 **네트워크 전체가 전이**되고(`load_trunk_weights`가 `skipped 0`), prior를 최대한
물려받은 상태에서 잰다. prior 불일치의 크기를 재는 것이 목적이므로 prior를 일부러
잘라내면 안 된다.

| 항목 | 값 | 비고 |
|---|---|---|
| source 정식화 | **binary** (free / not-free) | target과 동일. head가 전이된다 |
| source loss | `soft_boundary`, §3과 동일 파라미터 | `d_bev_g`를 SynWoodScape 라벨에서 계산해 추가한다 |
| source lifting | `Y=4`, §3과 동일 | 옛 체크포인트가 `Y=1`이라 못 쓰는 이유이기도 하다 |
| source 격자 | 240×240, 전 8 / 후 4 / 횡 ±6 m, **5 cm** | **셀 크기가 같아야 전이가 성립한다** (아래) |
| source 데이터 | `dataset/synwoodscape_2head_roi_8_4_6_h08` (500샘플, 4-cam) | 이미 구축되어 있다 |
| source 시드 | **0~4, 5개** | target 시드와 **1:1**로 짝짓는다 |
| 전이 범위 | 전체 (encoder + BEV decoder + head) | 형상이 전부 맞으므로 자동 |
| target | §3 확정 설정, `INIT_CHECKPOINT=<짝지은 source>` | |
| 대조군 | `01_overall/fixed_split/final_s0`~`s4` | 시드 1:1 |

**셀 크기를 맞춘 것이 이 실험을 성립시킨다.** BEV decoder는 전부 convolution이라 격자가
240이든 120이든 가중치는 그대로 붙는다. 전이되는 것은 "필터 하나가 몇 **미터**를 보는가"이고,
5 cm/셀이 같으므로 그것이 보존된다. 범위가 2배인 것은 SynWoodScape의 ego가 풀사이즈
승용차라 좁은 격자에서는 절반이 제 차체에 덮여 상수 GT가 되기 때문이다(`grid.py:33`).

**source 시드를 5개 쓰는 이유.** 하나를 공유하면 target 5런이 공통 원인을 공유해 산포가
총 변동을 과소평가하고, 결론이 "이 체크포인트 하나가 도움이 안 됐다"로 약해진다.
사전학습이 런당 ~35분이라 5개를 만들어도 3시간이다. 한계를 적는 것보다 없애는 편이 낫다.

사전학습 5런(~3시간) + target 5런(~35분).

#### 예상 결과와 그 해석

- **사전학습이 진다 (예상).** `Y=1` 시절 9개 지표 중 7개가 그랬다. 그러면 초기 epoch의
  `fatal_rate`·`free_miss_rate` 궤적을 함께 보고해 **기전이 관측 가능성 prior임을 보인다** —
  구체적으로 사전학습 팔의 ep1 예측 free 면적이 관측 영역(28 %)을 크게 넘는지 확인한다.
  이것이 이 실험의 핵심 증거이므로 **ep1~5 곡선을 반드시 보존한다.**
- **사전학습이 이긴다면** 그대로 보고한다. 그때는 `01`의 확정 설정(target-only)이 최선이
  아니었다는 뜻이므로, 논문 본문에서 그 사실을 숨기지 않고 "저비용 합성 사전학습이 통한다"로
  쓴다. 확정 설정을 소급해 바꾸지는 않는다.
- 차이가 σ_seed보다 작으면 **차이 없음**으로 보고한다.

### 6.4 산출물과 판정 규약

`01_overall`(§4.5, §4.6)과 같은 계약을 따른다.

- 주 지표 `iou_free`와 **constant-map baseline 대비 margin**을 함께 보고한다. 대조군이
  `01`의 고정 split이므로 baseline은 같은 값(0.5180)이고 팔 간에 공유된다.
- 보조 지표: `fatal_rate`, `free_miss_rate`, `f1@10/20/40cm`, `range_mae`, `range_bias`,
  `range_missed_obstacle_rate`. 동작점 τ=0.5.
- **paired difference**를 주 통계로 쓴다(시드 1:1). 평균 차이와 시드별 차이를 둘 다 싣는다.
- σ_seed(0.0018)보다 작은 차이는 우열로 쓰지 않는다.
- 확률맵 무결성 검사(허용오차 1e-3)를 통과해야 bundle을 만든다.
- 결과는 `docs/paper_package/02_projection_and_prior/`에 `01_overall`과 같은 구조로 쌓는다.

### 6.5 실험 2 외의 미동결 사항

- `04` Orin warm-up, 반복 횟수, latency percentile, 전력 측정 명령

---

## 6b. 실험 02b — 원본 기하 사전학습 (`02b_native_source_prior`)

**[2026-09-22 시작]** 산출물 `runs/paper_final/02b_native_source_prior/`.

### 무엇을 묻나

실험 02의 축 B(`source_prior`)는 SynWoodScape를 **타깃 과제에 맞춰 가공해서** 썼다 —
전방8/후방4/횡±6 m를 타깃과 **같은 0.05 m/cell**로. 셀을 맞춘 것은 의도였다(BEV decoder가
전부 convolution이라 전이되는 것이 "필터 하나가 몇 미터를 보는가"이고, 셀이 같으면 보존된다).
그래서 그 실험이 잰 것은 **"도메인 내용의 불일치"뿐**이다.

**남이 공개한 주행 BEV 모델을 그대로 가져다 쓰는 상황은 그렇지 않다** — 범위도 해상도도
그쪽 관행을 따른다. 이 실험이 그 상황을 만든다.

| | source 격자 | 셀 | 라벨 |
|---|---|---|---|
| 02의 `source_prior` | 8/4/±6 m (240×240) | 0.05 m = 타깃과 같음 | **사람이 보정** |
| **02b의 `source_prior_native`** | **30×30 m (200×200)** | **0.15 m = 타깃의 3배** | 자동 크롭 |

### 설계 근거 — 넷 다 실측이다

1. **소스가 실제로 주는 범위는 30 m다.** `_BEV.png`는 ego 위 z=15 m의 FOV 90° 핀홀이 찍은
   1024×1024이고 지면에서 정확히 30.0 m를 덮는다(`bev_crop.py` 유도, LiDAR 교차검증 0.9800).
   **지금 라벨이 그 이미지를 잘라낸 것**이므로 새 데이터가 아니라 덜 자르는 것이다.
2. **200×200이어야 한다.** decoder가 /8로 줄였다가 skip으로 복원하므로 격자가 8로
   나눠떨어져야 한다 — **100×100은 forward에서 26 대 25로 깨진다**(실측). 0.15 m/cell은
   자율주행 관행(Simple-BEV nuScenes 기본이 200×200)이다.
3. **가중치는 전부 전이된다.** 200×200 → 120×120에서 **668 텐서, skipped 0**으로 기존 팔과
   같다. BEV 쪽에 격자 크기에 의존하는 가중치가 없다.
4. **soft target이 살아남는다.** 셀이 3배 커져도 대역 안에 셀 4개가 남고 target 엔트로피가
   **0.2972 nats**로 0.05 m일 때(0.2965)와 사실상 같다. `δ`·`σ`를 미터로 그대로 둔다.

### 라벨

`dataset/synwoodscape_native_roi_15_15_15_h08/` (500장,
`tools/build_synwoodscape_native_labels.py`). **관측 82.5 % · 관측 셀 중 free 60.6 %**
(기존 8/4/±6은 93.8 % / 87.6 %, 로봇 타깃은 27.8 % / 95.1 %).

가시성 규칙(gather-column z=0~0.8 m)은 원 생성 스크립트가 저장소에 없어 문서의 규칙으로
다시 구현했고 기록본과 **셀 단위 99.0 % 일치**한다. **절대값이 다르므로 기록본과 한 표에
섞지 않는다.**

### ⚠ 유보 둘 — 결과를 읽을 때 반드시 같이 읽는다

1. **수동 보정이 없다.** 보정본은 8/4/±6에만 존재한다. 그래서 **02의 `source_prior`와의
   차이에는 기하와 라벨 품질이 섞여 있다.** **주 비교는 scratch(`01_overall/fixed_split`)
   대비**다. 기하만 분리하려면 "보정 없이 8/4/±6" 팔을 하나 더 만들어야 한다(+5런).
2. **BEV가 정사영이 아니다.** z=15 m 핀홀이라 높이 h인 면이 15/(15−h)배로 확대된다.
   횡 15 m에서 1.5 m 물체는 바깥으로 약 1.7 m 밀린다 — **넓힐수록 주변부 라벨이 나빠진다.**

### 보조항

**`arc_huber`, `λ_R=0.3`이다.** 대조군과 02의 기존 팔이 그것으로 학습돼 있으므로 여기만
`cumulative_l1`로 바꾸면 비교가 성립하지 않는다. **캠페인 전체를 옮길 때 이 팔도 같이 다시
돈다** — 연휴 재학습이 60런(01의 40 + 02의 20)에서 **70런**이 된다.

### 실행

```bash
python tools/build_synwoodscape_native_labels.py          # 라벨 500장 (완료)
python tools/run_paper_final_native_prior.py --plan_only
python tools/run_paper_final_native_prior.py --gpu=0 --seeds=0   # 정찰 2런, 약 25분
python tools/run_paper_final_native_prior.py --gpu=0             # 전체 10런, 약 2시간
```

> **[2026-09-22 사용자 결정] 먼저 seed 0 하나만 돌린다.** 기다릴 시간이 없어 방향만 보고,
> 5시드 전체는 다음 재학습 때 함께 돌린다.
>
> **⚠ n=1은 판정이 아니다.** 캠페인 판정 규약(`|Δ| > σ_seed` **그리고** `|Δ| ≥ 2·SE`)은
> n=1에서 SE가 정의되지 않아 **적용할 수 없다.** 시드 하나의 값은 평균에서 대략 σ_seed만큼
> 떨어져 있는 것이 정상이므로, `iou_free` 기준 **±0.002 규모의 차이는 읽지 않는다.**
> 이 정찰에서 읽을 수 있는 것은 **부호와 대략의 크기**뿐이다.

### 정찰 결과 (seed 0, 고정 epoch 40) — **판정이 아니다**

| 지표 | scratch | 사전학습 8/4/±6 | **사전학습 30 m** | σ_seed |
|---|---:|---:|---:|---:|
| `iou_free` ↑ | 0.81056 | 0.81454 | **0.80570** (−0.0049) | 0.0018 |
| `fatal` ↓ | 0.12213 | 0.11356 | 0.12346 (+0.0013) | 0.0040 |
| `free_miss` ↓ | 0.08503 | 0.08595 | 0.08767 (+0.0026) | 0.0053 |
| `f1@10cm` ↑ | 0.59091 | 0.59010 | **0.58016** (−0.0108) | 0.0061 |
| `f1@20cm` ↑ | 0.82854 | 0.81930 | 0.82042 (−0.0081) | — |
| `range_mae` ↓ | 0.21799 | 0.21715 | 0.22282 (+0.0048) | 0.0038 |
| `MOR` ↓ | 0.04083 | 0.03566 | 0.04285 (+0.0020) | — |

원본 기하 사전학습이 **일곱 지표 전부에서 scratch보다 나쁘다.** 다만 이 지표들은 같은 예측에서
파생되므로 **일곱 개의 독립 확인이 아니다** — 방향이 함께 움직이는 것이 당연하다.

### ⚠⚠ seed 0은 이 비교에서 **실측으로** 믿을 수 없다

5시드가 있는 기존 팔로 직접 쟀다. `iou_free`의 (사전학습 − scratch) 시드별 차이:

| seed | 0 | 1 | 2 | 3 | 4 | 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Δ`iou_free` | **+0.00398** | −0.00386 | −0.00295 | −0.00558 | −0.00160 | **−0.00200** |

**seed 0만 보면 부호가 평균과 반대이고, 다섯 중 유일한 양수이며 가장 극단값이다.**
시드별 차이의 sd가 0.0036이므로 **한 시드는 ±0.004쯤 쉽게 흔들린다.** 원본 기하 팔의
−0.0049도 그 폭 안에 있다.

**그래서 이 정찰에서 말할 수 있는 것은 하나다** — *"원본 기하 사전학습이 도움이 된다는
근거는 없다."* **"더 해롭다"는 아직 쓸 수 없다.** 5시드가 필요하다.

### 곁가지 — 소스 과제의 난이도가 많이 다르다

| 사전학습 | 소스 `iou_free` (seed 0, ep40) |
|---|---:|
| 8/4/±6 @5cm | **0.9864** |
| 30 m @15cm | 0.8731 |

**기존 사전학습의 소스 과제는 거의 포화다.** 좁은 ROI에 승용차 차체가 크게 들어차 쉬운
문제였다는 뜻이고, 그렇게 쉬운 과제에서 배운 특징은 옮길 것이 적다. 원본 기하 쪽이 덜
포화인데도 전이가 더 낫지 않다는 점은 5시드에서 다시 볼 값어치가 있다.

### 예상 (정찰 전에 적어 둔 것)

**방향을 모른다.** 02의 결론은 *"사전학습이 해로운 이유는 source가 격자의 93.8 %를 관측하는데
로봇은 27.8 %만 관측하기 때문"*이었다. 원본 기하는 관측 비율을 82.5 %로 낮춰 타깃 쪽으로
다가가지만, free 비율은 87.6 → 60.6 %로 타깃(95.1 %)에서 **멀어진다.** 두 힘이 반대다.

---

## 8. 연휴 실행 계획 (2026-09-23~27) — **[2026-09-22 사용자 승인]**

**목표 하나.** `arc_huber`를 논문에서 완전히 걷어낸다. 캠페인의 모든 런을
`RANGE_LOSS_MODE=cumulative_l1`, `λ_R=0.15`로 다시 돌린다.

### 대상 70런 — 기존 트리를 덮어쓰지 않는다

새 루트 `runs/paper_final_cumulative/`에 쓴다. **기존 `runs/paper_final/`은 그대로 둔다** —
덮어쓰면 "두 형태가 얼마나 달랐나"를 나중에 확인할 수 없고, 지금 논문에 들어가 있는 숫자의
출처도 사라진다.

| 실험 | 런 | 1런 소요 | 합계 |
|---|---:|---:|---:|
| `01_overall` 고정 split | 5 | 5분 | 25분 |
| `01_overall` LOSO (7 fold × 5 시드) | 35 | 5분 | 175분 |
| `02` SynWoodScape 사전학습 (8/4/±6) | 5 | 19분 | 95분 |
| `02` source_prior · pinhole120 · pinhole150 | 15 | 5분 | 75분 |
| `02b` SynWoodScape 사전학습 (원본 30 m) | 5 | 17분 | 85분 |
| `02b` source_prior_native | 5 | 5분 | 25분 |
| **합계** | **70** | | **약 8시간** (GPU 1개) |

**GPU 0만 쓴다.** GPU 1은 다른 사용자의 작업(75 GB)이 올라와 있다.

디스크는 **약 66 GB**가 필요하다. 런마다 체크포인트가 **둘**이다(고정 epoch의
`model-000000040.pth`와 `model_best-*.pth`, 각 470 MB) -- 계획 단계에서 하나로 세어
34 GB로 적었던 것을 실측으로 정정했다(941 MB/런 × 70). 2.8 TB가 비어 있어 문제는 없고,
기존 `runs/paper_final/`이 이미 61 GB를 쓰고 있는 것과 같은 규모다.

### ✅ Phase 0 — 완료 (2026-09-22)

**`tools/train_synwoodscape.py`가 `range_loss_mode`를 노출하지 않는다.** 지금 그대로 돌리면
**미세조정만 `cumulative_l1`로 가고 사전학습 둘은 `arc_huber`에 남는다.** 그러면 "걷어낸다"가
절반만 된다.

1. `train_synwoodscape.py`에 `range_loss_mode` 인자를 추가하고 loss 호출에 전달
2. 두 사전학습 셸(`train_synwoodscape_binary_pretrain.sh`,
   `train_synwoodscape_native_pretrain.sh`)에 `RANGE_LOSS_MODE` 환경변수 노출
3. 세 큐 러너가 `RANGE_LOSS_MODE=cumulative_l1`, `LAMBDA_R=0.15`를 넘기게 한다
4. **1 epoch 스모크로 실제로 그 항이 쓰였는지 확인**한다 -- `config.json`의
   `range_loss_mode`와 로그의 `loss_range`를 둘 다 본다
5. 테스트 470개 통과 확인 후 커밋

**결과.** `tools/paper_final_aux_loss.py`가 세 러너의 공통 정본이 됐고, 두 사전학습 셸이
`RANGE_LOSS_MODE`/`LAMBDA_R`을 환경변수로 받는다. 기본값은 `arc_huber`/0.3이라 **러너를
그냥 돌리면 기존 캠페인이 그대로 재현된다**(기존 manifest와 대조해 확인: 차이는 새로 기록된
칸뿐이고 값은 같다).

1-epoch 스모크 두 개로 플래그가 **실제 계산 경로를 바꾸는지** 확인했다:

| | `arc_huber` | `cumulative_l1` |
|---|---|---|
| `config.json` `range_loss_mode` | `arc_huber` | `cumulative_l1` |
| `range_cumulative_mae` 스칼라 | 없음 | **0.9101 (= `loss_range`와 일치)** |
| train `loss_total` (1 epoch) | 1.2022 | 0.8101 |

`range_cumulative_mae`는 누적 분기에서만 기록되는 태그다. 회귀 테스트 9개
(`tests/tools/test_paper_final_aux_loss.py`)가 **사전학습 런을 포함해** 모든 런이 같은
보조항을 받는지 고정한다. 테스트 479개 통과.

### ✅ Phase 1 — 학습 완료 (2026-09-22 17:18 → 2026-09-23 02:01, 8시간 43분)

```bash
bash tools/run_paper_final_cumulative_campaign.sh   # 70런, 순서 고정
```

순서가 중요하다. **사전학습이 먼저**여야 그것에서 시작하는 팔이 막히지 않는다
(01_overall → 02 → 02b, 각 실험 안에서는 러너가 팔 순서를 보장한다).

큐는 **완료된 런을 건너뛰고 불완전한 런을 덮어쓰지 않는다**(`blocked_incomplete`). 중간에
죽어도 같은 명령을 다시 실행하면 이어진다.

#### 멈춤 방지 — `tools/paper_final_run_command.py`

러너 셋은 `subprocess.run`을 **timeout 없이** 부르고 있었다. 학습이 교착되면 큐가 그
자리에 영원히 선다 -- 사람이 없는 동안 8시간이 1런에서 날아간다. 두 한도를 건다.

| 한도 | 값 | 무엇을 잡나 |
|---|---|---|
| **침묵** | 20분 무출력 | 주 감지기. 가장 느린 런도 epoch이 30초 미만이다 |
| **총 시간** | 2시간 | 출력은 나오는데 안 끝나는 경우. 최장 런(19분)의 6배 |

**프로세스 그룹째 죽인다** -- `bash`만 죽이면 `python`이 GPU를 붙든 채 고아로 남아 다음
런이 연쇄 실패한다. 회귀 테스트가 손자 프로세스가 실제로 죽는지까지 잰다.
런마다 stdout이 `<루트>/<실험>/run_logs/<런>.log`에 남는다.

#### 실행 결과 — 70/70 정상

| 실험 | 런 | 소요 |
|---|---:|---|
| `01_overall` (고정 5 + LOSO 35) | 40/40 | 3시간 46분 |
| `02_projection_and_prior` | 20/20 | 3시간 01분 |
| `02b_native_source_prior` | 10/10 | 1시간 56분 |

**watchdog 발동 0회, Traceback·OOM 0건.** 70런 전부 `config.json`을 다시 읽어
`range_loss_mode=cumulative_l1`, `λ_R=0.15`, `num_epochs=40`, 최종 체크포인트 존재를
확인했다. 디스크 **65 GB**(정정한 추정 66 GB와 일치).

#### 실행 전 리허설 (1 epoch × 7런)

8시간을 태우기 전에 전 경로를 관통시켰다. 확인한 것:

- 세 러너 모두 `cumulative_l1`/λ_R=0.15로 런을 만든다 (70/70 검사, 새 루트 밖을 가리키는
  경로 0건)
- **체크포인트 의존 사슬** -- `source_prior`·`source_prior_native`가 **새 루트의**
  사전학습을 문다(전이 668텐서, 건너뜀 0)
- Phase 2 전 경로 rc=0. 무결성 재채점 최대 오차 **2.9e-4 < 허용치 1e-3**
  (실패는 "5런 기대, 1런 발견" 같은 **구조 검사**뿐 -- 1시드 리허설이므로 정상)

### ⚠ Phase 2에서 먼저 정한 것 — **판정 눈금을 낮추지 않는다**

재학습한 대조군의 고정 split σ_seed가 **0.0018 → 0.0007**로 2.6배 줄어 보였다. 눈금이
작아지면 같은 차이가 더 쉽게 "유의"가 되므로, 그대로 쓰면 주장이 공짜로 강해진다.
그래서 **5표본 sd 추정이 얼마나 흔들리는지**를 먼저 쟀다.

| 추정 방식 | 옛(`arc_huber`) | 새(`cumulative_l1`) | 자유도 |
|---|---:|---:|---:|
| 고정 split 5시드만 | 0.0018 | **0.0007** | 4 |
| LOSO fold별 σ_seed 평균 (7 fold) | 0.0024 | 0.0025 | -- |
| **8군 묶음** (고정 split + LOSO 7 fold) | **0.0025** | **0.0026** | 32 |

LOSO fold별 σ_seed 추정치 14개가 **0.0010~0.0049**에 흩어진다 -- 같은 절차를 반복해도
5표본 sd는 5배 범위로 흔들린다는 뜻이다. 자유도 32의 묶음 추정으로 보면 **0.0025 →
0.0026, 사실상 변화 없다.**

**결론: `cumulative_l1`이 시드 재현성을 개선했다는 근거는 없다.** 고정 split의 0.0007은
운 좋은 한 번의 추정이다. 판정 눈금은 사전 등록값 **0.0018을 유지**하고, 아슬아슬한 칸은
더 보수적인 묶음값 **0.0026**으로도 따져서 둘 다 적는다.

### Phase 2 — 무결성과 분석 (약 1시간)

```bash
bash tools/run_paper_final_cumulative_analysis.sh
```

확률맵을 내보내 **독립 경로로 다시 채점**하고 학습 로그와 대조한다(허용치 1e-3).
**하나라도 실패하면 패키지를 만들지 않는다.** 그다음 짝지은 비교를 낸다.

**σ_seed를 새 대조군에서 다시 잰다.** 판정 규칙(`|Δ̄| > σ_seed` **그리고** `|Δ̄| ≥ 2·SE`)이
이 값에 걸려 있는데, 대조군을 재학습했으니 예전 0.0018을 그대로 쓰면 "차이 없음" 경계가
옛 런의 산포로 정해진다. 그래서 `01`을 먼저 돌리고 그 고정 split 5시드의 `iou_free` 표준편차를
`02`·`02b`에 넘긴다(`report_paired_arms.py --sigma_seed`).

constant-map baseline은 라벨과 val split만으로 정해지므로 바뀌지 않는다 -- 그래도 새 트리의
것을 가리켜 대조할 수 있게 `--baseline_path`를 열어 뒀다.

### Phase 3 — 패키지 (약 1시간)

```bash
bash tools/run_paper_final_cumulative_packages.sh            # 대조만 (기본)
APPLY=1 bash tools/run_paper_final_cumulative_packages.sh    # 검토 뒤 반영
```

**기본은 덮어쓰지 않는다.** 임시 폴더에 만들고 칸 단위 대조만 출력한다.

| 패키지 | 무엇을 하나 |
|---|---|
| `01_overall` | 새 루트로 재생성 → **CSV diff** → 바뀐 칸만 목록으로 |
| `02_projection_and_prior` | 같음 (`--control`도 새 루트를 가리킨다) |
| `02b_native_source_prior` | **신규 생성.** 표를 세 줄로 낸다 — scratch / 가공한 사전학습 / 원본 기하 |
| `03_boundary_uncertainty` | **손대지 않는다** — 아래 표로 실증했다 |

**논문 문장은 바뀌지 않고 자릿수만 바뀐다**는 예상을 diff로 검증한다. 만약 결론이 뒤집히는
칸이 나오면 **그것 자체가 보고 대상**이다 -- 조용히 숫자만 갈아끼우지 않는다.

#### `03`을 건드리지 않는 근거 (30런 config.json 실측)

| 칸 | `λ_R` | `range_loss_mode` | 보조항이 계산되나 |
|---|---:|---|---|
| `A_ce` · `B_perset` · `C_hard` · `C_soft` | 0.0 | (기록 없음) | **아니오 — 항 자체가 만들어지지 않는다** |
| `D_range` | 0.3 | (기록 없음 = 당시 기본값 `arc_huber`) | 예 |
| `E_cumulative` | 0.15 | `cumulative_l1` | 예 |

논문 사다리는 `A_ce → B_perset → C_hard → C_soft → E_cumulative` 다섯 칸이고 `D_range`는
빠진다. 그래서 **사다리의 어느 칸도 `arc_huber`로 학습되지 않았다.** 재학습이 불필요하다.
(`D_range`가 `arc_huber`였다는 사실은 `03` 패키지 provenance에 이미 글자로 적혀 있다.)

#### 검증 도구 셋 — 눈으로 훑지 않는다

| 도구 | 무엇을 잡나 |
|---|---|
| `tools/diff_paper_package_csv.py` | CSV 칸 단위 대조. **판정 열**(`verdict`·`sign_agreement`·`better_direction`)이 바뀌면 따로 센다 |
| `tools/audit_paper_prose_numbers.py` | **산문에 박힌 숫자.** 옛 CSV에는 있고 새 CSV에는 없는 값을 `stale`로 뽑는다 |
| `tools/check_results_tree_purity.py` | **옛 결과 트리가 섞인 것.** 산출물의 경로 문자열을 훑는다. 만들자마자 `experiment_manifest.json`의 `control_arm`이 옛 경로를 박고 있는 것을 잡았다 |

산문 감사가 필요한 이유는 규모다 -- `REPORT.md`·`README.md`에 소수 셋넷 자리 숫자가
**416개** 박혀 있다(01 125 · 02 146 · 03 145). CSV만 갈고 산문을 두면 논문이 데이터와
어긋난 채 작성되는데, 그건 눈으로 잡히지 않는다.

지금(재학습 전) 추적률: **01 122/125 · 02 136/146 · 03 143/145**. 남는 것은 팔 사이
차이처럼 손으로 계산해 적은 값이라 사람이 본다.

### Phase 4 — 기록

원장 §4·§6·§6b의 결과 표를 갱신하고, `AGENTS.md`의 캠페인 블록과 메모리를 맞춘다.
**철회된 주장(안전 개선·정확도 개선·재현성 개선)을 되살리지 않는지 확인한다.**

### 못 하는 것

- **`04_edge_deployment` — [2026-09-22 사용자 결정] 이번에는 건너뛴다.** Orin 실측이라 장비가
  필요하고, 나중에 따로 논의해서 테스트한다. **Results 초안도 이 절을 빼고 쓴다.**
  (`Y=4` 재실측이 필수라는 점은 그대로다 — 기존 20.2 FPS는 `Y=1` 값이다.)
- **02b의 유보 둘은 재학습으로 해소되지 않는다** — 수동 보정 없음, BEV가 정사영 아님(§6b).

### 중간 보고

각 phase가 끝날 때마다 원장에 기록하고 커밋한다. 돌아오셨을 때 `git log`와 이 절만 보면
무엇이 됐고 무엇이 막혔는지 알 수 있어야 한다.

---

## 7. 부록 — 원래의 미동결 목록 (이력)

`01_overall` 실행 전에 적어 둔 항목이다. §6이 이를 대체한다.

- `02` DS-native 대 undistort+pinhole의 virtual pinhole FOV와 리샘플링 규약
  → **해소.** §6.2에서 화각 120°/150° 두 팔로 동결했다. 120°는 정면 각해상도를 어안과
  맞춘 값(2.58 대 2.55 px/deg)이고, 150°는 커버리지를 96.7 %까지 올려 준 유리한 조건이다.
- `02` SynWoodScape pretraining의 source 정식화·source seed 처리·전이 범위
  → **해소.** §6.3에서 binary·시드 5개(target과 1:1)·전체 전이로 동결했다.
- `02` front-only의 고정 3-camera task 평가와 front 공통 가시영역 보조 평가
  → **기각.** §6.1. 결과가 기하로 이미 정해져 있고 방어할 결정이 아니다. 커버리지 표만
  setup 절에 싣는다.
- `04` Orin warm-up, 반복 횟수, latency percentile, 전력 측정 명령 → 미해소, §6.5
