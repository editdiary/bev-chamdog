# Training and Implementation Details — 캠페인 v3 학습 설정 정본

> **이 문서를 받은 사람/agent에게.** 논문의 "Training and Implementation Details"를 쓰는 데
> 필요한 값이 전부 여기 있다. **§7의 "쓰면 안 되는 값" 목록을 반드시 먼저 읽을 것** —
> 실행 기록 파일(`config.json`)에는 그 설정에서 실제로 동작하지 않는 값도 들어 있어서,
> 그대로 옮겨 적으면 틀린 M&M이 된다.
>
> 모든 값은 **실제로 돈 런(`runs/99_full_campaign/`, 2026-10-02~03)의 `config.json`·학습 로그와
> 소스 코드**에서 확인했다. 파라미터 수와 특징맵 크기는 모델을 실제로 생성해 측정했다.

> **[2026-10-03] 캠페인 v3로 고쳐 썼다.** 옛 캠페인(`docs/archive/paper_package_v2/`)과 다른 점은 셋이다.
> ① 광선 보조항 `L_range`를 **뺐다**(목적함수는 `½L_F + ½L_N + λ_B·L_B`), ② 학습 길이 40 →
> **100 epoch**(사전학습도 100), ③ 지표 개편(`metrics.md`). 옛 패키지의 숫자와 섞지 않는다.

**적용 범위.** 실험 01(40런), 02(20런), 02b(10런)는 **완전히 같은 학습 설정**을 쓰고 축
하나씩만 바꾼다(카메라 투영 모델 / 초기화). 실험 03(20런)은 **목적함수만** 바꾼다 — 그
네 칸의 정의는 `03_boundary_uncertainty/REPORT.md`에 있고, 그 밖의 설정은 이 문서와 같다.
`03`의 `C_soft` 칸이 곧 아래 §9.2의 목적함수다.

---

## 1. Optimizer와 learning rate schedule

| 항목 | 값 |
|---|---|
| optimizer | **AdamW** (β₁ = 0.9, β₂ = 0.999 — PyTorch 기본값) |
| 최대 학습률 | **1e-4** |
| **schedule** | **OneCycleLR** — cosine도, step도, 고정도 아니다 |
| **warmup** | **있다.** 전체 스텝의 **5 %** 동안 선형 상승 (`pct_start=0.05`) |
| decay | 그 후 **선형** 감소 (`anneal_strategy="linear"`), **cosine 아님** |
| weight decay | **1e-7** |
| momentum cycling | 끔 (`cycle_momentum=False`) |
| 갱신 주기 | **매 optimizer step**(배치마다). epoch 단위가 아니다 |
| 총 스텝 | `num_epochs × steps_per_epoch + 10` — 학습 길이에 묶여 있어 100 epoch이면 schedule도 100 epoch에 걸쳐 펴진다 |

> ⚠️ **가장 틀리기 쉬운 지점.** "AdamW with a learning rate of 1e-4"로만 쓰면 상수 학습률로
> 읽힌다. **one-cycle schedule, 5 % linear warmup, linear decay**를 반드시 함께 적어야
> 재현된다.

**weight decay 1e-7에 대한 주의.** 이것은 Simple-BEV upstream 기본값이고 **실질적으로
정규화가 없는 값**이다. "we use weight decay for regularisation"처럼 쓰면 과장이다.
정규화 손잡이를 스윕해서 전부 기각한 이력이 있으므로, 그냥 값만 보고하는 편이 정확하다.

---

## 2. 배치 크기와 학습 길이

| 항목 | 값 |
|---|---|
| batch size | **8** |
| epoch | **100** (모든 런, SynWoodScape 사전학습 포함) |
| seed | **5개** (0, 1, 2, 3, 4) |
| `drop_last` | 마지막 배치가 **1개일 때만** 버린다 (BatchNorm이 배치 1에서 실패한다) |

**epoch만 쓰지 말고 스텝 수를 같이 쓸 것.** 데이터셋이 작아서 100 epoch이어도 고정
split은 2,400 스텝이다.

| 프로토콜 | train 프레임 | val 프레임 | epoch당 스텝 | 총 스텝 |
|---|---|---|---|---|
| **고정 split** (01의 주 결과, 02·02b·03의 전 팔) | 192 | 75 | 24 | **2,400** |
| **LOSO fold** (01의 일반화 평가, 7 fold) | 229~231 | 36~41 | 29 | 2,900 |
| SynWoodScape 사전학습 (02·02b의 source) | 400 | 100 | 50 | 5,000 |

---

## 3. Augmentation

**광도(photometric) 증강만 쓴다. 기하(geometric) 증강은 쓰지 않는다.**

- **train split에만** 적용한다. validation은 항상 원본이다.
- 카메라 3대가 **같은 파라미터를 공유**한다(한 프레임 안에서).
- **BEV 라벨은 건드리지 않는다.**
- 샘플마다 아래 범위의 **균등분포**에서 뽑는다.

| 파라미터 | 범위 | 적용 순서 |
|---|---|---|
| gamma | 0.80 ~ 1.25 | 1 |
| brightness | 0.80 ~ 1.20 | 2 |
| contrast | 0.80 ~ 1.20 | 3 |
| saturation | 0.80 ~ 1.20 | 4 |
| additive Gaussian noise σ | 0.00 ~ 0.02 | 5 |

감마를 먼저 거는 이유: 실제 카메라 파이프라인에서 감마는 센서 응답 쪽에 있고 밝기·대비는
그 뒤의 보정에 해당한다.

**좌우 반전(horizontal flip)은 구현돼 있고 기하 정합성도 테스트로 고정돼 있지만, 확정
설정에서는 껐다**(`flip_augment=False`). "no geometric augmentation"이라고 쓰면 된다.

---

## 4. 정밀도와 학습 안정화

| 항목 | 값 |
|---|---|
| **mixed precision (AMP)** | **사용하지 않는다.** 전 구간 **fp32**다 |
| **gradient clipping** | **사용한다.** L2 norm **5.0** (`torch.nn.utils.clip_grad_norm_`) |
| gradient accumulation | 없음 |
| EMA / SWA | 없음 |

AMP 미사용은 `autocast`·`GradScaler`·`float16`·`bfloat16`을 학습 경로 전체에서 검색해
확인했다(해당 없음).

> ⚠️ **재현성 문구 주의.** 시드는 `torch.manual_seed`와 `numpy.random.seed`로 고정하지만
> **cuDNN 결정성 플래그는 설정하지 않는다**(`torch.backends.cudnn.deterministic`,
> `benchmark` 모두 미설정). 따라서 **"deterministic"이라고 쓰면 안 된다.**
> **"seeded; every configuration is trained with five seeds"**가 정확한 표현이다.
> 실측 시드 간 표준편차는 고정 split `iou_free` 기준 **0.0004**다(`01_overall`). 차이의
> 유의성은 이 값이 아니라 **시드로 짝지은 t-검정의 95 % 신뢰구간**으로 판정한다
> (`evaluation_protocol.md`).

---

## 5. 실행 환경

| 항목 | 값 |
|---|---|
| GPU | **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** |
| GPU 개수 | **1장** (단일 GPU) |
| 분산 학습 | **없음** (DDP / DataParallel 미사용) |
| framework | **PyTorch 2.7.0+cu128** |
| CUDA | 12.8 |
| Python | 3.11.15 (conda 환경 `bev-chamdog`) |
| numpy | 1.26.4 |
| dataloader workers | 8 |

---

## 6. 학습 시간 (실측 중앙값)

| 런 종류 | 에폭당 | 런당 (100 epoch) |
|---|---|---|
| 고정 split (로봇, 192프레임) | 8.3 초 | **약 14분** |
| LOSO fold (로봇, 229~231프레임) | 8.8 초 | 약 15분 |
| 핀홀 투영 (02) | 8.5 초 | 약 14분 |
| SynWoodScape 사전학습, 변환 기하 (02, 240×240 격자, 카메라 4대) | 30.1 초 | 약 50분 |
| SynWoodScape 사전학습, 원래 기하 (02b, 200×200 격자) | 27.7 초 | 약 46분 |

전체 wall-clock: 2026-10-02 18:04 → 10-03 21:40, **90런 약 27.6시간**(분석 포함, 재시도 0회).
에폭 시간은 각 런 로그의 `time` 필드 중앙값이다.

---

## 7. `config.json`에 기록돼 있지만 **실제로 동작하지 않는** 값

실행 기록은 trainer의 **모든 인자**를 저장하므로, 그 설정에서 쓰이지 않는 값도 들어 있다.
**그대로 옮겨 적으면 틀린 M&M이 된다.**

| 키 | 기록된 값 | 실제 |
|---|---|---|
| `max_class_weight` | 20 | **쓰이지 않는다.** 클래스 가중치가 계산은 되지만 soft-boundary 경로로 전달되지 않는다 (§8-① 참고) |
| `label_smoothing` | 0.0 | **쓰이지 않는다.** cross-entropy 경로 전용이다. soft-boundary의 대응물은 `label_eps`(= 0.0) |
| `lambda_r` | 0.0 | **보조항 없음.** 0이면 그 항이 아예 만들어지지 않는다 |
| `range_loss_mode` | `arc_huber` | **쓰이지 않는다.** `lambda_r = 0`이라 보조항이 없다. 이 값을 논문에 옮기면 안 된다 |
| `delta_r_m`, `huber_beta_m` | 0.15, 0.15 | 같음 — 보조항 전용 |
| `sigma_alpha` | None | 옛 파라미터. `sigma_m`을 쓰면 비활성 |
| `delta_r_over_m` | None | 보조항 전용 |
| `val_tail_fraction` | 0.2 | `val_sequences`를 명시했으므로 쓰이지 않는다 |
| `frame_split_fraction` | 0.0 | 프로브 전용 경로. 꺼져 있다 |
| `frame_block_len` | 0 | 같음 |
| `n_theta` | None | 보조항 전용(광선 수) |
| `label_constants` | 영역 비율·경계 하한 | **학습에 쓰이지 않는 기록값.** 라벨에서 한 번 계산해 손실 곡선 해석용으로 적어 둔다 (`evaluation_protocol.md`) |
| `keep_checkpoints` | 6 | 저장 정책일 뿐 학습과 무관 |

---

## 8. 질문에는 없지만 Training Details에 필요한 것 다섯

### ① 클래스 가중치를 썼다고 쓰면 안 된다 — 가장 중요하다

`config.json`의 `max_class_weight=20`은 **soft-boundary 경로에서 전달되지 않는다.**
대신 목적함수가 `½·L_F + ½·L_N` 형태로 **각 영역에서 따로 평균**을 내므로, 셀 개수와
무관하게 두 클래스가 정확히 절반씩 기여한다. 즉 **per-set 평균이 역빈도 클래스 가중치를
대체**한다. 배치 구성과 데이터셋 통계에 의존하지 않는다는 것이 이 설계의 요점이다.
(학습 배너에도 "soft_boundary에서는 쓰이지 않는다"고 출력된다.)

### ② 입력 정규화는 결국 표준 ImageNet이다

코드가 `−0.5`를 뺐다가 `Segnet.forward`에서 `+0.5`로 되돌린 뒤 ImageNet 통계
(`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)를 적용한다. 이 왕복은 upstream
Simple-BEV에서 온 것이라, 코드만 보면 입력 범위를 `[-0.5, 0.5]`로 오해하기 쉽다.
**순효과는 표준 ImageNet 정규화 하나**이므로 그렇게 쓰면 된다.

### ③ 체크포인트를 validation으로 고르지 않는다

**고정 epoch 100(마지막 epoch)**을 주 결과로 쓴다. validation으로 epoch을 고르면 평가 대상에 맞춰 고르는
선택 편향이 생기고, LOSO에서는 held-out 시퀀스가 곧 평가 대상이라 특히 그렇다.
validation-best는 진단용으로만 병기한다. 평가 동작점은 **τ = 0.5**이고 문턱 조정은 없다.

### ④ 초기화를 `scratch`로 줄여 쓰지 않는다

encoder는 **ImageNet 사전학습**이고, BEV decoder와 출력 head만 무작위 초기화다.
정확한 표현은 **"target-only training without BEV pretraining"**이다.

### ⑤ 실험 02에만 해당하는 프로토콜 둘

- **대조군을 다시 돌리지 않았다.** 실험 01의 고정 split 5런을 **시드 1:1로 짝지어**
  재사용한다(02·02b 공통). 논문에 **"paired by seed"**라고 명시하면 통계 절이 깔끔해진다.
  실험 03은 자기 안의 `A_ce` 칸이 대조군이다.
- SynWoodScape **사전학습도 위 하이퍼파라미터를 그대로** 쓴다(100 epoch 포함). 다른 것은
  데이터뿐이다 — 02는 BEV 격자 240×240(전 8 m / 후 4 m / 횡 ±6 m, **셀은 같은 5 cm**),
  02b는 원래 기하 200×200(전후좌우 15 m, 셀 15 cm). 입력 512×384(원본 종횡비), 카메라 4대.

---

## 9. 모델·목적함수 (Implementation Details에 같이 들어갈 값)

### 9.1 네트워크

| 항목 | 값 |
|---|---|
| 아키텍처 | Simple-BEV (`Segnet` 변형, 출력 채널만 `num_classes`로 교체) |
| 2D encoder | **ResNet-101**, ImageNet 사전학습, **출력 stride 8** |
| encoder 구조 | `conv1…layer2`(stride 8) → `layer3`(stride 16) → `UpsamplingConcat`로 stride 8 복원 → 1×1 conv로 채널 128 |
| 입력 해상도 | **512 × 288** (원본 1280 × 720을 PIL `BILINEAR`로 축소, 종횡비 16:9 유지) |
| 특징맵 | **64 × 36**, 128채널 (실측) |
| latent dim | 128 |
| 카메라 | 3대 (`front`, `left`, `right`). `rear`는 캘리브레이션에만 있고 라벨 생성에 쓰이지 않았다 |
| 카메라 모델 | **Double Sphere. 어안을 펴지 않는다** — BEV 복셀 중심을 DS 투영으로 이미지에 직접 투영한다 |
| BEV 격자 | **120 × 120**, 셀 **5 cm**, 전방 4 m / 후방 2 m / 좌우 ±3 m |
| lifting | **Y = 4** 높이 평면, 범위 −0.25 ~ 1.75 m, **복셀 중심 표본 = 0 / 0.5 / 1.0 / 1.5 m** |
| 다중 카메라 융합 | 유효 마스크 가중 평균 (`reduce_masked_mean`) |
| BEV decoder | Simple-BEV `Decoder`, 출력 head는 1×1 conv로 **2채널**(free / not-free) |
| 파라미터 | **41.00 M 전부 학습** (그중 encoder **37.04 M = 90.3 %**) — 실측 |

### 9.2 목적함수

`L = ½·L_F + ½·L_N + λ_B·L_B`  — **광선 보조항 없음**

부호 있는 수직 거리 `d`(**라벨만으로 결정되며 하이퍼파라미터가 아니다**)로 셀을 셋으로
나눈다.

| 영역 | 조건 | 목표 |
|---|---|---|
| `Ω_F` | `d > δ` | hard BCE, 목표 1 |
| `Ω_N` | `d < −δ` | hard BCE, 목표 0 |
| `Ω_B` | `|d| ≤ δ` | **soft** BCE, 목표는 절단 gaussian CDF of `d` |

| 하이퍼파라미터 | 값 |
|---|---|
| `δ` (경계 대역 반폭) | **0.30 m** |
| `σ` (라벨 경계 불확실성) | **0.10 m** (`k = δ/σ = 3`) |
| `λ_B` (경계 항 가중치) | **0.5** |
| `κ` (대역 target을 0.5로 섞는 계수) | 1.0 (= 섞지 않음) |
| `label_eps` (균일 label smoothing) | 0.0 (= 없음) |

`L_F`·`L_N`은 각 영역 **안에서 평균**(per-set mean)한다. 그래서 칸 개수와 무관하게 두 영역이
절반씩 기여한다. `L_B`도 대역 안 평균이다.

---

## 10. 고쳐 쓸 수 있는 영문 초안

> **Network.** We build on Simple-BEV. A ResNet-101 encoder, initialised from ImageNet,
> maps each 512 × 288 fisheye image to a 64 × 36 feature map of 128 channels (output
> stride 8; the stride-16 `layer3` output is upsampled and concatenated with the stride-8
> features before a 1 × 1 projection). BEV features are obtained by projecting the centre
> of every voxel of a 120 × 120 × 4 grid — 5 cm cells covering 4 m forward, 2 m rearward
> and ±3 m laterally, sampled at 0, 0.5, 1.0 and 1.5 m above the ground — into each camera
> **through the Double Sphere model of that camera**, i.e. without rectifying the fisheye
> images. Features from the three cameras are combined by a validity-masked mean, and a
> convolutional BEV decoder predicts a two-channel free / not-free logit map. The model has
> 41.0 M parameters, of which 37.0 M are the encoder; the BEV decoder and the output head
> are randomly initialised, so the model is trained on the target data only, **without any
> BEV pretraining**.
>
> **Objective.** Let `d` be the signed perpendicular distance from a cell centre to the
> ground-truth free-space boundary, computed from the labels alone. Cells with `d > δ` and
> `d < −δ` receive a hard binary cross-entropy towards 1 and 0 respectively, **each
> averaged over its own set so that the two classes contribute equally regardless of their
> cell counts**; cells with `|d| ≤ δ` receive a soft target given by a truncated Gaussian
> CDF of `d`. We use `δ = 0.30 m` and `σ = 0.10 m`, and weight the boundary term by
> `λ_B = 0.5`. No other loss term is used.
>
> **Training.** We train with AdamW (weight decay 1e-7), batch size 8, for 100 epochs — 2,400
> optimiser steps on the fixed split — under a one-cycle schedule that ramps the learning
> rate linearly to 1e-4 over the first 5 % of steps and then decays it linearly. Gradients
> are clipped to an L2 norm of 5. Training is in full fp32; we do not use mixed precision.
> The only augmentation is photometric (gamma, brightness, contrast, saturation and
> additive Gaussian noise, shared across the three cameras and applied to the training
> split only); no geometric augmentation is used. Every configuration is trained with five
> seeds, and we report the final (epoch-100) checkpoint at an operating point of τ = 0.5; the
> epoch is never selected on validation. Training a single run takes about 14 minutes on
> one NVIDIA RTX PRO 6000 (PyTorch 2.7.0, CUDA 12.8).

**이 초안을 고칠 때 지켜야 할 것 넷.**

1. `scratch`로 줄여 쓰지 않는다 → *target-only training without BEV pretraining* (§8-④)
2. 학습률을 "1e-4"로만 쓰지 않는다 → **one-cycle**을 함께 (§1)
3. 클래스 가중치를 썼다고 쓰지 않는다 → **per-set 평균이 대체** (§8-①)
4. **보조항을 썼다고 쓰지 않는다** → `config.json`에 `range_loss_mode`가 남아 있어도 `λ_R = 0`이다 (§7)

---

## 11. 출처 (값을 다시 확인하고 싶을 때)

| 무엇 | 어디 |
|---|---|
| 실제로 돈 런의 전체 인자 | `runs/99_full_campaign/01_overall/fixed_split/logs/final_s0/config.json` |
| optimizer·scheduler·clipping | `tools/train_robot_bev.py` (`OneCycleLR`, `clip_grad_norm_`) |
| 증강 범위 | `projects/datasets/photometric.py` |
| 입력 해상도·정규화 | `projects/datasets/robot_simplebev.py`, `third_party/models/simple_bev/nets/segnet.py` |
| 목적함수 | `projects/common/soft_boundary.py`, 설계 정본 `docs/archive/research/soft_boundary_loss_design.md` |
| BEV 격자·lifting | `projects/bev_gt/grid.py`, `projects/datasets/simplebev_vox.py` |
| 실험별 해석된 설정 | `docs/99_paper_results/*/provenance/environment_and_config.json` |
| 동결 설정 요약 | [`setup.md`](setup.md) §4 |
| 지표 정의·보고 규칙 | [`metrics.md`](metrics.md) |
