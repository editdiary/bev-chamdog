# Evaluation Protocol — 지표 정의·집계·통계 정본

> **이 문서를 받은 사람/agent에게.** 논문의 M&M "Evaluation Protocol" 절과 Results 각 절의
> 지표 선택에 필요한 것이 전부 여기 있다. **§5(집계)와 §6(마스크)을 반드시 읽을 것** —
> 지표마다 집계 단위가 다르고, 평가 마스크가 둘인데 취급이 서로 반대다. 두 곳이 논문에서
> 가장 틀리기 쉽다.
>
> 학습 설정(optimizer·증강·환경)은 [`training_details.md`](training_details.md)에 있다.
> 이 문서는 **무엇을 어떻게 재는가**만 다룬다.
>
> 모든 정의는 소스 코드에서 확인했고, 인용한 수치는 실제 런에서 측정했다. 출처는 §12.

> ### [2026-10-03] 캠페인 v3 — 정의도 수치도 새것이다
>
> 2026-10-02에 지표를 바꿨고(사용자 확정), 이 폴더의 CSV와 아래 인용 수치는 그 정의로 다시 돌린
> **캠페인 v3**(`runs/99_full_campaign`, 90런, 100 epoch, 광선 보조항 없음) 값이다.
>
> | 옛 (`docs/archive/paper_package_v2/`) | 새 (이 폴더) |
> |---|---|
> | `fatal_rate` (= 1 − precision, **micro**) | **Precision** (프레임 **macro**) |
> | `free_miss_rate` (= 1 − recall, **micro**) | **Recall** (프레임 **macro**) |
> | `occupied_f1_{10,20,40}cm` (**micro**) | **BF@{0.10, 0.20, 0.30} m** (프레임 **macro**) |
> | `range_mae` · `range_bias` · `range_missed_obstacle_rate` | **뺐다** |
> | — | **IoU$_\text{non-free}$** (보조), **거리 고리별** 네 지표 (보조) |
> | 판정: $|\bar\Delta|>\sigma_\text{seed}$ 그리고 $\ge 2\,\mathrm{SE}$ | **짝지은 t-검정 95 % 신뢰구간**(§7) |
>
> 정의의 코드 정본은 `projects/common/metric_spec.py`다. 옛 패키지의 숫자와 섞지 않는다.

---

## 0. Results 구성과의 대응

논문 Results가 네 절로 계획돼 있다. 절마다 필요한 지표가 다르므로 먼저 대응표를 둔다.

| Results 절 | 근거 실험 | 프로토콜 | 주 지표 | 보조 |
|---|---|---|---|---|
| **1. Overall BEV Perception Performance** | `01_overall` | **고정 split**(대표) + **LOSO**(일반화) | IoU$_\text{free}$ + **margin** | Precision·Recall 쌍, BF$@\tau$ |
| **2. 설계 선택 ablation** (§11 제목 제안) | `02_projection_and_prior` | 고정 split, **시드 짝지음** | 짝지은 $\Delta$IoU$_\text{free}$ | Precision·Recall, BF$@\tau$ |
| **3. Effect of Boundary-Uncertainty-Aware Learning** | `03` (loss ladder) | 고정 split | 사다리별 IoU$_\text{free}$ + 재현성 | 동작점 안정성 |
| **4. Edge Deployment** | `04` (Orin) | 지연·FPS·전력 | (정확도 지표 아님) | — |

**M&M의 Evaluation Protocol 절에는 §1~§8을 쓰고, 절별 지표 선택은 §9를 따른다.**

---

## 1. 표기와 라벨 분할

BEV 격자 $\mathcal{G}$는 $120\times120$, 셀 5 cm, 전방 4 m / 후방 2 m / 좌우 ±3 m다.
프레임 $t$에서 라벨이 주는 것은 셋이다 — 통과성 $o_t$, 관측 $v_t$, 유효 $m_t$.

**$m_t(c)=1$인 셀 안에서 완전 분할된다.**

$$
F_t=\{c: o_t(c)\wedge v_t(c)\},\qquad
O_t=\{c:\neg o_t(c)\wedge v_t(c)\},\qquad
U_t=\{c:\neg v_t(c)\}
$$

**이 과제의 정의: 통행 가능(free)은 "통과할 수 있으면서 실제로 관측된" 곳뿐이고,
관측되지 않은 곳은 전부 통행 불가다.** 즉 이진 분류의 not-free는 $O_t\cup U_t$다.
이것은 배포 시 안전 규약이다 — 못 본 곳을 비어 있다고 가정하지 않는다.

모델은 free 확률 $p_t(c)$를 내고, **동작점 $\theta=0.5$**에서

$$\hat F_t=\{c: p_t(c)\ge\theta\}$$

로 이진화한다. 문턱 조정은 하지 않는다.

**$O_t$의 예측 대응물 $\hat O_t$는 별도 head가 아니라 $\hat F_t$의 경계에서 유도한다**
(§4.3 참고). 모델의 출력은 2채널(free / not-free) 하나뿐이다.

> ⚠️ **기호 충돌 주의.** 구현이 **동작점과 경계 허용오차를 둘 다 `τ`로 부른다**
> (`report_threshold_sweep.py`의 `--taus`는 동작점이다). 논문에서는 반드시 갈라 쓸 것 —
> 이 문서는 **동작점 $\theta$**, **BF 허용오차 $\tau$**(BF$@\tau$)로 쓴다.

---

## 2. 두 프로토콜

| | **고정 split (주 결과)** | **LOSO (일반화)** |
|---|---|---|
| 묻는 것 | 대표 성능 | 처음 보는 시퀀스로의 일반화 |
| train | `raws2,raws3,rawos1,rawos2,rawos4` — 192프레임 | 6개 시퀀스 — 229~231프레임 |
| val | `raws1,rawos3` — 75프레임 | 남은 1개 시퀀스 — 36~41프레임 |
| 반복 | 시드 5개 | **7 fold × 시드 5개** |

**두 프로토콜의 수치를 하나로 합치거나 직접 우열 비교하지 않는다.** 학습 데이터 크기도
묻는 질문도 다르다. 값이 우연히 비슷해도 마찬가지다(실측 margin이 +0.2978 대 +0.2936으로
비슷하지만 합치지 않는다).

**LOSO에서 반드시 밝힐 것 둘.**

1. **fold 간 산포가 시드 산포의 9.4배다**(0.0260 대 0.0027). fold 차이를 모델의
   일반화 능력만으로 읽지 않는다 — 장면 난이도·표본 오차·라벨 품질이 섞여 있고 분리할 수 없다.
2. **fold는 독립 표본이 아니다.** 학습 데이터를 6/7씩 공유한다. 따라서 $\mathrm{SD}/\sqrt{7}$
   은 표준오차가 **아니며**, LOSO에 SE를 쓰지 않는다.

---

## 3. 주 지표가 단독으로 설 수 없는 이유 — constant-map baseline

**이 논문에서 가장 중요한 평가 규약이다.**

온실은 통로 배치가 규칙적이라, **이미지를 전혀 보지 않고** 학습 시퀀스의 칸별 다수결
free 지도 $M^\star$ 하나를 통째로 답으로 내는 것만으로도 상당한 IoU가 나온다.

$$
M^\star(c)=\Big[\tfrac{1}{|T_\text{train}|}\textstyle\sum_{t\in T_\text{train}}\mathbb{1}[c\in F_t]\ \ge 0.5\Big]
$$

실측값이다.

| 프로토콜 | constant-map baseline |
|---|---|
| 고정 split | **0.5180** |
| LOSO 7 fold | **0.385 ~ 0.685** (fold 간 SD **0.141**) |

LOSO 기준선의 fold 간 SD 0.141은 **모델의 fold 간 SD(0.026)의 5.4배**다. 즉 기준선이
장면에 크게 휘둘리고 모델은 그렇지 않다.

따라서 **주장은 IoU 값이 아니라 margin이다.**

$$\mathrm{margin}=\mathrm{IoU}_\text{free}-\mathrm{IoU}^{\text{const}}_\text{free}$$

> **규칙: IoU$_\text{free}$를 baseline·margin 없이 단독 보고하지 않는다.** 본문 문장도
> "0.81을 달성했다"가 아니라 "상수 지도 대비 +0.29"로 쓴다. LOSO에서는 **raw IoU 순위와
> margin 순위가 실제로 뒤집힌다** — 기준선 없는 표는 결론이 반대가 된다.

---

## 4. 지표 정의 — 의도별로

지표를 넷으로 나눈다. **각 묶음이 다른 실패 양상을 잡도록 설계돼 있고, 섞어 읽으면 안 된다.**

### 4.1 겹침 — Free-space IoU (**1차 지표**)

**의도:** "통행 가능 영역을 전체적으로 얼마나 맞게 그렸나." 허용오차 없다.

$$
\mathrm{IoU}_\text{free}=\frac{1}{|T'|}\sum_{t\in T'}\frac{|\hat F_t\cap F_t|}{|\hat F_t\cup F_t|},
\qquad T'=\{t\in T:\ |\hat F_t\cup F_t|>0\}
$$

모든 집합 연산은 $m_t=1$인 셀로 제한된다. 합집합이 빈 프레임은 IoU가 정의되지 않으므로
평균에서 뺀다(0점으로 세면 "그 클래스가 없는 장면을 완벽히 맞혔는데 0점"이 된다).

**왜 free 하나만 재고 mIoU·accuracy를 쓰지 않는가.** 평가 마스크 안에서 not-free가 약
80 %라 **"전부 not-free"가 accuracy 0.80을 받는다.** 과거에 실제로 트리비얼 해가 학습된
모델을 이긴 전례가 있다(`iou_drivable`에서 "전부 drivable" 0.935 대 모델 0.891).
mIoU도 not-free 쪽 IoU가 구조적으로 높아 평균이 부풀려진다. **보고하지 않는다.**

### 4.2 안전과 보수성 — Precision·Recall, 반드시 한 쌍으로

**의도:** 같은 오분류라도 **비용이 다르다.** 막힌 곳을 뚫렸다고 하면 충돌이고(precision이
떨어진다), 뚫린 곳을 막혔다고 하면 경로를 잃을 뿐이다(recall이 떨어진다). 두 방향을 따로 센다.

$$
P_t=\frac{|\hat F_t\cap F_t|}{|\hat F_t|},\qquad
R_t=\frac{|\hat F_t\cap F_t|}{|F_t|},\qquad
\mathrm{Precision}=\frac{1}{|T'|}\sum_{t\in T'}P_t,\quad
\mathrm{Recall}=\frac{1}{|T'|}\sum_{t\in T'}R_t
$$

$T'$은 §4.1과 같다(예측·GT free가 **둘 다** 빈 프레임만 뺀다). **한쪽만 빈 프레임은 그 비율을
0으로 센다** — 예: GT free가 있는데 예측 free가 하나도 없으면 $P_t=0$. 그 프레임을 빼면 "아무것도
주장하지 않는" 퇴행 해가 평균에서 빠져나간다.

**Precision이 planner가 실제로 지는 위험량이다** — "통행 가능하다고 **믿은** 곳 중 맞은 비율".
(2026-10-02 이전의 `fatal_rate`는 $1-$Precision을 셀을 모아(micro) 잰 것이었다.)

> ⚠️ **둘을 같은 동작점에서 한 쌍으로만 읽는다.** 문턱을 옮기면 서로 반대로 움직이므로,
> 하나만 좋아진 것을 "안전 개선"이라 부르면 안 된다. **이 프로젝트에 같은 주장을
> 동작점 이동으로 철회한 전례가 있다.** macro에서는 free를 몇 셀만 예측하고 맞힌 프레임이
> $P_t=1$을 받아 큰 프레임과 같은 표를 행사하므로, **Precision 단독은 더욱 속기 쉽다.**

### 4.3 경계 정밀도 — Boundary F-measure, BF$@\tau$

**의도:** "장애물 경계를 몇 cm 안에서 맞췄나." **전체 셀이 아니라 경계 셀만 잰다** — 경계는
격자의 약 1.1 %다. 면적 IoU로는 잴 수 없다 — $O_t$는 광선이 멈춘 **두께 1셀 표면**이라 한 칸만
밀려도 교집합이 무너진다(실측 `iou_obstacle` 0.312가 이미지를 보지 않는 규칙 0.473에 졌다).

**경계의 정의.** GT 경계 $O_t$는 라벨의 관측된 비통과 셀이고, 라벨의 관측 $v_t$가 **ego 원점
raycast**이므로 $O_t$는 **ego에서 보이는 free 공간의 첫 표면**이다. 예측 경계 $\hat O_t$는
별도 head 없이 $\hat F_t$에서 같은 규칙(방위각마다 광선이 처음 멈춘 셀)으로 유도한다.

> ⚠️ **DAVIS 계열 BF와 다르다** — 그쪽은 분할 경계 **전체(contour)**를 쓰지만 여기 경계는
> **ego-visible frontier**다. 물체 뒤편 윤곽은 경계 집합에 없다. 논문에 이 정의를 한 줄로
> 적는다: *"BF@τ, where the boundary is the ego-visible free-space frontier."*

**GT를 두껍게 만들지 않고 지표에 허용오차를 준다.** 라벨 두께는 센서의 물리적 정의가
결정하게 두고, "몇 cm까지 맞은 것으로 볼지"를 $\tau$로 명시적으로 노출한다.

$$d(p,\mathcal{S})=\min_{g\in\mathcal{S}}\lVert p-g\rVert_2$$

$$
P_{t,\tau}=\frac{\big|\{p\in\hat O_t: d(p,O_t)\le\tau\}\big|}{|\hat O_t|},\qquad
R_{t,\tau}=\frac{\big|\{g\in O_t: d(g,\hat O_t)\le\tau\}\big|}{|O_t|},\qquad
\mathrm{BF}@\tau=\frac{1}{|T_O|}\sum_{t\in T_O}\frac{2P_{t,\tau}R_{t,\tau}}{P_{t,\tau}+R_{t,\tau}}
$$

$T_O$는 $\hat O_t$와 $O_t$가 **둘 다** 빈 프레임을 뺀 집합이다. 한쪽만 비면 그쪽 비율을 0으로
본다($\hat O_t=\varnothing$이면 $P=0$, 따라서 BF$=0$). **프레임마다 F를 내고 평균한다** — macro
P·R의 조화평균이 아니다(DAVIS와 같은 방식).

**$\tau\in\{0.10,\ 0.20,\ 0.30\}$ m** (= 2 / 4 / 6 셀). **최댓값을 soft-boundary 대역 반폭
$\delta=0.30$ m와 맞췄다**(2026-10-02, 옛 눈금은 0.10/0.20/0.40). $\tau=0.05$(1셀)는 라벨 자체의
이산화 오차와 구별되지 않아 넣지 않았다.

> ⚠️ **$\tau_\max=\delta$라서 생길 수 있는 지적.** "제안 손실의 대역폭과 같은 눈금에서 평가했다"는
> 반론이 가능하다. 답은 둘이다 — 0.10·0.20 m는 $\delta$보다 좁고 거기서도 비교되며, CE 대조군도
> 같은 $\tau$에서 채점된다.

**여러 $\tau$를 나란히 두는 것이 핵심이다.** 하나만 고르면 그 값이 숨은 하이퍼파라미터가
되고, 여러 개를 두면 **"얼마나 밀렸나"의 분포**가 읽힌다 — 허용오차를 넓힐수록 차이가
사라지면 그것이 **"경계가 흐려졌다(끊긴 것이 아니라)"의 서명**이다.

BF의 $P_\tau$·$R_\tau$도 CSV에 함께 남는다(`bf_{τ}_precision`/`_recall`) — 경계가 **어느
쪽으로** 틀렸는지(과잉 예측 대 누락) 볼 곳이 여기다.

> ⚠️ **"정확도"가 아니라 "일치도"다.** 라벨이 LiDAR + 사람의 육안 보정이라 **절대 경계
> 오차는 측정 불가능**하다. "10 cm 이내로 정확하다"가 아니라 **"라벨 경계와 10 cm 이내에서
> 일치한다"**로 쓴다.

### 4.4 보조 — IoU$_\text{non-free}$ (논문 본문에 쓰지 않는다)

$$\mathrm{IoU}_\text{non-free}=\frac{1}{|T''|}\sum_{t\in T''}\frac{|\neg\hat F_t\cap\neg F_t|}{|\neg\hat F_t\cup\neg F_t|}$$

($m_t=1$ 안에서, $T''$은 합집합이 빈 프레임을 뺀 것). **변별력이 낮다** — not-free가 $m_t$
안의 약 83 %라 **"전부 not-free"라는 자명해가 이미 ≈ 0.83을 받는다.** 모델은 0.945다
(캠페인 v3 고정 split). CSV에만 남긴다.

> **방위각 자유거리 지표(`range_mae`·`range_bias`·`range_missed_obstacle_rate`)는 2026-10-02에
> 뺐다**(사용자 결정). 계산 코드는 옛 런 재채점용으로 남아 있다.

### 4.5 거리 고리별 성능 (보조 — 부록 권장)

**의도:** 근거리(충돌이 일어나는 곳)와 원거리(경로 계획에 쓰는 곳)의 성능이 다른지.

$\mathcal{G}$를 로봇 중심으로부터의 거리로 나눈 고리 $\mathcal{R}_k$로 $m_t$를 좁혀 **IoU$_\text{free}$,
IoU$_\text{non-free}$, Precision, Recall**을 같은 식으로 다시 잰다. **경계는 0 / 1.5 / 3 / 4 m다**
(0–2 / 2–4가 아니다). 격자가 전방 4 / 후방 2 / 좌우 3 m라 **3–4 m 고리는 전방만 남은 부분
고리**다. 고리 안에 예측·GT free가 둘 다 없는 프레임은 그 고리의 평균에서 빠진다.

### 4.6 학습 거동 — 손실 곡선 (모든 실험이 같은 항으로 기록, 2026-10-02)

**의도:** "가중 BCE는 경계 손실이 처음부터 오르고 soft-BCE는 수렴한다" 같은 학습 거동 진술.
모든 런이 같은 세 영역으로 손실을 나눠 기록한다 — 경계에서 0.30 m보다 먼 free $\Omega_F$,
먼 non-free $\Omega_N$, 경계 대역 $\Omega_B$($|d|\le0.30$ m, 유효 셀의 약 22 %).

- `loss_total`, `loss_{free,non_free,boundary}` — **모델이 실제로 받은 손실**과 그 영역별 값. 같은
  이름이지만 손실 함수가 다르므로 **값의 크기는 런끼리 비교하지 않고 모양(오르나·수렴하나)만
  비교한다.** soft-BCE의 경계 항은 0이 아니라 하한(target 엔트로피, `config.json`의
  `label_constants.loss_boundary_floor` ≈ 0.265)으로 수렴한다
- **`bce_{free,non_free,boundary}` — 논문에 싣는 값.** 모든 런의 예측을 **같은 함수**(가중치 없는
  BCE, 정답 0/1)로 잰 것이라 값끼리 직접 비교한다. "다른 함수라서 수렴해 보일 뿐"이라는 반론은
  이쪽으로 막는다. 영역 셀 비율은 `label_constants.frac_*`(약 10 / 67 / 22 %)

**집계:** 영역 평균은 epoch 안의 그 영역 모든 셀의 평균이고, 총 손실 `loss`는 그 epoch의 모든
셀을 한 집합으로 본 목적함수다. 그래서 가중 BCE는 $\sum_r f_r\,L_r$, soft-BCE는
$\tfrac12 L_F+\tfrac12 L_N+\lambda_B L_B$가 `loss`와 정확히 같다.

---

## 5. 집계 단위 — **전부 프레임 macro다** (2026-10-02 통일)

| 지표 | 집계 | 식 |
|---|---|---|
| **IoU$_\text{free}$**, IoU$_\text{non-free}$ | **프레임마다 계산 → 프레임 평균** (macro) | §4.1, §4.4 |
| Precision, Recall | **프레임마다 계산 → 프레임 평균** (macro) | §4.2 |
| BF$@\tau$ (와 그 $P_\tau$·$R_\tau$) | **프레임마다 계산 → 프레임 평균** (macro) | §4.3 |
| 고리별 네 지표 | 고리마다, **프레임 macro** | §4.5 |

**프레임을 평균에서 빼는 규칙은 하나다: 예측과 GT가 둘 다 빈 경우뿐.** 한쪽만 비면 0점이다.
macro라서 **배치를 어떻게 자르든 같은 값**이 나온다(구현이 배치마다 "합과 프레임 수"를 넘긴다).

> **2026-10-02 이전에는 IoU만 macro이고 나머지는 전부 micro였다**(셀·광선·경계 카운트를 모든
> 프레임에서 모아 한 번에 나눴다). 현재 이 폴더의 CSV는 그 옛 집계다. 같은 체크포인트에서 두
> 집계의 차이는 Precision −0.001, Recall +0.0001, BF@10cm·20cm +0.005 수준이었다(2026-10-02
> 재채점, `01` 고정 split `final_s0`).

### 시드·fold 집계 순서

| 프로토콜 | 순서 |
|---|---|
| 고정 split | 시드 5개 → **mean ± SD** |
| **LOSO** | **① fold 안에서 시드 5개 평균 → ② 7 fold의 macro mean ± SD** |

LOSO는 fold를 먼저 접는다. 시드를 먼저 접으면 프레임 수가 다른 fold가 다르게 가중된다.

---

## 6. 평가 마스크 — **둘이고, 취급이 서로 반대다**

| 마스크 | 정체 | 크기 | 취급 |
|---|---|---|---|
| **permanent blind** | 어떤 카메라 광선도 닿지 않는 셀 + ego 테이블에 가린 셀 | **813셀 = 5.65 %**, bbox **33×27** | **제외하지 않는다.** $v=0$ 강제 → $U$ → **not-free로 채점** |
| **invalid** | 수집 아티팩트(뒤쪽 카트 손잡이·미는 사람) | **384셀 = 2.67 %**, bbox 32×12 | **완전히 제외.** $m=0$이라 모든 지표에서 빠진다 |

둘이 겹치는 126셀을 정리하면 — **평가에 포함되되 항상 not-free인 셀 687개(4.77 %)**,
**완전히 빠지는 셀 384개(2.67 %)**다.

**사각지대를 제외하지 않는 이유는 과제 정의다.** 이 과제에서 "보이지 않는 곳"은 통행
불가이므로, 그것을 채점에서 빼면 **다른 과제를 재게 된다** — 라벨이 not-free라고 선언한
셀의 78.5 %가 빠진다.

**invalid를 제외하는 이유는 배포 시 존재하지 않기 때문이다.** 남겨 두면 모델이
"후방은 항상 가려져 있다"를 학습한다.

---

## 7. 통계 처리

### 보고 형식

- **mean ± SD**를 쓰고 **SE를 쓰지 않는다**(LOSO 이유는 §2).
- 소수 **3자리**, SD도 같은 자리. 시드 SD가 0.002 수준이라 4자리는 표만 지저분해진다.
- 시드 산포는 **SD와 n을 함께**(n = 5를 표 각주에).
- 시드 SD는 **참고값**이다(고정 split IoU$_\text{free}$ **0.0004**). 우열 판정에 쓰지 않는다 —
  판정은 아래 짝지은 검정으로만 한다.

### 짝지은 비교 (Results 절 2·3)

`02`·`02b`는 대조군을 다시 돌리지 않고 `01`의 고정 split 5런을 **시드 1:1로 짝지어** 쓴다.
`03`은 자기 안의 `A_ce` 칸이 대조군이다. 초기화·데이터 순서에서 오는 산포가 차이에서
상쇄되므로 훨씬 작은 효과를 볼 수 있다.

$$\Delta_s = M^{\text{arm}}_s - M^{\text{ctrl}}_s,\qquad s=0,\dots,4$$

$$\bar\Delta=\frac{1}{5}\sum_s\Delta_s,\qquad
\mathrm{SE}=\frac{\mathrm{sd}(\Delta)}{\sqrt{5}},\qquad t=\bar\Delta/\mathrm{SE}\ (\mathrm{df}=4)$$

**[2026-10-02 사용자 확정] 판정 규칙: 양측 짝지은 t-검정의 95 % 신뢰구간
$\bar\Delta \pm 2.776\,\mathrm{SE}$가 0을 포함하지 않으면 "유의한 차이"다.** 그 밖은 전부
"유의한 차이 없음"이다. 지표마다 따로 검정하고 다중비교 보정은 하지 않는다(캡션에 명시).

- n = 5에서 부호 검정·Wilcoxon 같은 비모수 검정은 95 %에 **도달할 수 없다**(가장 작은 양측
  p가 0.0625). 그래서 t-검정을 쓴다.
- LOSO fold에는 검정을 걸지 않는다(fold는 독립 표본이 아니다, §2).

**표 형식.** 논문 표에는 지표의 mean ± SD만 싣고 **유의한 칸에 `*`**를 붙인다.
$\bar\Delta$·CI·p값·부호 일치 시드 수는 전부 CSV(`paired_differences.csv` 등)에 기록하고
본문이나 부록에서 필요할 때 인용한다.

---

## 8. 체크포인트 선택과 무결성

### 고정 epoch

**epoch 100(마지막) 고정**이고 validation으로 고르지 않는다(학습 길이 근거는 원장 §5
"2026-10-02 (3)"). validation으로 고르면 평가 대상에 맞춰 고르는 선택 편향이 생기며,
**LOSO에서는 held-out 시퀀스가 곧 평가 대상**이라 편향이 직접 들어온다. 제안 설정에서는
실측 차이가 작다 — 고정 split에서 best-epoch 대 epoch-100 `iou_free` 차이가 **+0.0005**다.
**단 실험 03의 CE 칸(`A_ce`·`C_hard`)은 epoch 30 안팎에서 정점을 찍고 내려온다**(그 REPORT §4).

### 무결성 검사 (재현성 절에 한 줄 권장)

저장한 **문턱 적용 전 확률맵**을 독립 경로로 다시 채점해 학습 로그와 대조한다.
허용오차 $10^{-3}$, **실패 0건이어야** 결과를 쓴다.

| 실험 | 검사 | 실패 | 최대 절대차 |
|---|---|---|---|
| `01_overall` 고정 split | 확률맵 10개 | 0 | 5.80e-4 |
| `01_overall` LOSO | 확률맵 70개 | 0 | 7.05e-4 |
| `02_projection_and_prior` | 확률맵 30개 | 0 | 2.64e-4 |
| `02b_native_source_prior` | 확률맵 10개 | 0 | 2.61e-4 |
| `03_boundary_uncertainty` | 확률맵 40개 | 0 | 4.73e-4 |

"표의 숫자와 저장된 예측이 같은 것을 가리킨다"는 확인이다.

---

## 9. Results 절별 지표 선택

### 절 1 — Overall BEV Perception Performance

| | 싣는 것 |
|---|---|
| 주 표 (고정 split) | IoU$_\text{free}$ **+ baseline + margin**, Precision·Recall 쌍, BF$@\{0.10, 0.20, 0.30\}$ |
| 보조 (부록·CSV) | IoU$_\text{non-free}$, 고리별 네 지표, BF의 $P_\tau$·$R_\tau$ |
| 보조 표 (LOSO) | fold 7행 × (baseline, IoU, **margin**), macro mean ± SD, fold 간 SD |
| 본문 단서 | fold 간 SD가 시드 SD의 9.4배 · fold는 독립 표본 아님 · 데이터 267프레임 |

**LOSO 표에 fold별 baseline 열을 반드시 넣는다.** 없으면 raw IoU 순위가 실력 순위로
오독되는데, 실제로는 뒤집힌다.

### 절 2 — 설계 선택 ablation

| | 싣는 것 |
|---|---|
| 주 표 | 팔별 7지표 mean ± SD, **유의한 칸에 `*`**(§7). CI·p는 CSV |
| 축 A 전용 열 | **BEV 커버리지 %**와 **정면 각해상도 px/deg** ← 없으면 결과가 이상해 보인다 |
| 안전 | Precision·Recall 쌍과 BF$@\tau$ (옛 MOR은 2026-10-02에 지표와 함께 뺐다) |

### 절 3 — Effect of Boundary-Uncertainty-Aware Learning

**정본은 [`03_boundary_uncertainty/REPORT.md`](../03_boundary_uncertainty/REPORT.md)다.**

| | 싣는 것 |
|---|---|
| 사다리 | **`A_ce → B_perset → C_hard → C_soft` 넷** ([2026-10-02 사용자 결정] 광선 보조항을 뺐다 — soft-boundary BCE까지) |
| 주 비교쌍 | **`A_ce` 대 `C_soft`** |
| 인과 귀속 | **`C_hard` ↔ `C_soft`** — 대역 target만 다르다. 이 쌍이 없으면 집계 방식과 구분되지 않는다 |
| 주장 축 | **목적함수 수렴**(경계 대역 CE의 되올림 113 % → 2.4 %) · **경계 국소화**(`A_ce` 증가분의 96 %가 대역 안) · **val loss로 골랐을 때의 손해**(`iou_free` 0.0137 → 0.0008) · **문턱 재현성**(같은 recall을 만드는 τ의 시드 SD가 구간 전체에서 9~14배 작다) |
| 정확도 | τ=0.5에서 `iou_free` +0.0083*. **같은 recall에서는 +0.0019~+0.0041로 줄고 낮은 recall 쪽 3개 앵커에서만 유의**하다. 인과 귀속은 `C_soft` 대 `C_hard`(+0.0095*) |
| **반드시 함께 싣는 비용** | **BF$@0.10$의 손해** — τ=0.5에서 −0.0064*, 같은 recall에서 −0.014~−0.040(전 앵커 유의). BF$@0.30$은 +0.0054*로 뒤집히므로 **"경계가 번진다"로 쓴다** |
| **쓰면 안 되는 표현** | **"안전 개선"**(Precision·Recall은 동작점과 함께) · **동작점을 밝히지 않은 "정확도 개선"** · **"재현성 개선"**(문턱 재현성만 해당) |

**Precision 개선을 단독으로 쓰지 않는다.** 쓰려면 **같은 Recall에서의** 표를 반드시 함께 싣는다.

### 절 4 — Edge Deployment

정확도 지표가 아니라 지연·FPS·전력·메모리다. **기존 20.2 FPS는 `Y=1` 값이므로 `Y=4`
재실측이 필요하다.**

---

## 10. 용어·기호 점검표

| 점검 | 올바른 표현 |
|---|---|
| 동작점과 허용오차 | **다른 기호로** — 동작점 $\theta=0.5$, BF 허용오차 $\tau\in\{0.10,0.20,0.30\}$ m |
| IoU 단독 | 금지. **항상 baseline·margin과 함께** |
| BF$@\tau$ | "정확하다" 아님 → **"라벨 경계와 $\tau$ 이내에서 일치한다"**. 경계 = ego-visible frontier(DAVIS contour 아님) |
| 집계 | **전부 프레임 macro**라고 쓴다(§5) |
| Precision·Recall | **항상 쌍으로**. 하나만 좋아진 것은 "안전 개선"이 아니다 |
| mIoU·accuracy | **보고하지 않는다**(§4.1) |
| 사각지대 | "제외했다"가 아니라 **"not-free로 채점했다"**(§6) |
| 보조 loss 항 | **없음**(2026-10-02 결정). 이 폴더의 모든 런은 $\lambda_R=0$이다 |

---

## 11. 아직 정하지 않은 것 — 판단이 필요하다

### ⑴ 합성 사전학습 ablation을 Results 절 2에 넣을까

**권고: 넣는다. 캠페인 v3에서는 "해가 된다"가 측정됐다**(`02`·`02b` REPORT).

| 짝지은 $\bar\Delta$ (대조군: 사전학습 없음) | 변환 기하 사전학습 (`02`) | 원래 기하 사전학습 (`02b`) |
|---|---|---|
| IoU$_\text{free}$ | −0.0060* | −0.0110* |
| Precision | −0.0009 | −0.0071* |
| Recall | −0.0069* | −0.0068* |
| **BF$@0.10$** | **−0.0353*** | **−0.0345*** |

(`*` = 95 % CI가 0을 뺀다, 부호는 둘 다 5/5 시드. 원자료 `02*/data/table2_*.csv`)

**가장 크게 잃는 것은 경계다** — BF$@0.10$ 손해가 IoU 손해의 3~6배다. 기전 후보는 감독
밀도 차이로 측정돼 있다: source 라벨은 격자의 93.8 %를 관측하고 target 라벨은 27.8 %만
관측한다(`domain_prior.csv`). 1 epoch 때 사전학습 모델은 precision 0.392(대조군 0.695),
recall 0.952(대조군 0.922)로 **free를 넓게 선언한다**. 이 초반 차이는 3 epoch이면 거의
닫히지만 **epoch 100까지 남는 손해가 있다.**

**쓸 수 있는 문장:** "합성 어안 데이터(SynWoodScape)로 BEV를 사전학습하면 target 성능이
오히려 낮아지며, 손해는 경계 정밀도에 집중된다. source를 target 기하에 맞추든(5 cm, 8 m)
원래 기하(15 cm, 15 m)로 두든 같다." 원인을 감독 밀도로 **단정하지는 않는다** — 외형·기하·
격자 차이와 분리한 실험이 없다.

### ⑵ 절 2의 제목

현재 임시 제목 "Sensor/task adaptation ablation"은 내용과 맞지 않는다 — 센서 구성을 바꾸는
실험(front-only)은 기각됐고, 남은 둘은 **입력을 어떻게 소비하는가**와 **가중치를 어디서
가져오는가**다.

절 3이 **연구 대상**으로 이름 붙었으므로("Effect of Boundary-Uncertainty-Aware Learning")
절 2도 같은 방식이 자연스럽다.

| 축 A만 넣을 때 | 둘 다 넣을 때 |
|---|---|
| **"Effect of Native Fisheye Projection"** ← 절 3과 대칭, 권장 | **"Input Representation and Transfer"** |
| "Native Fisheye Projection vs. Rectification" | **"Design Choices: Camera Model and Initialization"** ← 권장 |
| "Effect of the Camera Projection Model" | "Ablation of Core Design Choices" (+ 하위 절 둘) |

둘 다 넣으신다면 **하나의 우산 제목을 억지로 만들기보다 하위 절 둘로 나누는 편**이 읽기
좋다 — 두 축이 정말로 다른 것을 묻기 때문이다.

### ⑶ 거리 구간별 IoU를 넣을까

**[2026-10-02] 넣기로 했다(CSV, 부록 권장).** 다음 캠페인부터 고리마다 네 지표가 학습 로그와
`fixed_split_ring_summary.csv`로 나온다. 경계는 **0 / 1.5 / 3 / 4 m**다. 0–2 / 2–4로 다시
자르려면 `projects/common/metric_spec.py`의 `RING_EDGES_M`을 캠페인 **전에** 바꾼다.

---

## 12. 출처

| 무엇 | 어디 |
|---|---|
| **지표 이름·눈금·tag 정본** | `projects/common/metric_spec.py` |
| IoU·Precision·Recall 정의와 macro 집계 | `projects/common/free_space_metrics.py` |
| 라벨 분할 $F/O/U$ | `projects/common/free_space.py` |
| BF$@\tau$ 정의와 macro 집계 | `projects/common/occupied_metrics.py` (`boundary_f_scores`) |
| 경계 유도(광선) | `projects/common/polar.py` (`frontier_cells`) |
| 평가 마스크 | `projects/datasets/robot_simplebev.py` (`build_bev_masks`) |
| LOSO 집계 순서 | `tools/report_loso.py` |
| 짝지은 판정 | `tools/report_paired_arms.py` |
| 실험별 수치 | `docs/99_paper_results/*/data/*.csv` |
| 무결성 결과 | `docs/99_paper_results/*/provenance/integrity.json` |
| 보고 규칙 요약 | [`metrics.md`](metrics.md) |
| 학습 설정 | [`training_details.md`](training_details.md) |
