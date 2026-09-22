# 실험 01 — Overall Performance: 실험·결과·해석·논문 구성

> 대상 실험 `paper_final / 01_overall` · 실행일 2026-09-18 · Git `78da736`
> 이 문서는 **서버 접속 없이 읽을 수 있게** 쓰였다. 인용한 모든 수치는 같은 폴더의
> `data/*.csv`에 있고, 그 CSV는 `make_package.py`가 결과 bundle에서 생성한 것이다.

---

## 0. 세 줄 요약

1. 고정 split에서 `iou_free` **0.815 ± 0.001**(시드 5개). 이미지를 보지 않는 constant-map
   기준선 0.518 대비 **+0.297**다.
2. 7개 시퀀스를 하나씩 빼고 학습하는 LOSO에서 macro **0.830**, 기준선 대비 **+0.293**.
   fold 간 표준편차 0.025는 fold 내부 시드 표준편차 0.0025의 **10.1배**다.
3. **raw `iou_free`로 fold를 줄세우면 결론이 뒤집힌다.** 기준선이 통로 폭에 좌우되기
   때문이다. 주 비교량은 raw 값이 아니라 기준선 대비 마진이다.

---

## 1. 무엇을 물었고 어떻게 쟀는가

### 1.1 과제

전방 주시 카메라 3대(front·left·right)로 로봇 주변의 **BEV(bird's-eye view) free space**,
즉 위에서 내려다본 격자에서 "지금 갈 수 있는 칸"을 예측한다. 격자는 120×120, 한 칸 5 cm,
전방 4 m / 후방 2 m / 좌우 ±3 m다. 출력은 칸마다 `free` / `not-free` 두 값이다.

### 1.2 두 프로토콜, 두 질문

같은 모델을 두 가지 방식으로 평가했다. **둘은 묻는 것이 다르므로 숫자를 합치지 않는다.**

| | A. 고정 split | B. LOSO |
|---|---|---|
| 질문 | 대표 성능과 학습 시드 산포는? | 처음 보는 시퀀스에서 얼마나 버티나? |
| 구성 | 학습 5시퀀스 → 검증 2시퀀스 고정 | 7시퀀스 중 1개를 빼고 6개로 학습, 7번 반복 |
| 프레임 | 학습 192 / 검증 75 | 학습 229~231 / 검증 36~41 |
| 시드 | 0,1,2,3,4 | fold마다 0,1,2,3,4 |
| 런 수 | 5 | 35 |
| 논문 역할 | **대표 성능 + 이후 모든 ablation의 공통 대조군** | **일반화 주장 + 층별 분석** |

LOSO는 leave-one-sequence-out의 약자다. held-out 시퀀스로 epoch를 고르면 시험 대상에
맞춰 고르는 셈이라 편향이 생기므로, **두 프로토콜 모두 고정 epoch 40만 주 결과**로 쓴다.
validation 최고 epoch 값은 진단용으로만 병기한다.

### 1.3 동결 설정

모델·loss·최적화 설정은 네 실험이 공유하며 [`../common/setup.md`](../common/setup.md) §4에
있다. 요지만 적으면 Simple-BEV + ResNet-101 stride 8, BEV 사전학습 없음, `Y=4` lifting,
soft-boundary loss, AdamW 40 epochs다.

**이 실험이 확인한 것:** 40런 전부 그 설정으로 돌았고, 런 이름·시드·split을 뺀
**설정 차이는 0건**이었다. 해석된 전체 값은 `provenance/environment_and_config.json`에 있다.

### 1.4 지표

정의와 보고 규칙은 [`../common/metrics.md`](../common/metrics.md)에 있다. 이 실험이
보고하는 것만 추리면:

- **주 지표 `iou_free`** — 예측 free 영역과 정답의 겹침 비율. **반드시 constant-map
  baseline 및 margin과 함께 읽는다.** 온실은 배치가 규칙적이라 이미지를 안 보고 지도
  하나만 외워도 0.39~0.69가 나오기 때문이다.
- **안전 한 쌍** `fatal_rate`(막힌 칸을 갈 수 있다고 오인) / `free_miss_rate`(뚫린 칸을
  막혔다고 오인). **같은 동작점 τ=0.5에서 한 쌍으로만** 읽는다.
- **경계** `f1@10/20/40cm` — 라벨과의 일치도이지 물리적 정확도가 아니다.
- **광선** `range_mae` / `range_bias` / `range_missed_obstacle_rate`.
  `range_bias`는 **양수가 위험한 쪽**이다(부호 규약은 `common/metrics.md` §3 — 문헌마다
  반대로 쓰므로 주의).

## 2. 결과가 믿을 만한지

논문 숫자를 꺼내기 전 네 단계를 통과했다. 원본은 `provenance/integrity.json`이다.

| 검사 | 결과 |
|---|---|
| 40런 완주 | `completed` 37 + `skipped_complete` 3 = 40, 실패 0건 |
| 설정 균일성 | 40개 config에서 런 이름·시드·split 외 차이 **0건** |
| 기하 일관성 | `height.json` 40개가 모두 `{Y=4, −0.25~1.75 m}` 단일 값 |
| **확률맵 재채점** | 학습 로그와 별도 저장한 확률맵을 **독립 경로로 다시 채점**해 대조. 8건(고정 split 1 + LOSO fold 7) 전부 `failures=0`, 최대 절대 차이 **3.47e-4**(고정) / **5.16e-4**(LOSO), 허용치 1e-3 이내 |

네 번째가 핵심이다. 표에 쓰는 숫자와 저장된 예측이 같은 것을 가리킨다는 확인이고,
이게 깨지면 결과를 쓰지 않는다는 것이 이 캠페인의 규칙이다.

---

## 3. 결과 A — 고정 split

원본: `data/table1_fixed_split.csv`, 시드별 원자료 `data/fixed_split_per_seed.csv`

| 지표 | 값 (mean ± SD, n=5) |
|---|---|
| **`iou_free`** | **0.8146 ± 0.0007** |
| constant-map baseline | 0.5180 |
| **margin** | **+0.2966** |
| `fatal_rate` | 0.1179 ± 0.0010 |
| `free_miss_rate` | 0.0833 ± 0.0012 |
| `f1@10cm` | 0.6069 ± 0.0051 |
| `f1@20cm` | 0.8355 ± 0.0041 |
| `f1@40cm` | 0.9273 ± 0.0022 |
| `range_mae` | 0.2070 ± 0.0047 m |
| `range_bias` | +0.0463 ± 0.0025 m |
| `range_missed_obstacle_rate` | 0.0385 ± 0.0018 |

**epoch 선택에 둔감하다.** validation 최고 epoch에서 고르면 `iou_free` 0.8126 ± 0.0013으로
고정 epoch 40과 0.0007 차이다(최고 epoch은 런마다 27/15/30/40/36으로 흩어졌는데도 그렇다).
논문에서 "고정 epoch을 썼다"는 선택을 방어할 근거가 된다.

**`range_bias`가 양수(+0.052 m)라는 것**은 장애물까지의 거리를 평균적으로 5 cm 정도
**멀게** 본다는 뜻이고, 이는 자유 공간의 과대추정 — 즉 **안전에 불리한 쪽**이다. 본문에서
언급하는 편이 정직하다.

> **부호 규약 주의.** 구현의 정의는 `dr = r_pred − r_gt`이고 `dr > 0`이 "장애물을 실제보다
> 멀다고 예측 = free 과대추정 = 위험한 쪽"이다. **다른 문헌은 같은 사건을 "장애물 거리의
> 과소추정"이라 부른다.** 논문에 `over`/`under`라는 단어를 그대로 옮기면 부호가 뒤집혀
> 읽힌다. 정의를 한 줄로 명시하고 쓰는 것이 안전하다.

---

## 4. 결과 B — LOSO 7 fold

원본: `data/table2_loso.csv`, fold 요약 `data/loso_fold_summary.csv`,
fold×시드 35행 원자료 `data/loso_per_fold_seed.csv`, macro `data/loso_macro.csv`

| fold | 조명 | 통로폭 | n_val | baseline | `iou_free` | **margin** | `fatal` | `free_miss` |
|---|---|---|---|---|---|---|---|---|
| `raws1` | 햇빛 | 좁음 | 38 | 0.630 | 0.824 ± 0.001 | +0.193 | 0.099 | 0.088 |
| `raws2` | 햇빛 | 좁음 | 41 | 0.670 | **0.872** ± 0.001 | +0.202 | 0.059 | 0.089 |
| `raws3`† | 햇빛 | 넓음 | 36 | 0.385 | 0.820 ± 0.005 | **+0.434** | 0.102 | 0.097 |
| `rawos1` | 가림막 | 좁음 | 39 | 0.685 | 0.848 ± 0.002 | **+0.164** | 0.087 | 0.088 |
| `rawos2` | 가림막 | 좁음 | 40 | 0.607 | 0.840 ± 0.002 | +0.233 | 0.090 | 0.105 |
| `rawos3` | 가림막 | 넓음 | 37 | 0.391 | 0.806 ± 0.004 | +0.415 | 0.127 | 0.084 |
| `rawos4` | 가림막 | 넓음 | 36 | 0.388 | **0.801** ± 0.002 | +0.414 | 0.103 | 0.120 |
| **macro (7)** | | | 267 | 0.537 | **0.830** | **+0.293** | 0.095 | 0.096 |

† `raws3`는 **외삽 fold**다. (햇빛 × 넓은 통로) 조합의 유일한 시퀀스라, 이 fold를 빼면
학습 데이터에 그 조합이 아예 없다.

산포 요약:

| | fold 간 SD | fold 내부 시드 SD 평균 | 비 |
|---|---|---|---|
| `iou_free` | 0.0251 | 0.0025 | **10.1×** |
| `fatal_rate` | 0.0206 | 0.0022 | 9.3× |
| `free_miss_rate` | 0.0129 | 0.0039 | 3.3× |
| baseline | 0.1413 | — | — |

---

## 5. 해석과 인사이트

사실은 위까지다. 아래는 그 사실을 읽는 방식이다.

### 5.1 시드 재현성이 매우 좋다 — 이게 앞으로의 실험을 가능하게 한다

시드만 바꿨을 때 `iou_free`가 0.0018(0.18 %p)밖에 안 흔들린다. fold 안에서도 0.0024로
같은 수준이다. 실용적 함의는 이것이다: **앞으로 할 비교 실험에서 0.005 정도의 차이도
시드 잡음과 구별할 수 있다.** 카메라 모델 교체나 사전학습 효과처럼 작은 차이를 다투는
ablation에서, 시드 5개면 충분하다는 근거가 된다.

### 5.2 fold 선택이 시드 선택보다 10배 중요하다

fold 간 SD가 시드 SD의 10.3배다. **서로 다른 fold의 숫자를 맞대어 조건을 비교하면 안
된다**는 뜻이고, 이게 이후 모든 ablation을 고정 split에서 하기로 한 이유다. 논문 본문에
한 줄로 넣어두면 "왜 ablation은 LOSO로 안 했나"라는 질문을 미리 막는다.

### 5.3 (핵심) raw 점수 순위와 실제 실력 순위가 뒤집힌다

`rawos1`은 `iou_free` 0.848로 7개 중 2위지만 margin은 +0.164로 **꼴찌**다.
`raws3`는 0.822로 공동 4위인데 margin은 +0.437로 **1위**다.

원인은 통로 폭이다. **좁은 통로 4개 fold의 기준선은 평균 0.648, 넓은 통로 3개는 0.388**이다.
통로가 좁으면 자유 공간의 모양이 시퀀스마다 비슷해서 상수 지도 하나로도 잘 맞는다. 넓으면
프레임마다 달라져서 상수 지도가 무너진다. 즉 **raw `iou_free`가 높은 fold는 모델이 잘한
fold가 아니라 문제가 쉬운 fold**다.

`figures/fig1_margin_inversion.pdf`가 이 그림이다. fold를 모델 점수 내림차순으로 세웠는데
막대 길이(=마진)가 그 순서와 전혀 맞지 않는 것이 보인다.

### 5.4 모델은 장면에 둔감하고 상수 지도는 그렇지 않다

같은 그림의 두 번째 읽기다. **기준선은 fold마다 0.385~0.685로 요동치고(SD 0.141) 모델은
0.801~0.872에 모여 있다(SD 0.025).** 5.7배 차이다.

이건 단순한 성능 주장보다 강한 주장을 지지한다 — **모델이 장면 배치를 외운 게 아니라
이미지를 실제로 보고 있다.** 외운 지도는 장면이 바뀌면 무너지는데 모델은 버티기 때문이다.
논문에서 "이미지 의존성"을 주장할 때 이 대비를 쓰는 게 별도 ablation보다 설득력 있다.

### 5.5 외삽 fold가 버텼다

`raws3`는 학습에 없던 (햇빛 × 넓은 통로) 조건인데 `iou_free` 0.822로 macro 평균 0.830보다
0.008 낮은 데 그쳤고, 마진은 가장 컸다. 다만 **fold 하나짜리 값**이므로 과하게 밀어붙이지
않는 게 좋다. "조명과 통로 폭의 미지 조합에서도 성능이 유지됐다" 정도가 안전한 표현이다.

### 5.6 두 프로토콜의 마진이 가깝다 — 그래도 합치지 않는다

고정 split +0.2966, LOSO macro +0.2935. 셋째 자리에서 갈린다(차이 0.0031).

**[2026-09-23]** 이전 캠페인(`arc_huber`)에서는 이 둘이 +0.2938 대 +0.2937로 소수 넷째
자리까지 일치했고, 그때도 "우연의 일치로 보는 게 맞다"고 적었다. 목적함수만 바꿔 다시
학습하자 그 일치가 사라졌다 — **그 해석이 옳았다는 사후 확인이다.** 본문에서 이 일치를
근거로 쓰지 않기로 한 판단은 그대로 유지한다.

학습 데이터 크기(192 대 229~231 프레임)와 묻는 질문이 다르다. 본문에서 "두
프로토콜이 일관된 마진을 보였다" 정도로 언급할 수는 있으나, 평균을 내거나 우열을
비교하면 안 된다.

### 5.7 안전 지표는 한 쌍으로만 읽는다

`fatal_rate`(장애물을 통과 가능으로 오인)와 `free_miss_rate`(통로를 막힘으로 오인)는
**문턱값을 옮기면 서로 반대로 움직인다.** 둘 중 하나만 좋아진 것을 "안전 개선"이라
부르면 안 된다 — 이 프로젝트에는 과거 같은 주장을 동작점 이동으로 철회한 전례가 있다.
항상 같은 동작점(τ=0.5)에서 한 쌍으로 보고한다.

---

## 6. 논문에 어떻게 넣을까

### 6.1 Table 1 — 대표 성능 (고정 split)

데이터: `data/table1_fixed_split.csv`

**baseline을 별도 행으로 올리는 것**이 핵심이다. 열에 숨기면 독자가 그냥 지나친다.

| Method | IoU<sub>free</sub> ↑ | Δ vs. const. map | Fatal ↓ | Free-miss ↓ | F1@20cm ↑ | Range MAE ↓ |
|---|---|---|---|---|---|---|
| Constant-map baseline | 0.518 | — | — | — | — | — |
| **Ours** | **0.815** (0.001) | **+0.297** | 0.118 (0.001) | 0.083 (0.001) | 0.836 (0.004) | 0.207 m (0.005) |

각주: *Fixed epoch 40; mean over 5 training seeds, SD in parentheses. Split: 192 train /
75 validation frames.*

`f1@10cm`와 `f1@40cm`, `range_bias`, `missed_obstacle_rate`는 본문 표에 다 넣으면
과밀하다. **부록 표로 빼고** 본문에는 위 6열만 두는 것을 권한다.

### 6.2 Table 2 — LOSO 일반화 (겸 층별 분석)

데이터: `data/table2_loso.csv`

이 표 하나가 일반화 결과이면서 동시에 조명·통로폭 층별 분석이다. **별도의 ablation 절이
필요 없다.** 열 순서를 `baseline → IoU → Δ`로 두면 독자의 눈이 자연스럽게 마진으로 간다.

각주에 반드시 넣을 것:

- *† Extrapolation fold: the only sequence with (sunlit × wide corridor), so that
  combination is absent from training.*
- *macro = mean of the seven fold means, not a pooled mean over 35 runs.*
- *SD across folds is descriptive. The folds share 6/7 of their training data, so
  SD/√7 is **not** a standard error.*

마지막 각주가 중요하다. 리뷰어가 가장 먼저 시도할 계산이 `SD/√7`이다.

### 6.3 Figure 1 — 마진의 역전

파일: `figures/fig1_margin_inversion.{pdf,png,svg}` · 생성 스크립트: `figures/make_margin_inversion.py`
· 입력 데이터: `data/figure_margin_inversion.csv`

단일 패널 dumbbell이다. fold를 모델 `iou_free` 내림차순으로 세우고, 각 fold마다 기준선 점과
모델 점을 선으로 잇는다. 선 길이가 마진이다. 뒤에 깔린 옅은 띠는 각 계열의 fold 간 범위다.

이 그림 하나가 §5.3과 §5.4를 동시에 말한다 — 막대 길이가 세로 순서와 어긋나는 것(역전),
주황 띠가 파랑 띠보다 훨씬 넓은 것(모델이 장면에 둔감함).

권장 캡션:

> **Figure 1.** Per-fold LOSO performance against the constant-map baseline. Folds are
> ordered by model IoU<sub>free</sub> (top = highest); bar length is the margin over the
> baseline. The two orderings disagree: `rawos1` ranks second by raw IoU but last by
> margin, while the extrapolation fold `raws3` (†) ranks fourth by raw IoU and first by
> margin. Shaded bands give each series' range across folds — the baseline varies 5.7×
> more than the model (SD 0.141 vs. 0.025), indicating the model is not exploiting a
> memorized layout.

**그림을 직접 고치지 말 것.** SVG/PDF를 손으로 편집하면 스크립트와 갈린다. 색·크기·정렬을
바꾸려면 `make_margin_inversion.py`를 고치고 다시 돌린다. matplotlib 외 의존성은 없다.

### 6.4 표현 규칙과 본문 단서

일반 규칙은 [`../common/metrics.md`](../common/metrics.md) §4~§5에 있다. 이 실험에
특히 걸리는 것 넷만 다시 짚는다.

- **Table 2 각주에 `SD/√7`을 쓰지 말라고 명시한다.** 리뷰어가 가장 먼저 시도할 계산이고,
  7개 fold가 학습 데이터를 6/7씩 공유하므로 표준오차가 아니다.
- **`macro`는 fold 평균의 평균**이지 35런을 한데 모은 평균이 아니다. 각주에 적는다.
- **`raws3`에 † 표시**와 "학습에 (햇빛 × 넓은 통로) 조합이 없는 유일한 fold"라는 각주.
- **`f1@10cm`·`f1@40cm`·`range_bias`·`missed_obstacle_rate`는 부록으로** 뺀다. 본문
  Table 1에 9개 지표를 다 넣으면 과밀하다.

### 6.5 LaTeX 뼈대

`booktabs` 기준 Table 1:

```latex
\begin{table}[t]
\centering
\caption{Overall performance on the fixed split. Mean over five training seeds
         (SD in parentheses), fixed epoch 40, 192 train / 75 validation frames.}
\label{tab:overall-fixed}
\small
\begin{tabular}{lcccccc}
\toprule
Method & IoU$_\text{free}$ $\uparrow$ & $\Delta$ vs.\ const.\ map
       & Fatal $\downarrow$ & Free-miss $\downarrow$
       & F1@20cm $\uparrow$ & Range MAE $\downarrow$ \\
\midrule
Constant-map baseline & 0.518 & --- & --- & --- & --- & --- \\
\textbf{Ours} & \textbf{0.815} (0.001) & \textbf{+0.297}
              & 0.118 (0.001) & 0.083 (0.001) & 0.836 (0.004) & 0.207\,m (0.005) \\
\bottomrule
\end{tabular}
\end{table}
```

Table 2는 행이 8개라 `\resizebox`보다 `\small` + 열 축약(조명/통로폭을 `Sun`/`Shade`,
`Narrow`/`Wide`로)이 낫다. CSV를 그대로 읽어 생성하려면 `data/table2_loso.csv`의 열 이름이
그대로 헤더가 되게 해 두었다.

---

## 7. 이 폴더에 없는 것 (서버에 남아 있는 것)

논문 작성에 필요 없어서 뺐다. 다시 필요하면 서버에서 꺼낸다.

| 있는 곳 | 무엇 | 크기 |
|---|---|---|
| `runs/paper_final/01_overall/*/ckpt/` | 40런 체크포인트 | 약 38 GB |
| `runs/paper_final/01_overall/*/analysis/predictions/` | **threshold 적용 전 확률맵 80개**(`p(free)`, GT, valid mask, 거리장) | 수 GB |
| `runs/paper_final/01_overall/*/logs/` | TensorBoard 이벤트 원본 | — |

확률맵을 남겨둔 이유가 있다. **새 문턱값이나 새 영역 지표를 계산할 때 재학습이 필요 없다.**
예를 들어 "같은 `free_miss`에서 `fatal`을 비교" 같은 동작점 곡선은 이 확률맵만으로 그린다.
그 작업은 서버에서 해야 하고, 결과 CSV만 이 폴더에 추가하면 된다.

재현 명령은 `provenance/environment_and_config.json`과 캠페인 원장
(`docs/paper_final_experiments.md`)에 있다.
