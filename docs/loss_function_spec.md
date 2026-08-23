# loss function 명세 — 지금 이 순간의 정확한 형태

**이 문서의 역할은 하나다: "현재 loss가 수식으로 정확히 무엇인가"에 답한다.**

- **왜 별도 문서인가.** [`soft_boundary_loss_design.md`](soft_boundary_loss_design.md)는 설계
  근거와 스윕 결과의 정본이고 그래서 길다(14개 절, 철회 기록 포함). 그 안에서 "그래서 지금
  쓰는 식이 뭐냐"를 뽑아내려면 §2–§5, §13, 그리고 §10.3·§11의 철회 표까지 대조해야 한다.
  여기는 **대조 없이 읽히는 명세**다.
- **여기에 실험 결과·설계 대안·기각 사유를 쓰지 않는다.** 그건 설계 문서가 정본이다. 이
  문서는 "무엇인가"만 말하고 "왜 이것인가"는 각 절 끝에서 설계 문서 절 번호로 넘긴다.
- **코드가 최종 권위다.** 식과 코드가 어긋나면 코드가 맞고 이 문서가 틀린 것이다. 각 절에
  대응 함수를 적어 두었다.

---

## 1. 전체 식

$$
\boxed{\;L \;=\; \tfrac12 L_F \;+\; \tfrac12 L_N \;+\; \lambda_B\,L_B \;+\; \lambda_R\,L_{\text{range}}\;}
$$

| 항 | 무엇에 걸리나 | 성격 |
|---|---|---|
| $L_F$ | 확실히 drivable한 셀 $\Omega_F$ | hard BCE, 목표 1 |
| $L_N$ | 확실히 non-drivable한 셀 $\Omega_N$ | hard BCE, 목표 0 |
| $L_B$ | GT 경계 ±$\delta$ 안의 셀 $\Omega_B$ | **soft** BCE, 목표는 $d$의 함수 |
| $L_{\text{range}}$ | 방위각 광선당 자유거리 스칼라 | dead zone + Huber 회귀 |

앞의 세 항은 **셀 단위**이고 마지막 하나만 **광선 단위**다. 그 차이가 $L_{\text{range}}$를
넣은 이유다(§8).

확정 계수: $\lambda_B = 0.5$, $\lambda_R = 0.3$.

구현: `projects/common/soft_boundary.py::compute_soft_boundary_loss` 하나가 네 항을 모두
합친다. $L_{\text{range}}$는 `projects/common/range_loss.py::compute_range_loss`가 계산해
`range_term` 인자로 넘겨진다 — **총합을 밖에서 더하지 않는다.** 두 곳에서 더하면 `share_*`
(각 항의 기여 몫, 정의상 합이 1)이 조용히 무의미해진다.

---

## 2. 기호

| 기호 | 뜻 | 모양 / 단위 |
|---|---|---|
| $z$ | 모델 logits | $(B, 2, H, W)$, 채널 1이 `free` |
| $p_i$ | $\mathrm{softmax}(z)_{1,i}$ = 셀 $i$의 예측 $P(\text{free})$ | $[0,1]$ |
| $d_i$ | GT free 경계까지의 **부호 있는 수직 거리** | m, `+`가 free 쪽 |
| $v_i$ | `valid` 마스크 (라벨이 존재하는 셀) | $\{0,1\}$ |
| $b_i$ | `permanent_blind` — ego 아래 정적 사각 원반 | $\{0,1\}$, 프레임 불변 |
| $\delta$ | 불확실 대역 반폭 | **0.15 m** (3셀) |
| $\sigma$ | 경계 위치 오차의 표준편차 | $\alpha\delta$ = **0.075 m** |
| $\alpha$ | 모양 매개변수 $\sigma/\delta$ | **0.5** |
| $\theta_j$ | 방위각 광선, $j = 1..720$ | $\theta = 0$이 전방 |
| $\Delta r$ | 광선 위 표본 간격 | **0.025 m** (0.5셀), 200 step |
| $\delta_R^{\pm}$ | 자유거리 허용 반폭 (과대/과소) | **0.20 / 0.20 m** |
| $\beta$ | Huber 전환점 | **0.10 m** |

격자는 120×120, 셀 0.05 m, ego 기준 전방 4.0 m / 후방 2.0 m / 좌우 ±3.0 m
(`ROBOT_GRID_SPEC`). 최대 반경 4.975 m.

정식화는 **binary**다 — `free` / `not_free` 두 채널만 예측한다. `occupied`는 예측 `free`의
ego 기준 경계에서 유도해 보고할 뿐 loss에 등장하지 않는다.

---

## 3. 부호 있는 거리장 $d$

구현: `soft_boundary.py::signed_distance_field`. **라벨에서 전처리 단계에 계산되어 배치에
`d_bev_g`로 실려 온다** — 모델 출력과 무관한 상수다.

$$
d_i \;=\; \begin{cases}
+\,\mathrm{dist}(x_i, \Gamma) & x_i \in \mathcal{F}^{\text{ext}} \\[2pt]
-\,\mathrm{dist}(x_i, \Gamma) & x_i \notin \mathcal{F}^{\text{ext}}
\end{cases}
\qquad
\mathrm{dist}(x, \Gamma) = \min_{q \in \Gamma} \lVert x - q \rVert_2
$$

- $\mathcal{F}^{\text{ext}} = \mathcal{F} \cup \{b_i = 1\}$ — GT free 영역에 `permanent_blind`
  원반을 **합쳐서** 거리를 잰다. 원반은 프레임마다 같은 정적 경계이므로, 합치지 않으면 그
  테두리가 경계 $\Gamma$로 잡혀 대역이 오염된다. 합치는 것은 **거리 계산에서만**이고 그
  셀들의 감독은 §4가 따로 정한다.
- $\Gamma = \partial\mathcal{F}^{\text{ext}}$ — 그 영역의 경계.
- 실제 구현은 EDT 두 번이다: `distance_transform_edt(F, sampling=cell_m)`와
  `distance_transform_edt(~F, sampling=cell_m)`을 부호로 합친다.

**왜 수직 거리이고 반경 거리가 아닌가.** 두 정의를 실측 비교했을 때 Jaccard가 0.305였고,
반경 정의는 경계에서 15 cm 안인 셀의 11 %를 놓쳤다. 경계가 광선을 스치듯 지나가는 통로
측벽에서 반경 거리는 크지만 수직 거리는 작기 때문이다. soft label의 근거가 "경계면이
공간상 몇 cm 밀릴 수 있다"이므로 재야 할 것은 수직 거리다. (설계 문서 §2.2)

**부수 효과로 이게 `f1@τ`와 같은 거리 연산자다.** `occupied_metrics._distance_field_m`도
같은 `distance_transform_edt(sampling=cell_m)`를 쓴다. 다른 것은 두 가지뿐 — `f1@τ`는 크기만
쓰고 부호를 안 쓰며, 대상 집합이 free 경계가 아니라 `occupied` 표면이다. **올리려는 지표와
loss가 같은 거리를 재는 것**이 이 정의를 고른 첫 번째 이유다.

**반 셀 오프셋.** 거리는 셀 *중심* 사이로 재므로 경계에 붙은 셀도 $|d| = 0.05$이고 0이 되지
않는다. 실제 경계면은 두 중심 사이(약 $0.025$ m)에 있어 크기가 반 셀만큼 크게 나오는데,
**양쪽에 대칭으로 걸리므로 target의 $y = 0.5$ 교차점은 경계면에 그대로 남는다.**

---

## 4. 영역 분할

구현: `soft_boundary.py::region_masks`. 세 집합은 서로 겹치지 않고 합쳐서 `valid`를 정확히
덮는다.

$$
\begin{aligned}
P &= \{i : v_i = 1 \wedge b_i = 0\} && \text{(분할 대상)} \\
\Omega_F &= \{i \in P : d_i > \delta\} \\
\Omega_N &= \{i \in P : d_i < -\delta\} \;\cup\; \{i : v_i = 1 \wedge b_i = 1\} \\
\Omega_B &= P \setminus (\Omega_F \cup \Omega_N) = \{i \in P : |d_i| \le \delta\}
\end{aligned}
$$

**`permanent_blind`가 특별 취급되는 이유는 두 요구가 동시에 성립해야 하기 때문이다.**

- *분할에서 빼야 한다*: 원반은 ego 근처라 §3의 합집합 때문에 $d$가 큰 양수로 나온다. 그대로
  두면 라벨이 `unknown`(= not_free)인 셀이 $\Omega_F$에 들어가 **free로 학습된다.**
- *그래도 감독해야 한다*: 지표는 `valid` 전체에서 채점되고 원반은 `valid`다. 감독을 빼면
  모델이 원반에 free를 찍어도 loss가 벌하지 않는데 `fatal_rate`는 나빠진다.

그래서 **분할에서 제외한 뒤 hard 목표 0으로 $\Omega_N$에 넣는다.**

$\Omega_F$/$\Omega_N$의 목표를 라벨이 아니라 **상수 1/0**으로 두는 것도 선택이다. 라벨이
$d$의 부호와 어긋나는 셀이 실측 0.00~0.01 %이므로 라벨을 다시 읽지 않는다. (설계 문서 §4.1)

실측 셀 비율(val, `sb_r30` ep40): $|\Omega_F|$ 15.3 %, $|\Omega_N|$ 73.1 %, $|\Omega_B|$ 11.6 %.

---

## 5. hard 항 — $L_F$, $L_N$

$$
L_F = \frac{1}{|\Omega_F|}\sum_{i \in \Omega_F} \bigl(-\log p_i\bigr),
\qquad
L_N = \frac{1}{|\Omega_N|}\sum_{i \in \Omega_N} \bigl(-\log(1 - p_i)\bigr)
$$

구현상 $\log p$는 `F.log_softmax(z, dim=1)`에서 직접 가져온다(수치 안정성). 집합이 비면 0을
돌려준다 — `nan`이 총합을 오염시키면 안 된다.

**per-set 평균이 역빈도 클래스 가중치를 대체한다.** 두 항을 각자 자기 집합 크기로 나눈 뒤
½씩 더하면, 셀 하나의 유효 가중치는 그 셀이 속한 집합 크기의 역수에 비례한다:

$$
\tfrac12 L_F + \tfrac12 L_N
= \sum_i w_i \cdot \text{BCE}_i,
\qquad
w_i = \frac{1}{2|\Omega_{c(i)}|}
$$

즉 **역빈도 가중치와 같은 것을 배치 자신의 영역 개수로 계산한 것**이다. 데이터셋 수준
상수(`max_class_weight=20`)에 의존하지 않고, 배치 구성이 바뀌어도 두 클래스가 정확히 절반씩
기여한다. 그래서 `run_batch_soft_boundary`는 `class_weights`를 **받지 않는다** — 둘을 같이
걸면 클래스 보정이 두 번 들어간다. (설계 문서 §3.1)

> 대조군 `weighted_ce`는 이 자리에서 데이터셋 전역 역빈도(로봇 train split에서
> `not_free` 1.00 / `free` 3.91)를 쓰는 단일 CE 항 하나다. 상한 20은 실제로 걸리지 않는다.

---

## 6. soft target $y$

구현: `soft_boundary.py::soft_target`.

### 6.1 유도

경계의 참 위치가 라벨에서 $\varepsilon$만큼 밀려 있다고 본다: $R_{\text{true}} = R_{gt} + \varepsilon$.
셀이 실제로 drivable할 확률은

$$
y_i \;=\; P(\text{셀 } i \text{가 drivable}) \;=\; P(\varepsilon > -d_i)
$$

**사전분포를 무엇으로 두느냐가 target의 형태를 정한다.** 두 형태 모두 정확한 사후확률이다.

| 사전분포 | target | 손잡이 |
|---|---|---|
| $\varepsilon \sim \mathrm{Unif}(-\delta, \delta)$ | $y = \frac12\!\left(1 + \dfrac{d}{\delta}\right)$ | $\delta$ 하나 |
| $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, 단 $\lvert\varepsilon\rvert \le \delta$ | 아래 식 | $\delta$, $\sigma$ |

**현재 쓰는 것은 두 번째(gaussian)다:**

$$
\boxed{\;
y_i \;=\; \frac{\Phi(d_i/\sigma) - \Phi(-\delta/\sigma)}{\Phi(\delta/\sigma) - \Phi(-\delta/\sigma)}
\;}
\qquad \Phi(x) = \int_{-\infty}^{x}\!\frac{e^{-t^2/2}}{\sqrt{2\pi}}\,dt
$$

이것은 **절단 정규 사전분포에서의 정확한 사후확률**이다:

$$
P(\varepsilon > -d \mid |\varepsilon| \le \delta)
= \frac{\Phi(\delta/\sigma) - \Phi(-d/\sigma)}{\Phi(\delta/\sigma) - \Phi(-\delta/\sigma)}
= \frac{\Phi(d/\sigma) - \Phi(-\delta/\sigma)}{\Phi(\delta/\sigma) - \Phi(-\delta/\sigma)}
$$

(둘째 등호는 $\Phi(-x) = 1 - \Phi(x)$.) **정규화가 필요한 이유**: 생 $\Phi(d/\sigma)$는 유한
구간에서 0/1에 도달하지 않아 $d = \pm\delta$에서 목표가 튄다($\sigma = \delta$면 대역 끝에서
0.841인데 바로 옆 셀은 hard 1.0이다). 정규화하면 그 불연속이 사라지면서 물리적 해석도
잃지 않는다.

구현상 $\Omega_B$ 밖에서도 계산되지만 값에 의미가 없다 — 호출부가 마스크로 걸러낸다. 출력은
`clamp(0,1)`.

### 6.2 왜 $\sigma$를 미터로 주지 않고 $\alpha = \sigma/\delta$로 주는가

정규화된 좌표 $u = d/\delta$를 넣으면

$$
y = \frac{\Phi(u/\alpha) - \Phi(-1/\alpha)}{\Phi(1/\alpha) - \Phi(-1/\alpha)}
$$

**$\delta$가 식에서 사라진다.** 즉 **모양은 $\alpha$만 정하고 $\delta$는 대역 폭만 정한다 —
두 손잡이가 직교한다.** $\sigma$를 절대값으로 주면 $\delta$를 바꿀 때 모양이 조용히 같이
바뀌고, "폭을 넓힌 효과"와 "모양을 둔하게 한 효과"가 섞인다.

`resolve_sigma`는 $\sigma$와 $\alpha$를 **동시에 받으면 실패한다.** 조용히 하나를 이기게 두면
로그의 config와 실제로 쓰인 모양이 갈리고, 그러면 스윕 표가 통째로 무의미해진다.

### 6.3 두 극한 — 선형은 별 형태가 아니다

$$
\alpha \to 0 \;\Rightarrow\; y \to \mathbb{1}[d > 0] \quad (\text{hard step}),
\qquad
\alpha \to \infty \;\Rightarrow\; y \to \tfrac12\!\left(1 + \tfrac{d}{\delta}\right) \quad (\text{정확히 선형})
$$

둘째 극한: $\Phi(x) \approx \frac12 + \varphi(0)x$로 전개하면 분자 $\to \varphi(0)(d+\delta)/\sigma$,
분모 $\to \varphi(0)\cdot 2\delta/\sigma$이므로 $y \to (d+\delta)/2\delta$. **선형 target은
별도 형태가 아니라 이 계열의 한쪽 끝**이다.

**아래로는 격자 양자화가 한계를 만든다.** 5 cm 격자에서 셀 중심의 최소 $|d|$가 1셀이므로
$\alpha \lesssim 0.2$면 셀이 실제로 놓이는 자리에서 target이 이미 1.0에 붙어 soft target이
이름만 남는다($\delta = 0.15$, $\alpha = 0.15$에서 최근접 셀 target이 0.987이다).

확정값 $\alpha = 0.5$ → $\sigma = 0.075$ m. 이 값에서 셀 중심들의 target은 대략
$d = \pm0.05$에서 0.75/0.25, $d = \pm0.10$에서 0.93/0.07이다.

---

## 7. soft 경계 항 $L_B$와 엔트로피 하한

$$
L_B = \frac{1}{|\Omega_B|}\sum_{i \in \Omega_B}
\Bigl(-\bigl[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\bigr]\Bigr)
$$

logits에 대한 gradient는 정확히

$$
\frac{\partial L_B}{\partial z_{1,i}} = \frac{p_i - y_i}{|\Omega_B|},
\qquad
\frac{\partial L_B}{\partial z_{0,i}} = -\frac{p_i - y_i}{|\Omega_B|}
$$

즉 **$p_i = y_i$에서 gradient가 정확히 0이다.** hard CE는 $p \to 1$까지 계속 밀지만 이 항은
"$y_i$만큼만 확신하라"고 요구하고 거기서 멈춘다. 그게 이 loss의 핵심 동작이다.

### 7.1 엔트로피 하한 — 이걸 모르면 곡선을 오독한다

$y_i$가 모델과 무관한 상수이므로 $L_B$의 최소값은 0이 아니다:

$$
\min_p L_B = \bar H = \frac{1}{|\Omega_B|}\sum_{i \in \Omega_B} H(y_i),
\qquad
H(y) = -\bigl[y\log y + (1-y)\log(1-y)\bigr]
$$

**그래서 경계 항의 진짜 진행도는 $L_B$가 아니라 KL이다:**

$$
\boxed{\;\mathrm{KL} \;=\; L_B - \bar H\;}
$$

이것이 $\mathrm{KL}(y \,\Vert\, p)$를 $\Omega_B$에서 평균한 값이고, 0이 도달 가능한 하한이다.
학습 루프는 `kl_boundary`와 `entropy_boundary`를 둘 다 로그한다 — **하한에 붙어 평평해지는
것을 "수렴 실패"로 오독하지 않기 위해서다.**

실측(val, `sb_r30` ep40, $\alpha = 0.5$): $\bar H = 0.299$ nats, $L_B = 0.995$, KL $= 0.696$.
$\lambda_B = 0.5$이므로 총 loss에 **상수 0.149가 얹혀 있다.**

> 선형 target이면 $y$가 $[0,1]$에 거의 균일해 $\bar H \approx 0.5$ nats다. gaussian
> $\alpha = 0.5$는 target을 0/1 쪽으로 밀기 때문에 하한이 그보다 낮다.

구현 주의: `target_entropy`는 `torch.xlogy`를 쓴다. `clamp` 후 `y·log y`로 쓰면 float32에서
$1 - 10^{-12}$가 정확히 1.0으로 반올림되어 $(1-y)\log(1-y)$가 $0 \cdot (-\infty)$ = `nan`이
된다. 그 `nan`은 $\Omega_B$ 밖 셀에서 생기는데 **마스크로 곱해도 사라지지 않고** 총합을
오염시킨다. `xlogy(0,0) = 0`이라 그 경로가 아예 없어진다.

### 7.2 $\lambda_B = 0.5$는 작은 값이 아니다

$\Omega_B$가 셀의 11.6 %뿐이므로 **셀당 가중치**는

$$
\frac{\lambda_B / |\Omega_B|}{\tfrac12 / |\Omega_N|}
= \frac{0.5 / 0.116}{0.5 / 0.731} \approx 6.3
$$

$\Omega_N$ 셀의 6.3배다. 그래서 로그는 항의 기여 몫 `share_*`(정의상 합 1)와 셀 비율
`frac_*`을 **같이** 낸다. 하나만 보면 이 사실이 안 보인다.

실측 `share_boundary` = 0.844인데, 그중 상당 부분이 §7.1의 상수 하한이다. 상수는 gradient가
0이므로 **축소 가능한 부분만으로 다시 센 몫**도 함께 낸다(`share_boundary_kl` = 0.790).
이쪽이 학습을 실제로 지배하는 비율이다.

---

## 8. 보조항 $L_{\text{range}}$

구현: `range_loss.py::compute_range_loss`.

### 8.1 왜 필요한가

§5–§7의 세 항은 전부 **셀마다 독립인 BCE**다. 광선 방향으로 셀을 묶는 항이 하나도 없으므로
**"이 방향으로 free가 몇 m까지 이어진다고 예측했는가"를 어떤 항도 묻지 않는다.** 결과로
per-cell BCE가 거의 벌하지 않는 오차 모드가 남는다 — 벽 뒤에 free 섬 하나를 찍으면 셀 몇
개분 BCE만 물지만 그 방향의 자유거리는 통째로 망가진다.

### 8.2 광선 표본화

`polar.build_ray_index`가 만든 **고정 인덱스**로 격자를 광선으로 다시 읽는다. $n_\theta = 720$,
step 0.5셀 → $\Delta r = 0.025$ m, 200 step (최대 5.0 m 커버). `RayGather`가 그 인덱스를
torch로 한 번만 옮겨 두고 매 배치 gather만 한다.

$\mathrm{in}_{jk} \in \{0,1\}$은 표본 $k$가 격자 안인지, $g$는 gather 연산자다.

### 8.3 목표와 예측 — 둘 다 `arc`다

$$
a_{gt}(\theta_j) = \Delta r \sum_{k} \mathbb{1}\bigl[\text{free}^{gt}_{jk}\bigr]\cdot \mathrm{in}_{jk},
\qquad
\hat a(\theta_j) = \Delta r \sum_{k} p_{jk}\cdot v_{jk}\cdot \mathrm{in}_{jk}
$$

**목표는 `polar.first_free_range`의 $R_{gt}$가 아니다.** 두 양은 같지 않다 — $R_{gt}$는 ego
원점부터의 반지름인데 그 함수의 계약은 ego 아래 `permanent_blind` 원반을 **건너뛰고 잰다**는
것이므로, 실측에서 $R_{gt} - a_{gt}$가 평균 **0.758 m**(val 40프레임)다. 제안된
$\delta_R = 0.20$ m의 3.8배인 계통 편차이므로 그대로 쓰면

1. dead zone이 무력화되고,
2. loss가 그 상수를 줄이려고 **원반 안을 free로 예측하라고 요구한다** — 원반은 hard
   $\Omega_N$(목표 0)으로 감독되는 자리라 두 항이 정면으로 싸운다.

**예측과 목표에 같은 연산자를 쓰면 그 편차가 정의상 사라진다.** 부수적으로 720광선·0.5셀
표본화와 라벨 raycast의 이산화 불일치(실측 0.035 m)도 같이 상쇄된다. $a$와 $R_{gt}$는 거의
정적인 per-$\theta$ 상수만큼 차이 나므로 **$\delta_R$의 물리적 뜻("이 방향 자유거리를 몇
cm까지 용서하나")은 그대로 보존된다.**

합은 **광선 전체**에 걸린다 — 첫 장애물 앞까지가 아니다. GT free는 ego 원점 raycast의
결과라 광선 위에서 연속 구간 하나이므로($\text{free} = \neg\text{occ} \wedge \text{vis}$) 두
정의가 GT에서는 같지만, **예측에서는 다르고 그 차이가 이 항이 잡으려는 오차 모드다.**

$v_{jk}$로 `valid = 0` 셀을 예측 쪽에서 뺀다. GT free는 이미 0이지만(`decompose`가 `valid`를
요구한다) 예측 확률은 그 셀에서 감독되지 않아 아무 값이나 들어 있다.

### 8.4 어느 광선을 쓰나 — `RAY_OK`만

$$
\mathcal{R}_{OK} = \Bigl\{\, j \;:\; \underbrace{\exists k\,\text{free}^{gt}_{jk}}_{\text{free가 있다}}
\;\wedge\;
\underbrace{\exists k \ge k_1(j)\,\bigl(\mathrm{in}_{jk} \wedge \neg\text{free}^{gt}_{jk}\bigr)}_{\text{그 뒤 격자 안에서 막힌다}} \,\Bigr\}
$$

($k_1(j)$는 첫 free 표본의 인덱스.) `ray_is_ok`가 이것을 torch로 계산하며,
`polar.first_free_range`의 `RAY_OK` 판정과 **광선 하나까지 일치한다**(테스트로 고정).

실측 비율(val 40프레임): `RAY_OK` 35 % / `RAY_CENSORED` 12 % / `RAY_NO_FREE` 53 %.
**나머지 두 상태를 빼는 이유가 각각 다르다.**

- `RAY_NO_FREE`(53 %) — 목표가 상수 0이고 그 셀들은 이미 hard $\Omega_N$으로 잘 학습된다.
  넣으면 광선 평균의 절반이 상수 0인 항이 되어 $L_{\text{range}}$ 값이 읽히지 않고
  $\lambda_R$의 실효 크기가 2배 희석된다.
- `RAY_CENSORED`(12 %) — 격자 끝까지 free라 **하한만 아는** 광선이다. 양방향 loss를 걸면
  모르는 값을 목표로 삼는 것이 된다.

M3(`free_space_metrics.range_error`)가 "GT와 예측이 둘 다 `RAY_OK`인 광선만 회귀 통계에
넣는다"고 정한 규약과 같은 판단이다.

### 8.5 dead zone과 Huber

$$
e_j = \hat a(\theta_j) - a_{gt}(\theta_j)
\qquad\text{(양수 = 자유공간 과대예측 = \texttt{fatal} 방향)}
$$

$$
\boxed{\;
e^{\text{eff}}_j = \max\bigl(0,\; e_j - \delta_R^{+}\bigr) + \max\bigl(0,\; -e_j - \delta_R^{-}\bigr)
\;}
$$

$e_j$의 부호가 하나뿐이므로 두 항 중 하나만 0이 아니다. 그리고

$$
L_{\text{range}} = \frac{1}{|\mathcal{R}_{OK}|}\sum_{j \in \mathcal{R}_{OK}} \rho_\beta\!\bigl(e^{\text{eff}}_j\bigr),
\qquad
\rho_\beta(x) = \begin{cases} \dfrac{x^2}{2\beta} & x \le \beta \\[6pt] x - \dfrac{\beta}{2} & x > \beta \end{cases}
$$

(`F.smooth_l1_loss(e, 0, beta=β)`가 정확히 이 함수다.)

**dead zone이 이 항의 존재 이유다.** $|e_j|$가 허용 반폭 안이면

$$
e^{\text{eff}}_j = 0 \;\Rightarrow\; \rho_\beta = 0 \;\Rightarrow\;
\frac{\partial L_{\text{range}}}{\partial p_{jk}} = 0 \quad \text{(정확히 0)}
$$

즉 그 자유도에 **모델이 용량을 쓸 수 없다.** 단순 MAE로 걸면 main loss가 §6–§7에서 완화한
경계 불확실성을 보조항이 다시 강요한다 — `soft_boundary`가 경계 셀의 *확신*만 낮추고
*위치*는 라벨에 못박아 둔 것이 train/val KL 격차의 원인이므로, 거리에도 같은 못박기를 넣으면
병을 키운다.

**Huber를 미터 단위로 두는 이유**: 정규화된 스케일($R_{\max}$로 나누기)에서 $\beta$를 주면
실효 오차가 항상 $\beta$보다 훨씬 작아져 Huber가 순수 L2로 퇴화하고 "outlier에 덜
끌려간다"는 도입 이유가 사라진다. $\beta = 0.10$ m는 dead zone 통과 후의 전형적 잔차 규모다.

gradient는 확률까지 이렇게 흐른다:

$$
\frac{\partial L_{\text{range}}}{\partial p_{jk}}
= \frac{\Delta r \cdot v_{jk}\,\mathrm{in}_{jk}}{|\mathcal{R}_{OK}|}\cdot
\rho'_\beta\!\bigl(e^{\text{eff}}_j\bigr)\cdot s_j,
\qquad s_j = \operatorname{sign}\bigl(e^{\text{eff}}\text{를 만든 쪽}\bigr)
$$

**한 광선의 모든 표본이 같은 스칼라 gradient를 나눠 받는다** — 그게 "셀을 광선으로 묶는" 항이라는 뜻이다.

### 8.6 비대칭 $\delta_R^{+}$ (구현됨, 현재 대칭으로 사용)

`delta_r_over` 인자로 과대예측 쪽 관용만 좁힐 수 있다. `None`이면 $\delta_R^{-}$와 같아
대칭이다.

동기: 대칭 dead zone은 **"벽 안쪽 $\delta_R$까지 free 예측"을 무벌점으로 허용**하는데,
그쪽이 로봇에게 치명 방향이고(`fatal_rate`·`missed_obstacle`) 실측 `range_arc_bias`가 전
런에서 +0.063 ~ +0.086 m로 일관되게 그쪽으로 기울어 있었다. 즉 판정하기로 한 축을 loss가
무료로 허용하고 있었다.

$\delta_R^{-}$(보수 방향)는 넓게 두는 것이 맞다 — 그쪽 오차는 라벨 불확실성과 구별되지 않고
비용도 낮다.

> **현재 확정값은 대칭 $\delta_R^{+} = \delta_R^{-} = 0.20$이다.** $\delta_R^{+} = 0$은
> `fatal_rate`를 개선하지만 `free_miss_rate`를 악화시키는 교환이라 채택을 보류했다 — 로봇
> 운용 판단이므로 사용자 결정 대기다. 실측은 설계 문서 §13.8.

### 8.7 왜 배치의 라벨에서 매번 다시 계산하나

$a_{gt}$를 `__getitem__`에서 미리 계산해 배치에 실을 수도 있었지만 **좌우 반전 증강에서
조용히 틀린다.** $(n_\theta,)$ 배열의 좌우 반전은 공간 축 flip이 아니라
$\theta \to 2\pi - \theta$, 즉 **순열**(`roll(flip(x), 1)`)이다. 여기서는 이미 반전된 라벨
텐서에서 매번 다시 뽑으므로 **$\theta$ 축이 저장되지 않고 그 함정이 아예 생기지 않는다.**
비용은 고정 인덱스 gather 한 번(프레임당 720×200)이라 사실상 0이다.

---

## 9. $\lambda_R$은 왜 0.3인가 — gradient 비 캘리브레이션

$L_{\text{range}}$는 **미터** 단위이고 BCE는 **nats**다. 두 항의 절대값에 공통 스케일이 없으므로
$\lambda_R$을 숫자로 고르면 그 뜻이 정해지지 않는다. 그래서 logits에서의 gradient 비로 맞춘다:

$$
G(\lambda_R) = \frac{\bigl\lVert \partial(\lambda_R L_{\text{range}})/\partial z \bigr\rVert_2}
{\bigl\lVert \partial(\tfrac12 L_F + \tfrac12 L_N + \lambda_B L_B)/\partial z \bigr\rVert_2}
\;\;\overset{!}{\approx}\;\; 0.1
$$

$\lambda_R = 1$에서의 실측 $G$: **1.55**(초기) → **0.424**(ep10) → **0.273**(ep20) →
**0.147**(ep40). 학습이 진행되며 BCE gradient는 남는데 dead zone 안에 들어가는 광선이 늘어
$L_{\text{range}}$ gradient가 줄기 때문이다. 학습 중반 기준으로 $\lambda_R = 0.3$이
"gradient의 10 % 남짓"에 해당한다.

측정 도구: `python tools/measure_range_gradient.py --lambda_r=1.0 --checkpoint=<...>`.

실측 확인(train, `sb_r30` ep40): `share_range` = 0.0033. **loss 값의 몫으로는 0.3 %지만
gradient 비로는 10 % 대**다 — 두 숫자를 혼동하면 이 항이 아무 일도 안 하는 것처럼 보인다.

---

## 10. 확정 하이퍼파라미터

| 이름 | 값 | 셸 변수 | 근거 |
|---|---|---|---|
| loss | `soft_boundary` | `LOSS` | §11 |
| 정식화 | `binary` | `FORMULATION` | 진단 문서 §15 |
| $\delta$ | 0.15 m (3셀) | `DELTA_M` | 진단 문서 §26 (val loss 증가분의 90 %가 경계 ±20 cm) |
| target 형태 | `gaussian` | `SOFT_TARGET` | §6 |
| $\alpha = \sigma/\delta$ | 0.5 → $\sigma = 0.075$ m | `SIGMA_ALPHA` | §6.3 (아래는 격자 양자화 한계) |
| $\lambda_B$ | 0.5 | `LAMBDA_B` | §7.2 |
| $\lambda_R$ | 0.3 | `LAMBDA_R` | §9 |
| $\delta_R^{-}$ | 0.20 m (4셀) | `DELTA_R_M` | `f1@10cm` 허용오차(2셀)의 2배 |
| $\delta_R^{+}$ | 0.20 m (대칭) | `DELTA_R_OVER_M` | **미결** — §8.6 |
| $\beta$ | 0.10 m | `HUBER_BETA_M` | §8.5 |
| $n_\theta$ | 720 | — | `polar` 기본값 |
| epochs | 40 | `NUM_EPOCHS` | |

실행:

```bash
export PATH=/data/home/dhlee/miniconda3/envs/bev-chamdog/bin:$PATH   # torch 2.7.0+cu128 필수
EXP_NAME=my_run RUN_NAME=my_run \
LOSS=soft_boundary SOFT_TARGET=gaussian \
DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
NUM_EPOCHS=40 \
bash configs/train_robot_bev_finetune.sh
```

`LAMBDA_R=0`이면 **$L_{\text{range}}$가 계산조차 되지 않는다** — 대조군 런이 새 코드 경로를
타지 않아야 한다. `lambda_r > 0`인데 `loss != soft_boundary`면 trainer가 거부한다.

---

## 11. 코드 위치

| 무엇 | 어디 |
|---|---|
| 네 항의 합, `share_*`, KL | `projects/common/soft_boundary.py::compute_soft_boundary_loss` |
| $d$ (전처리) | `projects/common/soft_boundary.py::signed_distance_field` |
| 영역 분할 | `projects/common/soft_boundary.py::region_masks` |
| soft target, $\alpha$ 해석 | `projects/common/soft_boundary.py::{soft_target, resolve_sigma}` |
| 엔트로피 하한 | `projects/common/soft_boundary.py::target_entropy` |
| $L_{\text{range}}$ | `projects/common/range_loss.py::compute_range_loss` |
| 광선 gather, `RAY_OK` | `projects/common/range_loss.py::{RayGather, ray_is_ok}` |
| 배치 실행 (항 조립) | `projects/common/binary_metrics.py::run_batch_soft_boundary` |
| 학습 루프, 플래그 검증 | `tools/train_robot_bev.py::main` |
| $\lambda_R$ 캘리브레이션 | `tools/measure_range_gradient.py` |
| 계약 테스트 | `tests/common/test_soft_boundary.py`, `tests/common/test_range_loss.py` |

---

## 12. 테스트가 고정하는 성질

식이 코드와 어긋나는 것을 막는 것은 아래 성질들이다(`tests/common/test_range_loss.py`).

| 성질 | 왜 고정하나 |
|---|---|
| `ray_is_ok`가 `first_free_range`의 `RAY_OK`와 **광선 하나까지 일치** | torch로 다시 구현했으므로 원본과 갈리면 조용히 틀린다 |
| GT를 그대로 예측으로 넣으면 `arc_mae = 0` | §8.3의 0.758 m 계통 편차가 되살아나지 않았다는 증거 |
| dead zone 안에서 **gradient가 정확히 0** | §8.5, 이 항의 존재 이유 |
| $\delta_R^{+} < \delta_R^{-}$면 과대예측만 벌하고 과소예측은 여전히 평평 | §8.6의 부호 규약 |
| 벽 뒤 free 섬이 벌을 받고 `arc_bias`가 양수로 기운다 | §8.1의 오차 모드 |
| `valid = 0` 셀이 양쪽에서 빠진다 | §8.3 |
| `share_*`의 합이 1, `range_term=None`이면 항이 아예 없다 | §1의 "한 자리에서만 더한다" |
| $\rho_\beta$가 $\beta$에서 이차→선형 ($\beta$ 2배 → 이차 영역 loss 절반, 선형 영역 `loss = mae − β/2` 정확히) | §8.5. **`mean(ρ(e)) ≠ ρ(mean(e))`**(Jensen)이라 분포 무관 성질로 고정해야 한다 |

---

## 13. 이 문서가 답하지 않는 것

| 질문 | 정본 |
|---|---|
| 왜 이 설계인가, 무엇을 기각했나, 스윕 결과 | [`soft_boundary_loss_design.md`](soft_boundary_loss_design.md) |
| 지표 하나하나의 정의와 채택 사유 | [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) |
| 이 loss가 CE보다 나은가, 각 항이 얼마나 기여하나 | 설계 문서 §13.7, 그리고 `configs/ablation_loss.sh` → `tools/report_ablation.py` |
| 왜 binary 정식화인가, 경계 대역 loss 분해 | [`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) §15, §26 |
| 전체 서사와 현재 지점 | [`experiment_history.md`](experiment_history.md) §5 |
| 라벨이 무엇을 `vis=0`/`valid=0`으로 두나 | 진단 문서 §23 |
