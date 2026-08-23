응. 이 명세를 기준으로 보면 **우리가 논의했던 설계가 상당히 충실하게 반영돼 있고, 수식 자체에서 바로 보이는 치명적인 오류는 없습니다.** 오히려 CLAUDE와 작업하면서 우리가 처음 논의했던 것보다 몇 군데는 더 정교하게 다듬어진 상태예요.

다만 검토하면서 **“이건 잘했다”와 “여기는 반드시 확인해야 한다”가 꽤 명확하게 갈립니다.** 특히 `λ_B`의 실제 영향력, `RAY_OK`에서 invalid 처리, 그리고 `λ_R=0.3`의 초기 gradient 크기는 꼭 짚고 싶습니다.

---

# 1. 현재 전체 loss는 이렇게 이해하면 정확해

문서의 최종식은

[
\boxed{
L
=

\frac12 L_F
+
\frac12 L_N
+
\lambda_B L_B
+
\lambda_R L_{\text{range}}
}
]

이고 현재

[
\lambda_B=0.5,\qquad
\lambda_R=0.3
]

이야. 

각 항의 의미는:

[
L_F
]

확실히 free/drivable인 영역에 대한 hard BCE,

[
L_N
]

확실히 not-free인 영역에 대한 hard BCE,

[
L_B
]

boundary ±(\delta) 안의 영역에 대한 soft-target BCE,

[
L_{\text{range}}
]

ray 단위의 free-space geometry auxiliary.

즉 우리가 이야기했던

[
L_{\text{soft-BCE}}
+
\lambda_RL_{\text{range}}
]

를 조금 더 풀어 쓰면

[
L_{\text{soft-BCE}}
===================

\frac12L_F+\frac12L_N+\lambda_BL_B
]

인 셈이야.

그러니까 **전체 formulation은 우리가 논의한 것과 맞아.**

---

# 2. Soft boundary 쪽은 상당히 잘 되어 있어

특히 radial distance가 아니라 우리가 나중에 정리했던 **signed perpendicular/Euclidean distance**를 쓰고 있어.

[
d_i=
\begin{cases}
+\operatorname{dist}(x_i,\Gamma) & \text{free side}\
-\operatorname{dist}(x_i,\Gamma) & \text{not-free side}
\end{cases}
]

이고 현재

[
\delta=0.15m,\qquad
\sigma=0.075m,\qquad
\alpha=\frac{\sigma}{\delta}=0.5
]

로 되어 있어. 

이건 정확히 우리가 마지막에 정리한 방향이야.

또 normalized CDF도

[
\boxed{
y_i=
\frac{
\Phi(d_i/\sigma)-\Phi(-\delta/\sigma)
}{
\Phi(\delta/\sigma)-\Phi(-\delta/\sigma)
}
}
]

로 제대로 들어가 있어.

따라서

[
y(-\delta)=0,\qquad
y(0)=0.5,\qquad
y(+\delta)=1
]

이고 hard region과 값이 연속적으로 이어져. 

더 좋은 점은 이것을 단순 heuristic CDF normalization이라고만 쓰지 않고

[
\epsilon\sim
\mathcal N(0,\sigma^2),
\qquad |\epsilon|\le\delta
]

라는 **truncated Gaussian boundary uncertainty model**로 해석했다는 거야.

수식도 맞아.

---

# 3. Soft BCE의 entropy floor 처리도 맞아

이 부분도 잘 반영됐어.

[
L_B
===

-\frac1{|\Omega_B|}
\sum_i
[
y_i\log p_i+
(1-y_i)\log(1-p_i)
]
]

이고,

[
\min L_B
========

\frac1{|\Omega_B|}
\sum_iH(y_i)
]

라는 걸 명시하고,

[
\boxed{
KL_{\text{boundary}}
====================

L_B-\bar H
}
]

를 따로 logging하고 있어. 

이게 우리가 바로 직전에 이야기했던 내용 그대로야.

특히 학습 loss 자체에서는 entropy를 빼지 않고 **logging에서만 KL을 보는 것**도 맞는 선택이야.

왜냐하면

[
\nabla_\theta H(y)=0
]

이므로 학습 gradient에는 아무 차이가 없으니까.

---

# 4. (L_F,L_N)의 class balancing도 합리적이야

기존의 `free weight = 3.91` 같은 dataset-level weighting을 없애고,

[
L_F
===

\frac1{|\Omega_F|}
\sum_{i\in\Omega_F}-\log p_i
]

[
L_N
===

\frac1{|\Omega_N|}
\sum_{i\in\Omega_N}-\log(1-p_i)
]

로 각각 평균을 낸 뒤

[
\boxed{
L_{\text{region}}
=================

\frac12L_F+\frac12L_N
}
]

로 한 것도 좋아.

즉 batch마다 free/not-free가 얼마나 있는지와 무관하게 두 class가 aggregate level에서 동일한 weight를 받아. 기존 `class_weights`까지 같이 넣지 않는다는 것도 정확한 판단이야. 

---

# 5. Range loss에서는 우리가 처음 얘기했던 formulation보다 오히려 개선된 부분이 있어

우리가 처음에는

[
R_{gt}(\theta)
==============

\text{robot부터 first boundary까지 거리}
]

를 이야기했잖아.

그런데 현재 코드에서는 그걸 직접 쓰지 않고,

[
\boxed{
a_{gt}(\theta_j)
================

\Delta r
\sum_k
\mathbf1[\mathrm{free}^{gt}_{jk}]
}
]

그리고 prediction은

[
\boxed{
\hat a(\theta_j)
================

\Delta r
\sum_k
p_{jk}v_{jk}
}
]

로 되어 있어. 

이 변경은 **좋은 변경**이라고 봐.

문서에 나온 것처럼 `permanent_blind`를 first-range 계산에서 건너뛰기 때문에 기존 (R_{gt})와 arc의 차이가 평균 0.758 m나 있었다면, 서로 다른 operator를 GT와 prediction에 사용하는 건 잘못된 supervision을 만들 수 있었거든. 

지금처럼

> GT와 prediction에 정확히 동일한 연산자를 사용한다

가 더 안전해.

---

# 6. 다만 이제 이것을 엄밀하게는 “range loss”라고만 부르면 약간 혼동될 수 있어

현재

[
a_{gt}=\Delta r\sum free
]

는 정확히 **first boundary range**가 아니라

[
\boxed{\text{ray상의 total free-space arc length}}
]

야.

prediction도

[
\hat a=\Delta r\sum p
]

이니까 더 정확히는

[
\boxed{\text{predicted free probability mass를 거리 단위로 적분한 것}}
]

이라고 보는 게 맞아.

예를 들어

```text
GT:
1 1 1 1 0 0 0 0

Prediction:
1 1 1 1 0 0 1 0
              ↑
          잘못된 free island
```

이면 뒤의 free island 때문에

[
\hat a>a_{gt}
]

가 되고 range auxiliary가 벌을 줘.

이게 문서에서 말하는 의도이기도 해. 

그래서 저는 이름을 그대로 유지하더라도 논문이나 설명에서는

> **ray-wise free-space extent consistency loss**

또는

> **ray-wise free-space arc loss**

라고 설명하는 게 더 정확하다고 봐.

---

# 7. Dead-zone + Huber formulation도 정확해

현재

[
e_j
===

\hat a_j-a_j^{gt}
]

로 두고

[
\boxed{
e_j^{eff}
=========

\max(0,e_j-\delta_R^+)
+
\max(0,-e_j-\delta_R^-)
}
]

를 사용하지.

그리고

[
\boxed{
L_{\text{range}}
================

\frac1{|\mathcal R_{OK}|}
\sum_{j\in\mathcal R_{OK}}
\rho_\beta(e_j^{eff})
}
]

이고

[
\rho_\beta(e)=
\begin{cases}
\dfrac{e^2}{2\beta}, & e\le\beta[4pt]
e-\dfrac{\beta}{2}, & e>\beta
\end{cases}
]

야. 

이건 우리가 논의했던 것과 정확하게 일치해.

현재

[
\delta_R^+=\delta_R^-=0.20m
]

이기 때문에

[
|e_j|\le0.20m
]

이면

[
L_{\text{range},j}=0
]

이고 gradient도 정확히 0.

즉 auxiliary가

> “GT free-space extent를 centimeter 단위로 똑같이 맞혀라”

라고 하지 않고,

> **“20 cm 안이면 충분히 맞았다고 보고 더 이상 간섭하지 않겠다.”**

가 되는 거야.

이게 soft-boundary와 철학적으로 잘 맞아.

---

# 8. Huber를 (R_{\max})로 normalize하지 않은 것도 문제없어

이건 우리가 앞에서는 normalize하는 방안을 이야기했기 때문에 짚고 싶어.

현재 구현은 meter 단위를 그대로 유지하고

[
\beta=0.10m
]

를 쓰고 있어. 

이것도 **틀린 게 아니고 오히려 지금 방식에서는 충분히 합리적**이야.

왜냐하면 loss scale 차이는 뒤에서 (\lambda_R)을 gradient norm 기준으로 calibration하고 있기 때문이야.

따라서

[
\beta=10cm
]

라는 물리적 의미도 그대로 유지되고.

---

# 그런데 여기서부터는 제가 발견한 “확인해야 할 점”이야

## 9. 첫 번째로 가장 신경 쓰이는 건 (\lambda_B=0.5)야

이건 bug는 아닌데 **해석을 조심해야 해.**

겉으로 보면

[
\lambda_B=0.5
]

니까

> boundary를 절반 정도만 약하게 본다

고 생각하기 쉬워.

그런데 현재 loss는 각 영역에서 **각각 mean을 먼저 낸 뒤 합치고 있어.**

현재 cell fraction이

* (F): 15.3%
* (N): 73.1%
* (B): 11.6%

인데, 문서 스스로 계산했듯 boundary cell 하나의 loss weight는 non-free cell 하나보다 약 6.3배야. 

즉

[
\lambda_B=0.5
]

는 이 normalization 아래에서는 절대 작은 값이 아니야.

더 중요한 건 문서에

> `share_boundary_kl = 0.790`, 그래서 이쪽이 학습을 실제로 지배

한다고 적힌 부분이야. 

여기는 표현을 조금 수정하는 게 좋아.

**KL loss magnitude share ≠ gradient share**야.

문서 §9에서도 range에 대해서는 바로 그 차이를 인정하고 있거든.

`share_range=0.0033`이어도 gradient ratio는 10%가 될 수 있다고 되어 있어. 

그렇다면 boundary에도 똑같은 원칙을 적용해야 해.

나는 반드시 한 번

[
\boxed{
G_B
===

\frac{
\left|
\nabla_z(\lambda_BL_B)
\right|_2
}{
\left|
\nabla_z(\frac12L_F+\frac12L_N)
\right|_2
}
}
]

를 측정해볼 것을 권하고 싶어.

이게 예를 들어

[
G_B=2.0
]

이라면 soft boundary가 실제 gradient를 region loss보다 두 배 지배하고 있는 거니까,

> “불확실한 boundary는 main region보다 약하게 supervise한다”

라는 원래 의도와는 다르게 된 거야.

반대로 (G_B\sim0.3) 정도라면 아주 괜찮고.

**soft-boundary가 실제로 성능을 개선했다는 점 때문에 지금 당장 (\lambda_B)를 바꾸라는 이야기는 아니야.** 다만 gradient ratio는 꼭 측정할 가치가 있어.

---

# 10. 두 번째는 `RAY_OK` 판정에서 `valid`가 빠져 있는 부분

여기가 제가 수식상 가장 주의 깊게 검증하고 싶은 부분이야.

현재 문서에서는

[
\mathcal R_{OK}
===============

\left{
j:
\exists k,free_{jk}
;\wedge;
\exists k\ge k_1(j)
\left(
in_{jk}\wedge\neg free_{jk}
\right)
\right}
]

로 되어 있어. 

여기서 문제 가능성이 있는 게:

[
\neg free^{gt}
]

가 반드시

[
\text{known non-free}
]

를 뜻하느냐는 거야.

만약 `valid=0`인 cell도 `free_gt=False`가 된다면,

```text
Robot → free → free → invalid → invalid
```

같은 ray도

> free가 있고, 그 뒤에 `not free`가 있다

고 판정되어 `RAY_OK`로 들어갈 수 있어.

하지만 의미상 이건

```text
free → unknown
```

이므로 **censored ray**에 더 가깝지.

그렇다면 두 번째 조건은 개념적으로

[
\boxed{
in_{jk}
\wedge
v_{jk}
\wedge
\neg free^{gt}_{jk}
}
]

즉

> **뒤에 known non-free가 실제로 존재함**

이어야 해.

물론 너희 데이터 생성 구조상 `valid=0`이 그런 위치에 절대 등장하지 않는다면 현재 구현도 맞아.

하지만 이건 unit test 하나로 반드시 확인했으면 해.

```text
Case 1
free, free, valid-nonfree
→ RAY_OK

Case 2
free, free, invalid, invalid
→ RAY_CENSORED
```

가 되는지.

문서에는 "`first_free_range`의 RAY_OK와 정확히 일치한다"는 테스트가 있다고 되어 있는데,  이것은 **두 구현이 서로 일치함**을 증명할 뿐, `first_free_range`의 원래 semantics 자체가 원하는 것과 맞다는 것을 증명하지는 않아.

이 부분은 제가 실제 코드를 받으면 가장 먼저 볼 부분이야.

---

# 11. 세 번째는 (\lambda_R=0.3)이 초반에는 생각보다 강하다는 점

문서에서 굉장히 좋은 일을 해놨어.

[
G(\lambda_R)
============

\frac{
|\nabla(\lambda_RL_{\text{range}})|
}{
|\nabla L_{\text{main}}|
}
]

를 실제로 측정했어.

그리고 (\lambda_R=1)일 때

[
1.55
\rightarrow0.424
\rightarrow0.273
\rightarrow0.147
]

이라고 되어 있어. 

그러면 (\lambda_R=0.3)일 때는 대략:

[
\text{initial}: 0.465
]

[
epoch10: 0.127
]

[
epoch20: 0.082
]

[
epoch40: 0.044
]

정도가 돼.

중반에는 정말 우리가 원했던

[
5\sim20%
]

auxiliary gradient 범위에 잘 들어와.

그런데 **초기에는 46%**야.

즉 random initialization 직후에는 range loss가 main BCE의 거의 절반 크기의 gradient를 주는 셈이야.

나는 이것을 약간 조심할 것 같아.

특히 네가 range를

> 어디까지나 auxiliary

로 두고 싶다면, **warm-up**이 상당히 자연스러워.

예를 들어 처음 5 epoch 동안

[
\boxed{
\lambda_R(t)
============

0.3
\min\left(1,\frac{t}{5}\right)
}
]

처럼

```text
epoch 0 : 0
epoch 1 : 0.06
epoch 2 : 0.12
...
epoch 5 : 0.30
```

으로 올리는 거야.

그러면 초기에는 soft-BCE가 기본적인 segmentation을 만들고, 어느 정도 free-space 구조가 나온 뒤 range constraint가 들어오게 돼.

**지금 (\lambda_R=0.3)이 틀렸다는 건 아니지만, 현재 gradient 측정 결과 자체가 warm-up을 시험해볼 충분한 이유를 준다고 봐.**

---

# 12. Range arc loss의 구조적 한계도 알아둘 필요가 있어

현재

[
\hat a
======

\Delta r\sum_kp_k
]

이므로 한 ray의 scalar error가 모든 cell에 전달돼.

문서에도

> 한 광선의 모든 표본이 같은 scalar gradient를 나눠 받는다

고 정확히 적혀 있어. 

예를 들어

```text
GT:
1 1 1 1 | 0 0 0 0

Pred:
1 1 1 1 | 0 0 1 0
              ↑ error
```

라면 잘못된 cell은 사실 뒤의 한 cell이야.

그런데 range loss만 보면

[
\hat a>a_{gt}
]

라는 scalar만 알기 때문에 **ray 전체의 (p_k)를 낮추는 방향으로 gradient가 나가.**

즉 error localization은 못해.

하지만 이건 꼭 bug라고 볼 것은 아니야.

왜냐하면 바로 그 역할을 main BCE가 하고 있고,

[
\lambda_R
]

가 작다면

* BCE → 어디가 틀렸는지를 알려줌
* Range → ray 전체 geometry가 틀어졌음을 알려줌

이라는 complementary 구조가 되니까.

그래서 오히려 **(\lambda_R)가 너무 커지면 안 되는 또 하나의 이유**라고 생각하면 돼.

---

# 13. `RAY_CENSORED`를 제외한 것도 현재로서는 안전한 선택이야

GT가 grid 끝까지 free이면

```text
Robot → free → free → free → grid end
```

우리는 실제 boundary가 어디인지 모르잖아.

그래서 양방향 range regression에서 제외한 건 맞아. 

나중에 개선하고 싶으면 censored ray에 대해 **one-sided lower-bound loss**는 만들 수 있어.

예를 들면

[
L_{\text{censored}}
===================

\operatorname{Huber}
\left[
\max(0,a_{gt}-\hat a-\delta_C)
\right].
]

즉

> GT에서 grid 끝까지 free였으니 prediction이 그보다 너무 짧은 것은 벌하지만, 더 길다고 벌하지는 않는다.

라는 식이지.

하지만 지금은 안 넣은 게 더 좋아. 현재 질문에서는 실험 변수를 늘릴 이유가 없어.

---

# 14. 비대칭 dead-zone은 구현만 해두고 대칭으로 사용하는 것도 좋은 선택

현재

[
\delta_R^+=\delta_R^-=0.20m
]

인데 과대 free-space prediction은 safety상 더 위험하니까 나중에는

[
\delta_R^+<\delta_R^-
]

를 고려할 수 있도록 해둔 상태야. 

이것도 설계가 좋아.

하지만 지금은 대칭을 baseline으로 유지하는 게 맞다고 봐.

처음부터 asymmetric하게 하면

> range auxiliary 자체가 좋아진 건지
> safety bias를 넣어서 좋아진 건지

원인 분리가 안 되니까.

---

# 15. (L_{\text{mono}})가 없는 것도 전혀 문제없어

현재 단계에서는 오히려 없는 게 좋아.

우리가 이야기한 순서는

[
L_{\text{soft-BCE}}
]

→

[
L_{\text{soft-BCE}}+\lambda_RL_{\text{range}}
]

→ 필요하면

[
+\lambda_ML_{\text{mono}}
]

였으니까.

지금 구조가 딱 두 번째 단계야.

---

## 제 평가를 요약하면

| 항목                                      | 평가                          |
| --------------------------------------- | --------------------------- |
| Binary formulation                      | **맞음**                      |
| Signed perpendicular boundary distance  | **맞음**                      |
| Normalized Gaussian CDF                 | **맞음**                      |
| (\alpha=\sigma/\delta) parameterization | **좋음**                      |
| Hard (F/N) + soft boundary 분리           | **맞음**                      |
| Soft BCE entropy floor / KL logging     | **맞음**                      |
| 기존 class weight 제거                      | **맞음**                      |
| Range를 auxiliary로 추가                    | **맞음**                      |
| GT/pred에 동일한 arc operator 사용            | **좋은 개선**                   |
| Dead-zone                               | **맞음**                      |
| Huber                                   | **맞음**                      |
| meter 단위 Huber                          | **문제없음**                    |
| asymmetric range tolerance 지원           | **좋음, 현재 대칭 유지 적절**         |
| (\lambda_R) gradient calibration        | **아주 좋음**                   |
| (\lambda_B=0.5)                         | **gradient 영향 추가 확인 필요**    |
| `RAY_OK`의 invalid 처리                    | **반드시 semantics 확인 필요**     |
| (\lambda_R=0.3) 초기 gradient             | **조금 강할 수 있음 → warm-up 검토** |
| (L_{\text{mono}}) 없음                    | **현재 단계에서 맞음**              |

그래서 제가 지금 실제로 수정 여부를 판단하기 전에 **딱 세 가지를 추가 검증**하겠습니다.

첫째, boundary의 실제 gradient ratio:

[
\frac{|\nabla(\lambda_BL_B)|}
{|\nabla(\frac12L_F+\frac12L_N)|}
]

둘째, `free → invalid` ray가 `RAY_OK`로 잘못 들어가지 않는지.

셋째, `λ_R=0.3 fixed`와 `0→0.3 warm-up`의 validation 비교.

그 외에는 **현재 명세가 우리가 논의해온 loss formulation을 꽤 제대로 구현하고 있다고 판단해도 좋습니다.**

다만 이건 어디까지나 **`loss_function_spec.md`의 수식과 설계 논리를 검토한 결과**야. 문서 자체도 “코드가 최종 권위”라고 명시하고 있어.  실제 구현에 indexing, mask broadcasting, detach, reduction, sign, augmentation 처리 같은 bug가 없는지까지 확인하려면 `soft_boundary.py`, `range_loss.py`, 그리고 가능하면 `run_batch_soft_boundary`가 있는 `binary_metrics.py`를 같이 보면 돼. 그 세 파일이 있으면 다음에는 **수식 ↔ 코드 한 줄씩 대응시키면서 실제 구현 검증**까지 할 수 있어.
