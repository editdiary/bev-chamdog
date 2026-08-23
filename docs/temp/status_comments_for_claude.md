문서를 읽고 나면, 이제 loss 쪽은 꽤 많이 정리됐다고 봐도 될 것 같아. 다만 몇 군데는 CLAUDE의 해석을 조금 약하게 표현하는 편이 더 정확하고, 앞으로의 우선순위도 나는 문서와 약간 다르게 잡겠어.

가장 먼저 결론부터 말하면, **현재 결과는 “soft-boundary/range loss가 정확도를 올렸다”기보다는 “거의 같은 정확도에서 학습 결과의 분산과 safety–conservativeness trade-off를 바꿨다”**가 가장 정확한 요약이야. 실제로 CE와 최종 loss 사이에서 `iou_free`, `f1@10cm`, `f1@20cm`, `range_mae` 차이는 seed variation 안이고, 반면 fatal/missed-obstacle은 감소하고 free-miss는 증가했어.  

## 1. "% 개선"에 대한 네 불편함은 타당해

이건 나도 중요하다고 생각해.

예를 들어

[
fatal: 0.1364 \rightarrow 0.1239
]

이면 표현은 두 가지가 가능해.

[
\text{relative improvement}=9.2%
]

이지만,

[
\text{absolute improvement}=0.0125
]

즉 **1.25 percentage point**야.

마찬가지로

[
missed_obstacle:
4.39%\rightarrow3.69%
]

은 상대적으로는 약 16% 감소지만 실제로는 **0.70 pp 감소**야. 반면 그 대가인

[
free_miss:
8.47%\rightarrow9.86%
]

은 **1.39 pp 증가**고. 

그래서 논문이나 보고서에서는 나는 반드시

> fatal: −1.25 pp (−9.2% relative)
> missed obstacle: −0.70 pp (−16.1% relative)
> free miss: +1.39 pp (+16.4% relative)

처럼 **absolute를 먼저 쓰고 relative를 괄호에 둘 것** 같아.

그리고 이것을 단순히 "좋아졌다"고 부르는 것도 아직 조금 이르다고 봐.

### 아주 중요한 추가 실험이 하나 있어

현재 loss가 실제 representation을 개선한 것인지, 아니면 단순히 모델을 **조금 더 conservative한 operating point로 이동시킨 것인지** 분리해야 해.

free probability threshold를 지금 0.5라고 하자. 이걸

[
\tau=0.3,0.35,\ldots,0.7
]

처럼 sweep하면서 각 모델에 대해

[
(\text{free-miss}(\tau),;\text{fatal}(\tau))
]

curve를 그려봐.

예를 들어 CE도 threshold를 조금 높이면

[
fatal\downarrow,\qquad free_miss\uparrow
]

가 되겠지.

만약 CE와 D-range의 curve가 거의 겹친다면 현재 관찰된

> fatal 감소 + free-miss 증가

는 **representation improvement가 아니라 threshold/calibration shift**일 가능성이 커.

반대로 같은 free-miss에서 항상 D-range가 fatal이 낮다면 그때는 정말로

[
\boxed{\text{better safety–utility trade-off}}
]

를 얻은 거야.

이런 식으로 risk와 coverage/utility를 threshold에 따라 비교하는 방식은 selective prediction에서도 일반적인 평가 아이디어야. ([Proceedings of Machine Learning Research][1])

나는 이 실험을 **상당히 높은 우선순위**로 놓겠어.

---

# 2. "남은 오차의 98.9%가 bias"는 방향은 맞지만 표현이 너무 강해

문서에서는 세 seed ensemble이

[
0.7977\rightarrow0.8000
]

밖에 못 올렸으므로 perfect IoU까지 남은 gap의 1.1%만 줄였고, 따라서

> 남은 98.9%는 모든 seed가 공유하는 bias

라고 결론내렸지. 

나는 이것을 다음처럼 바꾸는 게 정확하다고 봐.

> **현재 동일 architecture/data/split/training recipe에서 seed-induced variation이 validation error에서 차지하는 부분은 매우 작다.**

여기까지는 강하게 동의해.

하지만

[
98.9%=\text{bias}
]

를 고전적인 bias–variance decomposition처럼 해석하면 안 돼.

왜냐하면 세 모델은 모두

* 같은 데이터
* 같은 train/val split
* 같은 architecture
* 같은 representation
* 같은 geometry
* 같은 label
* 같은 loss

를 공유하거든.

그러니까 세 모델이 공통으로 틀리는 것은

* label systematic error
* train/val domain difference
* architecture limitation
* camera information limitation
* geometry implementation error
* 데이터 다양성 부족
* single-frame라는 task limitation

중 무엇 때문인지 ensemble만으로는 구분되지 않아.

Deep ensemble은 여러 모델의 disagreement를 이용해 predictive uncertainty를 추정하는 유용한 방법이지만, 같은 training pipeline의 ensemble이 모든 종류의 model uncertainty를 제거하는 것은 아니야. ([NIPS Papers][2])

따라서 나는 문서에서

> "98.9% is shared error under the current modeling pipeline"

정도로 쓰고,

> "irreducible bias"

나

> "task ceiling"

이라고까지 부르지는 않겠어.

---

## 비용 거의 없이 이 주장을 한 단계 더 검증할 방법도 있어

이미 모델들이 있잖아.

* `A_ce` 3개
* `C_soft` 3개
* `D_range` 3개

그러면 새 학습 없이 이들을 섞어서 ensemble해봐.

예를 들어

[
\text{Ensemble}(A_{ce}^{1..3},C_{soft}^{1..3},D_{range}^{1..3})
]

처럼.

같은 loss의 seed ensemble보다 **loss formulation 다양성까지 포함한 ensemble**이 되는 거지.

그것도 0.800 근처에서 똑같이 멈춘다면

> "loss family와 initialization을 바꿔도 공통으로 남는 error"

라는 주장이 훨씬 강해져.

기존 Res50 checkpoint도 남아 있다면 그것까지 섞은 heterogeneous ensemble은 더 좋은 probe고.

그래도 이건 어디까지나 "현재 model family의 shared error"를 보는 실험이지 진짜 irreducible error를 증명하는 건 아니야.

---

# 3. §8.3(b)의 (\sqrt{\sigma_{\rm label}^2+\sigma_{\rm model}^2}) 논증은 나는 절반만 동의해

이 부분은 CLAUDE에게 중요한 코멘트를 돌려주고 싶어.

문서에서는

[
\sigma_{\rm label}=0.075m
]

인데 model range error가 약

[
\sigma_{\rm model}\approx0.236m
]

이므로

[
\sigma_{\rm effective}
======================

\sqrt{0.075^2+0.236^2}
\approx0.247m
]

라고 하고, 따라서 학습 target이 모델의 achievable precision보다 3.3배 날카롭다고 이야기해. 

**현상에 대한 intuition은 매우 좋아.**

즉

> 모델이 validation에서 ±24 cm 정도 흔들리는데 ±7.5 cm짜리 매우 sharp한 soft target을 요구하면 boundary KL이 좋아지기 어렵다.

라는 해석에는 동의해.

하지만 이것을

[
\boxed{
\sigma_{\text{target}}
======================

\sqrt{
\sigma_{\text{label}}^2+
\sigma_{\text{model}}^2
}
}
]

가 되어야 한다는 **Bayesian target derivation으로 받아들이면 안 돼.**

왜냐하면 GT target의 확률은 원칙적으로

[
P(\text{free}\mid \text{annotation})
]

이어야 하고, 이것을 결정하는 것은 label/world uncertainty야.

반면

[
\sigma_{\text{model}}
]

은 현재 모델이 못 맞추는 정도야.

모델이 부정확하다고 해서 ground-truth probability 자체가 더 불확실해지는 건 아니거든.

그걸 target에 넣어버리면 극단적으로는

> 못하는 모델일수록 더 쉬운 target을 준다.

라는 자기강화가 생길 수 있어.

따라서 나는 이 계산을

[
\boxed{\text{target 설계의 이론적 정답}}
]

이 아니라

[
\boxed{\text{현재 target sharpness가 model resolution과 불일치한다는 diagnostic}}
]

으로만 사용할 것 같아.

또 0.236 m는 radial range MAE이고 soft boundary는 perpendicular distance니까 같은 (\sigma)라고 보기 어렵고, Gaussian/zero-mean/independence 가정도 검증되지 않았어.

그래서 **(\delta) sweep을 해보자는 결론은 동의하지만, 0.247 m가 이론적으로 맞는 값이라는 근거로 쓰지는 않겠다**가 내 입장이야.

---

# 4. val loss 상승이 boundary 100%인 것은 이제 꽤 잘 이해된 현상이라고 봐

이 분석은 문서에서 매우 잘했다고 생각해.

epoch 4→40에서 total val loss가 +0.0839 올라가는데

[
\lambda_BL_B:+0.1176
]

이고,

나머지는 전부 감소해.

즉 사실상 **validation loss 상승은 전부 boundary KL overfitting**이야. 

그리고 동시에

[
IoU_{best}\rightarrow IoU_{40}
]

하락은 겨우

[
0.0007\pm0.0002
]

라고 했지. 

따라서 나는 지금부터

> val loss가 올라가니 모델 전체가 나빠지고 있다

라고 해석하지 않겠어.

더 정확히는

> **hard region은 계속 잘 맞아지고 있지만, train boundary의 deterministic label을 점점 더 외우면서 held-out boundary에 대한 confidence mismatch가 커지고 있다.**

라고 보는 게 맞아.

이걸 완전히 제거해야 할 failure라고까지 생각하지 않아.

다만 **진단 지표로는 계속 남겨두는 게 좋다.**

---

# 5. 그러면 “모델 안정성”을 뭘로 정의할 것인가?

나는 안정성을 하나의 숫자로 만들지 않을 거야.

적어도 네 연구에서는 네 가지로 분리하는 게 좋아.

## A. Retraining stability — 다시 학습해도 같은 결과가 나오는가

지금 이미 측정하고 있는 seed variance야.

하지만 metric SD만 보는 것보다 **prediction 자체의 disagreement**도 같이 봤으면 해.

예를 들어 각 seed 모델의 ray boundary prediction을

[
\hat R_s(\theta)
]

라고 하면

[
\boxed{
J_{\text{seed}}
===============

\operatorname{median}*{x,\theta}
\operatorname{Std}*{s}
[
\hat R_s(x,\theta)
]
}
]

를 계산할 수 있어.

그리고

[
P90(J_{\text{seed}})
]

도 같이.

이러면

> “seed가 바뀌었을 때 boundary가 평균 4 cm 흔들리고, worst 10%에서는 13 cm 흔들린다.”

처럼 **cm 단위로 stability를 이야기할 수 있어.**

나는 `fatal σ = 0.0007`보다 이런 숫자가 robot 연구에서는 훨씬 직관적이라고 생각해.

---

# 6. B. Checkpoint stability — 학습을 조금 더 해도 결과가 무너지지 않는가

이미 쓰고 있는

[
M_{\text{best}}-M_{\text{last}}
]

와 last-N epoch fluctuation은 좋은 지표야.

다만 `B_perset`의 교훈 때문에 반드시 **품질 조건을 붙여야 해.**

예를 들어

[
\mathcal E
==========

{
e:
IoU(e)\ge IoU_{\max}-0.005
}
]

처럼 acceptable-quality checkpoint 집합을 먼저 잡고,

그 안에서

[
\operatorname{Std}_{e\in\mathcal E}
fatal(e)
]

[
\operatorname{Std}_{e\in\mathcal E}
\hat R_e
]

를 측정해.

그러면

> "아무것도 안 배워서 안정적인 모델"

이 stability 1등이 되는 문제를 막을 수 있어.

문서의 `B_perset`이 바로 그 함정을 아주 잘 보여줬어. 안정성 지표는 최고인데 품질은 최악이었지. 

---

# 7. C. Temporal stability — 연속 프레임에서 boundary가 튀지 않는가

나는 이걸 네 연구에서는 **가장 중요한 stability metric 후보**로 봐.

로봇이 실제로 사용할 때 문제는

```text id="p2ncv3"
frame t:   boundary 2.1 m
frame t+1: boundary 1.6 m
frame t+2: boundary 2.2 m
```

처럼 예측이 출렁이는 거잖아.

로봇 pose를 알고 있으니 (t)와 (t+1) prediction을 같은 world/ego frame으로 warp할 수 있을 거야.

그 다음 overlap region에서

[
\boxed{
J_{\text{temp}}
===============

\operatorname{median}
|
\hat R_t-
\hat R_{t+1\rightarrow t}
|
}
]

와 P90을 봐.

또는 mask 수준에서는

[
IoU(
\hat F_t,
Warp(\hat F_{t+1})
)
]

를 사용할 수 있고.

이건 **GT 자체가 noisy해도 prediction consistency를 직접 평가할 수 있다는 큰 장점**이 있어.

단, dynamic object가 있다면 그 영역은 빼야 하고.

---

# 8. D. Perturbation stability — 입력이 조금 흔들려도 결과가 유지되는가

실제 운용에 가능한 perturbation만 줘.

예를 들어:

* 밝기/노출 변화
* 약한 blur
* camera 한 대 masking
* calibration uncertainty 범위 안의 작은 projection perturbation

각 perturbation (\eta)에 대해

[
J_{\text{pert}}
===============

|\hat R(x)-\hat R(T_\eta x)|
]

또는 mask disagreement를 측정해.

그럼 stability가

> seed에만 안정적

이 아니라

> 실제 sensor perturbation에도 안정적

이라는 주장이 가능해져.

---

# 9. 그리고 사실 내가 제일 먼저 할 stability 실험은 seed가 아니라 sequence stability야

현재 validation은 특정 두 sequence holdout이지.

그런데 greenhouse dataset에서는 sequence가 사실상 하나의 domain일 가능성이 커.

따라서

[
\sigma_{\text{seed}}
]

보다 훨씬 중요한 게

[
\boxed{
\sigma_{\text{sequence}}
}
]

일 수 있어.

7개의 sequence가 있으니 가능하다면 **leave-one-sequence-out**을 추천해.

각 sequence를 한 번씩 validation으로 두고 동일 config를 학습해서:

[
\text{mean IoU},\quad
\text{worst IoU},\quad
\text{std across sequence}
]

[
\text{mean fatal},\quad
\text{worst fatal}
]

을 봐.

특히 safety-critical system에서는 나는 평균보다

[
\boxed{\text{worst held-out sequence}}
]

를 매우 중요하게 보겠어.

모델이 seed 3개에서 같은 결과를 내더라도 greenhouse A에서는 잘 되고 greenhouse B에서는 무너지면 그건 안정적 모델이라고 하기 어렵거든.

---

# 10. 그래서 “안정적인 모델”을 나는 이렇게 정의하고 싶어

단순히

[
\text{small val-loss oscillation}
]

이 아니라,

[
\boxed{
\begin{aligned}
&\text{Quality} && \text{acceptable}\
&\text{Seed jitter} && \text{small}\
&\text{Checkpoint jitter} && \text{small}\
&\text{Temporal jitter} && \text{small}\
&\text{Sequence worst-case} && \text{acceptable}\
&\text{Safety–utility curve} && \text{good}
\end{aligned}
}
]

이 정도가 훨씬 연구적으로 탄탄해.

---

# 11. 그리고 나는 문서의 앞으로의 우선순위를 하나 바꾸고 싶어

문서는 다음 실험 1순위를 (\delta) sweep으로 잡았어. 

나는 **그 전에 geometry 문제를 먼저 확인하겠어.**

문서에

> `normalize_grid2d`는 pixel-index convention인데 `grid_sample(align_corners=False)`는 pixel-boundary convention

이라고 적혀 있잖아. 

PyTorch 공식 문서도 `align_corners=False`에서는 (-1,1)을 corner pixel의 **중심이 아니라 바깥 corner points**로 취급한다고 명확하게 설명해. 좌표 생성과 `grid_sample`의 convention은 반드시 일치해야 해. ([PyTorch Documentation][3])

이게 실제 bug라면 **systematic shared error**를 만드는 아주 좋은 후보야.

그러니까 내 순서는:

1. **`grid_sample` coordinate convention unit test + 수정 여부 확정**
2. **CE vs D-range threshold sweep → 진짜 safety improvement인지 확인**
3. **stability metric 정식화**

   * seed boundary jitter
   * temporal jitter
   * sequence-wise worst case
4. 그 뒤에도 필요하면 **(\delta=0.25)** 한 번 시험
5. 그 결과까지 별 변화가 없으면 **loss 연구 종료**

로 가겠어.

---

## 최종적으로 지금 연구를 어떻게 해석하느냐

나는 현재 결과를 실패로 보지 않아.

오히려 loss에 대해 꽤 좋은 결론이 나온 상태라고 생각해.

> **Hard CE의 generalization failure를 loss engineering으로 정확도 개선까지 연결하지는 못했다. 하지만 noisy boundary를 soft supervision으로 바꾸면 overconfidence가 신뢰하기 어려운 boundary band로 국소화되고, retraining variability와 safety-side decision variability를 줄일 수 있었다.**

이건 충분히 의미 있는 결과야. 특히 CE에서 val loss 증가가 확실한 free 영역에서 폭발하던 것이 soft-boundary에서는 ±15 cm boundary 영역으로 제한됐다는 관측은 꽤 설명력이 있어. 

다만 **“9% 개선”, “16% 개선” 같은 relative number를 headline으로 잡지는 않는 게 좋고**, “1.25 pp/0.70 pp의 safety improvement와 1.39 pp의 conservativeness cost”라고 말한 뒤, threshold-matched trade-off curve가 실제 dominance를 보여주는지 확인하는 게 가장 정직하고 강한 다음 단계야.

그리고 stability를 연구 결과로 가져가려면 **seed metric variance 하나보다는 ‘boundary jitter [cm]’ + ‘temporal consistency’ + ‘held-out sequence worst-case’**의 세 축을 중심으로 가는 걸 가장 추천해. 그러면 “안정성”이라는 다소 모호한 표현이 실제 robot navigation 관점에서 측정 가능한 물리량으로 바뀌어.

[1]: https://proceedings.mlr.press/v97/geifman19a.html?utm_source=chatgpt.com "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"
[2]: https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html?utm_source=chatgpt.com "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
[3]: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html?utm_source=chatgpt.com "torch.nn.functional.grid_sample — PyTorch main documentation"
