읽고 나면 지금 프로젝트의 위치가 전보다 훨씬 명확해졌어. 이번 문서는 단순히 실험을 더 많이 한 게 아니라, **무엇을 더 이상 연구할 필요가 없는지**와 **무엇이 새로운 핵심 가설인지**를 상당히 잘 분리했어.

내 결론부터 말하면, **loss 연구는 이제 닫는 게 맞다.** 앞으로의 핵심은 loss가 아니라 **camera feature → BEV lifting 과정에서 확보할 수 있는 공간 정밀도와 GT 자체의 거리별 정밀도를 분리해내는 것**이야. 다만 문서에서 새로 등장한 “lifting 해상도 바닥”은 매우 유력한 후보이지, 아직 원인으로 확정된 건 아니야.

---

## 지금까지 한 테스트가 결국 무엇을 말하는가

이번 T1~T6는 이렇게 이해하면 가장 쉬워.

T1의 질문은 **“우리가 발견한 `grid_sample` 좌표 버그가 성능을 막고 있었나?”**였고 답은 **아니다**야. 버그는 실제였고 고치는 게 맞지만, 15개 학습 결과에서 성능 변화는 seed noise 범위였어.  따라서 이것은 **성능 개선이 아니라 correctness fix**야. 이 구분을 한 것이 아주 잘한 판단이야.

T2는 **“soft-boundary loss가 모델을 안정적으로 만들었다는 기존 주장이 정확히 무엇을 의미하나?”**를 다시 본 거야. 결과는 꽤 중요해. CE에서는 재학습할 때 전체 경계가 통째로 앞뒤로 이동하는 global shift가 약 1.75 cm였고 최종 loss에서는 0.16 cm까지 줄었어. 그런데 ray별 local variability는 median 약 3.8 cm로 사실상 똑같아. 상태 자체가 달라지는 ray 비율도 개선되지 않았어. 

따라서 이제부터는

> “우리 loss가 prediction을 10배 안정화했다”

라고 말하면 안 되고,

> **“boundary-aware supervision substantially reduces between-run global range bias, while local ray-wise prediction variability remains essentially unchanged.”**

정도로 좁혀 쓰는 게 정확해.

같은 데이터로 seed만 달리한 ensemble의 disagreement는 특히 seed-driven variability를 보는 데 적합하다는 점도 최근 segmentation uncertainty 연구의 구분과 잘 맞아. ([DKFZ][1])

T3는 **“loss가 달라지면 서로 다른 해답을 찾아서 ensemble이 천장을 뚫을까?”**였어. CE/soft/range 9개를 섞어도

[
IoU=0.8018
]

에서 멈췄어. 

이건 굉장히 강한 **loss 연구 종료 신호**야. 물론 이것만으로 task ceiling을 증명하지는 못하지만,

[
\boxed{\text{현재 architecture/data/split에서 loss family가 핵심 bottleneck은 아니다}}
]

라고 말하기에는 충분해 보여.

---

# 이번 문서에서 진짜 중요한 건 T4와 T5야

여기서 연구 방향이 바뀌었어.

현재 image feature가 stride 8이고, 실제 calibration을 통해 feature pixel 하나가 BEV 공간에서 어느 정도 크기에 해당하는지 계산했더니:

[
0!-!1m:;6cm
]

[
1!-!2m:;10.5cm
]

[
2!-!3m:;15!-!16cm
]

[
3!-!4m:;20!-!24cm
]

정도가 나왔지. 

그리고 실제 boundary metric을 거리별로 나눠보니:

[
F1@10cm:
0.649\rightarrow0.535\rightarrow0.355
]

로 거리와 함께 크게 무너지는데,

[
F1@40cm:
0.927\rightarrow0.938\rightarrow0.888
]

은 거의 유지돼. 

이건 **굉장히 좋은 관찰**이야.

즉 모델이 멀리 있는 boundary 자체를 아예 못 찾는다면 (F1@40)도 같이 무너져야 해. 그런데 40 cm 정도 허용하면 잘 맞는다는 것은

> “경계의 존재와 대략적인 위치는 찾고 있지만, 10 cm 수준의 localization precision은 멀어질수록 나오지 않는다.”

라는 해석과 매우 잘 맞아.

---

## 그런데 “stride-8의 고유 해상도 바닥”이라는 표현은 조금 약하게 쓰는 게 좋아

여기서는 CLAUDE 문서를 조금 수정하고 싶어.

feature pixel 하나가 BEV에서 16 cm를 커버한다고 해서

[
\boxed{\text{16 cm보다 작은 차이는 원리적으로 절대 구분할 수 없다}}
]

고 말할 수는 없어.

CNN feature에는 receptive field와 channel 정보가 있고, bilinear interpolation도 있으며, network가 feature lattice 사이의 sub-pixel 정보를 일정 부분 이용할 수도 있으니까.

따라서 16~24 cm를 나는

> **hard resolution limit**

이라기보다

> **nominal spatial sampling scale / spatial granularity**

라고 부르는 게 더 정확하다고 봐.

즉 T4+T5는 현재

[
\boxed{\text{feature spatial sampling density가 bottleneck이라는 강한 가설}}
]

을 만들었지만,

[
\boxed{\text{원인 확정}}
]

까지는 아직 아니야.

이 distinction은 논문에서도 중요할 거야.

---

# 그리고 아직 경쟁 가설이 하나 남아 있다

문서가 스스로 정확하게 잡았듯이:

[
\text{원거리 F1@10 저하}
]

를 설명할 수 있는 것은 적어도 두 가지야.

### 가설 A — feature lifting resolution

멀어질수록 image feature 하나가 BEV에서 더 넓은 영역에 대응하기 때문에 정확한 localization이 어렵다.

### 가설 B — GT 자체의 거리별 localization error

IPM/annotation 과정의 오차가 원거리일수록 커져서, 모델이 맞아도 GT와 10 cm 이내로 일치하기 어렵다.

현재 결과만 가지고는 둘을 분리할 수 없어. 문서도 이 부분을 정확하게 인정하고 있어. 

**이 두 가설을 분리하는 것이 앞으로 가장 가치 있는 실험이야.**

---

# 거리 의존 (\delta)는 지금 다시 열지 않는 걸 추천해

문서에서 중요한 질문을 하나 던졌어.

feature spatial scale이 거리에 따라

[
6cm\rightarrow24cm
]

로 커진다면 soft-boundary의

[
\delta=15cm
]

도 distance-dependent하게 해야 하지 않느냐는 거지. 

나는 **아직 아니라고 생각해.**

왜냐하면 우리가 (\delta)를 처음 도입한 의미는

[
\boxed{\delta=\text{GT boundary uncertainty}}
]

였어.

그런데 지금 측정한 것은

[
\boxed{\text{model input representation의 spatial granularity}}
]

야.

둘은 다른 양이야.

모델의 representation이 좋지 않다는 이유로 target을 더 부드럽게 만들어버리면

> “모델이 못하니까 정답을 쉬워지게 한다”

가 될 수 있어.

그래서 나는 지금은

[
\boxed{\text{거리 의존 }\delta\text{ 실험은 보류}}
]

하겠어.

단, 나중에 **GT annotation error 자체도 거리에 따라 증가한다는 독립적인 측정**이 나오면 이야기가 완전히 달라져.

예를 들어 측정 결과가

[
\sigma_{\rm GT}(r)
==================

\begin{cases}
5cm,&r<1.5m\
10cm,&1.5<r<3m\
20cm,&r>3m
\end{cases}
]

처럼 나온다면 그때는

[
\delta(r)
]

가 매우 정당한 formulation이 돼.

즉 **feature resolution을 근거로 (\delta(r))를 만들지는 말고, label uncertainty를 근거로 만들어야 해.**

---

# 그래서 GT precision을 따로 측정하는 실험을 상당히 높게 평가해

이건 학습을 수십 번 하는 것보다 가치가 클 수도 있어.

전체 dataset을 다시 만들 필요는 없어. representative frame을 작게 골라서, **현재 annotation pipeline과 독립적인 boundary reference**를 하나 만들어봐.

예를 들어 거리 구간별로 실제 boundary localization 차이를 재서

[
e_{\rm label}(r)
================

d_\perp(
\Gamma_{\rm current},
\Gamma_{\rm reference}
)
]

의 median/P90을 구하면 돼.

그러면:

```text
거리      annotation boundary error
0–1.5m       ? cm
1.5–3m       ? cm
3–4m         ? cm
```

가 나오지.

이것만 있으면 지금 가장 큰 질문이 상당 부분 해결돼.

특히 네가 이미 LiDAR 기반 map을 GT 생성에 활용하고 있으니, 가능하다면 annotation tool과 다른 경로로 만든 고정밀 reference를 작은 subset에서 구성해보는 것도 좋은 방법이야.

---

# 다음 architecture 실험은 stride 8 → 4가 가장 맞다

이 부분에서는 CLAUDE의 우선순위에 동의해.

입력을 1.5배 키우면 pixel 수가

[
1.5^2=2.25
]

배가 되고 encoder가 전체 inference의 70~82%를 차지하니 계산량이 많이 올라가. 현재 benchmark도 그걸 보여줘. 

반면 stride-4 feature를 lifting에 사용하면 **camera encoder 입력 자체는 그대로 유지하면서 spatial sampling density를 직접 건드릴 수 있어.**

단, 나는 단순히 `layer1`만 가져오는 것보다 **high-resolution + semantic feature fusion**으로 설계할 것을 추천해.

예를 들면 개념적으로:

[
F_4 = \text{layer1}
]

[
F_8 = \text{layer2}
]

[
F_{16}=\text{layer3}
]

를 적절히 channel projection하고,

[
\boxed{
F_{\rm lift}^{stride4}
======================

Fuse(
F_4,
Up(F_8),
Up(F_{16})
)
}
]

처럼 만드는 거야.

왜냐하면 문서가 말했듯 stride-4 layer1은 위치 정보는 좋은데 semantic abstraction이 약할 가능성이 있으니까.

현대 camera-BEV에서도 image feature를 stride-8 수준으로 추출하는 것은 흔하고, 여러 scale의 feature를 결합해 spatial/semantic 정보를 보완하는 설계도 사용돼. ([Frontiers][2])

---

# 이 stride-4 실험은 “성능이 오르나?”가 아니라 특정 가설을 검증해야 해

이게 제일 중요해.

그냥

> IoU가 좋아졌는지 봅시다.

로 끝내면 안 돼.

우리가 검증하려는 가설은:

[
H:
\text{현재 원거리 boundary precision은 image-feature sampling density에 제한된다.}
]

야.

그러면 **사전에 예상되는 결과 패턴**이 있어야 해.

stride 4가 feature footprint를 대략 절반으로 줄여준다면 가장 먼저 움직여야 하는 건:

[
\boxed{F1@10cm,;1.5!-!3m}
]

이고 그 다음은

[
\boxed{F1@10/20cm,;3!-!4m}
]

야.

반면 이미 tolerance가 넓은

[
F1@40cm
]

은 별로 움직이지 않아야 해.

즉 이상적인 evidence는:

```text
                  stride8   stride4
near F1@10          ≈         ≈
mid  F1@10         ↑↑
far  F1@10/20      ↑↑
F1@40              거의 동일
```

이런 **거리×tolerance에 선택적인 improvement**야.

이 패턴이 나오면 feature-resolution hypothesis가 굉장히 강해져.

반대로 stride 4로 바꿨는데

[
F1@10(r)
]

profile이 전혀 움직이지 않는다면,

[
\boxed{\text{feature sampling density가 주 bottleneck이라는 가설을 기각}}
]

해도 좋아.

그때는 label precision 쪽 증거가 훨씬 강해져.

---

# 그래서 다음 실험은 이런 순서로 끝내는 걸 추천해

| 순서    | 테스트                                      | 정확히 무엇을 확인?                            | 결과에 따른 판단                 |
| ----- | ---------------------------------------- | -------------------------------------- | ------------------------- |
| **0** | 현재 loss + `pixel_center` baseline freeze | 출발점 고정                                 | loss/geometry tuning 종료   |
| **1** | Orin에서 현재 baseline latency               | 실제 계산 예산                               | stride-4/해상도 후보의 허용 비용 설정 |
| **2** | 작은 GT precision audit                    | label error가 거리와 함께 커지는가               | 커지면 label ceiling 증거      |
| **3** | stride-4 multi-scale lifting, 최종 3 seeds | feature sampling density가 bottleneck인가 | 거리별 F1 profile이 개선되면 채택   |
| **4** | stride-4가 유효한 경우만 input resolution ↑ 비교  | 추가 spatial density가 더 필요한가             | Orin 예산 대비 gain 판단        |
| **5** | 최종 모델만 temporal/sequence stability 평가    | 배포 모델의 robustness                      | 연구 최종 stability 결과        |

여기서 중요한 건 **3번에서 stride-4가 실패하면 4번을 무조건 할 필요는 없다는 것**이야.

단순 spatial density가 원인이 아니라는 신호니까.

---

# 실험 횟수도 이제 많이 줄여야 해

지금까지는 가설을 찾는 과정이라 15-run sweep 같은 게 필요했어.

이제는 그러지 않는 게 좋아.

예를 들어 stride-4도 처음엔 **1 seed로 implementation sanity check** 정도는 할 수 있지만, 그걸 scientific conclusion으로 쓰면 안 돼.

정식 판단은:

[
\text{stride8 corrected baseline: 기존 3 seeds}
]

vs

[
\text{stride4: 3 seeds}
]

이면 충분해.

그리고 **parameter sweep은 하지 마.**

지금 질문은 stride-4의 최적 architecture를 찾는 게 아니라

> “spatial resolution을 높이면 우리가 예상한 거리 구간의 precision이 실제로 올라가는가?”

니까.

---

# 경계 metric도 이번 문서의 결론대로 가는 게 맞아

세 후보 중에서는 나는 명확히 **(a)** 를 선택하겠어.

즉 primary boundary report:

[
\boxed{
\text{distance bin}
\times
\text{absolute tolerance}
}
]

야.

예를 들어:

| 거리     | F1@10 | F1@20 | F1@40 |
| ------ | ----: | ----: | ----: |
| 0–1.5m |       |       |       |
| 1.5–3m |       |       |       |
| 3–4m   |       |       |       |

이게 가장 정직해.

“feature footprint의 1배 이내면 success” 같은 **distance-normalized tolerance는 diagnostic으로는 재미있지만 primary metric으로 쓰지 않는 게 좋아.**

왜냐하면 정말로

> 모델이 멀리 있으면 못하니까 평가 기준도 느슨하게 해준다

는 비판을 받을 수 있거든.

다만 secondary analysis로

[
\frac{\text{boundary error}}
{\text{local feature footprint}}
]

를 보여주는 것은 좋아.

하지만 main metric은 cm를 유지하자.

---

# 그리고 `f1@10cm = 0.55`를 더 이상 “모델 성능 55점”처럼 해석하면 안 돼

이것도 이번 문서의 아주 좋은 결론이야.

전체 boundary 중 55%가 존재하는 1.5–3 m 구간에서 feature sampling scale이 이미 10–16 cm인데 평가 tolerance가 10 cm야. 

그러니 이 metric은

> “경계를 얼마나 잘 찾았나?”

뿐 아니라

> **“현재 pipeline이 10 cm localization을 할 수 있나?”**

를 동시에 재고 있어.

그래서 앞으로 나는 boundary quality를 두 가지 문장으로 구분할 것 같아.

> **Detection/coarse localization:** F1@40 cm ≈ 0.9 이상.

> **Fine localization:** 10 cm tolerance에서는 거리에 따라 급격히 저하.

이 표현이 훨씬 정확해.

---

# stability는 이제 이렇게 확정하면 좋아

T2 결과를 보고 나면 “stability score 하나”를 만들려고 하지 않는 게 맞아.

최종 모델에는 세 숫자를 그대로 쓰면 돼.

[
\boxed{
\sigma_{\rm global}
}
]

재학습할 때 전체 free-space boundary가 통째로 얼마나 이동하는가.

[
\boxed{
\sigma_{\rm ray}^{median},;\sigma_{\rm ray}^{P90}
}
]

같은 scene/ray에서 seed에 따라 boundary가 얼마나 흔들리는가.

그리고

[
\boxed{
D_{\rm state}
}
]

어떤 seed에서는 obstacle을 만나고 다른 seed에서는 censored/free라고 판단하는 **상태 disagreement 비율**.

이 세 개가 각각 다른 failure mode를 측정해.

현재 결과는:

> **global bias stability는 크게 개선**
> **local geometric stability는 변화 없음**
> **state/topological stability 역시 개선 증거 없음**

이라고 정리하면 완벽해. 

이걸 “10배 더 stable” 하나로 압축하면 오히려 연구가 약해져.

---

# temporal stability는 마지막에 한 번은 측정하는 게 좋다

이건 loss나 architecture를 더 만드는 실험이 아니라 **최종 모델 평가**로 넣으면 돼.

실제 로봇에서는 seed보다 더 중요한 질문이

> 바로 다음 프레임에서 free boundary가 갑자기 30 cm씩 튀는가?

니까.

pose로 (t+1) prediction을 (t) 좌표계에 warp해서

[
E_{\rm temporal}
================

\operatorname{median}
|
R_t(\theta)-R_{t+1\rightarrow t}(\theta)
|
]

와 P90 정도만 측정해도 굉장히 유용해.

architecture 선택에 쓰지는 말고 **마지막 모델의 operational stability characterization**으로 쓰면 충분해.

---

# T6에서 한 가지 주의

desktop benchmark에서 FP16이 FP32보다 느린 경우도 나오고 있지. 

따라서 저 숫자로 Orin latency를 비례 추정하면 안 돼.

지금 측정의 의미는 딱:

[
\boxed{\text{encoder가 계산량의 대부분을 차지한다}}
]

정도야.

실제 deployment 판단은 AGX Orin에서 intended runtime—예를 들어 실제로 사용할 PyTorch/TensorRT/FP16 경로—로 재야 해.

특히 stride-4 방식의 장점은 **input resolution을 늘리는 것보다 encoder FLOPs 자체는 훨씬 덜 건드릴 가능성이 있다는 것**이기 때문에, Orin 결과가 architecture 선택에 직접 연결될 거야.

---

# 버그 수정본을 default로 바꾼 판단도 맞아

이건 나는 명확하게 찬성해.

T1에서 performance gain이 없었다고 해서

> legacy가 성능이 같으니 그냥 유지하자

라고 하면 오히려 이상해.

좌표 convention이 틀렸다는 게 수학적으로 확인됐고 unit test에서 올바른 mapping도 확인됐으니 **새 코드의 기본값은 correct convention이어야 해.** 

대신 재현성 때문에:

* historical result → `legacy`
* 앞으로의 final experiments → `pixel_center`
* 코드에는 legacy reproduction flag 유지

정도로 명확하게 분리하면 돼.

과거의 50개 실험을 전부 다시 돌릴 필요는 없어. **앞으로 서로 비교할 핵심 baseline만 수정 버전으로 통일하면 충분해.**

---

## 내가 지금 이 프로젝트를 한 문장으로 정의한다면

현재까지 얻은 결론은:

> **Loss formulation은 평균 geometric accuracy의 bottleneck이 아니며, soft-boundary supervision은 주로 재학습 간 global boundary bias를 줄인다. 현재 남은 fine-boundary error는 seed나 loss에 공통적이고, 그 거리 의존 패턴은 camera feature의 spatial sampling scale 또는 GT localization precision과 일치한다.**

여기까지는 꽤 강하게 말할 수 있어.

그리고 다음 연구 질문은 이제 단 하나로 좁혀져.

[
\boxed{
\text{10–20 cm fine localization의 한계는 representation인가, label인가?}
}
]

나는 **이 질문을 해결하면 현재 2D occupancy 실험을 굉장히 깔끔하게 마무리할 수 있다고 봐.**

그래서 당장은 새로운 loss, (\delta(r)), (L_{\rm mono}), optimizer, regularization 같은 쪽으로 다시 돌아가지 않았으면 해. **작은 independent GT precision audit + 하나의 clean한 stride-4 intervention** 이 두 가지가 지금까지 한 수십 개의 loss 실험보다 훨씬 더 많은 것을 알려줄 가능성이 높아.

[1]: https://inrepo02.dkfz.de/record/313277?utm_source=chatgpt.com "Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation - DKFZ"
[2]: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1431346/full?utm_source=chatgpt.com "Frontiers | Camera-view supervision for bird's-eye-view semantic segmentation"
