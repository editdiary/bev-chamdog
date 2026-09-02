# Loss Function 재설계 효과 실험 — GPT 검토 코멘트 및 추가 검증 제안
## Claude 전달용

### 문서 목적
이 문서는 `loss_effect_results.md`와 `RESULTS.json`의 결과를 바탕으로, 현재 실험에서 **실제로 무엇을 주장할 수 있는지**, **어떤 해석은 조심해야 하는지**, 그리고 **추가로 어떤 테스트를 수행하면 결과 파트를 더 강하게 만들 수 있는지**를 정리한 검토 메모이다.

이번 결과를 해석할 때 가장 중요한 원칙은 다음과 같다.

> **이 실험의 핵심은 모델의 평균 정확도 향상이 아니라, boundary uncertainty를 반영하도록 objective를 재설계했을 때 validation objective의 거동과 재학습 시 operating point가 어떻게 달라지는지에 있다.**

특히 현재 GT boundary 자체에 annotation uncertainty가 존재하기 때문에, IoU/F1과 같은 지표를 실제 physical correctness의 절대적 척도로 과도하게 해석하지 않는다. 해당 지표는 **동일한 validation set 안에서 loss와 기존 quality proxy가 얼마나 정렬되는지 확인하는 보조 수단**으로 사용한다.

---

# 1. 현재 결과에서 가장 강하게 남는 핵심 결론

## 1.1 가장 확실한 변화는 validation objective의 수렴 거동이다

기존 BCE(`A_ce`)에서는 validation loss가 비교적 이른 epoch에서 최저점에 도달한 뒤 다시 크게 증가했다.

- `A_ce` rebound: **44.3 ± 11.1 %**
- `D_range` rebound: **0.9 ± 0.6 %**
- `C_soft` rebound: **1.1 ± 1.7 %**

즉, 새 objective 계열에서는 기존 BCE에서 보이던 large rebound가 사실상 사라졌다.

따라서 다음 주장은 충분히 강하게 할 수 있다.

> **Boundary-aware objective로 전환한 이후 validation objective의 convergence behavior가 크게 안정화되었다.**

다만 이를 곧바로

> “전체 학습이 안정화되었다”

라고 확장해서는 안 된다.

현재 실험에서 실제 spatial prediction의 seed-to-seed variability는 개선되지 않았기 때문이다.

---

## 1.2 더 중요한 변화: loss가 다시 model quality와 정렬되기 시작했다

단순히 loss curve가 매끄러워진 것보다 더 중요한 결과는 **validation loss minimum과 기존 quality proxy의 optimum이 다시 가까워졌다는 점**이다.

`loss minimum ↔ IoU optimum`의 평균 epoch 차이:

- `A_ce`: **23.0 epoch**
- `C_soft`: **2.2 epoch**
- `D_range`: **8.2 epoch**

만약 validation loss minimum으로 checkpoint를 선택했다고 가정했을 때의 `IoU_free` regret:

- `A_ce`: **0.0135**
- `C_soft`: **0.0001**
- `D_range`: **0.0007**

이는 다음과 같이 해석하는 것이 가장 적절하다.

> **기존 BCE에서는 validation loss가 실제 model-quality proxy와 일찍 분리되어 더 이상 유용한 training signal로 해석하기 어려웠다. 반면 soft boundary supervision을 적용한 이후에는 validation objective의 minimum이 실제 quality optimum과 다시 상당히 잘 정렬되었다.**

따라서 “loss가 안정적으로 보인다”보다 한 단계 더 강하게,

> **새 objective는 checkpoint selection에 사용할 수 있는 의미 있는 validation signal로 회복되었다.**

라고 정리할 수 있다.

### 표현상 주의
기존 결과 문서에 있는 “val label 없이도 멈출 자리를 안다”라는 표현은 수정하는 것이 좋다. Validation loss 역시 GT label을 사용하므로 정확하지 않다.

대신 다음과 같이 표현하는 것이 적절하다.

> **별도의 IoU/F1 등의 quality metric을 checkpoint selection signal로 사용하지 않더라도, validation objective 자체가 학습 진행 정도를 반영할 수 있게 되었다.**

---

# 2. Ablation에서 가장 중요한 해석

## 2.1 `B_perset` 결과가 보여주는 것

`B_perset`은 boundary band를 supervision에서 제외한 상태이다.

결과는 매우 중요하다.

- rebound는 **0.6 %** 수준으로 거의 없음
- 그러나 `IoU_free`는 약 **0.694**로 크게 저하

즉,

> **loss가 매끄럽고 rebound가 없다는 것 자체는 좋은 학습의 증거가 아니다.**

Boundary supervision을 없애면 optimization objective 자체는 매우 안정적으로 보일 수 있지만, 모델이 필요한 boundary information을 충분히 학습하지 못한다.

따라서 본 실험의 메시지는 단순히 “boundary가 문제이므로 boundary를 무시해야 한다”가 아니다.

---

## 2.2 `C_soft`가 보여주는 가장 중요한 insight

`B_perset`에 soft boundary term을 다시 추가한 `C_soft`에서는:

- rebound는 여전히 약 **1.1 %**
- loss ↔ IoU optimum gap은 **2.2 epoch**
- 기존 quality proxy도 다시 정상 수준으로 회복

이 흐름은 매우 의미 있다.

> **Boundary supervision 자체가 문제였던 것이 아니라, uncertainty가 큰 boundary를 확정적인 0/1 hard target으로 supervise하는 방식이 문제였을 가능성이 높다.**

즉, 실험 흐름을 다음과 같이 해석할 수 있다.

1. **Hard BCE**
   - boundary를 확정적인 target으로 학습
   - validation objective가 일찍 rebound
   - objective와 quality proxy가 크게 분리

2. **Boundary supervision 제거 (`B_perset`)**
   - objective는 매우 안정
   - 그러나 필요한 boundary information까지 제거되어 model quality가 붕괴

3. **Soft boundary supervision (`C_soft`)**
   - boundary information은 유지
   - hard certainty만 완화
   - stable objective와 meaningful supervision을 동시에 확보

이 세 단계는 현재 loss redesign의 설계 의도를 가장 직관적으로 보여주는 ablation이다.

---

## 2.3 단, “rebound를 없앤 원인이 per-set averaging이다”라고 단정하지 말 것

현재 `A_ce → B_perset` 단계에서는 동시에 두 가지가 바뀐다.

1. free / non-free를 별도로 평균하는 **per-set aggregation**
2. boundary band의 **hard supervision 제거**

따라서 현재 ablation만으로는 rebound 감소가

- per-set averaging 때문인지,
- hard boundary supervision 제거 때문인지,
- 두 효과가 함께 작용한 것인지

분리할 수 없다.

따라서 현재 문서에서는

> **“Hard boundary supervision을 제거하고 region-wise supervision으로 전환한 단계에서 rebound가 거의 사라졌다.”**

정도로 기술하는 것이 안전하다.

이 원인을 더 명확히 밝히고 싶다면 아래의 추가 control experiment를 권장한다.

---

# 3. `L_range`에 대한 현재 판단

`C_soft → D_range` 단계에서 평균 quality proxy의 추가 개선은 사실상 관찰되지 않았다.

또한 여러 reproducibility 관련 지표에서 `D_range`가 `C_soft`보다 일관된 추가 이득을 보이지 않았다.

특히 selected checkpoint 기준:

- `IoU_free` seed SD: 0.0006 → 0.0017
- fatal-rate seed SD: 0.0011 → 0.0034
- global bias SD: 0.25 cm → 1.05 cm
- loss ↔ IoU gap: 2.2 → 8.2 epoch

따라서 현재 결과로는 다음 정도가 적절하다.

> **Ray-wise auxiliary term은 평균 predictive quality나 reproducibility에 대해 명확하고 일관된 추가 이득을 제공하지 않았다.**

다만 “`L_range`가 확실히 해롭다”라고 단정하는 것은 아직 조심하는 것이 좋다.

그 이유는 selected-checkpoint 기반 결과가 `val IoU` 최대값 선택에 의해 upward selection bias를 갖기 때문이다. Fixed epoch 40에서는 일부 지표의 상대적 관계가 달라지는 경우도 있다.

따라서 논문 또는 최종 문서에서는:

> **“No consistent additional benefit was observed from the ray-wise auxiliary term.”**

정도의 표현이 가장 방어 가능하다.

---

# 4. 안정성의 의미를 반드시 분리해서 사용할 것

현재 결과는 다음과 같이 정리된다.

## 개선된 것

### A. Validation-objective convergence stability
매우 명확하게 개선.

### B. Loss-quality alignment
validation objective minimum이 기존 quality proxy optimum과 크게 가까워짐.

### C. Operating-point reproducibility
같은 `free_miss`를 만들기 위해 필요한 threshold의 seed variation이 크게 감소.

예:

- `A_ce`: threshold SD ≈ **0.069**
- `D_range`: ≈ **0.015**
- `C_soft`: ≈ **0.005**

이는 재학습할 때 probability scale / decision threshold가 훨씬 덜 흔들린다는 뜻이다.

## 개선되지 않은 것

### A. Spatial prediction reproducibility
ray-wise prediction variability는 개선되지 않음.

### B. Discrete free/non-free decision reproducibility
전체 disagreement는 비슷하고, boundary 근방에서는 오히려 조금 증가.

### C. Final configured loss의 broad retraining reproducibility
`D_range`에서는 여러 최종 metric의 seed SD가 BCE보다 일관되게 줄지 않음.

따라서 문서 전체에서 **“training stability”라는 하나의 넓은 표현으로 묶지 않는 것이 중요하다.**

추천 표현:

- **validation-objective convergence stability**
- **objective–quality alignment**
- **operating-point reproducibility**

피해야 할 표현:

- “전체 학습 안정성이 개선되었다”
- “prediction stability가 개선되었다”
- “retraining reproducibility가 전반적으로 개선되었다”

---

# 5. 모델 성능 자체는 결과 파트의 중심에서 빼는 것이 적절하다

이번 프로젝트에서는 GT boundary 자체에 상당한 annotation uncertainty가 존재한다.

따라서 IoU/F1의 작은 차이를 실제 physical prediction quality의 우열로 직접 해석하기 어렵다.

이번 결과 파트에서는 성능 향상을 주요 contribution으로 주장하지 않는 것을 권장한다.

대신 metric은 다음 용도로만 제한적으로 사용한다.

1. `B_perset`처럼 supervision을 제거했을 때 model behavior가 무너지는지 확인
2. validation objective minimum이 기존 quality proxy optimum과 정렬되는지 확인
3. 새 loss가 기존 수준의 model behavior를 유지하면서 objective behavior를 정상화하는지 확인

즉, 표현을 다음처럼 가져가는 것이 좋다.

> “새 loss가 더 정확한 모델을 만들었다”  
> → 사용하지 않음

대신:

> **“새 loss는 기존의 predictive behavior를 크게 바꾸지 않으면서 validation objective의 convergence behavior와 model-quality proxy 간의 정렬을 정상화했다.”**

정도로 정리한다.

---

# 6. 추가로 수행하면 가장 가치가 높은 테스트

아래는 중요도 순서이다.

## Priority 1 — 새로운 학습 없이 가능한 검증

### 6.1 지원되는 Python/NumPy 환경에서 headline analysis 재실행

현재 분석 환경은 Python 3.14 + NumPy 1.26 조합에서 실제 boolean-array bug가 발견되었다.

분석 스크립트는 수정되었고 probability map 기반 독립 검증도 통과했기 때문에 현재 결과를 폐기할 이유는 없다.

다만 publication-grade confidence를 위해:

- 지원되는 NumPy/Python 조합에서
- 저장된 probability map을 이용하여
- 핵심 headline analysis만 다시 실행

하는 것을 권장한다.

최소 재검증 항목:

- decision disagreement
- threshold sweep / operating-point matching
- ray-wise jitter
- 주요 fixed-epoch summary

**재학습은 필요하지 않음.**

### 추가 확인
`RESULTS.json`의 experiment root가 `runs/loss_effect_wrongenv`로 기록된 반면, 설명 문서는 `runs/loss_effect/`를 정본으로 설명한다. 단순 metadata/path naming 문제인지, provenance상 다른 산출물이 섞인 것은 아닌지 한 번 확인할 것.

---

## Priority 2 — Hard boundary vs Soft boundary를 직접 비교하는 control

현재 가장 중요한 미해결 질문:

> **rebound 문제의 원인은 per-set aggregation인가, hard boundary supervision인가?**

이를 분리하기 위한 새 control을 하나 추가하는 것을 권장한다.

예:

### `C_hard`

현재 `C_soft`와 동일한 영역 분할/aggregation을 사용하되 boundary target만 soft가 아닌 original hard 0/1로 사용한다.

\[
L_{\mathrm{hard\text{-}partition}}
=
\frac12 L_F
+
\frac12 L_N
+
\lambda_B L_B^{hard}
\]

\[
L_B^{hard}
=
-\frac{1}{|\Omega_B|}
\sum_{i\in\Omega_B}
[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
]
\]

이 control이 있으면:

- `B_perset`: boundary supervision 없음
- `C_hard`: hard boundary supervision
- `C_soft`: soft boundary supervision

을 직접 비교할 수 있다.

### 핵심 관찰

만약:

- `C_hard`에서 rebound가 다시 커지고
- `C_soft`에서 rebound가 작아지며
- 둘의 다른 조건은 동일

하다면,

> **“hard boundary certainty가 objective misalignment의 핵심 원인이고, soft target이 이를 해결했다.”**

는 주장을 훨씬 직접적으로 할 수 있다.

이 테스트는 현재 전체 narrative를 가장 강하게 만들 수 있는 추가 실험이다.

---

## Priority 3 — Boundary vs Confident Region의 Loss Trajectory 직접 분해

가능하다면 `A_ce`에서 epoch별 validation loss를 공간적으로 나누어 확인한다.

추천:

\[
L_{\mathrm{boundary}}(e)
\]

\[
L_{\mathrm{confident}}(e)
\]

그리고:

\[
r_B(e)
=
\frac{
L_{\mathrm{boundary}}(e)
}{
L_{\mathrm{boundary}}(e)+L_{\mathrm{confident}}(e)
}
\]

를 측정한다.

확인하고 싶은 것은:

- BCE total validation loss가 rebound할 때
- confident region은 계속 개선 또는 plateau인데
- boundary region loss만 증가하는가?

이것이 확인되면 최초 문제 정의와 최종 결과가 직접 연결된다.

> **“Validation loss rebound는 boundary-localized error에 의해 주도되었다.”**

라는 주장을 정량적으로 뒷받침할 수 있다.

가능하면 `C_soft`에서도 같은 decomposition을 적용하여 boundary component의 rebound가 완화되었는지 비교한다.

---

## Priority 4 — Validation Loss로 Checkpoint를 고른 모델을 독립 Test Set에서 평가

현재 `loss minimum ↔ IoU optimum` 분석은 동일 validation set 내부의 alignment를 본다.

이를 한 단계 더 강하게 검증하려면:

1. 각 seed에서 **validation loss minimum** checkpoint 선택
2. 별도로 **validation IoU maximum** checkpoint 선택
3. 두 checkpoint를 **독립 test sequence**에서 평가

한다.

핵심 질문:

> 새 loss에서 validation loss로 checkpoint를 골라도 test behavior가 quality-metric-based selection과 거의 동일한가?

이 결과가 나오면:

> **validation objective 자체가 practical checkpoint-selection criterion으로 사용 가능하다**

는 주장이 훨씬 강해진다.

단, test sequence가 학습/튜닝 과정에서 사용되지 않았다는 조건이 필요하다.

---

## Priority 5 — `C_soft` 중심의 Fixed-Epoch Summary

현재 가장 좋은 loss-design insight는 `C_soft`에서 나온다.

따라서 selected checkpoint뿐 아니라 **fixed epoch 40** 기준으로 다음을 A/C/D 세 설정에서 다시 한 표로 정리할 것을 권장한다.

- rebound
- loss-quality gap / regret
- operating threshold SD
- global bias SD
- ray variability
- decision disagreement

목적은 `C_soft`의 장점이 checkpoint-selection artifact가 아니라 고정된 training budget에서도 유지되는지 확인하는 것이다.

---

## Priority 6 — Seed 수 확대는 필요한 경우에만

현재 n=5는 broad trend를 보기에는 충분하지만 variance ratio의 정확한 추정에는 작다.

만약 논문에서 다음을 핵심 주장으로 넣고 싶다면:

> “soft boundary supervision improves retraining reproducibility”

A_ce와 C_soft만 seed를 추가해 n=10 이상으로 늘리는 것을 고려한다.

하지만 이번 문서의 중심을 **objective convergence stability**에 둘 경우에는 우선순위가 낮다.

---

# 7. Claude에게 우선 요청할 내용

다음 순서로 가능 여부를 확인하는 것을 추천한다.

1. **지원되는 NumPy/Python 환경에서 저장된 prediction으로 headline analysis 재검증**
2. **`C_hard` control 추가 가능 여부**
   - same partition / same weights
   - boundary target만 hard 0/1
3. **A_ce와 C_soft의 epoch별 boundary-vs-confident validation loss decomposition 가능 여부**
4. **validation-loss-min checkpoint를 독립 test sequence에서 평가할 수 있는지**
5. **fixed epoch 기준 A/C/D 핵심 stability summary 재정리**

위 5개 중 특히 **2번과 3번**이 현재 연구 narrative를 가장 직접적으로 강화한다.
