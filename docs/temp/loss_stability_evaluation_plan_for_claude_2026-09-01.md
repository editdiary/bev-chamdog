# Loss Function 재설계 효과 검증 계획
## Claude 전달용 실험 지시 문서

### 목적
기존 BCE 기반 학습과 Boundary-Uncertainty-Aware loss를 동일 조건에서 반복 학습하여, 현재 관찰된 **validation 목적 함수 수렴 안정화**가 실제 **retraining reproducibility**와 **prediction stability**의 개선으로 이어지는지 검증한다.

---

## 0. 현재 상황과 검증하고 싶은 주장

기존 BCE를 사용할 때 validation total loss는 초기 최저점 이후 다시 증가하고, 특히 free-space loss가 지속적으로 증가하는 형태를 보였다. 반면 새 loss에서는 total/free/non-free validation loss가 모두 감소 후 비교적 안정적으로 유지되는 형태가 관찰되었다.

이 그래프만으로 직접 주장할 수 있는 것은 **“학습 전체가 안정화되었다”**가 아니라, **“validation objective의 convergence behavior가 안정화되었다”**는 점이다.

이번 실험의 목적은 이 변화가 실제 모델 출력의 재현성과 일반화 안정성까지 이어지는지를 분리해서 확인하는 것이다.

> **핵심 원칙:** “안정성”을 하나의 숫자로 정의하지 말고,  
> **optimization objective / retraining reproducibility / geometric prediction / discrete decision / sequence generalization**으로 나누어 측정한다.

---

## 1. 비교 실험 기본 프로토콜

- **비교군 A:** 기존 BCE (현재 baseline으로 사용하는 weighted BCE 설정)
- **비교군 B:** 최종 Boundary-Uncertainty-Aware loss 설정
- 각 비교군을 **동일한 random seed 5개**로 반복 학습
- 가능하면 두 loss에서 동일한 seed ID 사용
- 다음 조건은 loss 이외 모두 고정
  - Architecture
  - Dataset split
  - Data augmentation
  - Optimizer
  - Learning-rate schedule
  - Batch size
  - Epoch 수
  - Preprocessing
- Checkpoint selection rule도 두 설정에서 동일하게 유지
- 단, 아래의 **checkpoint alignment** 분석을 위해 모든 epoch checkpoint 또는 충분한 epoch 기록을 보존
- 동일한 validation/test sample에 대해 모든 seed model의 prediction을 저장하여 seed 간 직접 비교가 가능하게 할 것

> **중요:** 서로 다른 loss는 정의와 scale이 다르므로 BCE의 loss 값과 새 loss의 loss 절대값 자체를 직접 비교하지 않는다.  
> 비교 대상은 각 objective가 epoch에 따라 보이는 **수렴 형태(convergence behavior)** 와 **재학습 결과의 변동성**이다.

---

# 2. 1차 검증: Validation Objective Convergence Stability

이 파트는 현재 그래프에서 이미 보이는 현상을 정량화하기 위한 것이다.

여기서 얻는 결론은 **optimization objective의 수렴 안정성**이며, prediction stability와는 별도로 해석한다.

---

## 2.1 Loss Rebound

각 seed \(s\)에 대해 validation loss의 최소값 이후 마지막 epoch까지 얼마나 다시 증가했는지 측정한다.

\[
e_{\min}^{(s)}
=
\arg\min_e L_{\mathrm{val}}^{(s)}(e)
\]

\[
R^{(s)}
=
\frac{
L_{\mathrm{val}}^{(s)}(E_{\mathrm{final}})
-
L_{\mathrm{val}}^{(s)}(e_{\min}^{(s)})
}{
\max\left(
\left|L_{\mathrm{val}}^{(s)}(e_{\min}^{(s)})\right|,
\epsilon
\right)
}
\]

### Soft target loss의 경우

Soft target을 포함하는 objective는 이론적으로 0이 아닌 target-entropy floor를 가질 수 있다.

가능하면 새 loss에 대해서는 boundary soft-target의 최소 가능한 cross-entropy를 계산하여 이를 제거한 **excess loss** 기준 rebound도 함께 보고한다.

\[
L_{\mathrm{excess}}(e)
=
L_{\mathrm{val}}(e)
-
L_{\mathrm{floor}}
\]

\[
R_{\mathrm{excess}}
=
\frac{
L_{\mathrm{excess}}(E_{\mathrm{final}})
-
\min_e L_{\mathrm{excess}}(e)
}{
\max\left(
\min_e L_{\mathrm{excess}}(e),
\epsilon
\right)
}
\]

### 보고값

- 각 loss family에서 5 seeds의 rebound
  - mean ± SD
  - median
  - 개별 seed 값
- 새 loss에서 rebound가 일관되게 감소하면:
  - **“validation objective convergence stability가 개선되었다”**고 주장 가능

---

## 2.2 Epoch-to-Epoch Fluctuation (보조 지표)

초기 급격한 하강 구간을 제외하고 plateau 구간 \(E_1 \sim E_2\)를 정한 뒤, epoch 간 loss 변동량을 계산한다.

\[
F^{(s)}
=
\frac{1}{E_2-E_1}
\sum_{e=E_1+1}^{E_2}
\left|
L_{\mathrm{val}}^{(s)}(e)
-
L_{\mathrm{val}}^{(s)}(e-1)
\right|
\]

이 값은 curve smoothness를 보여주는 보조 지표이며, rebound보다 우선순위는 낮다.

---

## 2.3 Loss-Minimum Epoch의 Seed Variability

각 seed마다 validation objective가 어느 epoch에서 최저점에 도달하는지 기록한다.

\[
e_{\mathrm{loss}}^{*(s)}
=
\arg\min_e L_{\mathrm{val}}^{(s)}(e)
\]

### 보고값

- \(e_{\mathrm{loss}}^{*}\)의 mean ± SD
- seed 간 range

단, 이 값은 objective가 실제 task quality와 잘 정렬되어 있을 때만 의미가 크므로 다음 분석과 함께 해석한다.

---

## 2.4 Validation Loss Minimum과 실제 Quality Optimum의 정렬

새 loss가 단순히 보기 좋은 curve를 만드는 것을 넘어, checkpoint selection signal로도 유용해졌는지 확인한다.

대표 quality metric \(Q\)에 대해:

\[
e_Q^{*(s)}
=
\arg\max_e Q^{(s)}(e)
\]

\[
\Delta e^{(s)}
=
\left|
e_{\mathrm{loss}}^{*(s)}
-
e_Q^{*(s)}
\right|
\]

\[
\mathrm{Regret}_Q^{(s)}
=
Q^{(s)}(e_Q^{*(s)})
-
Q^{(s)}(e_{\mathrm{loss}}^{*(s)})
\]

### 추천 \(Q\)

- IoU_free
- F1@10cm
- 필요 시 Range MAE는 부호 방향에 맞게 별도 처리

### 해석

- \(\Delta e\) 감소
- Regret 감소

→ validation loss minimum이 실제 prediction quality optimum과 더 잘 정렬됨

---

# 3. 2차 검증: Retraining Reproducibility

동일한 데이터와 학습 설정에서 random initialization / data ordering만 바뀌었을 때 최종 모델이 얼마나 비슷한 결과를 내는지 평가한다.

이 파트가 **재학습 안정성(retraining reproducibility)** 주장에 가장 직접적이다.

---

## 3.1 최종 Quality Metric의 Seed Variation

각 seed의 최종 또는 선택 checkpoint에 대해 최소 다음 지표를 계산한다.

- IoU_free
- F1@10cm
- Range MAE
- 필요 시 fatal rate / free-miss rate 등 기존 safety-related metric

각 metric \(Q\)에 대해:

\[
SD_Q
=
\mathrm{Std}_s
\left[
Q^{(s)}
\right]
\]

### 해석

평균 성능이 유지되면서 SD가 감소하면:

> 같은 설정을 다시 학습했을 때 최종 quality가 더 재현 가능해졌다.

반드시 **mean과 SD를 함께** 보고한다.

---

## 3.2 Global Range-Bias Variability

각 seed에서 predicted free-space extent가 전체적으로 안쪽 또는 바깥쪽으로 얼마나 치우치는지 측정한다.

sample index를 \(n\), valid ray direction을 \(\theta\), seed를 \(s\)라고 둔다.

\[
b_s
=
\mathrm{Mean}_{n,\theta\in\Theta_{\mathrm{valid}}}
\left[
\hat a_{n,\theta}^{(s)}
-
a_{n,\theta}
\right]
\]

그리고 seed 간 global bias variability를:

\[
V_{\mathrm{global}}
=
\mathrm{Std}_s(b_s)
\]

로 정의한다.

### 해석

\(V_{\mathrm{global}}\)이 감소하면:

> 재학습할 때 predicted free-space boundary 전체가 통째로 inward/outward shift하는 global operating-point bias가 감소했다.

주의:

이 결과만으로 **“모든 위치의 boundary가 안정화되었다”**고 해석하면 안 된다.

아래의 local variability와 반드시 분리해서 보고한다.

---

# 4. 3차 검증: Geometric Prediction Stability

동일한 validation sample을 여러 seed model에 입력했을 때 각 위치의 predicted free-space geometry가 얼마나 흔들리는지 측정한다.

---

## 4.1 Raw Ray-Wise Seed Variability

각 sample-ray 위치에서 seed 간 predicted extent의 표준편차를 계산한다.

\[
\sigma_{n,\theta}^{\mathrm{raw}}
=
\mathrm{Std}_s
\left[
\hat a_{n,\theta}^{(s)}
\right]
\]

모든 valid \((n,\theta)\)에 대해 계산한 뒤 다음을 보고한다.

- median
- mean
- P90

### 해석

- median 감소 → 일반적인 ray prediction의 seed variation 감소
- P90 감소 → 어려운 방향에서 발생하는 큰 variation 감소

특히 P90을 중요하게 볼 것.

---

## 4.2 Global Bias를 제거한 Local Ray-Wise Variability

Raw variability에는 seed별 global shift가 포함된다.

global bias와 local shape instability를 분리하기 위해 seed별 bias를 제거한 prediction을 추가로 계산한다.

\[
\tilde a_{n,\theta}^{(s)}
=
\hat a_{n,\theta}^{(s)}
-
b_s
\]

\[
\sigma_{n,\theta}^{\mathrm{local}}
=
\mathrm{Std}_s
\left[
\tilde a_{n,\theta}^{(s)}
\right]
\]

### 보고값

- \(\sigma^{\mathrm{local}}\) median
- mean
- P90

### 해석

- \(V_{\mathrm{global}}\) 감소
- \(\sigma^{\mathrm{local}}\) 그대로

→ **global bias만 안정화된 것**

반대로 둘 다 감소하면:

> geometric reproducibility가 전반적으로 개선되었다.

---

# 5. 4차 검증: Discrete Free/Non-Free Decision Reproducibility

거리 차이가 작더라도 seed에 따라 같은 cell이 `free ↔ non-free`로 뒤집히면 navigation 관점에서 의미 있는 불안정성일 수 있다.

---

## 5.1 Pairwise Disagreement Rate

threshold \(\tau\)에서 binary prediction을:

\[
\hat y_{n,i}^{(s)}(\tau)
=
\mathbf{1}
\left[
p_{n,i}^{(s)}
\ge
\tau
\right]
\]

로 정의한다.

seed pair 간 disagreement rate:

\[
D(\tau)
=
\mathrm{Mean}_{n,i}
\;
\mathrm{Mean}_{s<t}
\;
\mathbf{1}
\left[
\hat y_{n,i}^{(s)}(\tau)
\neq
\hat y_{n,i}^{(t)}(\tau)
\right]
\]

### 우선 분석

- \(\tau = 0.5\)
- 전체 map
- boundary-near region 별도

전체 map만 보면 쉬운 confident cell이 많아 disagreement가 희석될 수 있으므로, **boundary-near region을 별도로 계산하는 것을 권장**한다.

---

## 5.2 Operating-Point Matched Disagreement (권장 추가)

Loss 변경으로 confidence calibration 자체가 이동할 수 있으므로, \(\tau=0.5\) 결과만으로 구조적 안정성을 판단하면 안 된다.

각 seed의 threshold를 동일한 free-miss rate 또는 사전에 정한 operating point에 맞춘 뒤 disagreement를 다시 계산한다.

### 해석

- 고정 threshold에서는 개선
- matched operating point에서는 차이 없음

→ 개선의 일부가 **calibration / threshold shift** 때문일 가능성이 높음

---

# 6. 5차 검증: Operating-Point Stability

각 seed model의 confidence scale과 safety–conservativeness trade-off가 얼마나 재현 가능한지 확인한다.

---

## 6.1 Target Free-Miss Rate에서 필요한 Threshold의 Seed Variation

목표 free-miss rate \(q\)를 만족하는 seed별 threshold를:

\[
\tau_s^*(q)
=
\text{threshold achieving target free-miss rate } q
\]

라고 한다.

그 변동성을:

\[
V_\tau(q)
=
\mathrm{Std}_s
\left[
\tau_s^*(q)
\right]
\]

로 측정한다.

### 해석

\(V_\tau\)가 감소하면:

> 같은 operating condition을 만들기 위해 필요한 threshold가 seed마다 덜 달라진다.

즉, confidence operating point의 재현성이 좋아진다.

---

## 6.2 Matched Operating Point에서 Fatal Rate Variation

동일한 target free-miss rate에서 seed별 fatal rate를 비교한다.

\[
V_{\mathrm{fatal}}(q)
=
\mathrm{Std}_s
\left[
\mathrm{FatalRate}_s
\left(
\tau_s^*(q)
\right)
\right]
\]

### 보고값

- target free-miss rate별 fatal rate mean ± SD
- BCE vs proposed loss curve

이 분석은 \(\tau=0.5\)에서 보이는 개선이 단순 confidence shift인지, 실제 safety–conservativeness trade-off 개선인지 분리하는 데 중요하다.

---

# 7. 6차 검증: Sequence-Level Generalization Stability

전체 validation 평균만 보면 특정 sequence에서의 붕괴가 가려질 수 있으므로, sequence 단위로 결과를 분해한다.

각 validation/test sequence별로 다음을 계산한다.

- IoU_free
- F1@10cm
- Range MAE

그리고 다음 값을 비교한다.

- sequence 간 mean
- sequence 간 SD
- worst-sequence 성능
- best–worst gap
- 가능하면 각 sequence metric의 seed variation

### 해석

평균 성능이 비슷하더라도:

- worst-sequence 성능 개선
- sequence 간 편차 감소

가 나타난다면:

> 새 loss가 cross-sequence generalization의 안정성을 높였다는 근거가 될 수 있다.

---

# 8. 우선순위: 반드시 먼저 할 테스트

모든 분석을 한 번에 하기 어렵다면 아래 순서로 진행한다.

| 우선순위 | 분석 | 핵심 지표 | 확인하려는 의미 |
|---|---|---|---|
| 1 | Objective convergence | Loss rebound | 현재 관찰한 수렴 안정화를 정량화 |
| 2 | Global reproducibility | \(\mathrm{Std}_s(b_s)\) | seed별 전체 boundary bias 이동 여부 |
| 3 | Local geometric reproducibility | \(\sigma_{\mathrm{local}}\) median / P90 | global shift를 제외한 local boundary 흔들림 |
| 4 | Decision reproducibility | Pairwise disagreement | free/non-free 상태가 seed마다 뒤집히는지 |
| 5 | Final metric reproducibility | IoU/F1/MAE mean ± SD | 최종 quality의 재현성 |
| 6 | Operating-point stability | threshold SD / matched fatal | confidence shift와 실제 trade-off 분리 |
| 7 | Sequence stability | worst-seq / seq SD | 환경 변화에 대한 안정성 |

---

# 9. 결과 해석 규칙

| 관찰 결과 | 권장 주장 |
|---|---|
| Loss rebound만 명확히 감소 | **“Validation objective convergence가 안정화되었다.”** |
| Global bias std도 감소 | **“Retraining 간 global operating-point bias가 감소했다.”** |
| Raw/local ray variability까지 감소 | **“Predicted free-space geometry의 retraining reproducibility가 개선되었다.”** |
| Disagreement도 감소 또는 최소한 악화되지 않음 | **“Discrete free/non-free decision의 재현성도 개선되었다.”** |
| Sequence variation/worst-case까지 개선 | **“Cross-sequence generalization stability도 개선되었다.”** |
| 일부만 개선되고 일부는 동일/악화 | **“학습 안정성 전체가 개선되었다”라고 묶지 말고 개선된 component만 구체적으로 주장한다.** |

---

# 10. 통계 및 보고 방식

- 동일한 seed ID를 두 loss에서 사용했다면 **paired comparison**으로 정리
- 각 metric은 최소:
  - mean ± SD
  - 개별 seed 값
- \(n=5\)에서는 p-value 하나에 의존하지 말고:
  - effect size
  - seed별 변화 방향의 일관성
  - 실제 차이 크기
  를 함께 볼 것
- ray/cell 수가 매우 많더라도 statistical replicate는 기본적으로 **seed**라는 점을 주의
- 수많은 ray/cell을 독립 반복으로 취급하여 과도하게 작은 p-value를 만드는 **pseudo-replication**은 피할 것
- 가능하면 각 seed의 paired difference \((B-A)\)를 함께 표로 남길 것
- threshold-dependent metric은:
  - \(\tau=0.5\)
  - matched operating point
  를 구분해 보고
- 새 loss와 BCE의 **loss 절대값 자체는 직접 비교하지 않는다**

---

# 11. Claude에게 요청할 최종 산출물

테스트 완료 후 아래 결과물을 한 번에 정리해줄 것.

1. A/B 각 5-seed run의 experiment table
   - seed
   - checkpoint
   - 주요 metric

2. Validation loss curve 및 rebound summary

3. Global range-bias \(b_s\)와 seed std

4. Raw ray-wise variability 및 bias-removed local variability
   - median
   - mean
   - P90

5. Free/non-free pairwise disagreement
   - \(\tau=0.5\)
   - 가능하면 boundary-near region 별도

6. Threshold sweep
   - target free-miss rate에서 threshold variation
   - fatal-rate variation

7. Sequence별 metric table
   - mean
   - SD
   - worst-sequence

8. 각 분석마다 다음 4가지를 짧게 해석
   - 무엇을 검증하는 지표인지
   - 실제 결과
   - 가능한 주장
   - 주장하면 안 되는 범위

---

# 12. 최종적으로 답해야 할 질문

1. 새 loss는 validation objective의 rebound를 실제로 얼마나 줄였는가?
2. 같은 설정을 재학습했을 때 global free-space bias가 덜 바뀌는가?
3. global bias를 제거한 뒤에도 local ray geometry가 seed 간 더 일관적인가?
4. 같은 cell의 free/non-free 상태가 seed마다 덜 뒤집히는가?
5. 평균 prediction quality를 희생하지 않고 reproducibility가 개선되었는가?
6. \(\tau=0.5\)에서 보이는 차이가 calibration shift인지, matched operating point에서도 남는 실제 trade-off 차이인지?
7. 특정 sequence에서만 좋아진 것이 아니라 sequence 전반에서 안정성이 개선되었는가?
8. 최종 결론은 다음 중 어디까지 근거 있게 말할 수 있는가?
   - objective convergence stability
   - retraining reproducibility
   - prediction stability
   - generalization stability

---

# 13. 해석 시 가장 중요한 주의점

이번 테스트의 목표는 새 loss가 **“무조건 더 좋다”**는 결론을 만드는 것이 아니다.

핵심은:

> **어떤 종류의 variability를 줄였고, 어떤 종류에는 영향을 주지 못했는지를 분리해서 확인하는 것**

이다.

예를 들어:

- global bias variability는 크게 감소
- local ray variability는 그대로
- state disagreement는 개선되지 않음

이라면, 그 결과 자체가 중요한 결론이다.

이 경우에는 **“전체 학습 안정성이 개선되었다”**라고 과장하지 말고,

> **“새 loss는 retraining 간 global bias component를 선택적으로 감소시켰다.”**

처럼 실제로 확인된 범위만 주장해야 한다.
