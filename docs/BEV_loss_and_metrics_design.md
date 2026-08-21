# BEV Occupancy + Visibility 모델: Loss 및 평가 지표 설계

> 대상: 단일/다중 카메라 입력 → BEV 2채널 출력 (occupancy, visibility) 모델
> 환경: 수직 재배 온실, 좁은 작물 행 통로, map-free local perception

> **[2026-08-14 갱신] §0의 라벨 전제 세 가지가 실물과 다르다.**
> 이 문서의 **설계 원리(§1 이후)는 그대로 유효하고 실제로 그렇게 구현했지만**, §0 표의
> 라벨 형태는 자체 데이터셋 실물을 받아보니 아래와 같이 단순했다. 구현된 계약은
> [`docs/finetuning_guide.md`](finetuning_guide.md) §3이 정본이다.
>
> | §0의 전제 | 실물 (`dataset/sj_datasets/`) |
> |---|---|
> | `occ_gt`에 `255`(map 없음) | **없다.** 전 격자가 `{0, 1}`로 라벨돼 있다 → `valid`는 다른 데서 온다 |
> | `vis_gt`가 `[0,1]` 연속 신뢰도 | **binary `{0, 1}`.** BEV 2D raycast 결과다 |
> | `ego_mask`·`static_fov_mask`가 산출물에 포함 | **라벨에 없다.** `common/`의 PNG 두 장과 캘리브레이션에서 계산한 화각 원반으로 학습 코드가 만든다 |
>
> 설계가 예상하지 못한 것이 하나 더 있다: 라벨의 visibility에 **카메라 화각이 들어 있지
> 않다.** 그래서 `static_fov_mask`에 해당하는 것을 런타임에 계산해 얹어야 했고, 그 과정에서
> "배포에도 남는 가림(`vis=0`)"과 "수집 아티팩트(`valid=0`)"를 구분하는 규칙이 나왔다.

---

## 0. 전제: 라벨 구조

| 텐서 | 형태 | 값 | 생성 방식 |
|---|---|---|---|
| `occ_gt` | (H, W) uint8 | `0` obstacle / `1` drivable / `255` map 데이터 없음 | SLAM map의 충돌 슬래브 투영 + 동적 객체 덮어쓰기 |
| `vis_gt` | (H, W) float32 | `[0, 1]` 관측 신뢰도 | 카메라 기준 3D raycasting (누적 투과율) |
| `ego_mask` | (H, W) bool | 고정 상수 | 플랫폼 footprint (카트 + 릭 + 조작자 영역) |
| `static_fov_mask` | (H, W) float | 고정 상수 | extrinsic으로 계산한 FoV + 근거리 사각지대 |

**핵심 설계 원칙**: occupancy는 map 참값을 그대로 담고, "시야가 닿는 범위"는 **오직 loss weight에서만** 적용한다. 이렇게 하면 visible-only ↔ amodal 전환이 플래그 하나로 끝나고, 라벨 재생성이 필요 없다.

---

## 1. Loss 설계

### 1.1 세 가지 원칙

1. **occupancy는 관측된 셀에서만 학습한다.** `vis_gt`를 per-cell weight로 곱한다. 안 보이는 곳의 참값은 저장은 하지만 gradient에 기여시키지 않는다.
2. **visibility는 전 격자에서 학습한다.** "안 보인다"는 사실 자체가 학습 대상이므로 마스킹하지 않는다.
3. **오류 비용이 비대칭이면 loss도 비대칭이어야 한다.** visibility를 과대 예측하는 것(안 보이는데 보인다고 함)은 검증되지 않은 occupancy 값을 신뢰하게 만들므로, 과소 예측보다 훨씬 위험하다.

### 1.2 구현

```python
import torch
import torch.nn.functional as F

def compute_loss(occ_logit, vis_logit, occ_gt, vis_gt,
                 ego_mask, lambda_vis=0.5, neg_weight=3.0):
    """
    occ_logit, vis_logit : (B, 1, H, W)  raw logits
    occ_gt               : (B, 1, H, W)  {0, 1, 255}
    vis_gt               : (B, 1, H, W)  [0, 1]
    ego_mask             : (1, 1, H, W)  bool, 고정
    """
    valid = (occ_gt != 255) & (~ego_mask)          # GT가 존재하는 셀
    valid = valid.float()

    # ---------- occupancy: 관측 가중 BCE ----------
    w_occ = vis_gt * valid
    l_occ = F.binary_cross_entropy_with_logits(
        occ_logit, occ_gt.clamp(max=1).float(), reduction='none')
    l_occ = (l_occ * w_occ).sum() / (w_occ.sum() + 1e-6)

    # ---------- visibility: 비대칭 가중 BCE ----------
    # vis_gt가 낮은 셀(안 보이는 셀)을 더 무겁게 → false-high 억제
    aw = torch.where(vis_gt > 0.5,
                     torch.ones_like(vis_gt),
                     torch.full_like(vis_gt, neg_weight))
    w_vis = aw * valid
    l_vis = F.binary_cross_entropy_with_logits(
        vis_logit, vis_gt, reduction='none')
    l_vis = (l_vis * w_vis).sum() / (w_vis.sum() + 1e-6)

    return l_occ + lambda_vis * l_vis, {'occ': l_occ.item(), 'vis': l_vis.item()}
```

### 1.3 반드시 지켜야 할 것

| 항목 | 이유 |
|---|---|
| `binary_cross_entropy_with_logits` 사용 | `sigmoid` → `binary_cross_entropy`는 수치적으로 불안정. logits 버전은 log-sum-exp 안정화가 내장돼 있음 |
| `w.sum()`으로 정규화 | 유효 셀 수가 프레임마다 크게 다름 (통로 중앙 vs headland). 정규화 없이 `.mean()`을 쓰면 batch 내 프레임 기여도가 불균등해져 학습이 흔들림 |
| `+ 1e-6` | 전부 마스킹된 프레임(전방이 완전히 막힌 순간)에서 0으로 나누는 것 방지 |
| `vis_gt`를 연속값으로 유지 | BCE는 타깃이 `[0,1]` 실수여도 그대로 동작. 별도 처리 불필요 |
| `ego_mask`를 별도로 곱하기 | `vis_gt=0`으로 이미 걸러지지만, amodal 실험 시 `vis=1`로 채웠을 때 ego 영역까지 supervise되는 사고를 막음 |

### 1.4 하지 말아야 할 것

- **`occ_gt`를 시야로 잘라내 저장하기.** 같은 정보를 두 번 인코딩하는 것이고, 잘린 자리에 placeholder를 넣어야 해서 표현만 지저분해진다. 무엇보다 나중에 amodal 실험이 불가능해진다.
- **`vis_pred`로 occupancy loss를 마스킹하기.** 학습 시엔 반드시 `vis_gt`를 쓴다. 예측으로 마스킹하면 모델이 어려운 영역을 "안 보인다"고 선언해 loss를 회피하는 축퇴 해(degenerate solution)로 수렴한다.
- **처음부터 Focal / Dice / boundary weight 전부 넣기.** 어떤 항이 효과가 있었는지 알 수 없게 된다. plain BCE 베이스라인부터.

### 1.5 λ (lambda_vis) 튜닝

`l_occ`와 `l_vis`를 **따로 로깅**하고 스케일을 비교한다.

- visibility는 상대적으로 쉬운 task이므로 `l_vis`가 빠르게 작아지는 것이 정상이다. 그 자체는 문제가 아니다.
- 다만 `l_vis`가 초기부터 `l_occ`의 10배 이상이면 occupancy 학습을 방해한다. λ를 낮춘다.
- 반대로 `false_high` (§2.3) 지표가 나쁘면 λ 또는 `neg_weight`를 올린다.
- 시작값: `lambda_vis=0.5`, `neg_weight=3.0`

### 1.6 단계적 확장 (베이스라인 확보 후)

각 항목을 **하나씩** 추가하고 지표 변화를 측정한다.

**(a) 클래스 불균형** — 좁은 통로에서는 obstacle 셀 비율이 낮을 수 있다.
```python
l_occ = F.binary_cross_entropy_with_logits(..., pos_weight=torch.tensor([1.5]))
```
먼저 `iou_obstacle`과 `fatal_rate`를 확인한다. 불균형이 실제로 문제가 아니라면 넣지 않는다.

**(b) 경계 가중** — 통로 경계는 안전상 가장 비싼 영역이다.
```python
boundary = morphological_gradient(occ_gt == 0)       # obstacle 경계
w_occ = w_occ * torch.where(dilate(boundary, k=3), 2.0, 1.0)
```

**(c) 미관측 영역 weight floor** — `vis_gt=0` 영역은 gradient가 0이라 출력이 정의되지 않는다.
```python
w_occ = torch.clamp(vis_gt, min=0.05) * valid
```
"안 보이는 곳에서 극단적인 값을 내지 마라" 수준의 약한 정규화. 다만 이것은 **약한 amodal supervision**이기도 하다. 기본은 0으로 두고, 학습이 불안정하거나 시각화에서 이상 패턴이 보일 때만 켠다.

**(d) Soft/Rigid 분리** — 잎(스쳐도 됨)과 지주대·유인끈(충돌 시 파손)은 밀도가 비슷하지만 물리적 결과가 완전히 다르다. 임계값 하나로는 구분 불가. occupancy를 다중 클래스로 확장하는 것을 검토한다.

### 1.7 현재 구현된 loss (2026-08-18 기록. 미해결이던 부분은 §1.7 끝에서 해결됨)

> **§1.1–1.6은 2-head(occupancy + visibility) 시절의 설계다.** 그 정식화는 Phase 3 A/B 이후
> 코드에서 제거됐다([`free_space_metric_migration.md`](free_space_metric_migration.md) §8–9).
> 아래가 **실제로 돌아가는 loss**이고, 구현은 `projects/common/three_class_metrics.py`다.

**지금의 loss: valid 셀에 한정한 가중 3-class cross-entropy**

클래스 순서는 `(unknown, free, occupied)`이고, `valid=0`인 셀은 채점에서 빠진다.
클래스 가중치는 **하드코딩이 아니라 train split의 라벨에서 역빈도로 계산**된다
(`class_weights_from_labels`). 최빈 클래스를 1.0으로 정규화한 뒤 `MAX_CLASS_WEIGHT`로 자른다.

```python
weights = counts.sum() / counts.clamp(min=1.0)   # 역빈도
weights = weights / weights.min()                # 최빈 클래스 = 1.0
return weights.clamp(max=MAX_CLASS_WEIGHT)       # MAX_CLASS_WEIGHT = 20.0
```

실측값(2026-08-18):

| split | unknown | free | occupied |
|---|---|---|---|
| SynWoodScape train (400장) 셀 비율 | 6.21% | **82.21%** | 11.58% |
| → 가중치 (캡 안 걸림) | 13.24 | 1.00 | 7.10 |
| 로봇 train (190장) 셀 비율 | **78.58%** | 20.26% | **1.16%** |
| → 순수 역빈도 | 1.00 | 3.88 | **67.93** |
| → **실제 사용값 (캡 적용)** | 1.00 | 3.88 | **20.00** ← 잘림 |

**이 로직은 Simple-BEV 원본이 아니다.** 원본 `train_nuscenes.py:361`은
`SimpleLoss(2.13)` — 이진 BCE에 스칼라 `pos_weight` 하나를 하드코딩하고, 그 값도 데이터에서
유도한 것이 아니라 Lift-Splat-Shoot 논문에서 가져온 상수다(`# value from lift-splat`).
3-class 역빈도 가중은 이 프로젝트에서 커밋 `dd3d474`으로 추가했다.

**[당시 미해결] `MAX_CLASS_WEIGHT = 20`의 근거가 없다** (이 절 끝의 갱신 블록 참고):

- 역빈도 가중 CE 자체는 semantic segmentation의 표준 처방 중 하나다. 하지만 **순수 역빈도는
  그중 가장 공격적인 축**이고, 실무에서는 median-frequency balancing(SegNet 논문), √역빈도,
  log 스케일 역빈도처럼 완화한 변형을 더 자주 쓴다. 캡 20은 그 완화를 가장 거칠게 구현한 것에
  해당한다.
- **캡이 지금 실제로 작동 중이다.** 로봇 데이터에서 `occupied`를 67.93 → 20으로, 4샘플
  overfit split에서는 ~112 → 20으로 자른다(같은 문서 §7.2). 즉 "혹시 몰라 넣어둔 안전장치"가
  아니라 **loss 균형을 실제로 결정하고 있는 하이퍼파라미터**다.
- 그런데 이 값이 어떻게 정해졌는지가 코드·문서·테스트 어디에도 없다. 테스트
  (`tests/common/test_three_class_metrics.py:88`)는 "캡이 걸린다"는 동작만 확인한다.
- **하필 `fatal_rate`와 직결된다.** `occupied` 가중치를 낮추면 장애물을 free로 오인하는 방향
  (planner 안전상 가장 비싼 오류)으로 기울고, 높이면 반대로 free를 과소 예측해 `iou_free`와
  `free_miss_rate`가 나빠진다. 3-class vs 2-head A/B에서 **유일하게 후퇴한 지표가
  `fatal_rate`**였다는 점(§8)을 생각하면, 그 판정은 검증되지 않은 캡 값 위에서 내려진 것이다.
- 또 하나: 가중치가 split마다 다시 계산되므로 **train 구성이 바뀌면 loss 균형도 같이 바뀐다.**
  4샘플 overfit run의 `free` 가중치는 7.476, 190샘플 A/B는 3.878로 약 2배 달랐다(§7.2).
  두 run의 loss 값을 직접 비교할 수 없다는 뜻이다.

> **[2026-08-18 갱신] 이 항목은 더 이상 미결이 아니다.** 본학습을 돌렸고 결론이 나왔다:
> `MAX_CLASS_WEIGHT`는 상한을 고르는 문제가 아니고, CE가 면적 loss인데 `occupied`가 표면이라는
> 것이 근본 원인이다. 실측과 다음 단계 설계는
> [`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) §12–§13이 정본이다.
> 아래 목록은 그때 세운 후보들로서 남겨 둔다.

**나중에 할 일 (본학습으로 기준 숫자를 확보한 뒤)**

1. 캡 값을 스윕한다: 10 / 20 / 50 / 캡 없음. `iou_free`와 `fatal_rate`가 어떻게 교환되는지
   본다. fine-tuning이 약 8분이라 비용이 싸다.
2. 캡이 정답이 아닐 수 있다. 대안: median-frequency balancing, √역빈도 또는 log 역빈도로
   완화 곡선 자체를 바꾸기, Focal loss, Lovász-Softmax / Dice 계열(IoU를 직접 최적화하므로
   주 지표가 `iou_free`인 지금 구조와 궁합이 좋다).
3. **§1.6의 원칙이 여기에도 그대로 적용된다** — 하나씩 넣고 지표 변화를 측정한다.
   §1.6(a)가 이미 "불균형이 실제로 문제가 아니라면 넣지 않는다"라고 적어 뒀는데, 3-class로
   옮기면서 그 확인 단계를 건너뛰고 역빈도를 바로 넣었다.
4. 클래스 가중치에는 **CLI 플래그가 없다**(의도적이다 — 같은 데이터에 두 run이 다른 가중치를
   쓰면 비교가 무너진다). 스윕하려면 코드를 고쳐야 하고, 어떤 값으로 돌렸는지는 학습 배너에
   매 run 기록된다.

> 주의: 이 항목은 `iou_free`의 **정의**를 건드리지 않으므로 평가 지표 수정 안건과 독립적이다.
> 다만 loss를 바꾸면 학습 결과 자체가 달라지므로 `rescore_checkpoints.py`로는 복구되지 않고
> 재학습이 필요하다.

---

## 2. 평가 지표 설계

### 2.1 대원칙

> **평가 마스크는 항상 `vis_gt`로 정한다. `vis_pred`는 절대 쓰지 않는다.**

예측으로 마스킹하면 모델이 자기가 어려운 영역을 평가에서 제외해 지표를 올릴 수 있다. 자기 자신을 속이는 지표가 된다.

```python
eval_mask = (occ_gt != 255) & (~ego_mask) & (static_fov_mask > 0) & (vis_gt > 0.5)
```

정적 사각지대(`static_fov_mask`)와 ego 영역은 **장면과 무관한 상수 영역**이다. 모델이 공짜로 맞추므로 반드시 제외한다. 포함하면 지표가 부풀려지고 모델 간 비교가 무의미해진다.

### 2.2 Occupancy

```python
iou_drivable = IoU(occ_p[eval_mask] > 0.5,  occ_gt[eval_mask] == 1)
iou_obstacle = IoU(occ_p[eval_mask] <= 0.5, occ_gt[eval_mask] == 0)
```

**거리 구간별로 반드시 분해한다.**

| 구간 | 의미 |
|---|---|
| 0–3 m | 즉각 충돌 회피. 여기가 나쁘면 배포 불가 |
| 3–6 m | 정상 계획 범위 |
| 6–10 m | 조기 감지. 여기서 성능이 급락하는 지점이 곧 **안전 속도 상한**을 결정 |

합쳐서 하나의 mIoU만 보면 근거리의 쉬운 성능이 원거리 실패를 가린다. 온실은 통로 구조가 규칙적이라 전체 mIoU가 항상 높게 나오므로, 단일 숫자는 거의 정보가 없다.

### 2.3 Visibility

IoU보다 **비대칭 오류율**이 중요하다.

```python
m = (occ_gt != 255) & (~ego_mask) & (static_fov_mask > 0)

false_high = ((vis_p > 0.5) & (vis_gt <= 0.5) & m).sum() / ((vis_gt <= 0.5) & m).sum()
false_low  = ((vis_p <= 0.5) & (vis_gt > 0.5) & m).sum() / ((vis_gt > 0.5) & m).sum()
```

| 지표 | 의미 | 비용 |
|---|---|---|
| `false_high` | 안 보이는데 보인다고 예측 | **위험.** 검증되지 않은 occupancy 값을 신뢰하게 됨 |
| `false_low` | 보이는데 안 보인다고 예측 | 보수적. 정보 낭비뿐 |

두 값을 **절대 합치지 말고 따로 리포트한다.** 하나의 F1이나 IoU로 뭉개면 비대칭성이 사라진다.

추가로 연속값 품질을 보려면 `vis_gt`가 중간값인 셀(잎 커튼 뒤)에 한정해 MAE를 본다. 이 구간이 온실 특유의 반투과 매질을 제대로 다루는지 알려준다.

### 2.4 안전 지표 (가장 중요)

IoU는 로봇이 부딪히는지를 말해주지 않는다.

```python
# obstacle을 drivable로 착각한 셀
fatal = (occ_p > 0.5) & (occ_gt == 0) & eval_mask
fatal_rate = fatal.sum() / ((occ_gt == 0) & eval_mask).sum()

# 계획된 경로 상으로 한정하면 더 직접적
fatal_on_path = fatal[planned_path_cells].sum()
```

**판단 기준**: `iou_drivable`이 0.85여도 `fatal_rate`가 나쁘면 배포하지 않는다. 반대로 IoU가 다소 낮아도 `fatal_rate`가 좋으면 쓸 만하다. 두 오류의 비용이 다르므로 지표도 분리해야 한다.

### 2.5 상황별 슬라이스

전체 평균은 다수 케이스가 지배한다. 실패는 소수 케이스에서 나고, 그것이 사고로 이어진다. 아래를 **각각 별도로** 집계한다.

**가시성 패턴별**
- 통로 중앙 (좌우 폐색)
- Headland / 행 끝 (좌우 개방)
- 통로 진입·진출 (가시성 전이 구간) ← 가장 어렵고 가장 중요

**기하 케이스별**
- 늘어진 잎 (하단 뚫림): `occ=obstacle` + `vis=high` 조합을 맞추는가
- 샛길 (머리 위 구조물만): 통행 가능으로 예측하는가
- 낮은 장애물: 뒤쪽 지면을 visible로 유지하는가

**동적 객체**
- 사람 / 카트 존재 프레임에서의 obstacle recall

### 2.6 도메인 특화: 통로 폭 오차

각 종방향 위치에서 예측 통로 폭과 GT 폭의 차이를 잰다.

```python
width_error = predicted_corridor_width - gt_corridor_width   # signed
```

| 부호 | 결과 |
|---|---|
| 음수 (과소, 좁게 봄) | planner가 경로를 못 찾아 로봇이 정지 |
| 양수 (과대, 넓게 봄) | 작물 손상 / 충돌 |

IoU보다 물리적 의미가 직접적이라 임계값 튜닝 시 해석이 훨씬 쉽다. 특히 §1.6(d)의 밀도 임계값을 조정할 때 이 지표를 본다.

### 2.7 궤적 기반 검증 (GT 독립)

로봇이 실제로 지나간 셀은 **주행 가능했음이 물리적으로 증명된** 영역이다. Map 추측이 아니라 실측이다.

```python
# t 시점 예측 vs t 이후 실제 주행 궤적
traj_recall = (occ_p[t][future_traj_cells] > 0.5).mean()
```

이 지표의 특별한 가치는 **라벨 품질과 독립적**이라는 점이다. 라벨 파이프라인 자체에 버그가 있어도 (슬래브 높이 오설정, raycasting 오류, map drift) 이 지표는 그것을 잡아낸다. 라벨과 모델을 동시에 검증하는 유일한 수단이므로, 배포 전 필수 항목으로 둔다.

### 2.8 현재 구현된 지표 집합 (2026-08-18 확장) — 이 절이 코드의 정본이다

위 §2.1–2.7은 2-head 시절의 **설계** 문서다. 실제로 코드가 계산하는 것은 아래다.
`docs/free_space_metric_migration.md`가 M1–M4가 왜 이 형태인지의 근거 정본이고, 여기서는
2026-08-18에 추가된 것과 그 이유를 적는다.

| 지표 | 코드 | 역할 |
|---|---|---|
| `iou_free` (M1) | `free_space_metrics.iou_free` | **체크포인트 선택 기준.** 주 지표 |
| `fatal_rate` (M2) | 같은 모듈 | 1 − precision(free). "믿은 영역의 오류율" |
| `free_miss_rate` (M2b) | 같은 모듈 | 1 − recall(free). 보수성 |
| `range_*` (M3) | `range_error` | 방위각별 첫 장애물 거리 오차 `dr = r_pred − r_gt` |
| ring별 M1·M2 (M4) | `metrics_per_ring` | 셀을 거리 링으로 나눈 재측정 |
| **`f1@τ`** (신규) | `occupied_metrics.py` | **occupied 전용.** τ = 10/20/40 cm |
| **`range_mae` / `bias`** (신규) | `_delta_stats` | 평균 절대오차와 순수 편향 |
| **`missed_obstacle_rate`** (신규) | `_delta_rates` | M3 표본에서 빠진 광선의 비율 |
| **거리별 `range_mae`** (신규) | `summarize_range_error_by_gt_range` | 광선을 `r_gt`로 층화 |
| ~~`iou_occupied` / `iou_unknown` / `iou_free_known`~~ | ~~`free_metrics_from_masks`~~ | **2026-08-21 제거.** 아래 (d) 참조 |

#### (a) 왜 `f1@τ`를 추가했는가 — 가장 큰 구멍이었다

`iou_obstacle`을 버린 뒤(두께 1셀 표면에 IoU를 씌우면 한 칸 밀리는 것만으로 반토막, §1
진단에서 0.312가 frontier 규칙 0.473에 졌다) **occupied를 직접 재는 지표가 하나도 없었다.**
M1/M2/M2b는 전부 free 기준이고 M3는 방위각마다 첫 경계 하나만 본다.

해결 방향은 라벨을 두껍게 만드는 것이 아니라 **지표에 거리 허용오차를 주는 것**이다. GT를
dilation해서 IoU를 안정화하면 지표 문제를 라벨 정의로 감추는 셈이 되고, GT 두께가 센서·맵
구축의 물리적 정의를 벗어난다.

    precision_τ = |{p ∈ O_pred : d(p, O_gt) ≤ τ}| / |O_pred|
    recall_τ    = |{g ∈ O_gt   : d(g, O_pred) ≤ τ}| / |O_gt|

**실측 (로봇 2 epoch fine-tune, **옛** val=rawos3 기준. split이 바뀌어
`val = raws1,rawos3`가 됐으므로 새 런과 직접 비교하지 않는다):**

| | 값 |
|---|---|
| `iou_occupied` | **0.063** |
| `f1@10cm` | 0.449 (precision 0.334 / recall 0.683) |
| `f1@20cm` | **0.665** (precision 0.565 / recall 0.808) |
| `f1@40cm` | 0.843 (precision 0.801 / recall 0.889) |

면적 IoU 0.063과 `f1@20cm` 0.665의 격차가 이 지표를 넣은 이유 그대로다 — "몇 셀 밀렸다"와
"장애물을 못 찾았다"를 면적 IoU는 구별하지 못한다. precision(0.565) < recall(0.808)이라
모델이 occupied를 과잉 예측하고 있다는 방향까지 읽힌다.

#### (b) M3의 표본 선택 효과 — `missed_obstacle_rate`가 필수다

회귀 통계는 GT와 예측이 **둘 다** `RAY_OK`인 광선만 쓴다(censored를 `r_max`로 대체해 섞으면
통계가 그 상수에 눌린다). 그래서 **장애물을 완전히 놓친 광선은 표본에서 빠지고, 많이 놓치면
`mae`가 오히려 좋아진다.** 이것은 이 프로젝트가 처음에 잡은 결함(`iou_drivable`이 레이아웃
prior를 재고 있었다)과 같은 부류다 — 분모/표본 선택이 지표를 실제 위험과 반대로 움직인다.

`missed_obstacle_rate` = (GT는 `RAY_OK`, 예측은 `RAY_CENSORED`) / (GT가 `RAY_OK`). 로봇
2 epoch에서 **0.069** (834/12070 광선)다. `RAY_NO_FREE` 예측은 세지 않는다 — 그건 "전부
막혔다고 봤다"는 과잉 보수로 비용의 종류가 정반대다. 로그에서 `mae` 바로 옆에 붙여 둔다.

#### (c) 부호 규약 — 문헌과 반대다

`dr > 0`은 **장애물을 실제보다 멀다고 예측 = free의 과대추정 = 위험한 쪽**이고 코드에서
`over_*`다. 여러 문헌은 같은 사건을 "장애물의 과소추정(over-estimation of free는 아님)"이라
부르므로 `over`/`under`라는 단어만 옮겨 읽으면 부호가 뒤집힌다. `bias = mean(dr)`이 그
방향을 한 숫자로 보여준다: SynWoodScape pretrain은 −0.794(보수적), 로봇 fine-tune은
+0.047(약간 낙관적)로 **두 데이터셋에서 부호가 반대**다.

#### (d) `iou_free_known` / `iou_occupied` / `iou_unknown` — 2026-08-21에 제거

**세 항목 모두 계산과 로깅에서 빠졌다.** 아래는 왜 추가했고 왜 뺐는지의 기록이다.
근거 실측은 `docs/finetune_overfitting_diagnosis.md` §23이다.

`iou_free_known`은 `valid`를 GT가 관측한 셀(free ∪ occupied)로 좁힌 `iou_free`였다.
추가한 당시의 관찰은 이랬다.

- SynWoodScape pretrain: `iou_free` 0.855 = `iou_free_known` 0.855 (완전 일치).
  `fatal_rate`가 0.000이면 `pred_free ⊆ gt_free`이므로 unknown 영역에 free 예측이 없어
  정의상 같아진다.
- 로봇 fine-tune: `iou_free` 0.751 대 `iou_free_known` 0.882 (+0.131).

**뺀 이유는 task 정의와 충돌하기 때문이다.** (D) binary 정식화에서 drivable은 "보이면서
빈 곳"뿐이고 **보이지 않는 곳은 전부 non-drivable**이다(`binary_metrics.py`). 그런데 로봇
데이터에서 `unknown`은 valid 셀의 **78.5 %**다(§23.1). `unknown`을 채점에서 빼면 라벨이
non-drivable이라고 선언한 셀의 대부분을 무료로 넘기는 것이므로, 정의된 task가 아닌 다른
task를 재게 된다. 실제로 로봇 fine-tune에서 모델이 `unknown` 안에 칠한 free 셀의 **57.3 %는
amodal 주석 기준으로도 obstacle**이었다(§23.2) -- 즉 그 벌점은 절반 이상 정당했다.

`iou_occupied`/`iou_unknown`은 binary에서 둘 다 `free`의 결정론적 함수다(`occupied`는 예측
free의 경계에서 유도되고 `unknown`은 그 나머지다). 게다가 두께 1셀 표면의 면적 IoU는 한 칸
밀리면 반토막 나서 val 0.071까지 떨어졌다 -- 품질 신호로 읽을 수 없는 숫자다. 경계 정밀도는
위 (a)의 `f1@τ`가 재고, 그쪽이 이 역할을 완전히 대체한다.

#### (e) 채택하지 않은 후보

| 후보 | 판정 |
|---|---|
| `RMSE_range` | **거부.** "큰 오차 벌주기"는 `abs_p90`이 이미 하고, censoring으로 표본 구성이 흔들리는 상황에서 RMSE는 불안정하다. 둘을 다 두면 서로 다른 말을 할 때 판단 규칙이 없다 |
| Chamfer distance | **거부.** EDT에서 공짜로 나오지만 상한이 없고 한쪽 집합이 비면 정의가 깨진다. `f1@τ`가 같은 정보를 유계로 준다 |
| HD95 | **보류.** 역시 공짜지만 이 숫자로 바꿀 행동이 없다. 필요해지면 1줄 |
| occupied PR curve / AP | **보류.** 결정 규칙이 3-class argmax이고 배포도 그렇다. threshold 스윕은 배포와 다른 것을 잰다 |
| GT dilation | **거부.** 지표 문제를 라벨 정의로 감추는 것 (위 (a)) |

#### (f) 집계 규칙 — batch-size 불변이 계약이다

`f1@τ`는 **카운트를 모아 마지막에 한 번 나눈다**(micro-average). 배치별 F1을 평균하면
프레임당 occupied 셀 수가 수십 배 차이 나므로 batch 크기에 따라 값이 달라진다. 백분위수에서
이미 같은 함정을 한 번 밟았다(§9.4). `missed_obstacle_rate`도 합산 카운트에서 재계산한다.
거리별 층화는 **`r_gt`로** 묶는다 — `r_pred`로 묶으면 구간 정의가 모델에 따라 움직여 run 간
비교가 무의미해진다.

---

## 3. 흔히 놓치는 함정

| 함정 | 증상 | 대응 |
|---|---|---|
| `occ_pred`를 단독으로 시각화 | 그림자 영역이 그럴듯하게 채워져 보임 | conv의 매끄러움 prior에 의한 **외삽**. 검증된 적 없고 캘리브레이션도 안 됨. 항상 `occ_p × vis_final`을 본다 |
| 전 격자 mIoU 리포트 | 숫자가 높게 나옴 | 모델의 perception이 아니라 **레이아웃 prior**를 측정한 것 |
| 정적 사각지대 포함 평가 | visibility 지표가 매우 좋음 | 상수 영역이라 공짜로 맞음. 제외 |
| 얇은 obstacle 껍질 | 예측 노이즈로 빈틈 발생, planner가 통과 경로를 찾음 | costmap inflation 반드시 적용. radius를 넉넉히 |
| 학습 데이터가 전부 통로 안 | 모델이 "좌우는 unknown"을 prior로 학습 | 위치가 아니라 **가시성 패턴 기준**으로 샘플링 균형 |
| 조작자가 매 프레임 후방에 등장 | 모델이 "뒤에 항상 사람"을 상수로 출력 | 후방 카메라 제외 + ego_mask에 조작자 영역 포함 |
| 수집 릭과 배포 로봇의 카메라 높이/pitch 상이 | 배포 시 성능 급락 | extrinsic 기록, augmentation, 또는 extrinsic-conditioned 아키텍처 |

---

## 4. 개발 순서

### Phase 1 — 베이스라인
- [ ] plain BCE, `lambda_vis=0.5`, `neg_weight=3.0`
- [ ] `iou_drivable` (거리 3구간) + `fatal_rate` 만으로 사이클 돌리기
- [ ] `l_occ` / `l_vis` 스케일 로깅, λ 조정
- [ ] 매 에폭 정성 샘플 저장: `image` / `occ_p` / `vis_p` / `occ_p × vis_p` 4분할

### Phase 2 — 진단
- [ ] `false_high` / `false_low` 추가
- [ ] 상황별 슬라이스 집계 (§2.5)
- [ ] 통로 폭 오차 추가
- [ ] 실패 케이스에 따라 §1.6에서 항 하나씩 추가

### Phase 3 — 배포 전
- [ ] `traj_recall` 로 라벨 파이프라인 자체를 교차 검증
- [ ] `fatal_on_path` 로 경로 상 안전성 확인
- [ ] 롤링 로컬맵 융합 후 end-to-end 주행 테스트
- [ ] 접촉 방지는 근접 센서(범퍼/ToF/초음파)로 별도 보장 — BEV perception에 접촉 거리 안전을 맡기지 않는다

---

## 5. 요약

**Loss**
- occupancy = `vis_gt` 가중 BCE, 유효 셀로 정규화
- visibility = 비대칭 가중 BCE, 전 격자
- 시야 제한은 loss weight에서만. 라벨은 참값 그대로 보관

**평가**
- 마스크는 `vis_gt` 기준. 정적 사각지대·ego 제외
- 거리별 분해 필수. 단일 mIoU는 정보가 없다
- `fatal_rate`가 실질적 배포 판단 기준
- `traj_recall`이 라벨과 모델을 동시에 검증하는 유일한 축

**태도**
- 숫자로 안 보이는 실패(껍질 끊김, 통로 조기 종료, 유령 장애물)가 실제 사고 원인이다. 정성 검증을 매 에폭 유지한다.