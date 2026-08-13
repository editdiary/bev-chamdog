# BEV Occupancy + Visibility 모델: Loss 및 평가 지표 설계

> 대상: 단일/다중 카메라 입력 → BEV 2채널 출력 (occupancy, visibility) 모델
> 환경: 수직 재배 온실, 좁은 작물 행 통로, map-free local perception

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