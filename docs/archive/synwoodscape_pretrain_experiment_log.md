# SynWoodScape Pretraining Experiment Log

작성: 2026-08-14

이 문서는 SynWoodScape two-head pretraining에서 **실제로 돌린 실험과 그 결과, 그리고 거기서
얻은 판단 근거**를 남긴다. 목적은 "어떤 수치가 나왔다"가 아니라 **왜 그 결론에 도달했고, 무엇을
하지 않기로 했는지**를 재현 가능하게 기록하는 것이다. 같은 실험을 다시 돌리지 않기 위한 문서다.

실행 방법과 지표 정의는 `docs/archive/training_guide.md`, 다음 세션 인수인계는
`docs/archive/next_session_synwoodscape_twohead.md`를 본다.

---

## 0. 이 로그를 읽는 법

### 지표 규약이 중간에 두 번 바뀌었다

수치를 가로로 비교하기 전에 반드시 확인해야 한다.

| 시점 | 규약 | 영향 |
|---|---|---|
| ~ 2026-08-14 13:00 | obstacle IoU에 GT obstacle 없는 샘플이 0점으로 포함 | val obstacle IoU가 실제보다 약 0.10 낮게 보고됨 |
| 2026-08-14 13:45 ~ | 해당 샘플 제외 (commit 45bec64) | 이후 런과만 비교 가능 |
| 2026-08-14 14:09 ~ | `valid`/`vis` 분리 (commit d0e089a) | visibility 지표가 비로소 의미를 가짐. occupancy 지표는 불변 |

### 노이즈 바닥

동일 설정 반복 런으로 실측했다. **이 값보다 작은 변화는 신호가 아니다.**

| 지표 | 노이즈 바닥 | 근거 |
|---|---|---|
| `val_iou_obstacle` (GT 마스킹) | **약 0.002** | 첫 런 0.7512 vs 재현 런 0.7491 (동일 설정) |
| `val_iou_mean` | 약 0.001 | 같은 두 런 0.8699 vs 0.8689 |
| `deploy_iou_obstacle` | **약 0.01** | 세 런에서 GT 마스킹 대비 -0.0097 / -0.0004 / 0.0000 |

`deploy` 지표가 GT 마스킹보다 5배 노이즈가 크다. 두 head의 오차가 곱해지기 때문으로 보인다.
**`deploy`에서 0.01 미만 변화는 읽지 않는다.**

---

## 1. 전체 런 목록

| # | 런 이름 | 조건 | best ep | val obst IoU | train obst IoU | 격차 |
|---|---|---|---:|---:|---:|---:|
| 1 | `twohead_pretrain_..._224942` | 최초 baseline | 41 | 0.751\* | — | — |
| 2 | `..._diag_baseline_..._130110` | 진단 지표 추가 | 52 | 0.749\* | 0.777\* | 0.028\* |
| 3 | `..._fixed_iou_..._134525` | empty-GT 수정 | 53 | 0.8539 | 0.9703 | 0.1164 |
| 4 | `..._vis_fixed_..._140932` | + `valid`/`vis` 분리 | 47 | 0.8544 | 0.9575 | 0.1031 |
| 5 | `..._photo_aug_..._150556` | + photometric aug | 46 | **0.8607** | 0.9556 | 0.0949 |
| 6 | `..._wd1e-4_..._154937` | + `weight_decay=1e-4` | 47 | 0.8583 | 0.9529 | 0.0946 |
| 7 | `..._wd1e-3_..._161441` | + `weight_decay=1e-3` | 43 | 0.8601 | 0.9521 | 0.0920 |

\* 구 규약. 3번 이후와 비교 불가.

전 런 공통: `res101`, `bs16`, `lr 3e-4`, 60 epoch, `val_fraction=0.2` (train 400 / val 100),
`split_seed=0`, `use_fisheye=True`, `lambda_vis=0.5`. 한 런당 약 24분 (GPU 1).

---

## 2. 실험별 상세

### 실험 A — 진단 지표 도입 (런 2)

**가설**: obstacle IoU가 낮은 이유가 "정말 못 맞혀서"인지 "obstacle 영역이 작아서 지표가
민감한 것"인지 구분되지 않는다. GT obstacle 비율 bin별 IoU와 오차 방향을 나눠 보면 갈릴 것이다.

**결과**: 갈렸다. 그런데 의도치 않게 **지표 자체의 결함**이 먼저 드러났다.

```
val_obst_iou_bins empty:0.000/n14 tiny:0.552/n7 small:0.636/n10 medium:0.834/n20 large:0.950/n49
train   missed_obstacle 0.0003   false_obstacle 0.0014   ← 사실상 완벽
train   empty bin 0.000/n81                              ← 그런데 0점
```

train에서 오차가 거의 0인데 empty bin이 0.000이면, 그 bin은 모델 성능을 재는 게 아니다.

**인사이트**: 진단 지표의 첫 번째 효용은 모델을 진단하는 게 아니라 **지표를 진단하는 것**이었다.
새 지표를 붙일 때는 "이 값이 완벽한 모델에서 얼마가 나와야 하는가"를 먼저 확인해야 한다.

---

### 실험 B — empty-GT 샘플 제외 (런 3, commit 45bec64)

**문제**: `compute_iou`는 `intersection / (1e-4 + union)`이다. GT에 obstacle이 하나도 없는
샘플은 `intersection`이 항상 0이라, **obstacle 없음을 완벽히 맞혀도 IoU가 0**이다.
val 100개 중 14개, train 400개 중 81개가 해당한다.

**수정**: obstacle이 존재하는 샘플로만 평균. 해당 샘플은 `empty_false_alarm`(obstacle 없는
장면에서 obstacle로 오탐한 valid 셀 비율)으로 따로 본다.

**결과** — 같은 가중치, 측정만 바뀜:

| | 수정 전 보고값 | 실제 |
|---|---:|---:|
| train obstacle IoU | 0.777 | 0.969 |
| val obstacle IoU | 0.749 | 0.854 |
| train/val 격차 | 0.028 | **0.116** |

**인사이트**: 이게 이번 세션에서 가장 중요한 발견이다. 아티팩트가 **과적합을 가리고 있었다.**
격차가 0.028로 보일 때는 "일반화 잘 되는 모델"이었고, 0.116이 드러나자 문제의 성격이 완전히
바뀌었다. 이후 모든 판단(Dice 기각, augmentation 우선순위, weight decay 시도)이 이 숫자에서
출발한다. **지표를 고치기 전의 모든 개선 계획은 잘못된 문제를 풀고 있었다.**

`empty_false_alarm`은 실측 0.0000~0.0002였다. 즉 모델은 이 14개 샘플에서 원래부터 완벽했다.

---

### 실험 C — `valid`/`vis` 분리 (런 4, commit d0e089a)

**문제**: 데이터셋이 `vis_bev_g`와 `valid_bev_g`에 **같은 배열**을 넣고 있었다.
`compute_two_head_loss`는 visibility loss를 `asymmetric_weight * valid`로 마스킹하므로,
`valid == vis_gt`이면 invisible 셀의 weight가 전부 0이 된다.

**실측 확인**:

```
vis_neg_weight=   0.0 -> loss_vis=0.805862
vis_neg_weight=   3.0 -> loss_vis=0.805862     ← 파라미터가 아무 효과 없음
vis_neg_weight= 100.0 -> loss_vis=0.805862
always-visible logit -> loss_vis=0.00000000    ← 자명해가 정확히 전역 최적
(not gt_visible) and valid 셀 개수: 0          ← false_high 분모가 공집합
```

visibility head는 **상수 함수를 학습**하고 있었고, `vis_false_high`는 예측과 무관하게 항상
0.000이었다. 런 1~3의 "visibility는 이미 잘 된다"는 판단은 근거가 없었다.

**수정**: `valid`는 "라벨이 존재하는 셀"(SynWoodScape는 ROI 전체 = 1), `vis`는 "관측 가능
여부". occupancy는 `vis * valid`, visibility는 `valid`로 마스킹.

**A/B 결과** (런 3 vs 런 4, occupancy 설정 동일):

| | visibility 죽음 | visibility 살아남 |
|---|---:|---:|
| val obstacle IoU | 0.8539 | 0.8544 |
| train obstacle IoU | 0.9703 | 0.9575 |
| 격차 | 0.1164 | 0.1031 |
| `vis_false_high` | 0.0000 (측정 불가) | 0.0245 |

**인사이트 1**: visibility를 진짜 task로 만드는 **비용이 0**이다. occupancy는 노이즈 바닥
안에서 동일하고, 격차는 오히려 줄었다(살아난 auxiliary task가 약한 정규화로 작용).
two-head 구조를 유지할 근거가 실측으로 확보됐다.

**인사이트 2**: 모델 코드(`simplebev_two_head.py`)는 처음부터 `valid ≠ vis`를 전제로 올바르게
작성돼 있었고, 기존 테스트 이름도 `..._trains_visibility_everywhere_valid`였다. **계약은
문서화돼 있었지만 데이터셋이 그것을 어겼고, 아무 테스트도 그 어김을 잡지 못했다.**
지금은 `test_valid_mask_covers_whole_roi_and_is_not_the_visibility_mask`가 잡는다.

---

### 실험 D — 배포 조건 지표 (런 4~, commit d0e089a)

**동기**: `iou_obstacle`은 **GT** visibility로 마스킹하므로 occupancy head 자체의 품질을 잰다.
실제 로봇은 GT가 없고 **모델이 "보인다"고 한 영역만** 신뢰하게 된다. 그 조건에서의 성능이
빠져 있었다.

**설계**: `deploy_iou_*`는 **예측** visibility로 마스킹하고 GT occupancy 전체를 정답으로 쓴다.
가려진 곳을 보인다고 착각하면 그 영역의 틀린 occupancy가 점수에 그대로 반영된다.
`deploy_visible_coverage`(모델이 보인다고 주장한 valid 셀 비율)를 **반드시 같이** 본다 —
시야를 좁게 부를수록 IoU는 공짜로 올라가기 때문이다.

**결과** (런 4): `deploy_iou_obstacle` 0.8540 vs GT 마스킹 0.8544, coverage 0.9370
(실제 visible 비율 약 0.946).

**인사이트**: 예측 visibility가 충분히 정확해서 **자기 예측으로 마스킹해도 손실이 없다.**
"occupancy/visibility를 나눈 설계가 실제 목적에 부합하는가"에 대한 직접적인 답이고, 긍정이다.

---

### 실험 E — photometric augmentation (런 5, commit 709ff36)

**동기**: 실험 B에서 드러난 0.10 격차. 확인해보니 파이프라인에 **augmentation이 전혀 없었다**
(리사이즈와 정규화뿐). train 400장에 증강 0이면 격차는 당연하다.

**설계 판단**:
- 이미지 공간 기하 변환은 제외했다. 거울 반사는 회전이 아니라 유효한 extrinsic으로 표현되지
  않고, fisheye `radial_poly`에서는 좌우 반전이 단순한 intrinsic 수정으로도 표현되지 않는다.
- 광도 변환만 적용: brightness/contrast/saturation ±20%, gamma 0.8~1.25, gaussian noise σ≤0.02.
- **4개 카메라에 동일 파라미터.** 카메라별로 다르게 걸면 실제 리그에 없는 카메라 간 색차를
  학습하게 된다.
- train split에만 적용. val은 항상 원본.
- Simple-BEV에는 광도 augmentation이 없어서(random resize+crop과 카메라 셔플뿐) 직접 구현했다.
  `albumentations`/`kornia`는 미설치, `torchvision`은 이미 의존성이라 새 패키지 추가 없음.

**결과** (런 4 vs 런 5):

| | 없음 | 있음 | 변화 |
|---|---:|---:|---:|
| val obstacle IoU | 0.8544 | 0.8607 | **+0.0063** (노이즈의 3배) |
| `tiny` bin | 0.5661 | **0.5935** | **+0.0274** |
| `small` bin | 0.6371 | 0.6458 | +0.0087 |
| `medium` bin | 0.8361 | 0.8448 | +0.0087 |
| `large` bin | 0.9475 | 0.9492 | +0.0017 |
| `missed_obstacle` | 0.0402 | 0.0380 | -0.0022 |
| `false_obstacle` | 0.0058 | 0.0056 | -0.0002 |
| train/val 격차 | 0.1031 | 0.0949 | -0.0082 |

**인사이트 1**: 효과가 있고, **가장 약한 bin에 가장 크게** 들어갔다(`tiny` +0.027).
`missed_obstacle`이 줄면서 `false_obstacle`이 늘지 않았으므로 단순히 보수적으로 변한 것도 아니다.

**인사이트 2 (더 중요)**: **격차는 거의 안 줄었다** (0.103 → 0.095). 이건 음성 결과가 아니라
진단 정보다 — **모델이 외우고 있는 것은 외형이 아니다.** 같은 종류의 증강을 더 세게 걸어도
소용없을 것이라는 예측이 여기서 나온다.

---

### 실험 F — weight decay 스윕 (런 6~7)

**가설**: `weight_decay=1e-7`은 Simple-BEV가 nuScenes(28k 샘플)용으로 쓰던 값을 그대로 물려받은
것이다. 400장에는 사실상 정규화가 없는 것과 같으므로, 상식적인 범위(1e-4~1e-3)로 올리면
격차가 줄 것이다.

**결과**: **아무 일도 일어나지 않았다.**

| | 1e-7 | 1e-4 | 1e-3 |
|---|---:|---:|---:|
| val obstacle IoU | 0.8607 | 0.8583 | 0.8601 |
| train obstacle IoU | 0.9556 | 0.9529 | 0.9521 |
| 격차 | 0.0949 | 0.0946 | 0.0920 |

4자리수를 움직였는데 val 편차가 0.0024로 노이즈 바닥 수준이다.

**인사이트**: 남은 과적합은 **L2 정규화가 다루는 종류가 아니다.** 가중치 노름의 문제가 아니라
데이터 커버리지의 문제라는 것이 이걸로 좁혀졌다. `weight_decay`는 근거가 없으므로 1e-7 유지.

부수 효과로 `deploy` 지표의 노이즈 바닥을 얻었다(§0).

---

## 3. 하지 않기로 한 것과 그 근거

### Dice loss — 기각

**원래 근거**: `tiny`/`small` bin이 약하니 작은 영역의 손실 기여를 키우면 나아질 것이다.

**기각 이유**: bin별로 train과 val을 나눠 보니 근거가 무너졌다.

| bin | train | val | 격차 |
|---|---:|---:|---:|
| tiny | 0.7796 | 0.5935 | 0.186 |
| small | **0.9087** | 0.6458 | **0.263** |
| medium | 0.9670 | 0.8448 | 0.122 |
| large | 0.9879 | 0.9492 | 0.039 |
| `missed_obstacle` | **0.0008** | 0.0380 | 47배 |

**모델은 train에서 작은 장애물을 이미 거의 다 맞힌다.** train `loss_occ`는 0.0013까지 떨어졌다
(val 0.029, 20배 차이). Dice의 작동 원리는 학습 목적함수에서 작은 영역의 기여를 키우는 것인데,
**재분배할 학습 오차가 남아 있지 않다.** `tiny`/`small`의 약점은 최적화 실패가 아니라 일반화
격차다.

**두 번째 독립적 이유**: pretrain은 `obst_frac 0.125`로 obstacle이 소수 클래스인데, 온실
fine-tuning에서 분포가 달라지면 obstacle 전용 Dice는 따라가지 못한다. BCE는
`compute_pos_weight`가 split마다 `neg/pos`를 실측하므로 자동으로 따라가지만 단일 클래스
Dice는 못 따라간다. 굳이 넣는다면 클래스 대칭 형태여야 하지만, 첫 번째 이유 때문에 지금은
넣을 근거 자체가 없다.

> **[2026-08-14 정정]** 이 문단은 원래 "온실 fine-tuning은 반대다(obstacle 다수, drivable
> 소수)"라고 적었는데 **틀렸다.** 자체 데이터셋 실물 측정 결과, 그리드 전체 obstacle 비율은
> 17% → 23%로 늘지만 **loss가 보는 마스킹 영역 안에서는 12.4% → 5.4%로 줄어든다.** 온실
> 통로에서 raycast visibility가 첫 장애물에 닿으며 멈추므로 장애물 대부분이 관측 영역 바깥에
> 놓이기 때문이다. Dice 기각 결론과 "단일 클래스 Dice는 분포 변화를 못 따라간다"는 논지는
> 그대로지만, **방향에 대한 예측은 반대였다.**
>
> 교훈 하나가 더 붙는다: **"분포가 어떻게 바뀔 것이다"는 예측은 실측 전까지 논거로 쓰지 말
> 것.** 그리고 클래스 비율은 **loss가 실제로 보는 마스크 안에서** 재야 한다 — 그리드 전체로
> 재면 이 경우처럼 부호가 뒤집힌다.

### BEV flip — 기각했다가 **정정**, 현재는 미시도 상태의 유효한 선택지

처음에 `nets/segnet.py:136-139`의 `bev_flip_indices` 훅만 떼어 보고 "decoder 맨 끝, head
직전이라 head conv만 증강된다"고 판단해 기각했다. **이 판단은 틀렸다.**

그 훅은 독립적인 증강이 아니라 **되돌리는 쪽 절반**이다. `Segnet(rand_flip=True)`일 때
전체 동작은 flip → 처리 → unflip이다:

| 위치 | 동작 |
|---|---|
| `segnet.py:401-404` | encoder **입력 이미지**를 좌우 반전 |
| `segnet.py:405-406` | encoder **출력 feature**를 다시 반전해 원위치 |
| `segnet.py:428-431` | `feat_mem`(BEV voxel feature)을 X축·Z축으로 반전 — `bev_compressor` **이전** |
| `segnet.py:136-139` | decoder 끝에서 되돌림 |

따라서 증강 표면은 **encoder + bev_compressor + decoder 본체 전체**다. 그리고 이미지 반전이
encoder 직후에 상쇄되므로 projection은 원본 좌표계를 받고 **calibration은 전혀 개입하지
않는다.** GT도 손댈 필요가 없다(출력을 되돌리므로).

수학적으로 이건 데이터 증강이라기보다 **거울 등변성(equivariance) 정규화**다. conv는 mirror
등변이 아니므로, 뒤집어 넣고 되돌린 출력이 원래 출력과 같아지도록 강제하면 네트워크가 방향에
무관한 표현을 학습하게 된다. BEV 레이어 입장에서는 레이아웃 다양성이 늘어나는 효과다.

`train_nuscenes.py:272`에서 upstream 기본값은 `rand_flip=True`다. "원저자도 안 쓴다"고 한
것도 틀렸다 — **항상 쓰고 있었고, 우리만 `False`로 넘기고 있었다**
(`tools/train_synwoodscape.py:529`).

**아직 실험하지 않았다.** 진단(레이아웃 다양성 부족)과 정확히 맞는 후보이고 플래그 하나면
되지만, 검토해야 할 점이 둘 있다:

1. `feat_mem` 반전은 X축뿐 아니라 **Z축(전후)에도** 걸린다. 우리 리그는 FV/RV가 다르고
   ROI도 전방 8 m / 후방 4 m로 비대칭이라, 전후 등변성은 좌우만큼 자연스럽지 않다.
   `TwoHeadSegnet`에서 X축만 남기도록 재정의할 수 있다.
2. `self.rand_flip`은 `self.training`으로 gate되지 않아 **val에서도 반전이 일어난다.**
   기하적으로는 상쇄되지만 네트워크가 완전 등변이 아니므로 val 지표에 노이즈가 낀다.
   채택한다면 `self.training`으로 막아야 한다.

*물리적으로 올바른 전체 장면 미러링*(4개 이미지 전부 반전 + `cx' = W - cx` + MVL↔MVR 슬롯
교환 + extrinsic 횡방향 반사 + BEV GT 반전)은 별개의, 더 무거운 선택지다. `rand_flip`이
calibration을 우회하는 것과 달리 이쪽은 캘리브레이션을 실제로 다시 계산한다. `rand_flip`을
먼저 시도해 보고 판단하는 편이 순서상 맞다.

**교훈**: 코드를 인용해 결론을 내릴 때는 그 코드를 **호출하는 쪽**까지 봐야 한다. 훅 하나만
보고 "표면이 좁다"고 단정했는데, 실제로는 훅이 더 큰 메커니즘의 마지막 단계였다.

### weight decay 조정 — 기각

실험 F 참고. 근거 없이 하이퍼파라미터를 추가하지 않는다.

---

## 4. 종합 진단

세 갈래의 **서로 독립적인** 증거가 같은 결론을 가리킨다.

| 증거 | 배제되는 원인 |
|---|---|
| photometric 증강이 격차를 0.103 → 0.095밖에 못 줄임 | 외형(appearance) 암기 |
| weight decay 4자리수 변화에 무반응 | 가중치 노름 / L2로 잡히는 복잡도 |
| train small bin 0.909, `missed_obstacle` 0.0008 | 최적화 실패 / loss 설계 문제 |

남는 것은 **train 400장의 장면 레이아웃 다양성 부족**이다. 이건 loss나 정규화로 푸는 문제가
아니라 데이터 또는 레이아웃 증강의 문제다.

**그리고 이 단계에서는 더 풀 필요가 없다.** pretraining의 목적은 SynWoodScape 점수 최대화가
아니라 fine-tuning에 넘길 쓸만한 초기 가중치를 만드는 것이다.

---

## 5. 방법론적 교훈 (다음 단계에도 적용)

1. **새 지표는 모델보다 먼저 자기 자신을 검증받아야 한다.** "완벽한 모델이면 이 값이 얼마인가"를
   확인한다. empty bin은 완벽한 모델에서도 0이었다.
2. **train 지표를 항상 val과 같이 본다.** empty-GT 결함도, Dice 기각 근거도, 둘을 나란히 놓고
   나서야 보였다. val만 보면 "작은 장애물을 못 배운다"로 오진한다.
3. **노이즈 바닥을 먼저 실측한다.** 동일 설정 2회 반복이면 충분하다. 이게 없으면 0.002짜리
   변화를 개선으로 보고하게 된다. 지표마다 바닥이 다르다는 것도 확인됐다(deploy는 5배).
4. **음성 결과를 근거와 함께 남긴다.** weight decay는 누구라도 다시 시도해볼 만한 아이디어다.
   기록이 없으면 다음 세션이 24분을 다시 쓴다.
5. **계약은 문서가 아니라 테스트로 지킨다.** `valid`/`vis` 계약은 docstring과 테스트 이름에까지
   적혀 있었지만 데이터셋이 어겼고 아무도 못 잡았다.
6. **한 번에 한 변수만 바꾼다.** 그래서 weight decay 스윕도 `--augment=True`를 유지했다.

---

## 6. 산출물

**pretrain 체크포인트** (fine-tuning 초기값 후보):

```text
runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth
```

val: obstacle IoU 0.8607 / drivable IoU 0.9889 / `missed_obstacle` 0.0380 /
`deploy_iou_obstacle` 0.8510 @ coverage 0.9369.

런 7(`wd1e-3`, epoch 43)도 노이즈 범위 내 동률이라 어느 쪽을 써도 무방하다. 근거 없는 변경을
피하는 원칙에 따라 런 5를 기본으로 둔다.

**후속 — 이 체크포인트가 실제로 어떻게 전이됐나 (2026-08-14)**

`configs/train_robot_bev_finetune.sh`의 기본 `INIT_CHECKPOINT`가 이것이다.

- **이식은 무손실이다.** 240×240 4-cam → 120×120 3-cam 모델에 0 missing / 0 unexpected /
  0 shape mismatch로 로드된다. `Segnet`이 (Z, X)에 대해 완전 합성곱이고 카메라별 전용
  파라미터가 없어서다.
- **zero-shot 성능은 낮다.** 자체 데이터셋에서 fine-tuning 없이 drivable IoU 0.532 /
  obstacle IoU 0.051. 도메인 갭이 크다 — 시각화를 보면 **온실 바닥을 통째로 장애물로
  예측**한다. SynWoodScape의 drivable은 아스팔트 도로이고 흰색 온실 바닥은 그렇게 보이지
  않기 때문이다.
- 그럼에도 **38장 fine-tuning에서는 이 초기값 자체가 best checkpoint였다**(epoch 1).
  물량이 부족해 학습할수록 val이 나빠지는 구간이라, 초기 가중치의 품질이 그대로 남았다.

자세한 내용은 `docs/finetuning_guide.md` §8.

**커밋**:

```text
18d06a0 Record weight decay negative result, rule out Dice and BEV flip
709ff36 Add photometric augmentation for the camera images
af150b6 Document metric fixes and reprioritize improvement plan
d0e089a Separate valid mask from visibility, add deployment metrics
45bec64 Exclude obstacle-free samples from occupancy IoU
```

---

## 7. 3-class 진단용 pretrain (2026-08-18, 커밋 `ca954e3`)

**목적**: 본학습이 아니라 **어떤 지표로 best epoch을 고를지 결정하기 위한 진단 런**이다.
확장된 지표(occupied `f1@τ`, `range_mae`/`bias`, `missed_obstacle_rate`, 거리별 층화)가
실제로 epoch을 구별하는지 60 epoch 곡선으로 본다.

```text
run  threeclass_pretrain_photo_aug_res101_bs16_lr3e-04_260818_170231
400 train / 100 val | bs16 | lr 3e-4 | res101 | fisheye | augment=True | 60 epoch
33.5분 (평균 33.5s/epoch, 26.1~36.9s) | best epoch 48 (기준 `iou_free`)
trivial baseline 0.875 | constant-map baseline 0.866
```

### 7.1 `iou_free`의 문제는 "여유폭"이 아니라 **조기 포화**였다

이전 세션의 진단(`next_session_threeclass_training.md` §2.1(b))은 "constant-map baseline
0.866에 대해 여유폭이 +0.052뿐"이었는데, **그 숫자는 1~2 epoch 모델에서 나온 것이라 틀렸다.**
60 epoch 학습하면 `iou_free`가 **0.9862**까지 가고 여유폭은 **+0.120**이다.

진짜 문제는 다른 것이고, 확인됐다 — **2 epoch에 0.9617, 11 epoch에 0.9829**에 도달한 뒤
남은 49 epoch의 개선폭이 **+0.0033**이다. epoch 20~60 구간의 전체 spread가 **0.0043**으로,
이 프로젝트가 스스로 정한 노이즈 대역(0.02)의 **1/5**다. 즉 `iou_free`로 best epoch을 고르는
것은 실질적으로 임의 선택에 가깝다.

### 7.2 상대 변별력 — SynWoodScape에서는 **range 계열만** 구별한다

epoch 20~60 구간의 `spread / best` (지표 자기 스케일 대비 얼마나 움직이나):

| 지표 | spread | best | 상대 |
|---|---|---|---|
| `missed_obstacle_rate` | 0.0405 | 0.0218 | **186 %** |
| `range_mae` | 0.1309 | 0.0932 | **140 %** |
| `range_mae` 4–6 m | 0.1074 | 0.0966 | 111 % |
| `range_mae` 2–4 m | 0.0474 | 0.0708 | 67 % |
| `range_mae` 6–8 m | 0.0740 | 0.1929 | 38 % |
| `iou_occupied` | 0.0311 | 0.8289 | 3.8 % |
| `iou_free` | 0.0043 | 0.9863 | 0.4 % |
| `f1@20cm` | 0.0024 | 0.9873 | **0.2 %** |
| `f1@40cm` | 0.0015 | 0.9942 | 0.2 % |

`range_mae`는 0.2241 → 0.0932(epoch 20→52, 58 % 감소)로 계속 움직이는데 같은 구간에서
`iou_free`는 +0.2 %만 움직인다. 거리 지표의 변별력이 **두 자릿수 배** 크다.

### 7.3 `f1@τ`는 pretrain 선택 기준으로 쓸 수 없다 — 두 데이터셋의 `occupied`가 다른 물체다

로봇에서 `f1@20cm` 0.665 대 `iou_occupied` 0.063이었던 것을 근거로 `f1@20cm`을 선택 기준
후보로 올렸는데, **SynWoodScape에서는 0.9873으로 포화하고 spread가 0.0024뿐이다.** 원인을
실측했다:

| | occupied 비율 | frontier shell 비율 |
|---|---|---|
| SynWoodScape (60프레임) | 0.121 | **0.0135** |
| 로봇 (267프레임) | 0.060 | **0.998** |

**SynWoodScape의 `occupied`는 채워진 면적이고, 로봇의 `occupied`는 두께 1셀 표면이다.**
전자는 시맨틱 래스터화에서 나오고 후자는 raycast가 멈춘 자리다. 그래서 면적 IoU가
SynWoodScape에서는 0.827로 잘 작동하고 로봇에서는 0.063으로 무너진다. **occupied 계열 지표의
값은 두 데이터셋 사이에서 옮겨 읽을 수 없다.**

### 7.4 `abs_p50` / `abs_p90`은 양자화 때문에 순위를 매길 수 없다

epoch 20~60의 41개 값 중:

- `abs_p50`: **39개가 동일한 0.0500**, 나머지 0.075와 0.1 각 1개
- `abs_p90`: **34개가 동일한 0.2500**, 나머지 0.275(5) / 0.3(1) / 0.35(1)

광선의 반지름 표본 간격이 `step_cells=0.5 × 0.05 m = 0.025 m`라 백분위수가 그 격자 위의
값만 취한다. 세 개 남짓한 값만 갖는 지표로는 체크포인트를 고를 수 없다.
**`mae`를 추가한 것이 이 런에서 결정적이었다** -- 없었으면 range 계열 전체가 순위를 못 매겼다.

### 7.5 `range_mae`의 게이밍 위험은 이 런에서 실현되지 않았다

`range_mae`는 표본이 예측에 의존해(장애물을 놓친 광선이 `RAY_CENSORED`가 되어 빠진다)
"어려운 광선을 버리는" 체크포인트가 유리해질 수 있다는 것이 이론적 우려였다. 실측:

    Pearson r(range_mae, missed_obstacle_rate) = +0.915   (Spearman +0.682)

**둘이 함께 좋아진다.** 즉 이 런에서 모델은 광선을 버려서 거리 오차를 줄이는 것이 아니라
실제로 둘 다 개선했다. 위험은 여전히 구조적으로 존재하므로 `missed_obstacle_rate`를 항상
병기해 읽는다(로그 같은 줄에 있다).

### 7.6 val loss는 epoch 13부터 나빠지는데 기하 지표는 50까지 좋아진다

| | epoch 13 | epoch 60 |
|---|---|---|
| val loss | **0.1952** (최소) | 0.3385 (+73 %) |
| train loss | -- | 0.0231 (val의 1/15) |
| train `iou_free` | -- | 0.998 |
| val `range_mae` | ~0.24 | **0.0946** |

**클래스 가중 CE와 기하 지표가 "언제 멈춰야 하는가"에 대해 서로 다른 말을 한다.**
val `loss_occupied`가 1.7504(train 0.0738)까지 벌어지는데 `range_mae`는 계속 줄어든다.
결론 두 가지: (a) **val loss를 선택 기준으로 쓰면 안 된다** -- epoch 21을 고르는데 그 지점의
`range_mae`는 0.1444로 최적 대비 55 % 나쁘다. (b) 이월 안건인
`MAX_CLASS_WEIGHT`/loss 재설계(§1.7)에 이 과적합 양상을 근거로 추가한다.

### 7.7 그러나 **이 런에서는 어느 기준을 써도 결과가 같다**

val loss를 뺀 모든 후보가 epoch 44~52를 고르고, 그 체크포인트들은 서로 구별되지 않는다:

| epoch | 고르는 기준 | `iou_free` | `f1@20cm` | `range_mae` | `missed` |
|---|---|---|---|---|---|
| 44 | `f1@τ`, `iou_occupied` | 0.9862 | 0.9873 | 0.0965 | 0.0261 |
| 47 | `missed_obstacle_rate` | 0.9856 | 0.9869 | 0.0995 | 0.0218 |
| **48** | **`iou_free` (현행, 저장됨)** | 0.9863 | 0.9866 | 0.0942 | 0.0263 |
| 52 | `range_mae` | 0.9863 | 0.9868 | 0.0932 | 0.0242 |

차이가 전부 jitter 수준이다(`range_mae` jitter 0.0068, 48↔52 차이 0.0010).
**따라서 pretrain을 다시 돌릴 이유가 없고 `model_best-000000048.pth`를 그대로 쓴다.**
선택 기준 변경은 이 체크포인트를 위해서가 아니라 **앞으로의 런**을 위해 하는 것이다.

### 7.8 부수 관측

- **근거리는 사실상 완벽, 원거리가 병목.** `range_mae` 0–2 m는 **0.0101 m**(0.2셀)까지
  내려가는데 6–8 m는 0.2048 m(4셀)에서 멈춘다. **20배 차이.** BEV 변환이 거리에 따라 기하를
  얼마나 복원하는지가 그대로 보인다 -- 거리별 층화를 넣은 이유가 이것이다.
- **occupied precision은 오르고 recall은 내려간다** (20 cm 기준: precision 0.9730 → 0.9906,
  recall 0.9860 → 0.9832). 학습이 진행되며 occupied 예측이 보수적으로 변한다.
- `bias`가 epoch 60에서 **+0.010 m**로 거의 0이다(초기에는 음수 = 보수적). 편향이 사라지고
  분산만 남은 상태다.
