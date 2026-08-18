# Next Session Handoff: 3-class 본학습과 지표 수정

Last updated: 2026-08-18

> **이 문서를 새 세션에서 가장 먼저 읽는다.** 그다음
> [`docs/free_space_metric_migration.md`](free_space_metric_migration.md) §8–9를 읽으면 지금
> 상태의 근거가 전부 숫자로 있다.

## 한 줄 상태

**3-class 단일 head로 확정, 2-head 제거 완료. 양쪽 학습이 전체 규모로 돌아가는 것을 재검증했고
(2 epoch 실측), 코드 정독 문서화와 정리 작업 3건까지 끝냈다. 아직 본학습은 돌리지 않았다.**

## ▶ 지금 진행 중인 안건: 평가 지표 수정 — 남은 것은 §2.1(b) 하나다

§2.1(a) `range` 백분위수의 batch-size 의존은 **해결됐다**. 남은 것은 **(b) SynWoodScape에서
`iou_free`의 변별력이 좁아 pretrain의 best epoch 선택이 노이즈에 가까울 수 있다**는 문제이고,
**본학습 전에 결정해야 한다**(선택 기준을 나중에 고치면 pretrain을 다시 돌려야 한다).
논의에 필요한 숫자와 선택지는 전부 §2.1(b)에 있다. 아직 사용자와 결론을 내지 않았다.

## 브랜치와 git 상태

```bash
git branch --show-current
# feat/bev-free-space-task     <- merge/push는 사용자가 직접 한다 (AGENTS.md)
```

```text
27f555b Document the training pipeline and the open loss questions              <- HEAD
f766026 Clean up the three-class training path before the real run
1ec89ca Document the three-class training handoff for the next session
78e1fa1 Train the three-class formulation on both datasets and drop two-head
c306411 Record the three-class versus two-head comparison                       (Task 18)
```

워킹트리 clean. 테스트: `python -m pytest tests/ -q` -> **233 passed**, 실패 0.

`runs/`에는 `_archive_2-head/`만 있다 -- 본학습 결과물은 아직 하나도 없다.

## 지금 바로 돌릴 수 있는 것

```bash
conda activate bev-chamdog

# 1) 3-class pretrain -- 약 33분 (400 train / 100 val, bs16, 60 epoch, 33.2s/epoch 실측)
bash configs/train_synwoodscape_threeclass_pretrain.sh 2>&1 | tee runs/threeclass_pretrain.log

# 2) 위에서 나온 best 체크포인트로 fine-tuning -- 약 8분 (190 train / 37 val, bs8, 60 epoch)
INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \
  bash configs/train_robot_bev_finetune.sh 2>&1 | tee runs/threeclass_finetune.log
```

GPU는 **`CUDA_VISIBLE_DEVICES=0`**을 쓴다 (config 기본값이 이미 0이다). GPU1은 다른 프로세스가
90GB 가까이 점유하고 있어 쓰면 OOM이 난다.

**[2026-08-18 갱신] GPU0도 더 이상 비어 있지 않다.** 다른 사용자의 학습(chamnet)이 GPU0에서
35~56GB를 쓰고 있고, 우리 pretrain은 48GB를 쓴다 -- 합산 84GB/97.9GB까지 올라간다. 실제로
돌아가는 것은 확인했지만 상대 job이 peak를 치면 OOM 위험이 있다. 같은 이유로 epoch 시간도
24.8s -> **33.2s**로 늘어, pretrain 60 epoch은 25분이 아니라 **약 33분**으로 봐야 한다.
(2 epoch 실행으로 실측: pretrain·fine-tuning 모두 정상 완주, fine-tuning의 weight transfer가
`loaded 668 tensors, skipped 0` -- 3-class pretrain 체크포인트는 출력 head까지 전이된다.
§8.4가 A/B의 최대 교란으로 지목한 head 전이 비대칭이 본학습에서는 사라진다는 뜻이다.)

---

## 1. 이번 세션(2026-08-17~18)에 실제로 한 일

### 1.1 Task 17 -- 4샘플 overfit 게이트 (커밋 `6d0febf`)

3-class head/loss/라벨 변환 배선이 학습 신호를 흘리는지 확인하는 게이트. 세 구성 전부 통과:

| init | epochs(=steps) | 0.98 도달 | 최종 train `iou_free` |
|---|---|---|---|
| pretrain trunk | 2000 | epoch 99 | **1.000** |
| pretrain trunk | 200 | epoch 82 | 0.989 |
| from scratch (계획서 원본 명령) | 200 | epoch 50 | 0.994 |

2400 epoch 전체에서 `partition` 경고 0건, `loss_occ` 23.71 -> 0.096(occupied 클래스가 죽지 않았다).

**실행 전 내가 한 판단이 틀렸던 것을 기록해 뒀다**(§7.4): "200 step으로는 도달 불가"라고 보고
2000 epoch으로 올렸는데, 200으로 충분했고 from scratch가 오히려 빨랐다. 4장 암기에서는 pretrain
가중치가 제약으로 작동한다. **재실행할 일이 있으면 계획서 원본 명령 그대로가 맞다.**

### 1.2 Task 18 -- 2-head vs 3-class A/B (커밋 `c306411`)

`--head`와 loss만 바꾸고 나머지를 전부 고정해 3-class만 새로 학습, Task 13의 2-head 기준선과 대조.
인자를 하나씩 대조해 조건 일치를 확인했다(§8.1).

| 지표 | 2-head (ep 42) | 3-class (ep 30) | 우세 |
|---|---|---|---|
| **`iou_free`** ↑ | 0.765 | **0.774** | 3-class (+0.009, **노이즈 대역 안**) |
| **`fatal_rate`** ↓ | **0.124** | 0.135 | **2-head** |
| `free_miss_rate` ↓ | 0.139 | **0.115** | 3-class |
| `range_abs_p50 / p90` ↓ | 0.183 / 0.725 | **0.172 / 0.718** | 3-class |
| ring 0–1.5 / 1.5–3 / 3–4 m ↑ | 0.807/0.757/0.713 | **0.810/0.766/0.735** | 3-class (멀수록 격차↑) |
| best epoch | 42 | **30** | 3-class |

**판정: 통과. 단 근거는 `iou_free` 우세가 아니다.** +0.009는 계획서가 정한 0.02 노이즈 대역 안이라
근거로 쓸 수 없다. 통과로 본 실제 이유 세 가지: (a) `fatal_rate` 하나 빼고 전 지표가 3-class 우세,
(b) 3-class는 출력 head가 랜덤 초기화라는 **핸디캡을 지고** 이겼다, (c) 더 빨리 도달했다.

**유일한 후퇴 `fatal_rate`는 노이즈로 넘기지 않았다.** §3-1이 실측으로 확정한 대로 occupancy head가
기여하는 유일한 지표가 그것이고, planner 안전 관점에서 비싼 오류다. `free_miss_rate`가 반대로
개선된 것과 합쳐 읽으면 free에 대한 precision/recall 교환이다.

### 1.3 3-class pretrain 지원 + 2-head 전면 제거 (커밋 `78e1fa1`, 사용자 지시)

사용자가 "앞으로 3-class로만 학습·테스트한다"고 결정 -> 2-head 전부 삭제를 선택했다.

**pretrain이 진짜 블로커였다.** `tools/train_synwoodscape.py`가 `TwoHeadSegnet`을 하드코딩하고
2-head `run_batch`의 9-튜플을 직접 언패킹해서, **3-class pretrain 자체가 불가능**했다. 전환에
모델 교체 외에 필요했던 것:

- `default_class_weights` -> **`class_weights_from_labels(label_triples)`**: 옛 시그니처가 로봇
  전용이었다(`permanent_blind`/`rear_self_box`/로봇 loader). SynWoodScape는 그 개념이 없고
  `valid`가 전부 1이다. `(occ, vis, valid)` iterable만 받도록 일반화.
- **링 경계 분리**: pretrain 그리드는 전방 8 m, 로봇은 4 m. 로봇 기본값(0/1.5/3/4)이면 바깥
  절반이 어느 링에도 안 들어간다 -> `PRETRAIN_RING_EDGES_M = (0,2,4,6,8)`.
- **constant-map baseline 추가**: pretrain에는 `iou_free`의 트리비얼 baseline이 없었다. 이
  프로젝트의 출발점이 "모델이 블라인드 예측기에 지는데 아무도 몰랐다"(§1)라서 넣었다. 즉시 작동했다.
- **`evaluate_split` 공유**: 두 trainer의 val 루프가 거의 같아져 공유 모듈로 올렸다.
- **loss 항 로깅 구멍 수정**: 옛 로그는 `loss_occ`/`loss_vis` 두 칸이었고 3-class를 거기 끼워
  넣고 있어서 **`loss_unknown`이 아예 출력되지 않았다** -- unknown이 셀의 85% 이상인 데이터에서.
  이제 세 항이 다 찍힌다.

`bev_occupancy_metrics.py`는 593 -> 323행. **이름은 그대로 뒀다** -- Task 16에서 방금 개칭한 것을
하루 만에 또 바꾸면 같은 모듈이 git 이력에서 세 이름을 갖게 되고, 남은 내용에 대해 이름이 맞다.

### 1.4 계약이 바뀐 API (다음 세션에서 코드를 읽을 때 주의)

| 옛 것 | 새 것 |
|---|---|
| `default_class_weights(samples, permanent_blind, invalid, load_labels)` | `class_weights_from_labels(label_triples)` |
| `select_checkpoint_score(d_iou=, o_iou=, free_metrics=)` | `select_checkpoint_score(free_metrics)` |
| `format_epoch_log(... train_occ_loss=, train_d_iou=, ...)` (17개 인자) | `format_epoch_log(... train_loss_parts=, ...)` (dict로) |
| 각 trainer의 로컬 `_evaluate` | `bev_occupancy_metrics.evaluate_split` (공유) |
| `tools/train_robot_bev.py --head=...` | 없음 (3-class 전용) |
| `configs/train_synwoodscape_twohead_pretrain.sh` | `configs/train_synwoodscape_threeclass_pretrain.sh` |
| `TwoHeadSegnet`, `compute_two_head_loss`, `split_two_head_logits`, `visibility_error_rates` | 삭제 |
| occupancy 진단(`iou_drivable`/`iou_obstacle`/obstacle bin), deployment 지표 | 삭제 |

### 1.5 2026-08-18에 추가로 정리한 것 (사용자 지시)

- **그리드 상수 통합.** `SYNWOODSCAPE_TWO_HEAD_PRETRAIN_GRID_SPEC`(8/4/±6, 240×240)이
  `SYNWOODSCAPE_PRETRAIN_GRID_SPEC`으로 개칭되고, 학습에 쓰이지 않던 옛 동명 상수
  (5/3/±4, 160×160)는 삭제됐다. 이름은 같고 값이 다른 상수가 둘 있어 혼동을 일으켰다.
  **학습 경로는 변화 없다**(원래부터 8/4/±6을 썼다). 다만 GT 생성 도구 5개
  (`build_occupancy_gt` 등)의 `--spec synwoodscape_pretrain` 출력이 160×160 -> 240×240으로
  바뀐다 -- 실제 라벨(`..._roi_8_4_6_h08`)과 이제 일치한다.
- **`save_freq_epochs`를 10으로 통일** (pretrain이 5였다). 둘 다 `keep_latest=3`이라 최종
  잔존 체크포인트 수는 원래 같았고, `model_best`는 별도로 항상 저장되므로 판정에 영향 없다.
- **`runs/`가 `runs/_archive_2-head/`로 옮겨졌다**(사용자가 정리). 그래서
  `configs/train_robot_bev_finetune.sh`의 기본 `INIT_CHECKPOINT`가 **죽은 경로**다 --
  `INIT_CHECKPOINT=`를 반드시 넘기거나, 본학습 때 그 기본값을 새 3-class 체크포인트로 갱신한다.
- **SynWoodScape는 500장이다** (train 400 / val 100). 이전 세션 기록의 "501"은 오기였다.
- **안 쓰는 decoder head 3개 제거** (`feat_head`/`instance_center_head`/`instance_offset_head`).
  nuScenes instance segmentation용이라 이 태스크에는 라벨도 loss도 없다. 파라미터 459,267개
  (decoder의 12.0%)와 decoder forward 시간 44%가 사라진다 -- 임베디드 배포가 주 동기다.
  weight transfer가 `loaded 677` -> `loaded 668`로 줄어든 것이 확인이고, 초기화 RNG 소비량은
  원본과 같아 초기 가중치는 바뀌지 않는다. 상세: `training_pipeline_walkthrough.md` §6.1.
- **`occ & vis & valid` 중복 제거.** 두 trainer, `class_weights_from_labels`,
  `rescore_checkpoints`, `measure_label_geometry`가 각자 조합하던 것을 전부
  `free_space.decompose()`로 모았다. **`occ=1`이 free라는 규약을 해석하는 지점이 이제 한 곳이다.**
  검증: 리팩터 전후로 클래스 가중치·trivial·constant baseline·라벨 통계가 **전부 비트 동일**.
- **[주의] 학습은 seed를 고정해도 비트 재현되지 않는다.** 같은 코드·같은 seed로 두 번 돌린
  결과가 val `iou_free` 0.930 vs 0.931, `range_abs_p90` 1.525 vs 1.600이었다. GPU 커널
  비결정성(`grid_sample` backward의 atomic 등)이다. §4.2의 "0.02 이내는 노이즈" 규칙이
  seed를 맞춰도 유효하다는 뜻이다.

---

## 2. 다음 세션에서 답을 내야 하는 것

### 2.1 [진행 중] 평가 지표 수정 -- (a) 해결, **(b)가 남았다**

**이것이 지금 논의 중인 안건이다.** 판단 기준은 하나다:

- **`iou_free`의 정의를 건드리는 수정이면 재학습이 필요하다.** Task 11에서 체크포인트 선택
  기준이 `iou_free`가 됐고, **pretrain과 fine-tuning 양쪽 모두** 그 기준을 쓴다. 정의가
  바뀌면 어느 epoch이 best로 뽑히는지가 달라지므로 재채점으로 복구되지 않는다.
- **`iou_free`를 그대로 두는 수정이면 재학습이 필요 없다.** `tools/rescore_checkpoints.py`로
  저장된 체크포인트를 재채점해 판정만 다시 내리면 된다. **사용자는 `iou_free` 정의를 유지한다고
  확인했다**(2026-08-17).

이미 발견돼 이 범위에 들어가는 후보가 **두 개** 있다:

**(a) [해결 2026-08-18] `range_abs_p50`/`abs_p90`의 batch-size 의존**

`summarize_range_error`가 배치별 백분위수를 가중평균하던 것을, **`dr` 표본을 전부 모은 뒤 한
번만** 통계를 내도록 고쳤다(`projects/common/free_space_metrics.py`). `over_mean`/`under_mean`도
같은 표본에서 직접 내므로 부분집합 가중 규칙이 사라졌다.

재채점 실측(수정 후): bs4/8/16에서 `abs_p50` 0.175, `abs_p90` 0.700으로 **완전히 일치**.
`over_mean`만 0.416 vs 0.417로 갈리는데, 이것은 집계가 아니라 모델 forward 탓이다 -- 같은
체크포인트를 bs4/bs8로 추론하면 argmax가 다른 셀이 532,800개 중 11개(0.002%) 있다(cuDNN
알고리즘 선택). 집계의 batch-size 불변성은 단위 테스트가 동일 표본에서 정확한 일치로 고정한다.

**주의: §9.4와 §8에 기록된 옛 range 숫자(0.172 / 0.718 등)는 옛 정의의 값이다.** 새 정의와
비교하려면 `tools/rescore_checkpoints.py`로 재채점해야 한다. `iou_free`는 안 건드렸으므로
체크포인트 선택과 다른 지표는 그대로다.

**(b) SynWoodScape에서 `iou_free`의 변별력이 매우 좁다 (2026-08-18 실측, 새 발견)**

| | constant-map baseline | 1 epoch 모델 | 여유 |
|---|---|---|---|
| 로봇 (`rawos3`) | 0.398 | 0.697 | +0.299 |
| **SynWoodScape** | **0.866** | 0.918 | **+0.052** |

ego 주변이 대체로 도로라 레이아웃 prior만으로 거의 맞는다. 그런데 **pretrain의 체크포인트 선택
기준도 `iou_free`**다 -- 0.866~1.0이라는 좁은 구간에서 best epoch을 고르는 것이 사실상 노이즈로
고르는 것에 가까울 수 있다. **이 프로젝트가 애초에 잡으려던 문제("지표가 레이아웃 prior를 재고
있다")와 같은 종류가 pretrain 쪽에 남아 있다는 뜻이다.**

**2026-08-18 추가 실측 -- 문제가 더 좁다:** pretrain을 2 epoch만 돌려도 val `iou_free`가
**0.951~0.952**에 도달한다(세 번 실행: 0.951 / 0.952 / 0.952). 즉 여유폭 0.052 중 대부분을
2 epoch에서 먹고, 남은 58 epoch은 0.95~1.0의 더 좁은 구간에서 best를 고른다. 그리고 같은
seed로도 재현이 안 된다(§1.5 마지막 항목: 같은 조건 두 번이 0.930 vs 0.931). **선택 폭이
run-to-run 잡음과 같은 자릿수일 가능성이 크다.**

**논의해야 할 선택지 (아직 결정 안 됨):**

1. **그대로 둔다.** pretrain은 fine-tuning의 초기값을 주는 것이 목적이므로 best epoch이 다소
   임의여도 무해할 수 있다. 근거를 만들려면 여러 pretrain epoch에서 fine-tuning을 돌려
   "pretrain `iou_free`가 fine-tune 결과와 상관이 있는가"를 봐야 한다(fine-tuning이 8분).
2. **pretrain의 선택 기준을 바꾼다.** 후보: `fatal_rate`를 함께 보는 복합 기준, baseline 대비
   초과분(`iou_free - constant_baseline`), 원거리 링(`ring 4-6m`/`6-8m`)의 `iou_free`
   -- 레이아웃 prior가 가장 안 통하는 구간이다.
3. **pretrain 그리드/데이터를 손본다.** free 비율 82%가 근본 원인이므로 ROI를 넓히거나 free가
   적은 장면을 고르면 변별력이 올라간다. 가장 비싸다(라벨 재생성).

주의: 2번이나 3번을 고르면 **본학습 전에** 해야 한다 -- pretrain의 체크포인트 선택 기준을
바꾸는 것은 `iou_free` 정의를 바꾸는 것과 같은 부류라 재채점으로 복구되지 않는다.
돌린 뒤에 고치면 pretrain을 다시 돌려야 한다(약 33분이라 비용 자체는 크지 않다).

### 2.2 [완료 2026-08-18] pretrain / fine-tuning 코드 파악

**결과물: [`docs/training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md).**
데이터 파일 -> 텐서 -> 모델 -> loss -> 지표 -> 체크포인트 선택을 코드 위치와 함께 따라간다.
§6에 읽다가 걸리는 것들(안 쓰는 decoder head의 실측 비용, `pix_T_cams`가 왜 무시되는지,
range 백분위수의 batch-size 의존), §7에 파일을 읽을 순서가 있다.

<details><summary>원래 계획했던 범위 (참고용)</summary>

사용자 요구: **"어떤 데이터셋을 어떻게 읽어서 어떤 과정을 거쳐 학습이 되고 평가가 무엇으로
되는지 구체적으로 파악한 다음 본격적인 학습을 시키고 싶다."**

내가 하겠다고 제안한 범위(사용자가 아직 수락/거절하지 않았다): 두 학습 경로 각각에 대해
**"데이터 파일 -> 어떻게 읽혀 -> 어떤 텐서가 되어 -> 어떤 loss로 학습되고 -> 무엇으로
평가/체크포인트 선택되는가"**를 실제 코드를 따라가며 문서화. 라벨 마스킹 계약(`vis=0` vs
`valid=0`), 어안 투영이 어디서 일어나는지, `permanent_blind`/`rear_self_box`가 언제 적용되는지 포함.

읽어야 할 파일:

| | pretrain | fine-tuning |
|---|---|---|
| 실행 | `configs/train_synwoodscape_threeclass_pretrain.sh` | `configs/train_robot_bev_finetune.sh` |
| 스크립트 | `tools/train_synwoodscape.py` (352행) | `tools/train_robot_bev.py` (364행) |
| 데이터 | `projects/datasets/synwoodscape_simplebev.py` | `projects/datasets/robot_simplebev.py` |
| 모델 | `projects/models/simplebev_three_class.py` (양쪽 공용) | 동일 |
| loss/지표 | `projects/common/three_class_metrics.py` (양쪽 공용) | 동일 |
| 로깅/선택 | `projects/common/bev_occupancy_metrics.py` (양쪽 공용) | 동일 |

</details>

### 2.3 [이월] loss / 클래스 가중치 -- 본학습 **뒤에** 볼 것

3-class loss의 클래스 가중치는 train split에서 역빈도로 자동 계산되는데, 상한
`MAX_CLASS_WEIGHT = 20`이 **근거 없는 임의값이면서 실제로 작동 중이다** -- 로봇 train split에서
`occupied`를 67.93 -> 20으로 자른다. loss 균형을 실질적으로 결정하는 하이퍼파라미터인데
검증된 적이 없고, 하필 `fatal_rate`(§8 A/B에서 유일하게 후퇴한 지표)와 직결된다.

**정본: [`docs/BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7**
(실측 분포표, Simple-BEV 원본과의 차이, 대안 목록, 스윕 계획). 코드 쪽 포인터는
`projects/common/three_class_metrics.py`의 `MAX_CLASS_WEIGHT` 주석.

**순서상 본학습 뒤다.** 기준 숫자가 없으면 스윕 결과를 판정할 대조군이 없다. 그리고 이것은
`iou_free`의 **정의**를 건드리지 않으므로 §2.1의 지표 수정과 독립적이다 -- 단 loss를 바꾸면
재채점이 아니라 재학습이 필요하다.

### 2.4 본학습 -- 위 두 개가 끝난 뒤

명령은 이 문서 맨 위 "지금 바로 돌릴 수 있는 것"에 있다. 내가 권장한 순서는 **지표 수정 -> 코드
파악 -> 본학습**이고, 근거는 §2.1(b)다(선택 기준을 나중에 고치면 pretrain 재실행). 사용자가
"숫자를 먼저 보고 판단"을 택할 수도 있다 -- 아직 정하지 않았다.

---

## 3. 실행 시 반드시 지켜야 할 것

- **`merge`와 `push`는 사용자가 직접 한다.** 에이전트가 임의로 하지 않는다.
- `third_party/`와 `mmdetection3d/`는 git submodule -- **직접 수정 금지.**
- GPU는 **`CUDA_VISIBLE_DEVICES=0`**. GPU1은 점유 중이다.
- **실측 없이 숫자를 적지 않는다.** 게이트/판정 결과는 명령 출력을 문서에 붙여 남긴다.
- 커밋 메시지는 **영어**, 코드 주석·문서는 **한국어**(주변 스타일).
- 로봇 split은 별도 지시 없으면 `train = raws1,raws2,raws3,rawos1,rawos4`, `val = rawos3`.

## 4. 판정을 읽을 때 계속 유효한 주의사항

1. **`iou_free`만 보면 안 된다. `fatal_rate`를 항상 같이 읽는다.** 실측 근거: 같은 체크포인트에서
   `vis` head 단독이 `iou_free` 0.851, 2-head 결합이 0.850 -- occ head는 `iou_free`에 기여하지
   않고 `fatal_rate`만 개선한다(0.085 -> 0.056). (§3-1)
2. **val 37프레임이다.** `iou_free` 차이 0.02 이내는 노이즈로 취급한다. §8의 A/B 판정은 데이터가
   늘어난 뒤 재확인해야 한다(§8.4).
3. **seed 하나로 낸 판정이다.** `torch.manual_seed(0)`이 하드코딩이라 다중 seed 비교에는 코드
   변경이 필요하다.
4. **2-head 기준선 0.765는 영구히 역사적 값이다.** 2-head 코드가 없어 재채점할 수 없다. 지표를
   바꿔도 다시 산출되지 않는다. 사용자가 이 손실을 알고 선택했다.
5. **`fatal_rate`는 스위트에서 가장 덜 검증된 지표다.** 스펙의 0.0587과 재현값 0.055681이
   상대 −5.1% 어긋나 있고 원본 스크립트가 없어 원인을 확정할 수 없다(§2-1). 판정을 가른 유일한
   후퇴 지표가 마침 이것이다.
6. **`iou_drivable`의 −0.001은 알려진 배치평균 아티팩트다.** 의도적으로 그대로 뒀다(§2-2).
   단 2-head 제거로 이 지표 자체가 사라졌으므로 이제는 과거 기록을 읽을 때만 필요하다.

## 5. 리뷰를 어떻게 걸어야 하는가 (이번 계획 전체에서 관찰된 것)

**Task 1–12의 리뷰 findings가 사실상 전부 "테스트는 통과하지만 그럴듯한 오류도 같이 통과시킨다"
유형이었고 구현 결함은 0건이었다.** 그래서 리뷰 프롬프트에 **"구현을 실제로 망가뜨려서 테스트가
진짜 실패하는지 확인하라"**를 계속 명시 요구했다. 추론만 한 mutation은 그렇게 표시하게 하고
실행한 것과 구분한다. **이 요구를 넣은 리뷰만 결함을 찾아냈다.**

게이트 조건도 만능이 아니다 -- Task 8에서 확인됐듯 Phase 1 게이트 3조건은 실제 배선 오류가
있어도 성립했다. 게이트는 타당성을 거르지 정확성을 보증하지 않는다.

## 6. 관련 문서

- **코드 정독 가이드**: [`docs/training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md)
  -- 데이터 파일 -> 텐서 -> 모델 -> loss -> 지표 -> 체크포인트 선택. §6에 함정, §7에 읽는 순서.
- **loss 미해결 항목**: [`docs/BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7
  -- `MAX_CLASS_WEIGHT=20`의 근거 없음, 실측 분포, 대안 목록. §1.1–1.6은 옛 2-head 설계다.
- **근거 정본**: [`docs/free_space_metric_migration.md`](free_space_metric_migration.md)
  -- §1 진단, §6 2-head 기준선, §7 Task 17, §8 A/B, §9 2-head 제거
- 계획 실행 기록: [`docs/superpowers/plans/2026-08-17-bev-free-space-task-progress.md`](superpowers/plans/2026-08-17-bev-free-space-task-progress.md)
  -- §3에 실행 중 정해진 결정들, §4에 이월 발견사항
- 설계 정본: [`docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md`](superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md)
- 라벨 계약 정본: [`docs/finetuning_guide.md`](finetuning_guide.md) §1.3
  -- `occupancy_3class_npy`는 **검수 전용**이고 학습 입력이 아니다. 라벨 정본은 `occupancy_npy` 하나다.
- 공통 작업 지침: [`AGENTS.md`](../AGENTS.md)
