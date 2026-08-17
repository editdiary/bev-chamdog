# Next Session Handoff: 3-class 본학습과 지표 수정

Last updated: 2026-08-18

> **이 문서를 새 세션에서 가장 먼저 읽는다.** 그다음
> [`docs/free_space_metric_migration.md`](free_space_metric_migration.md) §8–9를 읽으면 지금
> 상태의 근거가 전부 숫자로 있다.

## 한 줄 상태

**3-class 단일 head로 정식화가 확정되고 2-head는 코드에서 제거됐다. pretrain과 fine-tuning
양쪽 다 전체 규모로 실행 가능한 것이 실측 확인됐고, 아직 본학습은 돌리지 않았다.** 남은 것은
사용자가 예고한 두 가지 — 코드 파악과 평가 지표 수정 — 그리고 그 뒤의 본학습이다.

## 브랜치와 git 상태

```bash
git branch --show-current
# feat/bev-free-space-task     <- merge/push는 사용자가 직접 한다 (AGENTS.md)
```

```text
78e1fa1 Train the three-class formulation on both datasets and drop two-head   <- HEAD
c306411 Record the three-class versus two-head comparison                      (Task 18)
6d0febf Record the three-class overfit sanity check                            (Task 17)
a03eef1 Record Task 14 through Task 16 free-space progress
259b952 Switch between the two-head and three-class formulations at runtime
```

테스트: `python -m pytest tests/ -q` -> **229 passed**, 실패 0.
(계획 종료 시점 242에서 줄었다 -- 2-head 전용 테스트가 사라졌고 새 테스트가 일부 추가됐다.)

## 지금 바로 돌릴 수 있는 것

```bash
conda activate bev-chamdog

# 1) 3-class pretrain -- 약 25분 (400 train / 100 val, bs16, 60 epoch, 24.8s/epoch 실측)
bash configs/train_synwoodscape_threeclass_pretrain.sh 2>&1 | tee runs/threeclass_pretrain.log

# 2) 위에서 나온 best 체크포인트로 fine-tuning -- 약 5분 (190 train / 37 val, bs8, 60 epoch)
INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \
  bash configs/train_robot_bev_finetune.sh 2>&1 | tee runs/threeclass_finetune.log
```

GPU는 **`CUDA_VISIBLE_DEVICES=0`**을 쓴다 (config 기본값이 이미 0이다). GPU1은 다른 프로세스가
93GB를 점유하고 있어 쓰면 OOM이 난다.

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

---

## 2. 다음 세션에서 답을 내야 하는 것

### 2.1 [사용자 예고] 평가 지표 수정 -- **수정 내용을 아직 못 들었다**

이것이 다음 세션의 첫 안건이다. 판단 기준은 하나다:

- **`iou_free`의 정의를 건드리는 수정이면 재학습이 필요하다.** Task 11에서 체크포인트 선택
  기준이 `iou_free`가 됐고, **pretrain과 fine-tuning 양쪽 모두** 그 기준을 쓴다. 정의가
  바뀌면 어느 epoch이 best로 뽑히는지가 달라지므로 재채점으로 복구되지 않는다.
- **`iou_free`를 그대로 두는 수정이면 재학습이 필요 없다.** `tools/rescore_checkpoints.py`로
  저장된 체크포인트를 재채점해 판정만 다시 내리면 된다. **사용자는 `iou_free` 정의를 유지한다고
  확인했다**(2026-08-17).

이미 발견돼 이 범위에 들어가는 후보가 **두 개** 있다:

**(a) `range_abs_p50`/`abs_p90`이 batch size에 의존한다 (§9.4, 실측 확정)**

`summarize_range_error`가 배치별 백분위수를 `n_paired_rays`로 가중평균한다
(`projects/common/free_space_metrics.py:166`). 배치별 중위수의 가중평균은 전체 분포의 중위수가
아니다. 같은 체크포인트를 재채점한 결과:

| | 학습 로그(bs8) | 재채점 bs4 | 재채점 bs8 |
|---|---|---|---|
| `range_abs_p50` | 0.172 | **0.177** | 0.172 |
| `range_abs_p90` | 0.718 | **0.701** | 0.718 |
| 나머지 전 지표 | -- | 전부 일치 | 전부 일치 |

**스위트에서 유일하게 batch-size 불변이 아니다.** 올바른 구현은 배치별 delta를 모아 마지막에 한
번 백분위수를 내는 것. `iou_free`를 건드리지 않으므로 재학습 불필요.
**그때까지: 두 run의 range를 비교할 때 batch size가 같은지 먼저 확인한다.**

**(b) SynWoodScape에서 `iou_free`의 변별력이 매우 좁다 (2026-08-18 실측, 새 발견)**

| | constant-map baseline | 1 epoch 모델 | 여유 |
|---|---|---|---|
| 로봇 (`rawos3`) | 0.398 | 0.697 | +0.299 |
| **SynWoodScape** | **0.866** | 0.918 | **+0.052** |

ego 주변이 대체로 도로라 레이아웃 prior만으로 거의 맞는다. 그런데 **pretrain의 체크포인트 선택
기준도 `iou_free`**다 -- 0.866~1.0이라는 좁은 구간에서 best epoch을 고르는 것이 사실상 노이즈로
고르는 것에 가까울 수 있다. **이 프로젝트가 애초에 잡으려던 문제("지표가 레이아웃 prior를 재고
있다")와 같은 종류가 pretrain 쪽에 남아 있다는 뜻이다.**

주의: 이것을 고치려면 **pretrain의 체크포인트 선택 기준**을 건드려야 하고, 그건 `iou_free`
정의를 바꾸는 것과 같은 부류다 -> **본학습 전에 결정하는 것이 낫다.** 돌린 뒤에 고치면 pretrain을
다시 돌려야 한다(25분이라 비용은 크지 않다).

### 2.2 [사용자 예고] pretrain / fine-tuning 코드 파악 -- 아직 안 했다

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

### 2.3 본학습 -- 위 두 개가 끝난 뒤

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

- **근거 정본**: [`docs/free_space_metric_migration.md`](free_space_metric_migration.md)
  -- §1 진단, §6 2-head 기준선, §7 Task 17, §8 A/B, §9 2-head 제거
- 계획 실행 기록: [`docs/superpowers/plans/2026-08-17-bev-free-space-task-progress.md`](superpowers/plans/2026-08-17-bev-free-space-task-progress.md)
  -- §3에 실행 중 정해진 결정들, §4에 이월 발견사항
- 설계 정본: [`docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md`](superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md)
- 라벨 계약 정본: [`docs/finetuning_guide.md`](finetuning_guide.md) §1.3
  -- `occupancy_3class_npy`는 **검수 전용**이고 학습 입력이 아니다. 라벨 정본은 `occupancy_npy` 하나다.
- 공통 작업 지침: [`AGENTS.md`](../AGENTS.md)
