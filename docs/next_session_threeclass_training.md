# Next Session Handoff: 3-class 학습

Last updated: 2026-08-18

> ## ▶ 새 세션은 여기서 시작한다
>
> **지금 막힌 곳: fine-tuning이 제대로 학습되지 않는다. 원인은 loss가 면적 기반(CE)인데
> `occupied`가 두께 1셀 표면이라는 것이고, 다음 할 일은 loss를 표면에 맞게 바꾸는 것이다.**
>
> 진단·실험 기록과 다음 단계 설계가 전부 여기 있다:
> **[`docs/finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md)**
> — §13이 "다음 세션에서 할 일"((C) 복합 loss → (D) 이진 정식화), §14가 산출물 정리 규약이다.
>
> 지표 정의의 정본은 [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §2.8.
> pretrain 쪽 분석은 [`synwoodscape_pretrain_experiment_log.md`](synwoodscape_pretrain_experiment_log.md) §7.
>
> ```
> git log --oneline -1        # HEAD
> python -m pytest tests/ -q  # 275 passed
> nvidia-smi                  # GPU0을 쓴다. 다른 사용자와 공유될 때가 있다
> ```
>
> **한 줄 상태**: pretrain(60 epoch)과 fine-tuning ablation 8개를 돌렸다. 지표 확장과 시각화
> 재작성은 끝났다. fine-tuning은 아직 쓸 만한 상태가 아니다.

## 어느 문서를 믿어야 하는가

| 문서 | 역할 | 상태 |
|---|---|---|
| **[`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md)** | **지금 막힌 곳·실험 기록·다음 단계(§13)·산출물 정리 규약(§14)** | **활성** |
| [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) | 지표 정의 정본(**§2.8**), loss 배경(§1.7) | 활성 (§1–2.7은 2-head 설계) |
| [`training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md) | 코드 정독 가이드 (파일 → 텐서 → 모델 → loss → 지표) | 활성 |
| [`synwoodscape_pretrain_experiment_log.md`](synwoodscape_pretrain_experiment_log.md) | pretrain 실험 기록. **§7**이 3-class 진단 런 | 활성 |
| [`finetuning_guide.md`](finetuning_guide.md) | fine-tuning 실행 절차 | 활성 |
| 이 문서 | 진입점, git/실행 명령, 결정 이력 | 활성 |
| [`free_space_metric_migration.md`](free_space_metric_migration.md) | 지표 이관의 근거 기록 | 기록용 — **숫자가 옛 split 기준** |
| `training_guide.md`, `training_improvement_plan.md` | 2-head 시절 문서 | **낡음** — 플래그·경로가 존재하지 않는다 |
| `next_session_synwoodscape_twohead.md`, `finetuning_preparation.md` | 옛 단계의 인수인계·준비 기록 | 기록용 |

## 지금까지 무엇을 확인했나 (요약)

| 확인한 것 | 결론 | 근거 |
|---|---|---|
| 이미지 경로가 정상인가 | **정상.** 이미지를 섞으면 `iou_free` 0.79 → 0.37 | 진단 §11 앞부분, `tools/measure_image_dependence.py` |
| photometric augmentation | 작지만 일관되게 개선. **채택** | 진단 §8 |
| weight decay | **막다른 길.** 0.5에서도 train `iou_free` 0.99 | 진단 §10 |
| SynWoodScape pretrain | **해롭다.** 없이 돌리면 9개 지표 중 7개 우세 | 진단 §11 |
| 클래스 가중치 상한 | **20도 1도 실패.** 상한을 고르는 문제가 아니다 | 진단 §12 |
| `iou_free`의 변별력 | pretrain 0.4 %, 로봇 2.1 % -- 양쪽 다 부족 | 진단 §6, pretrain log §7.2 |

## 브랜치와 git 상태

```bash
git branch --show-current
# feat/bev-free-space-task     <- merge/push는 사용자가 직접 한다 (AGENTS.md)
```

워킹트리 clean. `python -m pytest tests/ -q` -> **275 passed**.

## 지금 바로 돌릴 수 있는 것

```bash
conda activate bev-chamdog
nvidia-smi   # GPU0을 쓴다. 다른 사용자 job과 겹칠 때가 있어 먼저 확인한다

# fine-tuning (약 10분). split은 config 기본값이 train 192 / val 75다.
# INIT_CHECKPOINT는 필수 -- `none`이면 pretrain 없이(ImageNet trunk만) 돌린다.
#   실측으로 pretrain이 해로웠으므로(진단 §11) 당분간 `none`이 기준선이다.
EXP_NAME=ft_x AUGMENT=True MAX_CLASS_WEIGHT=1 INIT_CHECKPOINT=none \
  bash configs/train_robot_bev_finetune.sh 2>&1 | tee runs/console/ft_x.log

# 환경변수로 스윕한다 (config 파일을 복사하지 말 것)
#   AUGMENT | WEIGHT_DECAY | NUM_EPOCHS | MAX_CLASS_WEIGHT | LR
#   TRAIN_SEQUENCES | VAL_SEQUENCES | INIT_CHECKPOINT

# pretrain (약 25~33분). 지금은 재실행할 이유가 없다 -- epoch 48 체크포인트가 남아 있고
# 그것을 쓰지 않는 것이 더 낫다는 실측이 있다.
bash configs/train_synwoodscape_threeclass_pretrain.sh 2>&1 | tee runs/console/pretrain.log
```

**진단 도구:**

```bash
# 예측 패널 (2x2, 오차 지도 포함). --sort_by 로 나쁜 순 정렬
python tools/visualize_robot_predictions.py --ckpt=<...>.pth --sort_by=fatal_rate --limit=6

# 모델이 이미지를 쓰고 있는지 (이미지를 섞어 재채점)
python tools/measure_image_dependence.py --ckpt=<...>.pth

# 저장된 체크포인트를 새 지표로 재채점
python tools/rescore_checkpoints.py --checkpoint=<...>.pth

# 산출물 정리 (기본 dry-run). 규약은 진단 문서 §14
python tools/prune_runs.py --pattern='ft_*'
```

**남아 있는 체크포인트** (주기 저장분은 정리했다):

```
runs/synwoodscape_threeclass/ckpt/threeclass_pretrain_.../model_best-000000048.pth  # pretrain
runs/robot_bev/ckpt/ft_scratch_.../model_best-000000051.pth                        # 현재 최고
runs/robot_bev/ckpt/robot_finetune_.../model_best-000000033.pth                    # 진단 기준선
runs/robot_bev/ckpt/ft_aug_*, ft_wd*_*                                             # ablation
```

## 1. 이번 세션(2026-08-17~18)에 실제로 한 일

### 1.1 Task 17 -- 4샘플 overfit 게이트 (커밋 `0bb113c`)

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

### 1.2 Task 18 -- 2-head vs 3-class A/B (커밋 `0bb113c`)

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

### 1.3 3-class pretrain 지원 + 2-head 전면 제거 (커밋 `6ec96ee`, 사용자 지시)

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
- **옛 2-head 산출물은 아카이브로 옮겨진 뒤 2026-08-18에 삭제됐다**(재채점 불가). 그래서
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

**[2026-08-18 진행] 지표 확장으로 이 문제의 해결 경로가 생겼다.**

원인이 "free가 셀의 82%라 레이아웃 prior만으로 맞는다"는 것이므로, **occupied를 재는 지표는
그 포화가 없다.** 그래서 커밋 `4f71565`에서 occupied `f1@τ`, range `mae`/`bias`,
`missed_obstacle_rate`, 거리별 층화를 넣었다(정의와 실측은
[`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §2.8).

**전부 보고 전용으로 넣었고 선택 기준은 아직 `iou_free`다.** 근거 없이 기준을 바꾸는 대신
**진단용 pretrain 1회(약 33분)로 60 epoch 전부의 곡선을 보고** 결정한다 -- `val_freq_epochs=1`
이라 모든 지표가 매 epoch TensorBoard에 찍힌다.

새 지표가 실제로 변별력이 있다는 방증 (로봇 2 epoch fine-tune, 같은 체크포인트):

| 지표 | 값 | 읽는 법 |
|---|---|---|
| `iou_occupied` | 0.063 | 면적 IoU는 두께 1셀 표면에서 무의미하다 |
| `f1@20cm` | **0.665** | 같은 예측인데 값이 완전히 다르다 |
| `missed_obstacle_rate` | 0.069 | 이전에는 로그에 없던 숫자 |
| 거리별 `range_mae` | 0.276 / 0.242 / **0.457** | 원거리가 1.9배 나쁘다 -- 근/원거리가 분리된다 |
| `iou_free_known` − `iou_free` | +0.131 | 로봇에서는 갈리고 SynWoodScape에서는 완전 일치 |

**선택 기준 후보에 대한 현재 판단:**

1. **`f1@20cm` (occupied)** -- 가장 안전하다. 분모(`|O_pred|`, `|O_gt|`)가 고정이고 유계이며
   레이아웃 prior로 맞을 수 없다.
2. **`MAE_range`** -- 해석은 가장 직접적(단위가 meter)이지만 **선택 기준으로는 위험하다.**
   표본이 예측에 의존해서(장애물을 놓친 광선이 `RAY_CENSORED`가 되어 빠진다) 이 지표로 순위를
   매기면 "어려운 광선을 버리는" 체크포인트가 유리해진다. **보고용 headline으로는 좋고 선택은
   다른 지표에 맡긴다**가 결론. `missed_obstacle_rate`를 함께 조건으로 걸면 완화되지만
   복합 기준은 가중치가 또 임의값이 된다.
3. **원거리 링 `iou_free`** -- 추가 비용 0이지만 여전히 free 기준이라 포화를 완전히 벗지 못한다.
4. **그대로 둔다** -- pretrain은 초기값 제공이 목적이라 무해할 수도 있다. 여러 pretrain epoch에서
   fine-tuning을 돌려(8분/회) 상관을 실측하면 근거가 생긴다.

주의: 선택 기준을 바꾸는 것은 **본학습 전에** 해야 한다 -- `iou_free` 정의를 바꾸는 것과 같은
부류라 재채점으로 복구되지 않는다. 돌린 뒤에 고치면 pretrain 재실행이다(약 33분).

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

### 2.3 [해결됨 2026-08-18] loss / 클래스 가중치

3-class loss의 클래스 가중치는 train split에서 역빈도로 자동 계산되는데, 상한
`MAX_CLASS_WEIGHT = 20`이 **근거 없는 임의값이면서 실제로 작동 중이다** -- 로봇 train split에서
`occupied`를 67.93 -> 20으로 자른다. loss 균형을 실질적으로 결정하는 하이퍼파라미터인데
검증된 적이 없고, 하필 `fatal_rate`(§8 A/B에서 유일하게 후퇴한 지표)와 직결된다.

**정본: [`docs/BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7**
(실측 분포표, Simple-BEV 원본과의 차이, 대안 목록, 스윕 계획). 코드 쪽 포인터는
`projects/common/three_class_metrics.py`의 `MAX_CLASS_WEIGHT` 주석.

> **[답] 본학습을 돌렸고 결론이 나왔다.** 상한을 고르는 문제가 아니다 -- `occupied`가 셀의
> 1.1 %인데 val loss의 67 %를 만들고, 20은 과신(정답 확률 0.07 %)을, 1은 (일시적) 붕괴를
> 만든다. 근본 원인은 CE가 **면적** loss인데 `occupied`는 두께 1셀 **표면**이라는 것이다.
> 다음 단계는 loss 교체다:
> **[`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) §12–§13.**

### 2.4 [완료] 본학습

pretrain 60 epoch과 fine-tuning ablation 8개를 돌렸다. 결과는 각각
[`synwoodscape_pretrain_experiment_log.md`](synwoodscape_pretrain_experiment_log.md) §7,
[`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) §8–§12.

**지금 막힌 곳은 fine-tuning 품질이고, 다음 할 일은 진단 문서 §13이다.**

---

## 3. 실행 시 반드시 지켜야 할 것

- **`merge`와 `push`는 사용자가 직접 한다.** 에이전트가 임의로 하지 않는다.
- `third_party/`와 `mmdetection3d/`는 git submodule -- **직접 수정 금지.**
- GPU는 **`CUDA_VISIBLE_DEVICES=0`**. GPU1은 점유 중이다.
- **실측 없이 숫자를 적지 않는다.** 게이트/판정 결과는 명령 출력을 문서에 붙여 남긴다.
- 커밋 메시지는 **영어**, 코드 주석·문서는 **한국어**(주변 스타일).
- 로봇 split은 별도 지시 없으면 **`train = raws2,raws3,rawos1,rawos2,rawos4`(192프레임),
  `val = raws1,rawos3`(75프레임)**. 2026-08-18에 사용자가 `rawos2`(40프레임)를 추가하고
  `raws1`을 val로 옮겼다. **이 문서와 `free_space_metric_migration.md`에 기록된 로봇 숫자는
  전부 옛 `val = rawos3` 37프레임 기준이라 새 split의 값과 직접 비교할 수 없다** --
  constant-map baseline(0.398)도 train split이 바뀌었으므로 다시 실측된다.

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
- **loss 설계 현황과 다음 단계**: [`docs/finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) §13
  (loss 구현의 배경·실측 분포는 [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7)
  -- `MAX_CLASS_WEIGHT=20`의 근거 없음, 실측 분포, 대안 목록. §1.1–1.6은 옛 2-head 설계다.
- **근거 정본**: [`docs/free_space_metric_migration.md`](free_space_metric_migration.md)
  -- §1 진단, §6 2-head 기준선, §7 Task 17, §8 A/B, §9 2-head 제거
- 계획 실행 기록: [`docs/superpowers/plans/2026-08-17-bev-free-space-task-progress.md`](superpowers/plans/2026-08-17-bev-free-space-task-progress.md)
  -- §3에 실행 중 정해진 결정들, §4에 이월 발견사항
- 설계 정본: [`docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md`](superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md)
- 라벨 계약 정본: [`docs/finetuning_guide.md`](finetuning_guide.md) §1.3
  -- `occupancy_3class_npy`는 **검수 전용**이고 학습 입력이 아니다. 라벨 정본은 `occupancy_npy` 하나다.
- 공통 작업 지침: [`AGENTS.md`](../AGENTS.md)
