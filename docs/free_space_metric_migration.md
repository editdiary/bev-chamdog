# Free-space 지표 마이그레이션 -- Phase 1 재채점 결과

이 문서는 `docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md`
Phase 1 게이트(C7–C9)의 실측 기록이다. **재학습 없이** 기존 체크포인트를 새 지표
(`projects/common/free_space_metrics.py`)로 다시 재고, 옛 지표(`iou_drivable`/`iou_obstacle`)와
비교해 배선이 맞는지 확인한다. 아래 숫자는 전부 실제로 실행한 명령의 표준출력에서
그대로 옮긴 것이다(실측 없이 숫자를 적지 않는다).

## 1. 실행한 명령과 표준출력 전체

```bash
conda activate bev-chamdog
CUDA_VISIBLE_DEVICES=1 python tools/rescore_checkpoints.py \
    --checkpoint=runs/robot_bev/ckpt/robot_finetune_res101_bs8_lr1e-04_260817_122504/model_best-000000046.pth \
    --train_sequences=raws1,raws3,rawos1 --val_sequences=raws2
```

```
/data/home/dhlee/miniconda3/envs/bev-chamdog/lib/python3.11/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/data/home/dhlee/miniconda3/envs/bev-chamdog/lib/python3.11/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet101_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet101_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
/data/home/dhlee/miniconda3/envs/bev-chamdog/lib/python3.11/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
| 체크포인트 | val | iou_free↑ | baseline iou_free | all-free iou_free | fatal↓ | baseline fatal | (참고) iou_drivable | (참고) iou_obstacle |
|---|---|---|---|---|---|---|---|---|
| robot_finetune_res101_bs8_lr1e-04_260817_122504 | raws2 | 0.850 | 0.673 | 0.176 | 0.056 | 0.168 | 0.890 | 0.312 |

free_miss_rate (M2b, 참고, 보수성): 0.135
range: p50 0.144 m | p90 0.480 m | over 0.220 m | under 0.288 m | paired rays 8439
  ring 0.0-1.5m   iou_free 0.861  fatal 0.045
  ring 1.5-3.0m   iou_free 0.848  fatal 0.055
  ring 3.0-4.0m   iou_free 0.812  fatal 0.068
```

(Fix round 1: 위 stdout은 `conda activate bev-chamdog` 상태에서 실제로 재실행해 얻은 것으로
교체했다. 최초 버전은 경로가 `miniconda3/lib/python3.14/...`인 **base** 환경 출력을 잘못
붙여넣은 것이었다 -- 명령 블록은 `bev-chamdog`을 activate하라고 적어 두고 그 아래 결과는
다른 환경 것이었다는 뜻이라, "실측 없이 숫자를 적지 않는다"는 이 문서의 원칙에 대한 직접적인
위반이었다. 숫자 자체는 두 환경에서 바이트 단위로 동일했다 — `python -V`로 `Python 3.11.15`,
`which python`으로 `.../envs/bev-chamdog/bin/python`을 먼저 확인한 뒤 재실행했다.)

체크포인트: `runs/robot_bev/ckpt/robot_finetune_res101_bs8_lr1e-04_260817_122504/model_best-000000046.pth`
split: train = `raws1,raws3,rawos1` (113 프레임), val = `raws2` (41 프레임) — 시퀀스 단위
holdout이며 constant map은 train만으로 만든다(val을 섞으면 baseline이 leak되어 부풀려진다).

## 2. 2026-08-17 실측값과의 대조 (게이트 C7–C9)

| 항목 | 기대값 (2026-08-17 실측) | 허용 | 재채점 결과 | delta | 판정 |
|---|---|---|---|---|---|
| `iou_free` | 0.850 | ±0.01 | 0.850 | +0.000 | 일치 |
| `baseline_iou_free` | 0.673 | ±0.01 | 0.673 | +0.000 | 일치 |
| `fatal_rate` | 0.059 | ±0.005 | 0.056 | -0.003 | 허용 범위 안 |
| `baseline_fatal_rate` | 0.168 | ±0.01 | 0.168 | +0.000 | 일치 |
| `iou_drivable` | 0.891 | ±0.01 | 0.890 | -0.001 | 허용 범위 안 |
| `iou_obstacle` | 0.312 | ±0.01 | 0.312 | +0.000 | 일치 |

여섯 값 전부 허용 오차 안에서 재현되었다. 표에서 참고용으로만 요구된 `all_free_iou_free`는
기대표에 없었지만 0.176으로 측정되어, 스펙 §2.3의 "전부 free ~0.17"과도 일치한다.

**순서 조건**: `iou_free(모델) 0.850 > iou_free(constant) 0.673 > iou_free(all-free) 0.176` — 성립.
**fatal_rate 조건**: `fatal_rate(모델) 0.056 < fatal_rate(constant) 0.168` — 성립.

### 2-1. `fatal_rate`만 정확히 재현되지 않는다 -- 깨끗한 일치로 포장하지 않는다

다섯 값은 소수 넷째 자리까지 사실상 그대로(delta 0.000~0.001) 재현됐지만 `fatal_rate`는
정밀 측정값 **0.055681**(≈0.056)이고 스펙(`docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md:69`)의
**0.0587**과 상대 오차 약 −5.1%다 -- 나머지 다섯 값의 오차(최대 상대 0.1%대)의 약 30배다.
허용 오차(±0.005) 안에는 들어오고, 최초 재채점(base 환경, §1의 표)과 `bev-chamdog` 환경
재실행이 표에 찍히는 소수 셋째 자리(0.056)까지 동일해 환경 문제는 아니다. 하지만 0.0587을
만든 스크립트는 이미 없어져 이 차이의 원인을 다시 계산해 확인할 방법이 없다 -- 즉 **이
delta는 영구적으로 재유도 불가능하다.**

가장 그럴듯한 무해한 설명: `fatal_rate`는 정의상(`fatal_rate = |pred_free & ~gt_free| / |pred_free|`)
분모가 "모델이 free라고 믿은 셀 수" 하나뿐이고 분자는 그중 실제로 틀린 셀의 개수라, 경계 셀
몇 개가 어느 쪽으로 걸리느냐에 따라 퍼센트 단위로 흔들리는 구조다. 반면 `iou_free`는 분모가
`|pred_free ∪ gt_free|`로 훨씬 크고 완만하다. 여섯 지표 중 `fatal_rate`가 구조적으로 가장
민감한 지표이므로, 경계 셀 소수(추론 시 라이브러리 버전·CUDA 커널 등 부동소수 연산 순서
차이로 뒤집힐 수 있는 정도의 수)만 다르게 셋 수 있으면 `iou_free`는 소수 셋째 자리까지
그대로 두고 `fatal_rate`만 퍼센트 단위로 움직이는 이 패턴이 설명된다. **이것은 추정이지
증명이 아니다** — 원본 스크립트가 없는 이상 이 이상은 확인할 수 없다.

### 2-2. `iou_drivable`의 −0.001은 알려진 집계 방식 차이다

`tools/rescore_checkpoints.py`의 `iou_drivable`은 `np.mean(d_ious)`로 **배치 단위** 비가중
평균을 낸다(`tools/train_robot_bev.py`의 `_evaluate`와 동일 -- 게이트 조건 2가 비교하는
대상이 바로 이 값이므로 **코드는 고치지 않는다**). val=41프레임을 `batch_size=4`로 돌리면
마지막 배치가 1프레임이라 그 배치가 다른 배치와 같은 가중치(1/11)를 받는다.

실측(`CUDA_VISIBLE_DEVICES=1`, `bev-chamdog` 환경, 같은 체크포인트/split):

| 집계 방식 | 값 |
|---|---|
| 배치 단순평균 (`batch_size=4`, 현재 코드, 표에 실린 값) | 0.890353 → **0.890** |
| 표본 가중평균 (batch_size로 가중, 참값) | 0.890981 → **0.891** |
| 배치 단순평균 (`batch_size=1`로 재실행 -- 이때는 배치=표본이라 위 표본가중평균과 같아야 한다) | 0.890978 |

`batch_size=1` 재실행값(0.890978)이 표본 가중평균(0.890981)과 사실상 같다(부동소수 축적
순서 차이만큼만 다름) — 배치 크기 효과라는 설명이 재실행으로도 확인된다. 즉 표의 −0.001은
드리프트가 아니라 **알려진 집계 방식 차이**이고, 진짜 표본 평균은 0.891로 스펙과 정확히
일치한다.

## 3. §2.4 대비표 재확인

스펙 §2.4는 "SynWoodScape에서는 옛 지표(`iou_drivable`)가 정상이고, 자체 로봇 데이터에서는
깨진다"는 대비를 아래처럼 기록했다(SynWoodScape 쪽은 이번 재채점 대상이 아니라 스펙에
이미 실측된 값을 그대로 인용한다):

| 데이터셋 | `vis` coverage | "전부 drivable" baseline | 모델 실측 | 판정 |
|---|---|---|---|---|
| SynWoodScape (n=500) | 0.938 | 0.875 | **0.989** | 옛 지표 **정상** (+0.11) |
| 자체 로봇 (n=154) | 0.17~0.28 | **0.935** | 0.891 | 옛 지표 **깨짐** (−0.04) |

이번 재채점(`iou_drivable` 0.890, val=raws2 41프레임)은 로봇 행의 0.891과 사실상 같은 값이라
이 대비가 그대로 재확인된다: 로봇 데이터에서 "전부 drivable" baseline(0.935)이 학습된
모델(0.890)을 여전히 이긴다. 반면 같은 체크포인트를 `iou_free`로 재면 모델 0.850이
constant map 0.673을 크게 앞선다 — **결함은 모델이 아니라 옛 지표에 있었다**는 결론이
새 재채점 도구로도 그대로 유지된다.

## 3-1. 헤드 분해: `iou_free` 0.850의 신호는 대부분 visibility head에서 온다

`iou_free`가 constant map을 크게 이긴다는 것(0.850 ≫ 0.673)이 "두 head의 결합이 학습됐다"는
증거인지, 아니면 "한 head만 학습되고 나머지는 결합에 묻어간 것"인지는 표 하나만 봐서는 알 수
없다. 같은 체크포인트·같은 마스크·같은 val split에서 `free_pred`의 정의만 바꿔 실측했다
(스크립트는 이 재채점 도구와 같은 로더·마스킹·모델 로딩 경로를 그대로 재사용했다):

| `free_pred` 정의 | `iou_free` | `fatal_rate` |
|---|---|---|
| `σ(occ)>0.5 & σ(vis)>0.5 & valid` (출하 버전, 위 표와 동일) | 0.849975 → **0.850** | 0.055681 → **0.056** |
| `σ(vis)>0.5 & valid` (visibility head만) | 0.851094 → **0.851** | 0.085062 → **0.085** |
| `σ(occ)>0.5 & valid` (occupancy head만) | 0.273171 → **0.273** | 0.729148 → **0.729** |

visibility head만으로도 `iou_free`가 0.851로 결합(0.850)보다 **오히려 높다** — 즉
`0.850 ≫ 0.673`은 visibility head가 학습됐다는 증거이지, occupancy head가 얼마나
기여했는지에 대해서는 이 지표만으로 아무것도 말해주지 않는다. occupancy head 없이는
`iou_free`가 거의 그대로인 반면(0.851 vs 0.850), `fatal_rate`는 0.085(occ 없음)에서
0.056(결합)으로 −35% 개선된다 — **여섯 개 게이트 지표 중 occupancy head의 기여를 실제로
드러내는 것은 `fatal_rate` 하나뿐이다.** occ 단독은 두 지표 모두에서 크게 나쁘다(0.273/0.729)
— GT `vis`가 0.17~0.28만 덮는 상태에서 occupancy만으로 free를 판정하면 안 보이는 영역까지
"보인다"고 가정하는 셈이라 당연한 결과다.

이 결과는 스펙 §2.3("배울 수 있는 신호는 전부 visibility head에 들어가 있었다")이 말로 남긴
주장을 정량으로 재확인한 것이다. `score_split:118`의 AND 결합 자체는 올바르게 배선돼 있다 —
이건 이 도구의 결함이 아니라 `iou_free`라는 지표의 성질이며, Phase 2 이후 두 head의 균형을
따로 볼 필요가 있다는 뜻이다.

## 4. 옛 지표를 삭제하지 않고 진단으로 남긴다

`iou_drivable`/`iou_obstacle`은 `score_split`이 여전히 계산해 표에 "(참고)" 열로 남긴다.
과거 실험 기록(로그·TensorBoard·이 문서 자체)이 전부 그 두 지표로 적혀 있어서, 지우면
과거 run과 지금 run을 나란히 읽을 방법이 없어지기 때문이다.

## 5. Phase 1 게이트 판정

세 조건 전부 성립 — Phase 2로 진행 가능하다. 다만 재현이 "여섯 값이 전부 깨끗하게 일치했다"는
뜻은 아니다 — §2-1에서 기록한 대로 `fatal_rate`는 다른 다섯 값보다 30배 큰 상대 오차(−5.1%)로
어긋나 있고, 원본 스크립트가 없어 그 delta의 원인을 확정할 수 없다. 허용 오차 안에 있다는
사실과 "정확히 재현됐다"는 주장은 다른 것이라 구분해 남긴다.

1. `iou_free(모델) > iou_free(constant) > iou_free(all-free)`: 0.850 > 0.673 > 0.176. **성립**
2. 재채점한 `iou_drivable`/`iou_obstacle`이 2026-08-17 실측값과 ±0.01 안: 0.890/0.891 (delta -0.001,
   §2-2에서 설명한 배치평균 vs 표본평균 차이 -- 표본평균으로는 0.891로 정확히 일치),
   0.312/0.312 (delta 0.000). **성립**
3. `fatal_rate(모델) < fatal_rate(constant)`: 0.056 < 0.168. **성립** (다만 0.056 자체가 스펙의
   0.0587과 −5.1% 어긋나 있다 -- §2-1 참고)

## 6. Phase 2 기준선 -- 2-head 재학습

Task 13은 새 로봇 split에서 2-head 기준선을 다시 세우는 게이트다. 이전 Phase 1의
`val=raws2` 재채점 값 0.850은 이 split에서는 비교 기준으로 쓰지 않는다. 아래 숫자가 이후
Phase 3 A/B에서 3-class 단일 head가 비교할 2-head 기준선이다.

실행 명령:

```bash
CUDA_VISIBLE_DEVICES=0 EXP_NAME=twohead_baseline_iou_free \
    TRAIN_SEQUENCES=raws1,raws2,raws3,rawos1,rawos4 VAL_SEQUENCES=rawos3 \
    bash configs/train_robot_bev_finetune.sh 2>&1 | tee runs/twohead_baseline.log
```

실행 조건:

| 항목 | 값 |
|---|---|
| train | `raws1,raws2,raws3,rawos1,rawos4` (190 samples) |
| val | `rawos3` (37 samples) |
| init checkpoint | `runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth` |
| constant-map baseline `iou_free` | 0.398 |
| 로그 | `runs/twohead_baseline.log` |
| best checkpoint | `runs/robot_bev/ckpt/twohead_baseline_iou_free_res101_bs8_lr1e-04_260817_213112/model_best-000000042.pth` |

best epoch 로그:

```text
epoch 042/60 | time    4.8s | val_iou_free↑ 0.765 (+0.001) (+0.367 vs baseline) | best_val_iou_free↑ 0.765 | checkpoint: new best
  bins  | val_obst_iou_bins empty:fa -/n0 tiny:-/n0 small:0.176/n21 medium:0.245/n16 large:-/n0
  train | iou_free↑ 0.967 | fatal↓ 0.001 | free_miss↓ 0.031 | loss_total↓ 0.0270 | loss_occ↓ 0.0050 | loss_vis↓ 0.0440 | (ref) iou_drivable↑ 0.973 | iou_obstacle↑ 0.711 | vis_false_high↓ 0.001 | vis_false_low↓ 0.027 | obst_frac 0.054 | false_obstacle↓ 0.024 | missed_obstacle↓ 0.004
  val   | iou_free↑ 0.765 | fatal↓ 0.124 | free_miss↓ 0.139 | loss_total↓ 0.3584 | loss_occ↓ 0.2191 | loss_vis↓ 0.2787 | (ref) iou_drivable↑ 0.890 | iou_obstacle↑ 0.206 | vis_false_high↓ 0.044 | vis_false_low↓ 0.135 | obst_frac 0.047 | false_obstacle↓ 0.085 | missed_obstacle↓ 0.472
  range | abs_p50↓ 0.183 | abs_p90↓ 0.725 | over↓ 0.346 | under 0.332 | rays 10659 censored 4153
  deploy| (pred visibility 기준) iou_drivable↑ 0.900 | iou_obstacle↑ 0.258 | visible_coverage 0.255
```

마지막 epoch 로그:

```text
epoch 060/60 | time    4.8s | val_iou_free↑ 0.762 (-0.002) (+0.364 vs baseline) | best_val_iou_free↑ 0.765 | checkpoint: -
  val   | iou_free↑ 0.762 | fatal↓ 0.132 | free_miss↓ 0.134 | loss_total↓ 0.4392 | loss_occ↓ 0.2793 | loss_vis↓ 0.3198 | (ref) iou_drivable↑ 0.891 | iou_obstacle↑ 0.200 | vis_false_high↓ 0.047 | vis_false_low↓ 0.132 | obst_frac 0.047 | false_obstacle↓ 0.083 | missed_obstacle↓ 0.493
  range | abs_p50↓ 0.193 | abs_p90↓ 0.760 | over↓ 0.349 | under 0.344 | rays 10699 censored 4153
```

게이트 판정:

1. 모델 best `iou_free`가 같은 split의 constant-map baseline보다 높다: 0.765 > 0.398. **성립**
2. 모든 epoch에서 `partition_defects == 0`: 학습 로그에 `partition` 경고가 없었다. **성립**
3. 이 기준선은 `val=rawos3` 37프레임 기준이다. split이 바뀌었으므로 Phase 1의 `val=raws2`
   0.850과 직접 비교하지 않는다.

## 7. Task 17 -- 3-class overfit 검증 (게이트 D10)

샘플 4장만으로 3-class 모델이 그 4장을 외우는지 본다. **일반화를 보는 시험이 아니다** --
Task 14–16에서 새로 만든 head/loss/라벨 변환(`decompose` -> `to_class_index` -> weighted CE)
배선이 실제로 학습 신호를 흘리는지만 확인한다. 통과 기준은 train `iou_free` >= 0.98.

### 7.1 데이터

`raws1`의 앞 4장(`sample_000000`..`sample_000003`)을 임시 루트에 복사해 `--dataset_root`로
넘겼다. `common_root`는 `dataset/sj_datasets/common`으로 독립이라 calib/mask는 원본을 쓴다.
`RobotBEVDataset`이 시퀀스에서 읽는 것은 `occupancy_npy`/`visibility_npy`/`rgb_images`뿐이다.

이 4장의 클래스 분포(masked, valid 14016셀 기준):

| 항목 | 값 |
|---|---|
| unknown (`~vis & valid`) | 49007셀 (85–88%/frame) |
| free (`occ & vis & valid`) | 6555셀 (11–14%/frame) |
| occupied (`~occ & vis & valid`) | **502셀 (0.90%)** |
| class weights (unknown/free/occupied), 4샘플 | `[1.0, 7.476, 20.0]` |
| class weights, 190샘플 train split (참고) | `[1.0, 3.878, 20.0]` |

두 가지를 남긴다. (a) `occupied`는 양쪽 split 모두 `MAX_CLASS_WEIGHT=20` 캡에 걸린다 --
순수 역빈도라면 4샘플에서 ~112다. (b) `free` 가중치가 7.476 대 3.878로 약 2배 다르다. 즉
**이 overfit run의 loss 균형은 Task 18과 동일하지 않다.** class weight는 CLI 플래그가 없어
맞추려면 코드 변경이 필요한데, 이 게이트가 묻는 것은 "이 4장을 외울 수 있는가"이므로 기본값
(4샘플에서 유도)을 그대로 썼다. Task 18의 A/B에는 190샘플 가중치가 쓰인다.

`iou_free`는 free 클래스 단독 IoU라 unknown 85–88%라는 다수 클래스에 부풀려지지 않는다.
게이트 지표로서 유효하다.

### 7.2 실행한 명령

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_robot_bev.py --head=three_class \
    --exp_name=overfit4_threeclass --dataset_root=<tmp>/overfit4 --train_sequences=seq \
    --val_sequences="" --val_tail_fraction=0.0 \
    --num_epochs=2000 --batch_size=4 --lr=1e-4 --num_workers=0 --save_freq_epochs=500 \
    --init_checkpoint=runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth \
    > runs/overfit4_threeclass.log 2>&1
```

`val_sequences`가 비어 있어 val split이 없다. 배너가 `val split=없음 (0 samples)`,
`constant-map baseline iou_free = nan`을 찍고 "체크포인트 선택 없이 마지막 epoch만 남는다"를
경고한다. `val_score`가 NaN이라 `model_best`는 저장되지 않는다 -- 의도된 동작이다.
trunk 전이는 `loaded 674 tensors, skipped 6`이고 skip은 전부 `segmentation_head`라
`main`의 `unexpected` 검사가 통과했다.

`--num_workers=0`을 썼다. 4샘플 기준 epoch당 1 batch뿐인데 `persistent_workers`가 아니라
epoch마다 worker를 새로 띄우면 spawn 비용이 학습 시간을 압도한다.

### 7.3 결과 -- 세 구성 전부 통과

| # | init | epochs (= optimizer steps) | `iou_free` >= 0.98 도달 | 최종 `iou_free` | 최종 `fatal` | 최종 `free_miss` | 최종 `loss_occ` |
|---|---|---|---|---|---|---|---|
| A | pretrain trunk | 2000 | **epoch 99** | **1.000** | 0.000 | 0.000 | 0.0962 |
| B | pretrain trunk | 200 | **epoch 82** | 0.989 | 0.008 | 0.003 | 1.4388 |
| C | from scratch | 200 | **epoch 50** | 0.994 | 0.005 | 0.001 | 0.9961 |

C가 계획서 Task 17 Step 1의 문자 그대로의 명령이다(`--init_checkpoint=None`). B는 그 epoch
예산에 pretrain trunk를 얹은 것, A는 여유를 크게 준 것이다. 로그는 각각
`runs/overfit4_threeclass.log`, `runs/overfit4_plan200.log`, `runs/overfit4_scratch200.log`.

**게이트 판정: 통과.** 세 구성 모두 train `iou_free` >= 0.98을 넘겼고, A는 epoch 528부터
1.000에 도달해 2000까지 유지한다(도달 후 1893/1902 epoch에서 >= 0.98). 즉 3-class 배선은
학습 신호를 정상적으로 흘린다. Task 18로 진행 가능하다.

부수 확인 두 가지:

- **`partition_defects == 0`** -- 세 run 전체(2400 epoch)에서 `partition` 경고가 한 번도
  없었다. 계획서 Step 2의 첫 진단 분기(Task 1 의심)는 해당되지 않는다.
- **`occupied` 클래스가 죽지 않았다** -- `loss_occ`가 A에서 23.71 -> 0.096으로 떨어진다.
  계획서 Step 2의 둘째 분기("`occupied`가 0에 붙어 있으면 Task 15의 클래스 가중치")도
  해당되지 않는다. 0.90%짜리 소수 클래스도 캡에 걸린 가중치 20으로 학습된다.

### 7.4 계획서와 달라진 점, 그리고 그 판단이 틀렸던 것

실행 전에 "4샘플 + `batch_size=4`면 epoch당 optimizer step이 1회이므로 계획서의 200 epoch은
200 step뿐이고, Task 13이 train 0.967에 ~1000 step 걸린 것에 비하면 도달 불가에 가깝다"고
판단해 2000 epoch(A)으로 올려 실행했다. step 수 계산 자체는 맞다
(`steps_per_epoch = max(1, len(train_loader))`, `tools/train_robot_bev.py:380`).

**그러나 결론은 틀렸다.** B와 C가 보여주듯 200 step으로 충분하고, 오히려 from scratch(C)가
pretrain trunk(B)보다 **빨리** 통과한다(epoch 50 대 82). 4장을 외우는 문제에서는 pretrain
가중치가 도움이 아니라 제약으로 작동한다. Task 13의 ~1000 step은 190장을 일반화하는
학습이라 4장 암기의 필요 step에 대한 근거가 되지 못했다.

기록해 두는 이유는 계획서를 고칠 필요가 없다는 것이 확인됐기 때문이다. **Task 17을 다시
돌린다면 계획서 원본 명령(C) 그대로가 맞다.** 200 epoch은 40초면 끝난다.

주의: A와 B/C의 lr 궤적은 같지 않다. `OneCycleLR`이 `num_epochs * steps_per_epoch + 10`으로
잡히므로 A는 warmup 100 step 후 1910 step에 걸쳐 완만히 감쇠하고, B/C는 warmup 10 step 후
200 step에 걸쳐 급히 감쇠한다. 세 run의 도달 epoch을 서로 직접 비교하지 않는다 -- 각각이
독립적으로 게이트를 넘겼다는 것만 읽는다.

### 7.5 이 게이트가 보증하지 않는 것

4장 암기는 **배선이 살아 있다**는 증거이지 정식화가 좋다는 증거가 아니다. `iou_free` 1.000은
train 4장에 대한 값이고 val이 없다. 3-class가 2-head보다 나은지는 Task 18에서만 답이 나오며,
§3-1에 기록된 대로 그때 `iou_free`만 보면 안 된다 -- occupancy head가 기여하는 유일한 지표가
`fatal_rate`다.

## 8. Phase 3 A/B -- 3-class 단일 head 대 2-head

Task 13의 2-head 기준선과 **`--head`와 loss만 바꾸고** 나머지를 전부 동일하게 유지해 3-class를
학습했다. 새로 학습한 것은 3-class 한 쪽뿐이다 -- 2-head는 §6에서 끝났다.

### 8.1 조건이 일치하는지 확인

Task 13은 `configs/train_robot_bev_finetune.sh`로, Task 18은 직접 명령으로 실행했으므로
인자를 하나씩 대조했다. `num_epochs=60`, `batch_size=8`, `lr=1e-4`, `weight_decay=1e-7`,
`num_workers=8`, `encoder_type=res101`, `augment=False`, `val_freq_epochs=1`,
`save_freq_epochs=10`, `log_dir`/`ckpt_dir`, init checkpoint, train/val split이 전부 같다.
차이가 나는 두 인자는 효과가 없음을 확인했다:

- `--val_tail_fraction` -- 스크립트는 0.2, 직접 명령은 기본값 0.0. `main`의 분기가
  `if val_names:`를 먼저 타므로 `val_sequences=rawos3`인 이상 이 값은 쓰이지 않는다.
- `--lambda_vis=0.5` / `--vis_neg_weight=3.0` -- 2-head 전용 loss 인자이고 3-class 경로에서
  호출되지 않는다. 기본값도 같은 값이다.

런타임 확인: 두 run 모두 train 190 samples / val `rawos3` 37 samples,
**constant-map baseline `iou_free` = 0.398**(동일한 숫자), class weights `[1.0, 3.878, 20.0]`,
trunk transfer `loaded 674 / skipped 6`(전부 `segmentation_head`).

실행 명령:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_robot_bev.py --head=three_class \
    --exp_name=threeclass_ab --train_sequences=raws1,raws2,raws3,rawos1,rawos4 --val_sequences=rawos3 \
    --num_epochs=60 --batch_size=8 --lr=1e-4 --weight_decay=1e-7 --num_workers=8 \
    --encoder_type=res101 --augment=False --val_freq_epochs=1 --save_freq_epochs=10 \
    --init_checkpoint=runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth \
    > runs/threeclass_ab.log 2>&1
```

best checkpoint: `runs/robot_bev/ckpt/threeclass_ab_res101_bs8_lr1e-04_260817_225457/model_best-000000030.pth`

### 8.2 결과 -- 두 run의 best epoch 대조

`val = rawos3` 37프레임. 2-head는 epoch 42, 3-class는 epoch 30이 best다.
링별 `iou_free`는 로그에 찍히지 않으므로 두 run의 TensorBoard event에서 각 best epoch step을
직접 읽었다.

| 지표 | 2-head (ep 42) | 3-class (ep 30) | delta | 우세 |
|---|---|---|---|---|
| **`iou_free`** ↑ | 0.765 | **0.774** | +0.009 | 3-class (노이즈 대역 안) |
| **`fatal_rate`** ↓ | **0.124** | 0.135 | +0.011 | **2-head** |
| `free_miss_rate` ↓ | 0.139 | **0.115** | −0.024 | 3-class |
| `range_abs_p50` ↓ | 0.183 | **0.172** | −0.011 | 3-class |
| `range_abs_p90` ↓ | 0.725 | **0.718** | −0.007 | 3-class |
| `range_over` ↓ | 0.346 | **0.322** | −0.024 | 3-class |
| `range_under` | 0.332 | 0.318 | −0.014 | 3-class |
| paired rays | 10659 | 11024 | +365 | -- |
| censored rays | 4153 | 4153 | 0 | 동일 |
| ring 0.0–1.5 m `iou_free` ↑ | 0.8065 | **0.8101** | +0.004 | 3-class |
| ring 1.5–3.0 m `iou_free` ↑ | 0.7567 | **0.7662** | +0.010 | 3-class |
| ring 3.0–4.0 m `iou_free` ↑ | 0.7130 | **0.7345** | +0.022 | 3-class |
| best epoch | 42 | 30 | -- | 3-class가 더 빨리 도달 |

`partition` 경고 0건, `NaN or Inf` 경고 0건, 60 epoch 전부 완주. `pytest tests/ -q` 242 passed.

### 8.3 판정 -- 통과. 단 근거는 `iou_free` 우세가 아니다

**게이트 조건 1(`3-class best iou_free >= 2-head 기준선`)은 성립한다: 0.774 >= 0.765.**
그러나 계획서가 같이 정한 "val 37프레임이라 `iou_free` 차이 0.02 이내는 노이즈" 규칙에 따르면
**+0.009이라는 우세 자체는 근거로 쓸 수 없다.** 이 두 문장을 함께 적는 이유는 숫자 하나로
통과 도장을 찍는 것을 막기 위해서다.

통과로 판정하는 실제 근거는 다음 세 가지다:

1. **`fatal_rate` 하나를 제외한 전 지표에서 3-class가 대등하거나 낫다.** 링 3개, range 4개,
   `free_miss_rate`가 모두 3-class 쪽이다. 개별 delta는 작지만 방향이 한쪽으로 몰려 있다.
2. **3-class는 핸디캡을 지고 이겼다.** 2-head는 `load_initial_weights`로 pretrain 체크포인트를
   **출력 head까지 통째로** 받는다(missing/unexpected 키가 있으면 예외를 던지므로 부분 로드가
   아니다). 3-class는 `load_trunk_weights`로 trunk만 받고 **출력 head는 랜덤 초기화**다.
   같은 조건이 아니라 3-class가 불리한 조건이었다.
3. **더 빨리 도달했다.** best가 epoch 30 대 42다.

**단 하나의 후퇴는 `fatal_rate`다 (0.124 -> 0.135, 상대 +8.9 %).** 이것을 노이즈로 넘기지
않는다. §3-1이 실측으로 확정한 바로 그 지표이기 때문이다 -- occupancy head가 `iou_free`에는
기여하지 않고 오직 `fatal_rate`만 개선한다(0.085 -> 0.056, 상대 −35 %). 즉 이 후퇴는 예측된
방향이며, "전용 occupancy head를 없앤 값"으로 읽는 것이 일관된 해석이다.

`fatal_rate`와 `free_miss_rate`가 반대로 움직인 것(+0.011 / −0.024)도 같은 이야기다. 3-class가
free를 더 후하게 예측해 놓치는 것은 줄고 잘못 free라 부르는 것은 늘었다. free에 대한
precision/recall 교환이고, **planner 안전 관점에서는 `fatal_rate` 쪽이 비싼 오류다.**

### 8.4 이 판정의 한계 -- 데이터 확장 후 재확인이 필요하다

**이 판정은 `val = rawos3` 37프레임 하나에 seed 하나로 내린 것이다. 데이터가 늘어난 뒤
반드시 재확인해야 한다.** 스펙 §12의 요구다. 구체적으로 네 가지가 미해결이다.

1. **37프레임.** 계획서가 정한 0.02 노이즈 대역 안에서 판정이 갈렸다. 프레임이 늘면
   `iou_free` 순서가 뒤집힐 수 있다.
2. **seed 1개.** `tools/train_robot_bev.py`가 `torch.manual_seed(0)`을 하드코딩하므로 다중 seed
   비교에는 코드 변경이 필요하다. Task 18은 "실행과 기록만"인 태스크라 계획대로 seed 하나로 돌렸다.
   두 run 모두 같은 seed라 재현은 되지만, 관측된 delta 중 seed 분산에 묻히는 것이 어느
   범위인지 모른다.
3. **head 전이 비대칭.** §8.3의 근거 2는 3-class에 유리한 방향의 교란이므로 판정을 통과 쪽으로
   기울이는 데 쓰는 것은 타당하다. 그러나 **이 비대칭이 제거되면 결과가 어떻게 되는지는 모른다.**
   그것을 제거하는 것이 Phase 4다.
4. **`fatal_rate`의 신뢰도.** §2-1에 기록된 대로 이 지표는 원본 스크립트 부재로 −5.1 % delta의
   원인이 확정되지 않았고, 분모가 `|free_pred|`인 작은 오탐 개수라 IoU보다 구조적으로 민감하다.
   판정을 가른 유일한 후퇴 지표가 마침 가장 덜 검증된 지표다.

### 8.5 다음 단계

Phase 4(SynWoodScape 3-class pretrain 재학습)로 진행할 근거가 성립한다. Phase 4는 §8.4의 3번을
제거하는 작업이므로, 이 A/B의 가장 큰 교란을 직접 해소한다.

Phase 4에서 반드시 들고 갈 것: **`fatal_rate`를 `iou_free`와 함께 주 지표로 읽는다.** 3-class
pretrain으로 출력 head까지 전이되면 §8.3의 유일한 후퇴가 사라지는지가 핵심 확인 항목이다.
사라지지 않으면 그것은 head 초기화 문제가 아니라 3-class 정식화가 occupancy 경계에서 실제로
잃는 것이라는 뜻이며, polar head(Phase 5)나 free 임계값 조정 같은 별도 수단이 필요해진다.
