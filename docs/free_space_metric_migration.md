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
