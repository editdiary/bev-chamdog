# BEV Free-space Task 재정의 — 실행 인수인계

> 이 문서는 [`2026-08-17-bev-free-space-task.md`](2026-08-17-bev-free-space-task.md) 계획의 **실행 상태**를
> 기록한다. 계획 자체가 무엇을 할지 정하고, 이 문서는 어디까지 했고 그 과정에서 계획에 없던
> 무엇이 정해졌는지를 정한다. 다른 세션·다른 에이전트가 이어받을 때 이 문서를 먼저 읽는다.

- 브랜치: **`feat/bev-free-space-task`** (Task 13 커밋 후 main 대비 58 커밋)
- 마지막 완료 커밋: **Task 13 문서 커밋**
- 테스트: **224 passed**, 실패 0
- 진행: **Task 1–13 완료. Phase 0·Phase 1·Phase 2 게이트 통과. 다음은 Task 14. Task 17 전에 멈추고 사용자와 논의한다.**
- 실행 방식: `superpowers:subagent-driven-development` (태스크마다 구현자 → 리뷰 → fix 루프 → 재리뷰)

---

## 1. 재개 방법

```bash
git checkout feat/bev-free-space-task
python -m pytest tests/ -q          # 224 passed 확인
```

`superpowers:subagent-driven-development` 스킬로 **Task 14부터** 이어간다. BASE는 현재 HEAD다.
Task 17은 GPU 학습이고 사용자가 그 전에 멈춰 논의하라고 지시했다.

태스크 브리프는 계획에서 기계적으로 추출한다:

```bash
<skill>/scripts/task-brief docs/superpowers/plans/2026-08-17-bev-free-space-task.md 14
```

실행 중 ledger는 `.superpowers/sdd/2026-08-17-bev-free-space-task/progress.md`에 있다.
**git-ignored 스크래치라 계획 완료 시 삭제된다** — 그래서 오래 남아야 하는 것은 전부 이
문서와 커밋 메시지에 옮겨 두었다. ledger가 없어도 이 문서로 재개할 수 있다.

### 절대 하지 않는 것

- **`merge`와 `push`는 사용자가 직접 한다.** 에이전트가 임의로 수행하지 않는다.
- `third_party/`와 `mmdetection3d/`는 git submodule — 직접 수정하지 않는다.
- **Task 13 · 17 · 18은 GPU 학습이다. 실행 전 사용자 확인을 받는다.**

---

## 2. 완료된 태스크와 커밋 범위

| Task | 내용 | 커밋 범위 | 결과 |
|---|---|---|---|
| 1 | `free_space.py` — free/occupied/unknown 4-way 분해 | `82a5409..ee632f4` | 167 passed |
| 2 | `free_space_metrics.py` — M1 `iou_free`, M2 `fatal_rate`, M2b `free_miss_rate`, 공유 집계기 | `ee632f4..78478e0` | 175 passed |
| 3 | `polar.py` — 방위각 광선 인덱스 캐시와 `r(θ)` 추출 | `78478e0..df99f74` | 182 passed |
| 4 | M3 `range_error` — occupied를 IoU가 아니라 거리로 측정 | `df99f74..be10001` | 188 passed |
| 5 | M4 거리 링 분해 (0–1.5 / 1.5–3 / 3–4 m) | `a6ddbaa..a0b3705` | 193 passed |
| 6 | 라벨 무결성 테스트 + `tools/measure_label_geometry.py` **[Phase 0 게이트]** | `a0b3705..2f702c2` | 197 passed |
| 7 | `baselines.py` — 이미지를 보지 않는 예측기 | `2f702c2..72b756e` | 201 passed |
| 8 | `tools/rescore_checkpoints.py` + 마이그레이션 결과표 **[Phase 1 게이트]** | `72b756e..d575c4b` | 208 passed |
| 9 | 2-head `run_batch` free-space 지표 배선 | `8d2d31d..acaf1eb` | 212 passed |
| 10 | 로그·배너·TensorBoard에 `iou_free`와 baseline 배선 | `3efb297..d996c9a` | 217 passed |
| 11 | checkpoint 선택 기준을 `iou_free`로 교체 | `de630ea..a0a3e53` | 220 passed + GPU0 smoke |
| 12 | 시각화 3-class 팔레트 + polar range overlay | `61b74fd..235ea12` | 224 passed + PNG 4장 생성 |
| 13 | 새 split 2-head 기준선 재학습 **[Phase 2 게이트]** | Task 13 문서 커밋 | best `val_iou_free` 0.765, 224 passed |

`a6ddbaa`(문서 커밋)는 Task 4와 5 사이에 들어갔다. `8d2d31d`(인수인계 문서 커밋)는
Task 8과 9 사이에 들어갔다.

### 게이트 통과 기록

**Phase 0** — `python -m pytest tests/ -q` → 197 passed, 실패 0.

**Phase 1** — `docs/free_space_metric_migration.md`에 명령과 stdout 전문이 있다. 세 조건 전부 성립:

| 지표 | 기대(2026-08-17 실측) | 재채점 | 판정 |
|---|---|---|---|
| `iou_free` | 0.850 | 0.849975 | ✅ |
| `baseline_iou_free` (constant map) | 0.673 | 0.672560 | ✅ |
| `all_free_iou_free` | ~0.17 | 0.175721 | ✅ |
| `fatal_rate` | 0.0587 | 0.055681 | ⚠️ §4 참조 |
| `baseline_fatal_rate` | 0.1678 | 0.167764 | ✅ |
| `iou_drivable` (참고) | 0.891 | 0.890353 | ✅ |
| `iou_obstacle` (참고) | 0.312 | 0.312019 | ✅ |

순서 조건 `0.849975 > 0.672560 > 0.175721` 성립, `fatal_rate` 조건 `0.055681 < 0.167764` 성립.

**이 재현이 중요한 이유**: 이 숫자들의 원래 출처는 브레인스토밍 중에 급조한 임시 스크립트였고
그 스크립트는 남아 있지 않다. Task 8이 만든 도구가 그 측정의 첫 영구 구현이며, 다섯 값이
1e-4 수준으로 독립 재현됐다. 평가 재설계 전체가 이 재현 위에 서 있다.

---

## 3. 실행 중 정해진 것 — 계획·스펙에 없던 결정

계획 문서만 읽어서는 알 수 없는 것들이다. 이후 태스크가 반드시 지켜야 한다.

### 3.1 M3의 방위각 bin 수를 **720**으로 올렸다 (사용자 판정, Task 6)

계획은 "roundtrip IoU 차이가 0.02 미만이면 360 유지"를 기준으로 삼았으나 실측 차이가
0.0319(마스크 보정 시 0.0401)로 기준을 넘었다. 스펙 §5.2가 원래 요구한 기준은 roundtrip이
아니라 **M3 지표 자체의 차이**였으므로 roundtrip 기준은 애초에 M3에 틀린 시험이었다.
720이 사실상 공짜이고 스펙 §2.5가 이미 "720 필요"라고 결론냈으며 아직 M3로 채점한 run이
하나도 없어 바꾸는 비용이 가장 싼 시점이라, 720을 채택했다.

**구조로 고정돼 있다**: `projects/common/polar.py`의 `DEFAULT_N_THETA = 720`이
`build_ray_index`의 기본값이다. **이후 태스크는 `n_theta`를 하드코딩하지 않는다.**
Task 8에서 브리프 코드가 `n_theta=360`을 박아 이 판정을 조용히 덮는 것이 발견돼 `None`으로
고쳤다 — 같은 실수를 반복하지 않는다.

### 3.2 `summarize_range_error`의 over/under 평균을 partition 크기로 가중한다 (사용자 판정, Task 4)

`over_mean`/`under_mean`은 부분집합 평균이므로 `n_paired_rays`가 아니라 각자의
`over_count`/`under_count`로 가중해야 한다. 수치 예: paired 100/100에서 over가
1건×1.00 m vs 99건×0.01 m일 때 정답 0.02 m, 구 공식 0.505 m로 **25배** 어긋난다.

### 3.3 `free_metrics_from_masks`를 단일 공유 집계기로 통합했다 (pre-flight, `82a5409`)

계획의 Task 9와 Task 15가 `compute_free_metrics`를 각각 정의하고 있었다. 사본이 둘이면
Phase 3의 A/B가 조용히 서로 다른 것을 비교하게 되므로 Task 2로 끌어올려 하나로 만들었다.
**2-head와 3-class가 반드시 이 함수를 공유해야 A/B가 공정하다.**

### 3.4 `two_head_metrics.py` 개칭은 Task 16까지 미룬다 (계획에 명시된 스펙 대비 편차)

스펙 §10은 Phase 0에서 `bev_occupancy_metrics.py`로 개칭하라고 적었으나, Phase 2의
기준선을 2-head 구성 그대로 뽑아야 해서 두 정식화가 Phase 3까지 공존한다. 그 기간에
개칭하면 rename diff가 정작 중요한 지표 변경 diff를 덮는다.

### 3.5 `occupancy_3class_npy`는 학습 입력이 아니다

사용자가 만든 3-class 라벨 파일은 **검수 전용**이다. 근거는
[`docs/finetuning_guide.md`](../../finetuning_guide.md) §1.3에 실측과 함께 기록돼 있다.
요약: 원시 visibility에서 파생돼 `permanent_blind`(5.6 %)가 빠져 학습 코드가 쓰는 값과
셀의 4.97 %가 어긋나고, `permanent_blind`를 구워 넣어도 `valid`(rear_self_box)는 3-class로
표현할 수 없다. 라벨 정본은 `occupancy_npy` 하나다.

### 3.6 epoch 누적에는 free-space mask 텐서를 저장하지 않는다 (Task 9 리뷰에서 고정)

Task 9에서 `run_batch`는 Task 10의 val-only M3/M4 계산을 위해 batch 단위 `free_metrics`에
`pred_free`/`gt_free`를 포함한다. 그러나 epoch 집계 리스트에는 이 dict를 그대로 append하지
않고 `append_free_metrics`로 scalar/count 키만 복사한다. 그렇지 않으면 epoch 동안 batch별
마스크 텐서를 붙잡아 메모리가 새고, 계획의 "`pred_free`/`gt_free`는 집계하지 않는다" 계약을
위반한다.

### 3.7 Task 10 로깅과 Task 11 checkpoint 선택은 분리한다

Task 10은 epoch log/TensorBoard에 `iou_free`와 baseline delta를 보이게 하는 작업이다.
Task 10 리뷰에서 `model_best` 선택까지 `iou_free`로 바꾼 것이 scope 위반으로 판정돼 되돌렸다.
따라서 Task 10 완료 시점의 두 trainer는 여전히 `val_score = 0.5 * (d_iou + o_iou)`로
checkpoint를 고른다. 단, `format_epoch_log`의 baseline delta는 checkpoint용 `val_score`가
아니라 `val_free_metrics["iou_free"] - baseline_iou_free`로 계산한다. Task 11이 이
선택 기준을 정식으로 바꾼다.

### 3.8 GPU 실행은 당분간 GPU0을 쓴다 (사용자 지시, Task 11)

Task 11 smoke에서 GPU1은 다른 프로세스가 점유해 OOM이 났다. 사용자가 "GPU1이 누가 쓰고
있어서 GPU0으로 계속 실행"하라고 지시했다. 이후 GPU 명령은 별도 지시가 없으면
`CUDA_VISIBLE_DEVICES=0`을 사용한다. Task 11의 1-epoch smoke는 GPU0에서 통과했다:
`constant-map baseline iou_free = 0.673`, `val_iou_free↑ 0.730 (+0.057 vs baseline)`,
`partition` 경고 없음, `model_best-000000001.pth` 저장.

### 3.9 현재 로봇 split (사용자 지시)

새 시퀀스 `rawos3`, `rawos4`가 추가됐다. 이후 자체 로봇 데이터셋 학습은 별도 지시가 없으면
`train = raws1,raws2,raws3,rawos1,rawos4`, `val = rawos3`로 실행한다. `rawos3`는 held-out
시퀀스이며 train에 넣지 않는다. `configs/train_robot_bev_finetune.sh` 기본값과 남은 GPU
task 명령은 이 split으로 맞췄다.

이 변경 이후 Task 13의 `0.850 ±0.03` 비교는 더 이상 적용하지 않는다. 그 0.850은 old
split(`val=raws2`)의 역사적 재채점 값이다. Task 13은 `rawos3` 기준 새 2-head baseline을
수립하고, 같은 split의 constant-map baseline보다 높은지와 `partition_defects == 0`을
게이트로 본다.

### 3.10 Task 12 시각화 산출물

Task 12에서 `runs/robot_bev/viz/task12_raws2/`에 raws2 샘플 4장의 PNG를 생성했다.
`raws2_sample_000000.png`를 직접 확인했고, GT/pred free-space 패널과 pred 패널 위
GT(흰색)/pred(노란색) range profile overlay가 표시된다. `draw_range_profile`의 `rays`
인자는 현재 내부에서 직접 사용하지 않지만, 호출 경로에서는 `r_m/status`가 같은 ray index에서
나온다. 상태 필터(`RAY_OK`만 draw)와 theta=90° 좌측(col 감소) 규약은 테스트로 고정했다.

---

## 4. 이후 태스크가 알아야 할 발견사항

### 4.1 헤드라인 0.850은 거의 전부 visibility head의 공로다 (Task 8 리뷰에서 실측)

같은 체크포인트·마스크·집계에서 `free_pred` 유도만 바꿔 측정한 결과:

| 예측 | `iou_free` | `fatal_rate` |
|---|---|---|
| `(σ(occ)>0.5) & (σ(vis)>0.5)` — 현재 배선 | 0.849975 | 0.055681 |
| `σ(vis)>0.5` 단독 | **0.851094** | 0.085062 |
| `σ(occ)>0.5` 단독 | 0.273171 | 0.729148 |

occupancy head를 아예 버리면 `iou_free`가 오히려 오른다. 즉 `0.850 ≫ 0.673`은 "two-head
시스템이 배웠다"가 아니라 **"visibility head가 배웠다"**의 증거다. occ head가 실제로
기여하는 유일한 지표는 `fatal_rate`다(0.085 → 0.056, 상대 −35 %).

**Phase 3 A/B 해석에 직접 영향을 준다.** 3-class 단일 head가 2-head를 이기는지 볼 때
`iou_free`만 보면 occ head의 기여가 보이지 않는다. `fatal_rate`를 반드시 함께 읽는다.
스펙 §2.3이 같은 주장을 말로 하고 있고, 이제 숫자로 확정됐다.

### 4.2 `fatal_rate`만 깨끗하게 재현되지 않았다 — 원인 규명 불가

0.055681 대 스펙 §2.3의 0.0587로 상대 −5.1 %, 나머지 다섯 값 오차의 약 30배다. 브리프의
허용(±0.005) 안이고 환경 차이도 아니지만(두 환경에서 동일), 0.0587을 만든 스크립트가
없어 영구히 확인할 수 없다. `fatal_rate`는 분모가 `|free_pred|`인 작은 오탐 개수라 IoU보다
구조적으로 민감하다는 설명이 그럴듯하지만 **추정이지 증명이 아니다.**
`docs/free_space_metric_migration.md` §2-1에 그대로 기록돼 있다.

이 값은 planner 위험 논거가 걸린 유일한 지표이므로, 앞으로 이 숫자가 흔들리면 먼저 의심한다.

### 4.3 `iou_drivable`의 −0.001은 알려진 집계 아티팩트다 — 고치지 않는다

`np.mean(d_ious)`가 **배치 단위** 비가중 평균이라 41프레임/batch 4에서 마지막 1프레임
배치가 1/41이 아니라 1/11 가중을 받는다. 배치 평균 0.890353 대 참 샘플 평균 0.890981.

**의도적으로 그대로 둔다** — `tools/train_robot_bev.py:_evaluate`를 정확히 재현해야 하고
Phase 1 게이트 조건 2가 그것과 비교하기 때문이다. 바꾸면 비교가 깨진다. Phase 2에서
`run_batch`를 건드릴 때 함께 정리할지 판단한다.

### 4.4 `rear_self_box`(`invalid`)는 좌우 대칭이 정확히 1.0000이다

좌우 반전만으로는 형태로 검출이 불가능하다. `permanent_blind`는 0.9912로 거의 대칭이라
면적 검사만으로는 못 잡고 위치 기반 검증이 필요하다 —
`tests/common/test_label_integrity.py`가 그렇게 돼 있다.

---

## 5. 남은 태스크

| Task | 내용 | 비고 |
|---|---|---|
| 14 | 3-class head (`TwoHeadDecoder` 패턴을 따른다) | |
| 15 | 가중 CE loss + `three_class_metrics.py` | |
| 16 | `--head` 스위치 + `two_head_metrics.py` 개칭 | §3.4 참조 |
| 17 | 과적합 게이트 | **GPU 학습 — 실행 전 중지하고 사용자와 논의** |
| 18 | 2-head vs 3-class A/B **[Phase 3 게이트]** | **GPU 학습 — 사용자 확인 필요**, §4.1 해석 주의 |

Phase 4(SynWoodScape 3-class pretrain 재학습)와 Phase 5(polar head)는 이 계획 범위 밖이며,
Phase 3 결과를 보고 별도 스펙·계획으로 다룬다.

---

## 6. 실행에서 관찰된 것 — 리뷰를 어떻게 걸어야 하는가

**Task 1–12에서 나온 리뷰 findings가 사실상 전부 "테스트는 통과하지만 그럴듯한 오류도 같이
통과시킨다" 유형이었고, 구현 결함은 0건이었다.** 계획에 완성 코드를 넣는 방식은 구현 오류를
잘 막지만 테스트 강도가 계획 작성자의 상한에 묶인다.

실제로 잡힌 것들:

- Task 3 — `d_col` 부호를 뒤집어도 roundtrip IoU 1.0. 원점 중심 원반이 좌우 대칭이라 원리적으로 안 보였다.
- Task 5 — 링 마스크의 row/col을 뒤바꿔도 16개 전부 통과. 유일한 fixture가 정사각이었다.
- Task 6 — `permanent_blind`를 통째로 지워도, transpose해도, 좌우로 뒤집어도 전부 통과.
- Task 6 — `≥ 0.99`를 지킨다던 polar roundtrip 테스트가 구멍 없는 원반을 써서 검증하려던 효과를 볼 수 없었다.
- Task 7 — `all_free_map`의 shape를 아무도 assert하지 않아 transpose가 통과.
- Task 8 — `score_split`을 테스트하는 것이 하나도 없었다. `sigmoid`를 빼도 208개 통과하고 게이트 3조건도 통과하며 6개 값 중 5개가 허용 범위 안이었다.
- Task 9 — `free_miss_rate` 분모를 `fatal_denom`으로 바꿔도, `run_batch` 반환을 8-튜플로 되돌려도 focused 테스트가 통과했다.
- Task 10 — `_baseline_iou_free`가 항상 NaN을 반환해도 최초 focused 테스트가 통과했다.
- Task 11 — 공용 `select_checkpoint_score` 테스트만으로는 두 trainer 호출부가 옛 평균식으로 되돌아가도 통과했다.
- Task 12 — `draw_range_profile`의 `RAY_OK` guard 제거와 좌우 부호 반전이 최초 focused 테스트를 통과했다.

**따라서 리뷰 프롬프트에 "구현을 실제로 망가뜨려서 테스트가 진짜 실패하는지 확인하라"를
계속 명시 요구한다.** 추론만 한 mutation은 그렇게 표시하게 하고, 실행한 것과 구분한다.
이 요구를 넣은 리뷰만 위 결함들을 찾아냈다.

게이트 조건 자체도 만능이 아니다 — Task 8에서 확인됐듯 **Phase 1 게이트 3조건은 실제 배선
오류가 있어도 성립한다.** 게이트는 타당성(plausibility)을 거르지 정확성을 보증하지 않는다.
