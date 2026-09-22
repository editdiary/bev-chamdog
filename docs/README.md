# docs/ 색인

**무엇을 찾는지에 따라 들어가는 문을 다르게 둔다.** 문서마다 *역할*과 *신선도*를 아래에
명시했으니, 어떤 문서를 믿고 어떤 문서를 인용만 할지 여기서 판단한다.

문서 상태는 네 가지뿐이다.

| 상태 | 뜻 |
|---|---|
| 🟢 **정본** | 지금 유효하고 계속 갱신한다. 내용이 바뀌면 여기를 고친다 |
| 🔵 **운영 메모** | 유효하지만 갱신이 느슨하다. 실행 명령·함정 목록처럼 실무용 |
| 📚 **참고** | 사실 조사·학습 자료. 시점 의존성이 낮다 |
| 🗄 **아카이브**(`archive/`) | 완료된 단계의 기록. **인용은 되지만 갱신하지 않는다.** 여기 적힌 플래그·경로·숫자를 그대로 쓰면 안 된다 |

---

> ## ▶ [2026-09-21] `L_range`를 누적형으로 바꿨다 — **논문은 `D`가 아니라 `E`를 쓴다**
>
> 기존 `arc_huber`는 광선을 종점 스칼라 하나로 합쳐 비교해서, `δ_R`이 경계의 수직 거리
> band가 아니라 ray 방향 arc 오차에 걸리고 광선 안의 양·음 오차가 상쇄된다. 그래서
> `L_cell`과 **같은 soft target**의 **누적 radial arc profile L1**(`cumulative_l1`)로 바꿨다.
> `δ_R`과 `β`는 새 식에 없다.
>
> **`D_range`와 `E_cumulative`는 이어지는 계단이 아니라 `C_soft`에서 갈라지는 대체 팔**이고,
> **[사용자 결정] 논문 ablation 표에는 `E`만 싣고 `D`는 뺀다.** 확정 표기는
> `RANGE_LOSS_MODE=cumulative_l1`, `λ_R = 0.15`다(`λ_R` 0.3 → 0.15는 **gradient 기여를
> 맞춘 결과**이지 "절반으로 줄인" 것이 아니다).
>
> `runs/loss_effect/` 30런에서 짝지은 판정 규약을 통과한 것은 셋이다 — `iou_free` **+0.0020**
> (대 `D`), `range_mae` **−0.0053/−0.0033 m**(둘 다 5/5, 가장 단단하다), 그리고
> `missed_obstacle` **+0.0010**(**나빠진다**). **안전은 개선이 아니고**(같은 `free_miss`에서
> 곡선이 겹친다) **되올림도 안 움직이는데**, 그것이 "되올림의 원인은 경계의 hard target"을
> 오히려 보강한다.
>
> 작성 규칙과 표는 [`loss_effect_results.md`](loss_effect_results.md) **§16**, 수식은
> [`loss_function_spec.md`](loss_function_spec.md) §8.8, 서사는
> [`experiment_history.md`](experiment_history.md) §5다.
> **⚠ `runs/paper_final/`의 완료된 캠페인은 `arc_huber`로 학습돼 있다** — 논문 구성에 대한
> 사용자 판단이 남아 있다(결과 문서 §16.7 말미).
>
---

## 🟢 정본

| 문서 | 역할 | 이것만은 알아야 한다 |
|---|---|---|
| [`experiment_history.md`](experiment_history.md) | **서사 정본.** 2-head → 3-class → binary 전환 이유, 기각된 가설, 최종 결론과 근거, 방법론 교훈, 다음 수집 권고 | 개별 숫자의 근거 정본은 아니다 -- 각 절이 원본 문서 절 번호를 가리킨다 |
| [`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) | **근거 정본.** §1–§14 과적합 진단, §15–§19 binary 전환, §20–§21 분할·held-out test, §22 오차 구조, §23 `unknown`의 정체와 지표 축소, §24 시드 분산, §25 요인 구조·CV(**§25.6 LOSO를 읽는 법**, **§25.7 config 동결**, §25.8 사전 예상), §26 경계 대역 loss 분해, §27 lifting 표본 간격, **§28 프로브 실측·라벨 출처·결론 문장**, **§29 stride 8→4(사전 선언 + §29.9 결과=기각)**, **§30 안정성 축 둘**, **§31 LOSO 결과**, **§32 Orin 벤치마크(§32.6 TensorRT future work)**, **§33 lifting 높이 축 Y=1→4(실행 절차·코드 계약)** | 가장 길고(약 2900줄) 가장 자주 인용된다. **새 실측은 여기 절을 추가한다** |
| [`loss_function_spec.md`](loss_function_spec.md) | **loss 형태 정본.** 현재 loss의 수식·기호·확정 하이퍼파라미터·코드 위치. §3 거리장, §4 영역 분할, §5 hard 항, §6 soft target 유도, §7 엔트로피 하한과 KL, §8 기본 `arc_huber`와 실험형 `cumulative_l1`, §9 mode별 λ_R 캘리브레이션 | **설계 근거와 실험 결과는 여기에 쓰지 않는다** -- 그건 아래 설계·결과 문서가 정본이다. 두 문서가 어긋나면 코드가 맞다. **누적형은 §8.8이고, 논문에 쓰는 것은 이쪽이다**(결과 문서 §16.7) |
| [`soft_boundary_loss_design.md`](soft_boundary_loss_design.md) | **loss 설계·결과 정본.** §2–§5 정식화, §6 구현 상태, §7 왜 하이퍼파라미터를 라벨에서 못 얻나, §8 실행법, §9–§10 스윕 10런 실측, §11 확정/기각, §12 보조 loss 계획, §13 `L_range`, §14 판정 프로토콜, §15 n=3 ablation, **§16 threshold sweep -- 최종 결론(정본)**, §17–§18 cm 재진술·교차 앙상블, **§19 `L_range` dead zone 진단**, **§20 경계 target 네 축 스윕(2026-08-27)**, **§21 `(σ, k)` 재매개변수화와 확정 config 변경(2026-08-28)** | **loss 연구는 §16으로 종결됐고 §20이 그 판단의 정량적 근거다.** 결론만 볼 거면 **§16.4**와 **§20.8** 두 표다. **⚠ §21이 확정 config를 바꿨다(δ=0.30, σ=0.10) -- §20.9의 "확정 config는 안 바뀐다"는 철회됐고, `α`는 더 쓰지 않는다.** **⚠ §20.1: `kl_boundary`는 config끼리 비교할 수 없다 -- 옛 절의 그 비교는 인용 금지이고 정본 지표는 `tools/report_boundary_calibration.py`의 `공통 kl`이다.** **§15.5(3)의 "안전 개선"은 §16.2가 철회했고, §13.6~§13.8의 σ 값은 §15.4가 철회했다 -- 순위만 인용한다.** **다음 할 일은 이 문서가 아니다** -- 계획된 실험은 전부 끝났고 남은 것은 논문 집필이다(위 2026-08-26 블록) |
| [`loss_effect_results.md`](loss_effect_results.md) | **loss 영향력 대조 실험의 결과 정본.** 본 실험 25런(2026-09-02)과 `E_cumulative` 5런(§16, 2026-09-21, 총 30런). 현재 확정 config·정본 환경에서 일곱 축을 분리해 판정한다 | 본 실험 결론은 §0·§9. **되올림 원인은 hard boundary target**이고 재현성은 고정 epoch에서도 확인해야 한다. **§16이 논문용 정본이다** -- `D`를 빼고 `E`를 쓰고, 통과한 주장은 `iou_free` +0.0020과 `range_mae` −0.0033 m 둘뿐이며 `missed_obstacle`은 나빠진다. §15는 `λ_R=0.1526`의 재현 시도다. 본 실험 데이터는 `runs/loss_effect/analysis/RESULTS.json`, probe는 `runs/cumulative_ray_loss/` |
| [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) | **지표 정의 정본.** 지표 하나하나의 정의와 채택/기각 사유 | 지표를 추가·삭제·변경할 때 **먼저 읽고 여기에 기록한다** |
| [`finetuning_guide.md`](finetuning_guide.md) | fine-tuning 실행 절차 | 실행 전 데이터 점검 체크리스트가 여기 있다 |
| [`training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md) | 코드 정독 가이드(데이터 → 텐서 → loss → 지표). **§3 항목 5에 수용영역·FLOPs 실측**, §6.5~§6.6에 upstream 함정 둘 | 코드를 처음 만질 때 |
| [`setup_guide_pro6000.md`](setup_guide_pro6000.md) | 환경 세팅 정본 (RTX PRO 6000 Blackwell) | **PyTorch는 `cu128` 빌드여야 한다** -- sm_120에서 cu118 커널은 실행되지 않는다 |
| [`git_workflow.md`](git_workflow.md) | 브랜치·커밋 규칙 | `merge`/`push`는 사용자만 수행한다 |
| [`project_structure.md`](project_structure.md) | 폴더 구조 + **`tools/` 색인**(무엇을 재는 도구이고 그 숫자가 어느 절에 실렸나. 완전하지 않으며 없는 도구는 docstring을 읽는다) | `third_party/`·`mmdetection3d/`는 submodule -- 직접 수정 금지. **재채점 시 `--encoder_type`을 맞춰 넘긴다** |

## 🔵 운영 메모

| 문서 | 역할 |
|---|---|
| [`paper_final_experiments.md`](paper_final_experiments.md) | **논문 최종 설정 재실험의 단일 진행 원장 -- 지금 진행 중인 작업은 여기가 정본이다.** 계획·동결 설정·실행 상태·무결성·결과를 계속 갱신한다. **`01_overall`(40런)·`02_projection_and_prior`(20런) 완료. 남은 것은 `03` 재정리와 `04` Orin 측정** |
| [`paper_package/`](paper_package) | **논문 작성용 패키지.** 표 CSV·그림·해석을 실험별로 모았고 노트북에 그대로 받아 쓴다. `common/`에 지표 정의·보고 규칙·동결 설정. **CSV와 그림은 생성 스크립트가 정본이고 손으로 고치지 않는다** |
| [`next_session_binary_and_verification.md`](next_session_binary_and_verification.md) | 2026-08-19 시점 인수인계에서 **운영 메모로 격하**. 도구 목록·실행 명령·시퀀스 이질성 실측·`cam0..3` 매핑 함정(**`left=cam3`이다**)처럼 다른 곳에 중복되지 않은 실무 정보가 남아 있다. §0의 "논의 중"은 종결됐다 |

## 📚 참고

| 경로 | 내용 |
|---|---|
| [`paper_experiment_compendium.md`](paper_experiment_compendium.md) | **논문 집필용 실험 총정리(2026-08-26).** 모든 실험을 목적→설계→결과(수치)→해석으로 한 곳에 모은 **파생 문서다 -- 정본이 아니다.** 숫자가 정본과 어긋나면 정본이 맞고, **새 실측을 여기에 추가하지 않는다**(정본에 절을 추가한 뒤 옮겨 적는다). 사실과 해석을 표기로 구분하고(§0.3), **철회·정정 16건을 §12에 모아 두었다** |
| [`dataset_analysis/`](dataset_analysis) | SynWoodScape 기하 조사, 수동 라벨링 ROI 결정, WoodScape 분석 |
| `env/` | 환경 고정용 `pip freeze` 스냅샷 |
| `superpowers/` | 스킬 시스템이 생성한 spec·plan (도구가 관리한다 -- 손으로 고치지 않는다) |

## 🗄 아카이브 — [`archive/`](archive)

**완료된 단계의 기록이다. 인용은 되지만 갱신하지 않고, 여기 적힌 플래그·경로·숫자를 그대로
쓰면 안 된다.** 지우지 않는 이유는 정본 문서들이 이 문서의 절 번호를 근거로 인용하기 때문이다.

| 문서 | 무엇의 기록인가 | 왜 수명이 끝났나 |
|---|---|---|
| [`archive/free_space_metric_migration.md`](archive/free_space_metric_migration.md) | 지표 교체(`iou_drivable`→`iou_free`), 2-head 대 3-class A/B, 2-head 제거 | 정식화가 그 뒤 binary로 또 바뀌었다. **여기 로봇 숫자는 옛 split(val=`rawos3` 37장) 기준** |
| [`archive/synwoodscape_pretrain_experiment_log.md`](archive/synwoodscape_pretrain_experiment_log.md) | SynWoodScape 2-head pretraining 실험 | pretrain이 **해롭다고 실측**돼(진단 §11) 현행은 from scratch다 |
| [`archive/finetuning_preparation.md`](archive/finetuning_preparation.md) | Phase 4 fine-tuning 준비 단계 | 준비가 끝났다. 실행은 `finetuning_guide.md`가 정본 |
| [`archive/next_session_threeclass_training.md`](archive/next_session_threeclass_training.md) | 2026-08-18 인수인계(3-class 본학습) | "막힌 곳"이 해결됐다 -- loss를 고치는 대신 정식화를 binary로 바꿨다 |
| [`archive/next_session_synwoodscape_twohead.md`](archive/next_session_synwoodscape_twohead.md) | 2026-08-14 인수인계(2-head pretraining) | 2-head 코드가 제거됐다 |
| [`archive/training_guide.md`](archive/training_guide.md) | 2-head 학습 실행 가이드 | `--lambda_vis`·`--vis_neg_weight`·`--head` 플래그가 전부 사라졌다 |
| [`archive/training_improvement_plan.md`](archive/training_improvement_plan.md) | 2-head 개선 계획서 | `TwoHeadSegnet`과 `iou_drivable`/`iou_obstacle` 전제. 둘 다 제거됐다 |
| [`archive/setup_guide.md`](archive/setup_guide.md) | RTX 3080 환경 세팅 | 서버를 RTX PRO 6000으로 이전했다 |

---

## 문서를 새로 쓸 때 / 고칠 때

- **새 실측은 정본에 절을 추가한다.** 새 문서를 만들면 정본이 둘이 되고, 그게 지금까지
  문서가 불어난 이유였다.
- **"다음 세션 인수인계" 문서를 새로 만들지 않는다.** 세 개가 쌓여 서로 다른 문서가
  "여기서 시작한다"고 주장하는 상태가 됐다. 현재 상태는 `experiment_history.md` §5에 적는다.
- **문서를 아카이브로 옮길 때는 링크를 함께 고친다.** 검증:

  ```bash
  python - <<'PY'
  import re, subprocess
  from pathlib import Path
  files = [f for f in subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split()
           if f.endswith(".md") and not f.startswith(("third_party/","mmdetection3d/","docs/superpowers/"))]
  for path in files:
      p = Path(path)
      for m in re.finditer(r"\]\(([^)#\s]+\.md)[^)]*\)", p.read_text()):
          if not (p.parent / m.group(1)).resolve().exists():
              print("깨진 링크", path, "->", m.group(1))
  PY
  ```

  **코드 주석에도 문서 경로가 들어 있다**(`docs/...md` 형태로 14곳). 옮길 때 같이 고친다.
- 상태가 바뀌면 **이 색인을 먼저 고친다.** 색인이 문서 목록과 어긋나면 색인이 없는 것보다 나쁘다.
