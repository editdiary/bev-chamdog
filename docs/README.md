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

## 처음 오는 사람이 읽을 순서

1. [`experiment_history.md`](experiment_history.md) — **여기서 시작한다.** 무엇을 시도했고
   무엇을 배웠고 왜 지금 여기인지가 한 문서에 있다.
2. [`../AGENTS.md`](../AGENTS.md) — 작업 규칙(소통·git·폴더·환경)
3. [`loss_function_spec.md`](loss_function_spec.md) — **지금 쓰는 loss가 수식으로 정확히 무엇인가.** 근거·결과는 없고 형태만 있다
4. [`soft_boundary_loss_design.md`](soft_boundary_loss_design.md) — **현재 진행 중인 작업.** §12가 다음 할 일
4. [`finetuning_guide.md`](finetuning_guide.md) — 실제로 학습을 돌리는 순서
5. [`training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md) — 코드가 데이터를
   어떻게 읽고 어떤 텐서가 되어 어떤 loss로 학습되는지

---

## 🟢 정본

| 문서 | 역할 | 이것만은 알아야 한다 |
|---|---|---|
| [`experiment_history.md`](experiment_history.md) | **서사 정본.** 2-head → 3-class → binary 전환 이유, 기각된 가설, 최종 결론과 근거, 방법론 교훈, 다음 수집 권고 | 개별 숫자의 근거 정본은 아니다 -- 각 절이 원본 문서 절 번호를 가리킨다 |
| [`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) | **근거 정본.** §1–§14 과적합 진단, §15–§19 binary 전환, §20–§21 분할·held-out test, §22 오차 구조, §23 `unknown`의 정체와 지표 축소, §24 시드 분산, §25 요인 구조·CV(**§25.6 LOSO를 읽는 법**), §26 경계 대역 loss 분해, §27 lifting 표본 간격, **§28 프로브 실측·라벨 출처·결론 문장**, **§29 stride 8→4 사전 선언** | 가장 길고(1500줄) 가장 자주 인용된다. 새 실측은 여기 절을 추가한다 |
| [`loss_function_spec.md`](loss_function_spec.md) | **loss 형태 정본.** 현재 loss의 수식·기호·확정 하이퍼파라미터·코드 위치. §3 거리장, §4 영역 분할, §5 hard 항, §6 soft target 유도, §7 엔트로피 하한과 KL, §8 `L_range`, §9 λ_R 캘리브레이션 | **설계 근거와 실험 결과는 여기에 쓰지 않는다** -- 그건 아래 설계 문서가 정본이다. 두 문서가 어긋나면 코드가 맞다 |
| [`soft_boundary_loss_design.md`](soft_boundary_loss_design.md) | **loss 설계·결과 정본.** §2–§5 정식화, §6 구현 상태, §7 왜 하이퍼파라미터를 라벨에서 못 얻나, §8 실행법, §9–§10 스윕 10런 실측, §11 확정/기각, §12 보조 loss 계획, §13 `L_range`, §14 판정 프로토콜, §15 n=3 ablation, **§16 threshold sweep -- 최종 결론(정본)** | **loss 연구는 §16으로 종결됐다.** 결론만 볼 거면 **§16.4** 한 표다. **§15.5(3)의 "안전 개선"은 §16.2가 철회했고, §13.6~§13.8의 σ 값은 §15.4가 철회했다 -- 순위만 인용한다.** 다음 할 일은 이 문서가 아니라 `experiment_history.md` §5에 있다 |
| [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) | **지표 정의 정본.** 지표 하나하나의 정의와 채택/기각 사유 | 지표를 추가·삭제·변경할 때 **먼저 읽고 여기에 기록한다** |
| [`finetuning_guide.md`](finetuning_guide.md) | fine-tuning 실행 절차 | 실행 전 데이터 점검 체크리스트가 여기 있다 |
| [`training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md) | 코드 정독 가이드(데이터 → 텐서 → loss → 지표) | 코드를 처음 만질 때 |
| [`setup_guide_pro6000.md`](setup_guide_pro6000.md) | 환경 세팅 정본 (RTX PRO 6000 Blackwell) | **PyTorch는 `cu128` 빌드여야 한다** -- sm_120에서 cu118 커널은 실행되지 않는다 |
| [`git_workflow.md`](git_workflow.md) | 브랜치·커밋 규칙 | `merge`/`push`는 사용자만 수행한다 |
| [`project_structure.md`](project_structure.md) | 폴더 구조 | `third_party/`·`mmdetection3d/`는 submodule -- 직접 수정 금지 |

## 🔵 운영 메모

| 문서 | 역할 |
|---|---|
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
