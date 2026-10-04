# docs/ 색인

**[2026-10-04] 정리했다.** 캠페인 v3가 끝나 논문 숫자의 정본이 `99_paper_results/`로 옮겨 갔고,
그 이전 단계의 연구 기록은 `archive/`로 보냈다. 지금 위에 남은 문서는 **현재 유효한 것**뿐이다.

| 상태 | 뜻 |
|---|---|
| 🟢 **정본** | 지금 유효하고 계속 갱신한다 |
| 📚 **참고** | 사실 조사·환경 기록. 시점 의존성이 낮다 |
| 🗄 **아카이브**(`archive/`) | 끝난 단계의 기록. **인용은 되지만 갱신하지 않는다.** 플래그·경로·숫자를 그대로 쓰면 안 된다 |

---

## 🟢 정본

| 문서 | 역할 |
|---|---|
| [`99_paper_results/`](99_paper_results) | **논문 숫자·표·그림·해석의 정본** — 캠페인 v3(90런, 100 epoch, 광선 보조항 없음, 새 지표). 실험별 `REPORT.md`, 공통 정의는 `common/`. CSV와 그림은 생성 스크립트가 정본이고 손으로 고치지 않는다 |
| [`paper_final_experiments.md`](paper_final_experiments.md) | **실행 원장** — 무엇을 언제 어떻게 돌렸고 무결성이 어땠나. v3는 §5 "2026-10-02 (8)". §4·§6의 표는 옛 캠페인 이력이다 |
| [`paper_campaign_protocol.md`](paper_campaign_protocol.md) | **캠페인을 다시 돌릴 때의 사용법** — Phase 0~4 절차, 비용 실측, **실제로 물린 지뢰 목록** |
| [`loss_function_spec.md`](loss_function_spec.md) | 손실함수의 수식·기호·코드 위치. **v3는 `L_range`를 쓰지 않는다**(§8은 이력) |
| [`training_pipeline_walkthrough.md`](training_pipeline_walkthrough.md) | 코드 정독 가이드(데이터 → 텐서 → loss → 지표) |
| [`project_structure.md`](project_structure.md) | 폴더 구조와 `tools/` 색인 |
| [`setup_guide_pro6000.md`](setup_guide_pro6000.md) | 학습 서버 환경 세팅(RTX PRO 6000, `cu128`) |
| [`git_workflow.md`](git_workflow.md) | 브랜치·커밋 규칙. `merge`/`push`는 사용자만 한다 |

## 📚 참고

| 문서 | 역할 |
|---|---|
| [`dataset_analysis/`](dataset_analysis) | SynWoodScape 기하 조사, 수동 라벨링 ROI 결정, WoodScape 분석 |
| [`env/`](env) | 학습 서버 `pip freeze` 스냅샷 |

## 🗄 아카이브 — [`archive/`](archive)

| 위치 | 무엇 |
|---|---|
| [`archive/research/`](archive/research) | v3 이전의 **연구 기록**. 과적합 진단·binary 전환·LOSO·Orin 벤치마크(`finetune_overfitting_diagnosis.md`), 서사(`experiment_history.md`), 논문용 총정리(`paper_experiment_compendium.md`), loss 설계·결과(`soft_boundary_loss_design.md`, `loss_effect_results.md`), 옛 지표 설계(`BEV_loss_and_metrics_design.md`). **Discussion의 근거를 찾을 때 여기를 본다.** 숫자는 옛 설정 값이다 |
| [`archive/paper_package_v2/`](archive/paper_package_v2) | 옛 캠페인(2026-09) 논문 패키지. 지표·목적함수가 v3와 달라 **숫자를 섞지 않는다** |
| [`archive/paper_campaign_v3_plan.md`](archive/paper_campaign_v3_plan.md) | v3 실행 전 계획서(결정 §6·§7) |
| [`archive/superpowers/`](archive/superpowers) | 2026-07~09 작업 계획·명세 |
| `archive/` 나머지 | 2-head·3-class 시절 인수인계·가이드, 옛 서버(RTX 3080) 세팅, fine-tuning 준비 기록 |

`docs/temp/`(외부 검토와 주고받은 메모)는 2026-10-04에 지웠다. git 이력에 남아 있다.

---

## 문서를 새로 쓸 때 / 고칠 때

- **새 실측은 정본에 절을 추가한다.** 새 문서를 만들면 정본이 둘이 되고, 그게 지금까지
  문서가 불어난 이유였다.
- **"다음 세션 인수인계" 문서를 새로 만들지 않는다.** 현재 상태는 원장 §5에 적는다.
- **문서를 옮길 때는 링크를 함께 고친다.** 검증:

  ```bash
  python - <<'PY'
  import re, subprocess
  from pathlib import Path
  files = [f for f in subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split()
           if f.endswith(".md") and not f.startswith(("third_party/","mmdetection3d/","docs/archive/"))]
  for path in files:
      p = Path(path)
      for m in re.finditer(r"\]\(([^)#\s]+\.md)[^)]*\)", p.read_text()):
          if not (p.parent / m.group(1)).resolve().exists():
              print("깨진 링크", path, "->", m.group(1))
  PY
  ```

  **코드 주석에도 문서 경로가 들어 있다**(`docs/...md` 형태). 옮길 때 같이 고친다.
- 상태가 바뀌면 **이 색인을 먼저 고친다.**
