# tools/

데이터 변환·분석·시각화·검증 등 보조 스크립트를 두는 폴더입니다. 학습/평가 실행 스크립트도 여기에 둡니다.

## 지금 쓰는 것 (캠페인 v3, 2026-10-04)

| 무엇 | 스크립트 |
|---|---|
| 학습 진입점 | `train_robot_bev.py`(자체 데이터), `train_synwoodscape.py`(사전학습). 셸 진입점은 `configs/` |
| 캠페인 드라이버 | `run_full_campaign.sh`(학습 → 끊긴 런 정리·재시도 → 분석), `run_full_campaign_packages.sh`(논문 패키지·그림) |
| 실험별 러너 | `run_paper_final_overall.py`, `run_paper_final_projection_prior.py`, `run_paper_final_native_prior.py` |
| 분석 | `export_run_scalars.py`, `export_val_predictions.py`, `verify_val_predictions.py`(확률맵 재채점), `summarize_repeats.py`, `report_paired_arms.py`, `report_loso.py`, `report_threshold_sweep.py`, `build_results_bundle.py` |
| 검증·정리 | `clean_incomplete_runs.py`(**지금 돌리는 캠페인 루트에만**), `audit_paper_prose_numbers.py`(산문 숫자 감사), `paper_final_run_command.py`(멈춤 감시) |
| 지연 측정 | `benchmark_inference.py` |

절차 정본은 `docs/paper_campaign_protocol.md` §9("지금 쓰는 도구 목록")다. 그 밖의 `report_*`·`probe`·`sweep`
관련 스크립트는 v3 이전 연구에 쓴 것이고 근거 문서는 `docs/archive/research/`에 있다.

> 재사용 가능한 로직은 `projects/`의 모듈로 두고, `tools/`의 스크립트는 그것을 호출하는 얇은 실행 계층으로 유지합니다. 실행은 저장소 루트에서 합니다.

## BEV occupancy GT 생성 스크립트 — 여러 세대가 공존함

시행착오를 거치며 같은 목적의 스크립트가 여러 개 남아 있습니다. **`build_hybrid_occupancy.py`가
현재 파이프라인**이고, 나머지는 폐기된 중간 단계입니다(코드는 참고용으로 남겨둠). 자세한 이유는
`docs/dataset_analysis/synwoodscape_geometry_findings.md` §3 참고.

| 스크립트 | 상태 |
|---|---|
| `build_hybrid_occupancy.py` | **현재.** class 값=top-down label, observed=raycast. 출력: `dataset/synwoodscape_occupancy_gt/` |
| `build_occupancy_gt.py` | 폐기 (visibility 축 없음 — 모든 cell에 항상 값이 채워짐) |
| `build_visibility_mask.py` | 폐기 (z=0 가설검정 — 근거리 캘리브레이션 오차로 오탐) |
| `build_raycast_occupancy.py` | 폐기 (어안 카메라 자신의 semantic label로 raycast — FV 카메라가 자기 차 본네트를 "road"로 잘못 라벨링하는 결함을 그대로 물려받음) |
