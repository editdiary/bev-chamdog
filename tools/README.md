# tools/

데이터 변환·분석·시각화·검증 등 보조 스크립트를 두는 폴더입니다. 학습/평가 실행 스크립트도 여기에 둡니다.

예상 용도:
- 벤치마크 데이터셋(WoodScape/SynWoodScape) 탐색·통계·시각화 — 현재 `woodscape_viz/`
- SynWoodScape / 자체 데이터셋을 학습용 형태로 변환하는 전처리 스크립트
- 데이터 로딩·BEV GT·어안 투영 검증 스크립트
- Simple-BEV 학습·평가 실행 스크립트 (`projects/`의 모듈을 조합)

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
