# 로드맵 (ROADMAP)

> 프로젝트의 큰 흐름을 잡기 위한 가이드입니다. 반드시 이대로 지켜야 하는 것은 아니며, 진행하면서 **유동적으로 변경**될 수 있습니다.

**최종 목표:** 자체 구축한 Fisheye 4-cam 데이터셋으로 BEV 모델을 학습·평가한다.

---

## Phase 0 — 개발 환경 세팅 & 데모 검증 ✅ 완료

- conda 환경 및 mmdet3d 스택 구축, 버전 고정 (→ `docs/setup_guide.md`)
- GPU 인식, CUDA 커스텀 연산, 모델 빌드, 추론 데모(`pcd_demo.py`) 정상 동작 확인

## Phase 1 — 벤치마크 데이터셋 분석 ⬅️ 다음 단계

- **WoodScape / SynWoodScape** 데이터셋 구조 파악
  - 폴더/파일 구성, 이미지·캘리브레이션·annotation 포맷
  - fisheye 카메라 모델(왜곡), 4-cam 배치, 좌표계
  - BEV 관점에서 어떤 정보가 어떻게 필요한지 정리
- 분석 노트는 `docs/dataset_analysis/`에 기록, 탐색 스크립트는 `tools/`에 정리

## Phase 2 — 자체 BEV 데이터셋 구축

- 데이터 수집 → 카메라 캘리브레이션 → 라벨링(annotation)
- mmdet3d에서 로드 가능한 데이터 포맷/디렉터리 구조 정의
- 벤치마크에서 파악한 구조를 참고해 자체 포맷 설계

## Phase 3 — mmdet3d 커스텀 통합

- `projects/`에 커스텀 dataset 클래스 · transform 작성 (registry 등록)
- `configs/`에 학습/추론 config 작성
- 소규모 샘플로 데이터 로딩·시각화 파이프라인 검증

## Phase 4 — BEV 모델 학습 & 실험

- baseline 모델 학습 (RTX 3080 제약에 맞춘 batch/accumulation 튜닝)
- 평가 지표 산출, 실험 반복 및 개선

## Phase 5 — 분석 · 개선 · 문서화

- 결과 분석, 실패 사례 정리, 개선 반복
- 재현 가능한 형태로 문서/코드 정리
