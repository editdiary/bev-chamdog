# bev-chamdog

자체 구축한 **Fisheye 4-cam 데이터셋**으로 **BEV(Bird's-Eye-View) 3D 검출 모델**을 학습하는 것을 최종 목표로 하는 프로젝트입니다. [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)를 학습 프레임워크의 주축으로 사용합니다.

## 🎯 목표

서라운드뷰 fisheye 카메라 4대로 촬영한 자체 데이터를 BEV 학습에 사용할 수 있는 형태로 구축하고, 이를 이용해 BEV 모델을 학습·평가한다.

## 🗺️ 접근 흐름

1. **벤치마크 분석** — `WoodScape` / `SynWoodScape` 데이터셋의 구조·annotation·fisheye 특성을 깊게 파악
2. **자체 데이터셋 구축** — 수집 → 캘리브레이션 → 라벨링 → mmdet3d 호환 포맷 정의
3. **mmdet3d 통합** — 커스텀 dataset 클래스 · transform · config 작성 (`projects/`)
4. **학습 & 실험** — baseline 학습, 하드웨어 제약(RTX 3080) 튜닝, 평가

전체 단계별 계획은 [`ROADMAP.md`](ROADMAP.md) 참고.

## 📁 폴더 구조

```
bev-chamdog/
├── README.md              # 프로젝트 개요 (현재 문서)
├── CLAUDE.md              # 개발 시 참고할 핵심 지침
├── ROADMAP.md             # 단계별 로드맵
├── constraints.txt        # 패키지 버전 고정 (numpy<2 등)
├── docs/                  # 상세 문서 (환경 세팅, git 워크플로, 구조, 분석 노트)
├── dataset/               # 데이터셋 (내용물은 git 미추적 → dataset/README.md 참고)
├── mmdetection3d/         # MMDetection3D (git submodule, v1.4.0 고정 — 직접 수정 금지)
├── configs/               # 커스텀 학습/추론 config
├── tools/                 # 커스텀 스크립트 (데이터 변환·분석·시각화)
└── projects/              # 커스텀 mmdet3d 모듈 (dataset/transform 등, registry 등록)
```

각 폴더의 상세 역할과 사용법은 [`docs/project_structure.md`](docs/project_structure.md) 참고.

## ✅ 진행 상황

- [x] **Phase 0** — 개발 환경 세팅 & mmdet3d 데모 검증 완료 (→ [`docs/setup_guide.md`](docs/setup_guide.md))
- [ ] **Phase 1** — 벤치마크 데이터셋(WoodScape/SynWoodScape) 분석 ← *다음 단계*
- [ ] **Phase 2** — 자체 BEV 데이터셋 구축
- [ ] **Phase 3** — mmdet3d 커스텀 통합
- [ ] **Phase 4** — BEV 모델 학습 & 실험

## 🚀 시작하기

### 1) 저장소 클론 (submodule 포함)

```bash
git clone --recursive https://github.com/editdiary/bev-chamdog.git
# 이미 클론한 경우:
git submodule update --init --recursive
```

`mmdetection3d/`는 submodule이므로 `--recursive` 없이 클론하면 빈 폴더로 받아집니다.

### 2) 개발 환경 세팅

conda 환경 및 패키지 버전 세팅 과정은 [`docs/setup_guide.md`](docs/setup_guide.md)에 정리되어 있습니다. (핵심: Python 3.9 / PyTorch 2.1.0+cu118 / mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 / **numpy<2**)

## 🔀 개발 / 버전 관리

`main`(안정) · `develop`(핵심 개발) 브랜치를 두고, 기능 개발은 항상 별도 브랜치에서 진행합니다. 자세한 규칙은 [`docs/git_workflow.md`](docs/git_workflow.md) 참고.
