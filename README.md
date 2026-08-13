# bev-chamdog

자체 구축한 **Fisheye 4-cam 데이터셋**으로 **BEV(Bird's-Eye-View) occupancy map**을 예측하는 모델을 학습하는 것을 최종 목표로 하는 프로젝트입니다. baseline 모델은 [Simple-BEV](https://github.com/aharley/simple_bev)이며, standalone PyTorch로 동작합니다.

## 🎯 목표

서라운드뷰 fisheye 카메라 4대의 이미지만으로 **로봇 주변의 BEV occupancy map**을 예측한다. 각 BEV 격자 셀이 **로봇이 지나갈 수 있는 영역(drivable)인지 아닌지**를 판단하는 task이다.

즉 3D bounding box 검출이 아니라, **주행성(traversability) 판단을 위한 BEV 상의 영역 예측**이 목표다. 세밀한 semantic 구분이나 3D 객체 검출은 이후 확장 과제로 분리한다.

## 🗺️ 접근 흐름

1. **벤치마크 분석** — `WoodScape` / `SynWoodScape`의 구조·annotation·fisheye 특성 파악 *(완료)*
2. **자체 데이터셋 구축** — 수집 → 캘리브레이션 → 라벨링 *(별도 프로젝트에서 완료)*
3. **SynWoodScape로 학습** — Simple-BEV baseline이 어안 4-cam 입력으로 동작하는지 검증 ← *현재 단계*
4. **자체 데이터셋으로 fine-tuning** — 3단계가 잘 되는 것을 확인한 뒤, SynWoodScape pre-training → 자체 데이터셋 fine-tuning

## 🔧 학습 프레임워크

현재 task에는 **MMDetection3D가 필요하지 않습니다.** Simple-BEV는 mmdet3d에 의존하지 않는 standalone PyTorch 구현이라, `third_party/models/simple_bev`를 직접 사용합니다.

`mmdetection3d/` submodule은 이후 3D 검출로 확장할 경우를 위해 남겨두었을 뿐, 현재 학습 경로에서는 사용하지 않습니다.

## 📁 폴더 구조

```
bev-chamdog/
├── README.md              # 프로젝트 개요 (현재 문서)
├── AGENTS.md              # AI agent 협업 시 참고할 핵심 지침
├── CLAUDE.md              # Claude 호환용 포인터 (정본은 AGENTS.md)
├── ROADMAP.md             # 단계별 로드맵
├── constraints.txt        # 패키지 버전 상한 고정 (numpy<2, opencv<5)
├── requirements.txt       # Simple-BEV 학습에 실제로 필요한 의존성
├── docs/                  # 상세 문서 (환경 세팅, git 워크플로, 구조, 분석 노트, study)
├── dataset/               # 데이터셋 (내용물은 git 미추적 → dataset/README.md 참고)
├── third_party/           # 참고용 외부 저장소 (git submodule — 직접 수정 금지)
│   ├── datasets/WoodScape #   WoodScape 공식 툴킷 (캘리브레이션 규약·시각화 참고)
│   └── models/            #   simple_bev(baseline), lift-splat-shoot, bevformer, BEVDet
├── mmdetection3d/         # MMDetection3D (git submodule, v1.4.0 — 현재 미사용)
├── configs/               # 커스텀 학습/추론 config
├── tools/                 # 커스텀 스크립트 (데이터 변환·분석·시각화)
└── projects/              # 커스텀 모듈 (데이터 로더·투영·모델 래퍼 등)
```

각 폴더의 상세 역할과 사용법은 [`docs/project_structure.md`](docs/project_structure.md) 참고.

## ✅ 진행 상황

- [x] **개발 환경 세팅** — conda 환경 및 버전 고정 (→ [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md))
- [x] **벤치마크 분석** — WoodScape 분석 완료 (→ [`docs/dataset_analysis/`](docs/dataset_analysis/)), SynWoodScape 구조 파악
- [x] **자체 데이터셋 구축** — 별도 프로젝트에서 수집·캘리브레이션·라벨링 마무리
- [ ] **SynWoodScape + Simple-BEV 학습** ← *현재 단계*
- [ ] **자체 데이터셋 fine-tuning**

## 🚀 시작하기

### 1) 저장소 클론 (submodule 포함)

```bash
git clone --recursive https://github.com/editdiary/bev-chamdog.git
# 이미 클론한 경우:
git submodule update --init --recursive
```

`third_party/`와 `mmdetection3d/`는 submodule이므로 `--recursive` 없이 클론하면 빈 폴더로 받아집니다.

### 2) 개발 환경 세팅

```bash
conda create -n bev-chamdog python=3.11 -y && conda activate bev-chamdog
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -c constraints.txt -r requirements.txt
pip check
```

> ⚠️ **PyTorch는 반드시 `cu128` 빌드**여야 합니다. GPU가 sm_120(Blackwell)이라 `cu118` 빌드로는 커널이 실행되지 않으며, `torch.cuda.is_available()`이 True로 나와도 실제 연산에서 죽습니다.

전체 절차·검증 스크립트·트러블슈팅은 [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md)에 정리되어 있습니다. mmcv·mmdet·mmdet3d는 현재 학습 경로에서 사용하지 않으므로 설치하지 않습니다. (이전 RTX 3080 환경의 세팅 이력은 [`docs/setup_guide.md`](docs/setup_guide.md)에 남겨두었습니다.)

학습 하드웨어는 **RTX PRO 6000 Blackwell / VRAM 96GB 단일 GPU**입니다. VRAM이 넉넉해 batch size 축소나 gradient accumulation 회피는 불필요하고, 오히려 데이터 로딩이 병목이 되기 쉽습니다.

## 🔀 개발 / 버전 관리

`main`(안정) · `develop`(핵심 개발) 브랜치를 두고, 기능 개발은 항상 별도 브랜치에서 진행합니다. 자세한 규칙은 [`docs/git_workflow.md`](docs/git_workflow.md) 참고.
