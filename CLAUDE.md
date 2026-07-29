# CLAUDE.md

이 문서는 개발 시 항상 참고할 **핵심 지침**입니다. 상세 내용은 `@docs/`의 개별 문서를 필요할 때만 찾아봅니다.

## [중요] 소통 규칙

- **사용자와의 모든 대화는 한국어로 한다.** 최종 답변뿐 아니라 **작업 중간의 진행 설명·판단 근거·질문도 한국어**로 쓴다.
- 예외는 두 가지뿐: **커밋 메시지는 영어**(아래 Git 규칙), 그리고 코드 안의 식별자·주석은 주변 코드 스타일을 따른다.

## 프로젝트 한 줄 요약

자체 구축한 Fisheye 4-cam 데이터셋으로 **BEV occupancy map**(각 BEV 격자 셀이 주행 가능한 영역인지 여부)을 예측하는 모델을 학습한다. baseline은 **Simple-BEV**.

## Task & 모델

- Task: 어안 4-cam 이미지 → **BEV occupancy map** (drivable / non-drivable). 3D bbox 검출과 세밀 semantic 구분은 **이후 확장 과제**로 분리.
- baseline: **Simple-BEV** (`third_party/models/simple_bev`) — mmdet3d에 의존하지 않는 standalone PyTorch 구현.
- 학습 순서: **SynWoodScape로 먼저 학습·검증 → pre-training → 자체 데이터셋 fine-tuning.**
- **MMDetection3D는 현재 학습 경로에서 사용하지 않는다.** `mmdetection3d/` submodule은 이후 3D 검출로 확장할 경우를 위해 남겨둔 것일 뿐이다.
- 상세: `@README.md`

## 개발 환경

- conda 환경명: **`mmdet3d`** (Python 3.9) — 초기 세팅 때 붙은 이름이며, 현재 학습 경로와는 무관하다.
- Simple-BEV 학습에 실제로 필요한 것: PyTorch 2.1.0+cu118 / torchvision 0.16.0+cu118 / **numpy<2 고정**.
- 같은 환경에 mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0도 설치되어 있으나 현재 쓰지 않는다.
- 패키지 설치/업데이트 시 `constraints.txt`로 numpy<2·opencv<5 강제, 이후 항상 `pip check`.
- 상세: `@docs/setup_guide.md`

## 하드웨어

- GPU: **RTX 3080 / VRAM 10GB** (단일 GPU)
- 공개 BEV 모델 config는 다중 GPU 기준이 많음 → `batch_size` 축소 또는 gradient accumulation 필요. **첫 학습 시 OOM 주의.**
- 상세: `@docs/setup_guide.md`

## 폴더 규칙

- 커스텀 코드는 `projects/`(데이터 로더·투영·모델 래퍼 등 파이썬 모듈), `configs/`(config), `tools/`(스크립트)에 둔다.
- **`third_party/`와 `mmdetection3d/`는 git submodule → 직접 수정하지 말 것.** Simple-BEV 동작을 바꿔야 하면 submodule을 고치지 않고 `projects/`에서 래핑한다.
- 상세: `@docs/project_structure.md`

## [중요] Git 규칙

- **`commit`은 자율적으로 수행해도 됨. 단, 큰 논리 단위로 묶어서 남긴다** (자잘한 변경마다 쪼개지 말 것).
- **커밋 메시지는 영어로 작성한다** (파일/문서 내용은 한글 무방).
- **`merge`와 `push`는 절대 임의로 수행하지 말 것 — 반드시 사용자가 직접 수행한다.**
- 기능 개발은 항상 새 브랜치(`feat/*` 등)에서 진행한다.
- 상세: `@docs/git_workflow.md`
