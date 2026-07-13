# CLAUDE.md

이 문서는 개발 시 항상 참고할 **핵심 지침**입니다. 상세 내용은 `@docs/`의 개별 문서를 필요할 때만 찾아봅니다.

## 프로젝트 한 줄 요약

자체 구축한 Fisheye 4-cam 데이터셋으로 BEV 3D 검출 모델을 학습한다. 학습 프레임워크는 MMDetection3D(v1.4.0).

## 개발 환경

- conda 환경명: **`mmdet3d`** (Python 3.9)
- 핵심 버전: PyTorch 2.1.0+cu118 / torchvision 0.16.0+cu118 / mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 / **numpy<2 고정**
- 패키지 설치/업데이트 시 `constraints.txt`로 numpy<2·opencv<5 강제, 이후 항상 `pip check`.
- 상세: `@docs/setup_guide.md`

## 하드웨어

- GPU: **RTX 3080 / VRAM 10GB** (단일 GPU)
- mmdet3d 기본 config는 다중 GPU 기준이 많음 → `batch_size` 축소 또는 gradient accumulation 필요. **첫 학습 시 OOM 주의.**
- 상세: `@docs/setup_guide.md`

## 폴더 규칙

- 커스텀 코드는 `projects/`(mmdet3d 등록 모듈), `configs/`(config), `tools/`(스크립트)에 둔다.
- **`mmdetection3d/`는 git submodule(v1.4.0 고정) → 직접 수정하지 말 것.** 확장은 `projects/`에서.
- 상세: `@docs/project_structure.md`

## [중요] Git 규칙

- **`commit`은 자율적으로 수행해도 됨. 단, 큰 논리 단위로 묶어서 남긴다** (자잘한 변경마다 쪼개지 말 것).
- **커밋 메시지는 영어로 작성한다** (파일/문서 내용은 한글 무방).
- **`merge`와 `push`는 절대 임의로 수행하지 말 것 — 반드시 사용자가 직접 수행한다.**
- 기능 개발은 항상 새 브랜치(`feature/*` 등)에서 진행한다.
- 상세: `@docs/git_workflow.md`
