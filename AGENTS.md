# AGENTS.md

이 문서는 AI agent와 협업할 때 항상 참고할 **핵심 지침**입니다. 특정 플랫폼에 종속되지 않는 공통 규칙만 둡니다. 상세 내용은 `docs/`의 개별 문서를 필요할 때만 찾아봅니다.

## [중요] 소통 규칙

- **사용자와의 모든 대화는 한국어로 한다.** 최종 답변뿐 아니라 **작업 중간의 진행 설명·판단 근거·질문도 한국어**로 쓴다.
- 예외는 두 가지뿐: **커밋 메시지는 영어**(아래 Git 규칙), 그리고 코드 안의 식별자·주석은 주변 코드 스타일을 따른다.

## 프로젝트 한 줄 요약

자체 구축한 Fisheye 4-cam 데이터셋으로 **BEV occupancy map**(각 BEV 격자 셀이 주행 가능한 영역인지 여부)을 예측하는 모델을 학습한다. baseline은 **Simple-BEV**.

## Task & 모델

- Task: 어안 이미지 → **BEV occupancy map** (drivable / non-drivable). 3D bbox 검출과 세밀 semantic 구분은 **이후 확장 과제**로 분리.
- baseline: **Simple-BEV** (`third_party/models/simple_bev`) — mmdet3d에 의존하지 않는 standalone PyTorch 구현.
- 학습 순서: **SynWoodScape로 먼저 학습·검증 → pre-training → 자체 데이터셋 fine-tuning.**
  - pretraining ✅ 완료 (SynWoodScape 4-cam, `radial_poly`, 240×240 그리드)
  - fine-tuning ⬅️ 현재 단계 — **자체 리그는 3-cam(front/left/right), Double Sphere,
    120×120 그리드다.** 리그에 카메라는 4대지만 rear는 라벨 생성에 쓰이지 않았다.
    실행 방법은 [`docs/finetuning_guide.md`](docs/finetuning_guide.md)가 정본.
- **MMDetection3D는 현재 학습 경로에서 사용하지 않는다.** `mmdetection3d/` submodule은 이후 3D 검출로 확장할 경우를 위해 남겨둔 것일 뿐이다.
- 상세: [`README.md`](README.md)

## 개발 환경

- conda 환경명: **`bev-chamdog`** (Python 3.11)
- **PyTorch는 반드시 `cu128` 빌드**(2.7.0+cu128)를 쓴다. GPU가 sm_120이라 `cu118` 빌드로는 커널이 실행되지 않는다. `torch.cuda.is_available()`이 True로 나와도 실제 연산에서 죽으므로 속지 말 것.
- 의존성은 `constraints.txt`(numpy<2·opencv<5) + `requirements.txt` 조합으로 설치하고, 이후 항상 `pip check`.
- **`third_party/models/simple_bev/requirements.txt`를 그대로 쓰면 안 된다** (Python 3.7~3.8 시절 핀 → 3.11에서 설치 실패). 루트의 `requirements.txt`를 쓴다.
- **mmcv / mmdet / mmdet3d는 설치하지 않는다.** prebuilt wheel이 torch 2.1까지만 존재한다. 이후 3D 검출로 확장할 때는 별도 conda 환경으로 분리한다.
- 상세: [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md) (이전 RTX 3080 환경 이력은 [`docs/setup_guide.md`](docs/setup_guide.md))

## 하드웨어

- GPU: **RTX PRO 6000 Blackwell / VRAM 96GB** (단일 GPU, sm_120)
- CPU 32코어 / RAM 250GB / `/dev/shm` 126GB
- 데이터셋은 루트 파티션이 아니라 **`/data`(3.7T)** 에 두고 symlink를 건다.
- VRAM이 넉넉해 `batch_size` 축소나 gradient accumulation 회피가 **불필요**하다. 오히려 **데이터 로딩이 병목**이 되기 쉬우므로 `num_workers`(8~16)와 GPU 활용률을 먼저 살핀다. 최대 batch size는 추측하지 말고 실측한다.
- 상세: [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md)

## 폴더 규칙

- 커스텀 코드는 `projects/`(데이터 로더·투영·모델 래퍼 등 파이썬 모듈), `configs/`(config), `tools/`(스크립트)에 둔다.
- **`third_party/`와 `mmdetection3d/`는 git submodule → 직접 수정하지 말 것.** Simple-BEV 동작을 바꿔야 하면 submodule을 고치지 않고 `projects/`에서 래핑한다.
- 상세: [`docs/project_structure.md`](docs/project_structure.md)

## [중요] Git 규칙

- **`commit`은 자율적으로 수행해도 됨. 단, 큰 논리 단위로 묶어서 남긴다** (자잘한 변경마다 쪼개지 말 것).
- **커밋 메시지는 영어로 작성한다** (파일/문서 내용은 한글 무방).
- **`merge`와 `push`는 절대 임의로 수행하지 말 것 — 반드시 사용자가 직접 수행한다.**
- 기능 개발은 항상 새 브랜치(`feat/*` 등)에서 진행한다.
- 상세: [`docs/git_workflow.md`](docs/git_workflow.md)
