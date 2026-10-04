# AGENTS.md

AI agent와 협업할 때 항상 참고할 **핵심 지침**이다. 특정 플랫폼에 종속되지 않는 공통 규칙만 둔다.
어떤 문서를 믿어야 하는지는 [`docs/README.md`](docs/README.md)가 정한다.

> **[2026-10-04] 다시 썼다.** 2026-08-26 ~ 10-03에 이 파일 위에 쌓였던 날짜별 상태 블록(loss 연구,
> 옛 캠페인 두 번, 철회 기록)은 [`docs/archive/agents_history_2026-08_to_10.md`](docs/archive/agents_history_2026-08_to_10.md)로
> 옮겼다. 거기 적힌 숫자·플래그는 옛 설정 값이다.

## 현재 상태 (2026-10-04)

- **논문 실험 캠페인 v3 완료.** 90런(`runs/99_full_campaign/`), 100 epoch, 광선 보조항 없음, 새 지표.
  **논문 숫자·표·그림·해석의 정본은 [`docs/99_paper_results/`](docs/99_paper_results/README.md)**다.
  실행 기록은 원장 [`docs/paper_final_experiments.md`](docs/paper_final_experiments.md) §5 "2026-10-02 (8)".
- **남은 실험: Jetson AGX Orin 엣지 배포 측정**(논문 Results 4절). 옛 측정 20.2 FPS(512×288 fp16,
  PyTorch)는 **`Y=1` 모델 값이라 `Y=4`로 다시 재야 한다.** TensorRT는 아직 해 보지 않았고 목표 FPS는 미정이다.
  옛 측정의 절차·환경 함정은 [`docs/archive/research/finetune_overfitting_diagnosis.md`](docs/archive/research/finetune_overfitting_diagnosis.md) §32.
- **가중치는 v3 마지막 체크포인트 90개만 남아 있다**(2026-10-04 정리, `runs/99_full_campaign/**/model-000000100.pth`).
  다른 트리는 로그·분석만 있어 다시 채점할 수 없다.
- **캠페인을 다시 돌릴 때**는 [`docs/paper_campaign_protocol.md`](docs/paper_campaign_protocol.md)부터 읽는다
  (Phase 0~4 절차, 실제로 물린 지뢰 목록).

## [중요] 소통 규칙

- **사용자와의 모든 대화는 한국어로 한다.** 작업 중간의 진행 설명·판단 근거·질문도 한국어로 쓴다.
- 예외는 두 가지뿐: **커밋 메시지는 영어**(아래 Git 규칙), 코드 안의 식별자·주석은 주변 코드 스타일을 따른다.

## 프로젝트 한 줄 요약

온실 로봇의 **전방 어안 카메라 3대**(front·left·right, Double Sphere 모델)로 **BEV free space**
(위에서 본 120×120 격자, 5 cm 칸에서 "지금 갈 수 있는 칸")를 예측한다. baseline은 **Simple-BEV**.

## Task & 모델 (확정 설정)

- **이진 분할**: 칸마다 free / not-free. 볼 수 없는 칸(사각지대)은 not-free로 채점한다.
- **모델**: Simple-BEV + ResNet-101(ImageNet, stride 8), 어안을 펴지 않고 DS 투영으로 직접 lifting,
  **`Y=4` 높이 평면**(−0.25~1.75 m). BEV 사전학습 없음(실험 02·02b에서 사전학습은 해로웠다).
- **목적함수**: `½L_F + ½L_N + λ_B·L_B` — 경계 대역(|d| ≤ δ)에 soft target. **δ = 0.30 m, σ = 0.10 m,
  λ_B = 0.5. 광선 보조항 `L_range` 없음**(`λ_R = 0`). 수식은 [`docs/loss_function_spec.md`](docs/loss_function_spec.md).
- **학습**: AdamW lr 1e-4(one-cycle), batch 8, **100 epoch, 마지막 epoch 고정으로 평가**(검증으로 고르지 않는다).
  상세는 [`docs/99_paper_results/common/training_details.md`](docs/99_paper_results/common/training_details.md).
- **지표**: `iou_free`(항상 constant-map 기준선·마진과 함께), Precision·Recall(같은 동작점에서 한 쌍으로),
  BF@{0.10, 0.20, 0.30} m, 보조로 `iou_non_free`·거리 고리별. **전부 프레임 macro.** 이름 정본은
  `projects/common/metric_spec.py`, 정의는 [`docs/99_paper_results/common/metrics.md`](docs/99_paper_results/common/metrics.md).
- **유의성**: 시드로 짝지은 양측 t-검정의 95 % 신뢰구간이 0을 빼면 유의(`projects/common/paired_stats.py`).
  논문 표에는 mean ± SD와 `*`만 싣는다.
- **MMDetection3D는 쓰지 않는다.** `mmdetection3d/` submodule은 이후 3D 검출 확장용으로만 남겨 두었다.

## 꼭 지킬 규칙 — 실제로 사고가 났던 것들

- **conda 환경은 `bev-chamdog`**(Python 3.11 + torch 2.7.0+cu128). `base`에서 돌렸다가 numpy가 큰 배열을
  조용히 망가뜨린 사고가 있었다. 학습·분석 스크립트는 다른 환경이면 시작하지 않는다(`ALLOW_ANY_ENV=1`로만 우회).
- **`Y`의 단일 출처는 `projects/datasets/simplebev_vox.vox_dims()`**다. 리터럴 `1`을 쓰지 않는다.
  체크포인트 옆 `height.json`이 없으면 `LEGACY_HEIGHT_BINS = 1`로 읽는다 — **이 상수는 바꾸지 않는다.**
- **긴 학습은 `setsid nohup … &`로 분리해 띄운다.** 도구의 실행 시간 상한에 학습이 같이 죽은 적이 있다.
- **지연·FPS를 재기 전에 `nvidia-smi`로 GPU 점유를 확인한다.** 공유 GPU가 측정을 두 번 망쳤다(최대 4.6배).
- **프로세스는 정확한 PID로 종료한다.** `pkill -f` 패턴이 자기 셸을 죽인 적이 있다. 실행 중인 bash 스크립트는 편집하지 않는다.
- **생성 스크립트가 정본이다.** 패키지 CSV·SVG·PDF를 손으로 고치지 않는다. 그림은 열어서 눈으로 확인한다.
- **결과가 예상과 달라도 사용자 확인 없이 새 실험 가지를 열지 않는다.**
- **git worktree를 쓰지 않는다**(submodule 때문에 정리가 안 되고 산출물을 잃은 적이 있다).
- 작업 트리를 다른 세션과 공유할 수 있다 — **`git stash`·다른 브랜치 checkout·`reset --hard`·`branch -f`·
  `push -f`를 하지 않고, `git add`는 파일을 명시한다.**

## 개발 환경

- **PyTorch는 반드시 `cu128` 빌드**를 쓴다. GPU가 sm_120이라 `cu118` 빌드는 `is_available()`이 True여도 실제 연산에서 죽는다.
- 의존성은 `constraints.txt`(numpy<2·opencv<5) + `requirements.txt`로 설치하고 `pip check`.
  `third_party/models/simple_bev/requirements.txt`는 쓰지 않는다(3.7~3.8 시절 핀).
- mmcv / mmdet / mmdet3d는 설치하지 않는다.
- 상세: [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md)

## 하드웨어

- 학습: **RTX PRO 6000 Blackwell / VRAM 96 GB** 단일 GPU(다른 연구실과 공유), CPU 32코어, RAM 250 GB.
  데이터와 `runs/`는 **`/data`(3.7 T)**에 둔다. 병목은 대개 데이터 로딩이다.
- 배포 대상: **Jetson AGX Orin.** 환경 함정 둘 — `efficientnet_pytorch`는 `--no-deps`로 설치, cuSOLVER 우회 필요.

## 폴더 규칙

- 커스텀 코드는 `projects/`(모듈), `configs/`(셸 진입점), `tools/`(스크립트)에 둔다.
- **`third_party/`와 `mmdetection3d/`는 git submodule — 직접 수정하지 않는다.** Simple-BEV 동작을 바꿔야 하면 `projects/`에서 래핑한다.
- 상세: [`docs/project_structure.md`](docs/project_structure.md)

## [중요] Git 규칙

- **`commit`은 자율적으로 해도 된다. 단, 큰 논리 단위로 묶는다**(자잘한 변경마다 쪼개지 않는다).
- **커밋 메시지는 영어로 쓴다**(파일·문서 내용은 한글 무방).
- **`merge`와 `push`는 절대 임의로 하지 않는다 — 사용자가 직접 한다.** 필요하면 명령만 알려 준다.
- 기능 개발은 새 브랜치(`feat/*`, `exp/*` 등)에서 한다.
- 상세: [`docs/git_workflow.md`](docs/git_workflow.md)
