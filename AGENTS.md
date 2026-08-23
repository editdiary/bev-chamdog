# AGENTS.md

이 문서는 AI agent와 협업할 때 항상 참고할 **핵심 지침**입니다. 특정 플랫폼에 종속되지 않는 공통 규칙만 둡니다. 상세 내용은 `docs/`의 개별 문서를 필요할 때만 찾아봅니다.

> **어떤 문서를 믿어야 하는지는 [`docs/README.md`](docs/README.md)가 정합니다.** 각 문서의
> 역할과 상태(🟢 정본 / 🔵 운영 메모 / 📚 참고 / 🗄 아카이브)가 거기 있습니다.
> **`docs/archive/`의 문서는 인용용이고, 거기 적힌 플래그·경로·숫자를 그대로 쓰면 안 됩니다.**

> **여기까지의 서사: [`docs/experiment_history.md`](docs/experiment_history.md)** (2026-08-21)
>
> 2-head → 3-class → binary로 정식화를 두 번 바꾼 이유, 각 단계에서 기각된 가설, 그리고
> **병목이 모델·loss·지표가 아니라 라벨이 정의한 task 자체라는 결론**과 그 근거가 여기 있다.
> 방법론 교훈 요약(§7)과 다음 데이터 수집 권고(§6)도 같은 문서다. **새 세션은 이것을 먼저
> 읽는다** -- 개별 숫자의 근거 정본은 각 절이 가리키는 원본 문서다.
>
> **현재 작업: soft-boundary loss -- [`docs/soft_boundary_loss_design.md`](docs/soft_boundary_loss_design.md)** (2026-08-21)
>
> 라벨의 불완전성을 **loss에 명시적으로 모델링**한다(경계 대역에 soft target). 여기에
> 방위각 자유거리 보조항 `L_range`를 더한 것이 **현재 확정 config**다(§13:
> gaussian δ=0.15 α=0.5 λ_B=0.5, λ_R=0.3 δ_R=0.20 β=0.10).
> **loss가 수식으로 정확히 무엇인지는 [`docs/loss_function_spec.md`](docs/loss_function_spec.md)가
> 정본이다** -- 근거·결과 없이 형태만 있어 대조 없이 읽힌다.
>
> **[정본] CE 대비 성과는 §15의 n=3 ablation이다** (`runs/ablation/`,
> `configs/ablation_loss.sh` → `tools/report_ablation.py`). 사다리 4칸
> `A_ce → B_perset → C_soft → D_range` × 시드 3개. 확정 세 가지:
> ① **재현성 4~9배** (시드 간 σ: `fatal` 0.0063 → 0.0007, `range_bias` 0.0190 → 0.0023).
> ② **되올림 2.4배 감소** (61.8 → 19.3 %, 엔트로피 하한 제거 후).
> ③ **안전 개선** (`missed_obstacle` −16.1 %, `fatal` −9.2 %) / 대가 `free_miss` +16.4 %.
> **정확도는 CE와 동일**(`iou_free`·`f1@τ`·`range_mae` 전부 노이즈). **암기는 안 줄었다.**
>
> **[철회 두 개 -- 옛 절의 σ를 인용하지 말 것]**
> (a) **σ_run은 하한이다**(§15.4). 같은 config·같은 시드의 재현 노이즈이므로 **서로 다른
> config 비교에는 부족하다** -- 시드 분산은 config마다 다르고 `ce`가 soft보다 4~9배 크다.
> §13.6·§13.7·§13.8의 **순위는 인용 가능, σ 값은 불가.**
> (b) **되올림은 loss 종류를 넘어 그대로 비교 불가**(§15.3). soft loss의 target 엔트로피
> 상수(`λ_B·H̄`=0.1496)가 비를 기계적으로 줄여 45 % 부풀려져 있었다. **주 근거는 loss가
> 등장하지 않는 축**(`iou 최고→끝 하락`·`fatal/iou 흔들림`·`train−val iou 격차`)**으로 옮겼다.**
>
> **[교훈] 안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이다** -- `B_perset`이
> 안정성 3개를 다 이기는데 품질은 전부 최악이다(§15.6). 반드시 품질과 같이 읽는다.
>
> `f1@10cm`은 주 판정에서 강등돼 있다(§14, [`docs/BEV_loss_and_metrics_design.md`](docs/BEV_loss_and_metrics_design.md) §2.9).
> `--loss=weighted_ce`는 대조군이므로 지우지 않는다.
>
> **미결(사용자 결정 대기): 비대칭 dead zone `δ_R⁺`** (§13.8). `δ_R⁺=0`이 `fatal_rate`를
> 개선하는 대신 `free_miss_rate`를 악화시키는 교환인데, **그 판정의 σ가 위 (a)로 무효가
> 됐고 n=3 ablation에 이 칸이 없어 미확인 상태다.** 현재 확정 config는 여전히 대칭
> `δ_R=0.20`이다.
>
> **과적합은 파라미터로 못 고친다**(§15.8, 진단 문서 §10.2·§17.1·§19·§20.2). `weight_decay`
> ·`res50`·`freeze_encoder`·`flip_augment`·`label_smoothing` 전부 기각. train을 20 % 버려
> 과적합을 2배로 만들어도 `iou_free`는 노이즈 안이었다. 지목된 다음 병목은 진단 문서
> §18.3(특징맵 표본 좌표, **미해결**)이다.
>
> **직전 작업 인수인계: [`docs/next_session_binary_and_verification.md`](docs/next_session_binary_and_verification.md)** (2026-08-19)
>
> binary 정식화 전환·과적합 손잡이 실측·파이프라인 검증까지의 상태와 다음 결정 사항.
> 근거 정본은 [`docs/finetune_overfitting_diagnosis.md`](docs/finetune_overfitting_diagnosis.md)
> **§15–§23** (§20–§21 분할·test, §22 오차 구조, §23 `unknown`의 정체와 지표 축소).
> 그 이전 단계(3-class 본학습) 인수인계는 [`docs/archive/next_session_threeclass_training.md`](docs/archive/next_session_threeclass_training.md).
> 학습 산출물 정리 규약은 같은 문서 §14 (`tools/prune_runs.py`).

## [중요] 소통 규칙

- **사용자와의 모든 대화는 한국어로 한다.** 최종 답변뿐 아니라 **작업 중간의 진행 설명·판단 근거·질문도 한국어**로 쓴다.
- 예외는 두 가지뿐: **커밋 메시지는 영어**(아래 Git 규칙), 그리고 코드 안의 식별자·주석은 주변 코드 스타일을 따른다.

## 프로젝트 한 줄 요약

자체 구축한 Fisheye 4-cam 데이터셋으로 **BEV free-space map**(각 BEV 격자 셀이 free / occupied / unknown 중 무엇인지)을 예측하는 모델을 학습한다. baseline은 **Simple-BEV**.

## Task & 모델

- Task: 어안 이미지 → **BEV 3-class free-space map** (`free` / `occupied` / `unknown`). 3D bbox 검출과 세밀 semantic 구분은 **이후 확장 과제**로 분리.
- **정식화는 3-class 단일 head 하나다.** 옛 2-head(occupancy + visibility) 정식화는 Phase 3 A/B 이후 코드에서 제거됐다 — `visibility = raycast(occupancy)`라 두 head가 같은 라벨의 두 인코딩이었기 때문이다. 근거와 A/B 결과는 [`docs/archive/free_space_metric_migration.md`](docs/archive/free_space_metric_migration.md) §8–9.
- **주 지표는 `iou_free`이고 `fatal_rate`를 항상 같이 읽는다.** 옛 `iou_drivable`/`iou_obstacle`은 이미지를 안 보는 트리비얼 예측기에 지는 지표였다(같은 문서 §1). 새 학습·평가는 반드시 constant-map baseline과 병기해 판단한다.
- baseline: **Simple-BEV** (`third_party/models/simple_bev`) — mmdet3d에 의존하지 않는 standalone PyTorch 구현.
- 학습 순서: **SynWoodScape로 먼저 학습·검증 → pre-training → 자체 데이터셋 fine-tuning.**
  - pretraining — 2-head 시절 것은 완료돼 있다 (SynWoodScape 4-cam, `radial_poly`, 240×240 그리드).
    **3-class pretrain은 아직 돌리지 않았다** — 코드는 준비됐다
    (`configs/train_synwoodscape_threeclass_pretrain.sh`). 그때까지 fine-tuning은 옛 2-head
    체크포인트에서 trunk만 받고 출력 head는 랜덤 초기화로 시작한다(배너 `weight transfer` 줄로 확인).
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
- 상세: [`docs/setup_guide_pro6000.md`](docs/setup_guide_pro6000.md) (이전 RTX 3080 환경 이력은 [`docs/archive/setup_guide.md`](docs/archive/setup_guide.md))

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
