# AGENTS.md

이 문서는 AI agent와 협업할 때 항상 참고할 **핵심 지침**입니다. 특정 플랫폼에 종속되지 않는 공통 규칙만 둡니다. 상세 내용은 `docs/`의 개별 문서를 필요할 때만 찾아봅니다.

> **어떤 문서를 믿어야 하는지는 [`docs/README.md`](docs/README.md)가 정합니다.** 각 문서의
> 역할과 상태(🟢 정본 / 🔵 운영 메모 / 📚 참고 / 🗄 아카이브)가 거기 있습니다.
> **`docs/archive/`의 문서는 인용용이고, 거기 적힌 플래그·경로·숫자를 그대로 쓰면 안 됩니다.**

> ## ▶ [2026-08-27] lifting 높이 축 `Y=1 → 4` **채택** -- 새 세션은 이것부터 안다
>
> **확정 config가 바뀌었다.** `Segnet(Z, Y, X)`의 `Y`가 1 → **4**이고 높이 범위는
> −0.25~1.75 m다(표본 높이 지면 기준 **0, 0.5, 1.0, 1.5 m**). n=3에서
> `iou_free` 0.7950 → **0.8104**, `f1@10cm` 0.5437 → **0.6085**, 지연 +3.2 %,
> 재현성 불변. **이 프로젝트에서 단일 변경으로 얻은 가장 큰 개선이다.**
> 근거 정본은 [`docs/paper_experiment_compendium.md`](docs/paper_experiment_compendium.md) §12.
>
> **왜 열렸나**: 라벨 생성 코드(`dataset/sj_datasets/common/temp/slab_label.py`)가
> occupancy를 **"로봇이 통과해야 하는 높이 구간"의 2D 기둥 누적**(지상 0.87~1.67 m)으로
> 정의하고 있었다. 라벨은 처음부터 지면 occupancy가 아니었는데 lifting은 `ego z=0` 한
> 평면만 표본했다. `Y=1`은 실험값이 아니라 로더 커밋의 하드코딩이었다.
>
> **⚠ 절단선**: **2026-08-27 이전의 모든 런은 `Y=1`이고 lifting 기하가 다르다.**
> `runs/ablation`·`runs/pixel_offset`·`runs/stride4`·`runs/frame_blocks`·LOSO·시드 스윕이
> 전부 그렇다. 옛 숫자와 한 표에 세우려면 `HEIGHT_BINS=1 HEIGHT_MIN_M=None
> HEIGHT_MAX_M=None`을 명시한다. 도구는 체크포인트 옆 `height.json`으로 자동 구분하고,
> 파일이 없으면 `LEGACY_HEIGHT_BINS = 1`로 읽는다(**이 상수는 바꾸면 안 된다**).
>
> **`Y`의 단일 출처는 `projects/datasets/simplebev_vox.vox_dims(grid_spec, height_bins)`다.**
> 리터럴 `1`을 다시 쓰지 않는다 -- 예전에 20곳에 퍼져 있었다.
>
> **남은 것**: 논문 집필, 그리고 **`Y=4`에서 Orin 재실측**(사용자). 20.2 FPS는 `Y=1` 값이다.
>
> **여기까지의 서사: [`docs/experiment_history.md`](docs/experiment_history.md)** (2026-08-27, 단계 J)
>
> 2-head → 3-class → binary로 정식화를 두 번 바꾼 이유, 각 단계에서 기각된 가설, 그리고
> **병목이 모델·loss·지표가 아니라 라벨이 정의한 task 자체라는 결론**과 그 근거가 여기 있다.
> 방법론 교훈 요약(§7)과 다음 데이터 수집 권고(§6)도 같은 문서다. **새 세션은 이것을 먼저
> 읽는다** -- 개별 숫자의 근거 정본은 각 절이 가리키는 원본 문서다.
>
> ## ▶ [2026-08-27] 경계 target 하이퍼파라미터 네 축 -- **전부 같은 축으로 붕괴했다**
>
> 사용자 질문 "`L_B`가 수렴을 안 하는데 최소한 수렴은 해야 정상 아닌가"에서 출발해
> **모양 α, 폭 δ, 대역 수축 κ, 전역 평활 ε**을 Y=4에서 12런으로 훑었다.
> 근거 정본은 [`docs/soft_boundary_loss_design.md`](docs/soft_boundary_loss_design.md) **§20**.
>
> **① [방법론 정정 -- 이게 제일 중요하다] `kl_boundary`는 config끼리 비교할 수 없다**(§20.1).
> 각 런이 **자기 `Ω_B`에서 자기 target에 대해** 잰 평균이라, δ가 커지면 쉬운 셀이 섞이고
> α·κ·ε이 바뀌면 target 자체가 쉬워진다. δ=0.45의 0.0870을 분해하면
> `0.0870 → 0.1052`(셀 통일) `→ 0.2691`(target 통일)이고 대조군은 0.4780이다 --
> **"96 % 개선"이 실제로는 44 %다.** **정본 지표는 `tools/report_boundary_calibration.py`의
> `공통 kl`이다**(같은 셀 `|d| ≤ 0.15`, 같은 target). 옛 절의 `kl_boundary` 비교는 인용 금지.
>
> **② 네 축이 하나로 붕괴한다.** 12런에서 **공통 kl과 예측 엔트로피의 r = −0.901, R² = 0.81**.
> **보정 개선의 81 %가 "얼마나 덜 확신하게 됐는가" 하나로 설명된다.** 어떤 손잡이인지는 거의 무관.
>
> **③ 능력은 안 움직인다.** `iou_free`는 엔트로피와 무관(r = −0.25, 전부 0.805~0.815),
> **`f1@10cm`은 음의 상관(r = −0.71)** -- 덜 확신할수록 나빠진다. **val/train 배율 23~27배로
> 암기 그대로.** **한 문장: 경계를 더 잘 알게 된 것이 아니라 모른다고 말하게 됐다.**
>
> **④ 확정 config는 안 바뀐다**(§20.9). δ=0.15가 `f1@10cm` 최고 근처다. 굳이 고르면
> δ=0.20~0.25 또는 ε=0.05~0.10(보정 −20~−46 %, 품질 잡음 안)인데 **확률을 쓰는 하류가 없으면
> 실익이 없다.** δ≥0.45는 `f1@10cm` −5.9σ에 자유공간의 69 %가 hard 감독을 잃는다
> (자유 셀 절반이 벽에서 32 cm 이내다).
>
> **⑤ 새 손잡이 둘이 구현됐고 기본은 꺼져 있다**(bit 단위 항등, 테스트 고정):
> `BAND_KAPPA`(κ, 기본 1.0)와 `LABEL_EPS`(ε, 기본 0.0). 형태는
> [`docs/loss_function_spec.md`](docs/loss_function_spec.md) §6.4·§6.5.
>
> **⑥ α=0.5의 채택 근거는 남아 있지 않다**(§20.2). 근거였던 f1 차이 0.0021은 σ_run 0.0037의
> 0.57배다. Y=4 재측정도 1.6σ. **α는 미결이고 n=3이 필요하다.** 이 절 전체가 **n=1**이다.
>
> **⑦ `L_range` 진단**(§19): 상쇄로 면제되는 "실제로 틀린" 광선은 val의 **0.6~0.8 %**뿐이라
> 실질 누수는 작다. 그런데 **val 오차의 71 %가 경계 대역에서 온다** -- 설계 목적이던
> far(벽 뒤 free 섬)는 분산의 2.9 %이고 이미 일반화된다. **이 항은 경계 오차를 반경 방향으로
> 다시 재고 있다.** 그리고 train `arc_mae`가 epoch 3에 dead zone에 들어가 `share_range`가
> 0.33 %로 떨어진다 -- **40 epoch 중 3 epoch만 살아 있다.**
>
> **현재 작업: soft-boundary loss -- [`docs/soft_boundary_loss_design.md`](docs/soft_boundary_loss_design.md)** (2026-08-21)
>
> 라벨의 불완전성을 **loss에 명시적으로 모델링**한다(경계 대역에 soft target). 여기에
> 방위각 자유거리 보조항 `L_range`를 더한 것이 **현재 확정 config**다(§13:
> gaussian δ=0.15 α=0.5 λ_B=0.5, λ_R=0.3 δ_R=0.20 β=0.10).
> **loss가 수식으로 정확히 무엇인지는 [`docs/loss_function_spec.md`](docs/loss_function_spec.md)가
> 정본이다** -- 근거·결과 없이 형태만 있어 대조 없이 읽힌다.
>
> **[정본] loss 연구는 §16으로 종결됐다.** 근거는 §15의 n=3 ablation(`runs/ablation/`,
> `configs/ablation_loss.sh` → `tools/report_ablation.py`, 사다리 4칸
> `A_ce → B_perset → C_soft → D_range` × 시드 3개)과 그것을 threshold sweep으로 재판독한 §16이다.
> **살아남은 주장은 하나다.**
>
> ① ✅ **재학습 재현성 5~22배** (시드 간 σ: `fatal` 0.0063 → 0.0007, `range_bias`
> 0.0190 → 0.0023). **τ ∈ [0.2, 0.8] 전 구간에서 성립**하고, 유효 threshold jitter로는 17배다.
> ② ✅ **되올림 2.4배 감소** (61.8 → 19.3 %, 엔트로피 하한 제거 후). val loss 발산이
> **±15 cm 경계 대역으로 국소화되고 크기가 절반 이하**가 된다(CE +0.2003 대 `D_range` +0.0839).
> ③ ❌ **[철회] "안전 개선"은 동작점 이동이었다**(§16.2). 같은 `free_miss`에서 `fatal`·
> `missed_obstacle` 곡선이 **CE와 시드 σ 안에서 겹친다.** **CE도 τ를 0.5 → 0.7로 올리면
> 같은 자리에 온다.**
>
> **정확도는 CE와 동일**하고 **τ를 최적화해도 그렇다**(최대 `iou_free` A 0.7992 / D 0.7979).
> **암기는 안 줄었다.** → **한 문장: 더 좋은 모델을 주지 않고, 같은 모델을 더 일관되게 준다.**
>
> **표기 규약: 절대값(pp) 먼저, 상대값은 괄호**(§16.5). `fatal −1.25 pp (−9.2 % relative)`.
> **"남은 오차의 98.9 %가 편향"은 과했다**(§16.6) -- "현재 pipeline이 공유하는 오차"가 맞고
> `task ceiling`이라고 부르지 않는다. **`σ_target = √(σ_label²+σ_model²)`는 유도가 아니라
> 진단이다**(§16.7).
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
> **[종결] 비대칭 dead zone `δ_R⁺`는 채택하지 않는다**(§16.8). 판정의 σ가 위 (a)로 무효가
> 됐고, n=3에 그 칸이 없고, 무엇보다 **"range 보조항이 좋아진 건지 safety bias를 넣어
> 좋아진 건지 원인 분리가 안 된다."** 확정 config는 대칭 `δ_R=0.20`이다.
>
> ## ▶▶ [2026-08-26] **계획된 실험이 전부 끝났다. 남은 것은 논문 집필이다.**
>
> **새 세션은 [`docs/paper_experiment_compendium.md`](docs/paper_experiment_compendium.md)를
> 읽는다.** 모든 실험을 목적→설계→결과(수치)→해석으로 모은 문서이고, **다른 파일을 열지
> 않아도 읽히도록 용어 정의까지 안에 들어 있다**(§0.6). 정본은 여전히 진단·설계 문서다.
>
> **끝난 것 (계획했던 남은 작업 넷 전부).**
>
> | 무엇 | 결과 | 정본 |
> |---|---|---|
> | `stride 8→4` 6런 | ❌ **표본 밀도 가설 기각, 미채택.** 사전 선언한 무늬가 안 나왔고 `iou_free`·`fatal_rate`도 노이즈. **기본 encoder는 `res101`(stride 8)** | 진단 **§29.9** |
> | **LOSO 7-fold × 시드 3 = 21런** | ✅ **7 fold 전부 상수 지도를 +0.158 이상 이긴다.** `σ_fold` = 0.0405(`σ_seed`의 **27배**)이고 **그 분산의 거의 전부가 통로 폭 하나로 설명된다**(계층 안 마진 std 0.012~0.030) | 진단 **§31** |
> | 안정성 축 둘 (perturbation + frame-gap) | ✅ **흔들림 세 원천이 전부 모델 오차보다 작다.** 지배적 불확실성은 불안정성이 아니라 정확도 | 진단 **§30** |
> | **Orin 실측** | ✅ **512×288 fp16 = 20.2 FPS**, 32.7 W, 359 MB. 8.2분 지속 부하에서 **throttling 없음** | 진단 **§32** |
>
> **남은 것 둘 -- 둘 다 사용자 몫이다.**
>
> | # | 무엇 | 상태 |
> |---|---|---|
> | 1 | **논문 집필** | compendium §18에 구성 제안과 주 기여 후보 셋이 있다 |
> | 2 | **Orin 목표 FPS** | 미정. 이 값이 없으면 "배포 가능" 진술만 못 쓴다(진단 §32.7) |
>
> **하지 않기로 한 것 (되돌리지 않는다).** TensorRT 변환(진단 §32.6에 정찰 결과와 막힐 지점
> 셋을 적어 뒀다) · 라벨 재수집·재어노테이션 · `traj_recall`/pose · `δ` 스윕 계열 ·
> ROI crop · BEV 격자 변경 · stride-4에서 파생되는 작업.
>
> > **[규칙] 결과가 예상과 달라도 새 가지를 열지 않는다**(진단 §28.8, 사용자 방침).
> > 현재 데이터·아키텍처에서 관측된 한계를 그대로 결과로 정리한다.
>
> **과적합은 파라미터로 못 고친다**(설계 §15.8, 진단 §10.2·§17.1·§19·§20.2). `weight_decay`
> ·`res50`·`freeze_encoder`·`flip_augment`·`label_smoothing` 전부 기각. train을 20 % 버려
> 과적합을 2배로 만들어도 `iou_free`는 노이즈 안이었다.
> **한때 "다음 병목"으로 지목된 §18.3(특징맵 표본 좌표)은 종결됐다** -- 고쳤고 성능 영향은
> 15런 스윕 51칸 전부 노이즈였다(§18.3.5).
>
> **옛 인수인계 문서**: [`docs/next_session_binary_and_verification.md`](docs/next_session_binary_and_verification.md)(2026-08-19)는
> **운영 메모**다 -- 도구 목록·`cam0..3` 매핑 함정(**`left=cam3`**) 같은 실무 정보만 본다.
> 3-class 시대 인수인계는 [`docs/archive/next_session_threeclass_training.md`](docs/archive/next_session_threeclass_training.md).
> 학습 산출물 정리 규약은 진단 문서 §14 (`tools/prune_runs.py`).

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
