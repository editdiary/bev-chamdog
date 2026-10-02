# 논문 본 실험 캠페인 v3 — 실행 계획 (검토용)

> **상태: 사용자 검토 대기 (2026-10-02 작성).** 승인 후 Phase 0부터 자율 실행한다.
> 절차의 정본은 [`paper_campaign_protocol.md`](paper_campaign_protocol.md)이고, 이 문서는 그
> 절차를 **이번 캠페인의 설정·규모·산출물**에 맞춰 구체화한 것이다. 맨 아래 §7이 **결정이
> 필요한 것**이다.

---

## 1. 무엇을 돌리나 — 확정 설정

이전 캠페인과 **비교하지 않는다**(사용자 결정). 아래 설정으로 처음부터 새로 만든다.

| 항목 | 값 | 근거 |
|---|---|---|
| 목적함수 | soft-boundary BCE `½L_F + ½L_N + λ_B·L_B` (δ = 0.30, σ = 0.10, λ_B = 0.5, κ = 1, ε = 0). **광선 보조항 없음** | 원장 §5 2026-10-02 (2) |
| 학습 길이 | **100 epoch**, 판정은 **고정 epoch 100** (best epoch은 진단용) | 사전 실험 `runs/99_epoch_exp` |
| 높이 | Y = 4, [−0.25, 1.75] m (표본 0 / 0.5 / 1.0 / 1.5 m) | 사전 실험 `runs/99_height_exp` |
| 모델·입력 | Simple-BEV, ResNet-101, 어안 3대, Double Sphere 원본 투영, 120×120 @ 5 cm | 확정값 |
| 지표 | IoU_free · IoU_non-free · Precision · Recall · BF@{0.10, 0.20, 0.30} m · 거리 고리별 — **전부 프레임 macro** | `projects/common/metric_spec.py` |
| 손실 로그 | 모델이 받은 손실 `loss_{total,free,non_free,boundary}` + 논문용 `bce_{free,non_free,boundary}` | 같은 파일 손실 절 |
| 시드 | 0, 1, 2, 3, 4 | |

## 2. 실험 구성과 비용

비용은 실측 epoch 시간(로봇 8.3 s, SynWoodScape 사전학습 27~29 s) × 100 epoch이다.

| 실험 | 런 | 내용 | 시간 |
|---|---|---|---|
| **1. Overall** `01_overall` | 고정 split 5 + **LOSO 7 fold × 5** = **40** | 대표 성능 + 시퀀스 일반화. constant-map 기준선 대비 margin | 약 9.8 h |
| **2. Ablations** `02_projection_and_prior` | 사전학습 5 + `source_prior` 5 + 핀홀 120° 5 + 핀홀 150° 5 = **20** | 어안 원본 대 가상 핀홀 / 합성 사전학습. 대조군 = 실험 1 고정 split(시드 짝) | 약 7.5 h |
| **2b. Native prior** `02b_native_source_prior` | 사전학습(원본 기하) 5 + 미세조정 5 = **10** | 사전학습을 SynWoodScape 원본 기하로 | 약 4.9 h |
| **3. Loss ablation** `03_boundary_uncertainty` | `A_ce → B_perset → C_hard → C_soft` 4 × 5 = **20** | 가중 BCE → per-set 평균 → hard 대역 → soft 대역 | 약 4.7 h |
| | **합계 90런** | | **약 27 h** |

- **디스크:** 런당 체크포인트 둘(마지막 + best) 약 0.95 GB → 약 85 GB(여유 2.7 TB).
- **GPU:** 0번 한 장으로 **순차** 실행한다(1번은 다른 사용자의 vLLM이 75 GB 사용 중). 한 장에 둘을
  동시에 돌리면 빨라질 수 있지만 측정해 본 적이 없고, 이번 사전 실험들이 전부 순차였다 — 주말 안에
  끝나는 규모라 검증된 방식을 쓴다.
- **사전학습 길이를 40으로 두면 약 23 h**가 된다(§7-①).

## 3. 진행 순서

```
Phase 0  준비 (약 1~2 h, 사람 확인 불필요)
  0-1  새 루트 runs/99_full_campaign/ 와 드라이버 셋 작성 (학습 / 분석 / 패키지)
  0-2  손잡이 점검: 세 큐 러너 + 03 사다리가 확정 설정을 받는지 plan-only로 기계 대조
  0-3  03 사다리에 런당 시간 상한 추가 (지금은 없다 -- 멈추면 무한 대기)
  0-4  패키지 생성기 넷에 --seeds / --fixed_epoch / 루트 인자 (리허설용, 기본값 = 본 실험)
  0-5  새 패키지 폴더 docs/99_paper_results/ 를 만들고 생성 스크립트(make_package.py, figures/*.py)를
       복사해 입력 경로를 runs/99_full_campaign/ 으로 바꾼다. 옛 docs/paper_package/ 는 건드리지 않는다
리허설 (약 30 min)
  모든 러너 1 epoch x 시드 0 (사전학습 포함) -> 분석 셸 전부 -> 패키지 생성기 넷 + 그림 전부
  확인: 런 설정 / 경로가 새 루트 밖으로 안 샘 / 사전학습 -> 미세조정 체크포인트 사슬 /
        무결성 수치 실패 0 / 생성기·그림이 새 열 이름으로 끝까지 돈다
  -> 리허설 산출물 삭제
Phase 1  본 학습 (약 27 h, 분리 세션 setsid + watchdog)
  01 고정 split -> 01 LOSO -> 02 -> 02b -> 03
  각 실험이 끝나는 즉시 그 실험의 Phase 2를 돌린다 -- 문제가 있으면 다음 실험이 도는 동안 발견된다
Phase 2  분석 (실험마다, 각 10~40 min)
  scalar 보존 -> 확률맵 내보내기 -> 독립 재채점(무결성 게이트) -> 집계 -> 짝지은 비교
Phase 3  패키지 재생성 (약 15 min)
  docs/99_paper_results/{01,02,02b,03} 의 data/·provenance/·figures/ 를 새 결과로
Phase 4  문서 (수 시간)
  REPORT.md 넷을 새 데이터로 다시 쓴다 + 산문 숫자 감사 + common/ 문서 + 원장·AGENTS.md·메모리
```

### 멈춤·실패 대응 (사람이 없는 동안)

| 상황 | 자동 대응 |
|---|---|
| 학습이 출력 없이 20분 / 한 런이 2시간 초과 | watchdog이 프로세스 그룹째 종료 → 그 런 실패로 기록, 다음 런 진행 |
| 런 실패(오류·OOM·종료) | 불완전 폴더를 확인 후 지우고 **한 번 재시도**. 두 번 실패하면 그 런만 빼고 계속, 보고에 적는다 |
| 무결성 검사 실패 | 그 실험의 패키지를 만들지 않는다. 원인을 조사·기록하고 다른 실험은 계속 |
| GPU를 다른 사람이 쓰기 시작함 | 시간 측정이 흔들릴 뿐 결과에는 영향 없음 — 기록만 남긴다 |
| 셸 도구 시간 상한(약 30분) | 학습은 분리 세션이라 영향 없음. 감시 작업만 30분 단위로 다시 건다 |

## 4. 저장되는 데이터

```
runs/99_full_campaign/
├── campaign_train.log · campaign_analysis.log      드라이버 전문
├── 01_overall/{fixed_split,loso}/
│   ├── logs/<run>/  events.out.tfevents.*           epoch마다 지표 25개 + 손실 7개 (train/val)
│   │                config.json                     설정 전부 + label_constants(영역 비율·경계 하한)
│   │                split_{train,val}_samples.txt   실제로 쓴 프레임 목록
│   ├── ckpt/<run>/  model-000000100.pth · model_best-*.pth · height.json
│   ├── run_logs/    런마다 콘솔 전문 (watchdog)
│   └── analysis/    scalars.csv(전 tag x 전 epoch) · configs.json · manifest.json
│                    predictions/*.npz(val 확률맵 best·last, fp16) + labels.npz
│                    verify_predictions.json(무결성) · constant_map_baseline.json
│                    summarize_repeats.txt · report_loso.json · RESULTS.json/csv
├── 02_projection_and_prior/<arm>/{logs,ckpt,analysis}/   + analysis/paired_arms.json
├── 02b_native_source_prior/<arm>/{logs,ckpt,analysis}/  + analysis/paired_arms.json
└── 03_boundary_uncertainty/{logs,ckpt,analysis}/
                     analysis/ 위 공통 + threshold_sweep_rows.csv(동작점 τ 스윕) · RESULTS.json/csv
```

`runs/`는 git에 들어가지 않는다(서버에만 있다). 논문에 필요한 것은 Phase 3에서 패키지로 옮긴다.

## 5. 마무리 때 남는 문서와 데이터

### 논문용 패키지 — `docs/99_paper_results/` (git, 노트북으로 가져가는 것). 옛 `docs/paper_package/`는 그대로 둔다

| 폴더 | 내용 |
|---|---|
| `01_overall/` | 고정 split 시드별·요약·Table 1, **거리 고리별 요약**, 보조 지표(BF의 P/R·고리) CSV, LOSO fold×시드·fold 요약·macro·Table 2, margin 역전 그림, val 곡선 |
| `02_projection_and_prior/` | 시드별 원자료, **짝지은 차이**(전 지표), Table 1(투영)·Table 2(사전학습), 커버리지 대 정확도 그림, 초기 epoch 궤적 그림 |
| `02b_native_source_prior/` | 세 줄 비교표(scratch / 가공 기하 / 원본 기하), 전 지표 판정표, 사전학습 수렴 확인 |
| `03_boundary_uncertainty/` | 사다리 정확도, 목적함수 수렴(되올림), **경계 국소화(세 영역 BCE)**, 동작점(같은 recall에서 precision), 문턱 산포, 짝지은 차이, 그림 셋 |
| `common/` | `metrics.md` · `evaluation_protocol.md`(지표 정의·집계·통계) · `training_details.md`(**새 목적함수로 다시 씀**) · `setup.md` |

각 실험 폴더: `data/*.csv`(생성기가 정본) · `provenance/`(환경·설정·무결성·런 목록) ·
`figures/*.{pdf,png,svg}` · **`REPORT.md`(새 데이터로 처음부터 다시 쓴다)** · `README.md`.
REPORT의 모든 소수 숫자는 `tools/audit_paper_prose_numbers.py`로 CSV·provenance까지 추적한다.

### 기록

- 원장 `docs/paper_final_experiments.md` §5에 캠페인 항목(설정·실행 기록·결과 요약·무엇이 주장되고 무엇이 안 되는지)
- `docs/paper_campaign_protocol.md`에 이번에 물린 것 추가
- `AGENTS.md` 상단 블록과 메모리 갱신
- 커밋은 단계마다(파일 명시). **푸시는 사용자가** 한다

## 6. 판정 규칙 (미리 고정)

- 주 수치: **고정 epoch 100**의 5시드 mean ± SD. `iou_free`는 항상 constant-map baseline·margin과 함께
- 짝지은 비교(02·02b·03): 시드끼리 짝지은 Δ에서 **|Δ̄| > σ_seed 그리고 |Δ̄| ≥ 2·SE** (SE = sd(Δ)/√5), p값 없음
- σ_seed는 §7-②에서 정한 대로 **캠페인 전에 고정**한다

---

## 7. 결정 (2026-10-02 사용자 답)

| # | 질문 | 결정 |
|---|---|---|
| ① | SynWoodScape 사전학습 길이 | **100 epoch** (전체 약 27 h) |
| ② | 판정 눈금 σ_seed | **확인 대기** — 제안: 값이 아니라 계산 규칙을 지금 고정한다. 이번 캠페인의 모든 5시드 묶음(약 10개)을 합친 시드 간 sd. 학습에는 영향이 없고 짝지은 비교의 "차이 있음" 판정에만 쓰인다 |
| ③ | 03의 `C_soft` | **새로 따로 돌린다**(4칸 x 5시드 = 20런). 01 대조군과 같은 설정이라 재학습 잡음(σ_run)도 5쌍으로 잰다 |
| ④ | 이름 | 학습 결과 **`runs/99_full_campaign/`**, 패키지 **`docs/99_paper_results/`**(이름 확인 대기). 옛 `docs/paper_package/`는 덮어쓰지 않는다 |
