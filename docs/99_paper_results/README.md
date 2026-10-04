# 논문 결과 패키지 — 캠페인 v3

`runs/99_full_campaign/`(90런, 2026-10-02 18:04 ~ 10-03 21:40)에서 생성한 논문용 숫자·표·그림이다.
**이 폴더가 논문 숫자의 정본이다.** 옛 패키지 `docs/archive/paper_package_v2/`는 지표·학습 길이·목적함수가
달라(아래) 숫자를 섞지 않는다.

| | 옛 (`paper_package`) | 새 (이 폴더) |
|---|---|---|
| 목적함수 | soft-boundary + 광선 보조항 | **soft-boundary BCE만** |
| 학습 길이 | 40 epoch | **100 epoch** (사전학습 포함) |
| 지표 | fatal / free_miss / F1@τ micro / range_* | **IoU_free·IoU_non-free·Precision·Recall·BF@{0.1,0.2,0.3}**, 전부 프레임 macro |
| 유의성 | 두 문턱(σ_seed, 2 SE) | **시드 짝지은 t-검정 95 % CI**, 표에는 `*` |

## 실험

| 폴더 | 질문 | 런 | 한 줄 결과 |
|---|---|---|---|
| [`01_overall`](01_overall/REPORT.md) | 대표 성능과 일반화 | 40 | 고정 split 0.816(기준선 +0.298), LOSO macro 0.830(+0.294) |
| [`02_projection_and_prior`](02_projection_and_prior/REPORT.md) | 어안 직접 투영 · 합성 사전학습 | 20 | 핀홀로 펴면 손해, 사전학습도 손해(경계에 집중) |
| [`02b_native_source_prior`](02b_native_source_prior/REPORT.md) | 사전학습 손해가 기하 맞춤 탓인가 | 10 | 아니다 — 원래 기하로도 같은 방향 |
| [`03_boundary_uncertainty`](03_boundary_uncertainty/REPORT.md) | soft 경계 target의 효과 | 20 | 학습 거동·문턱 재현성은 확실히 낫고, 정확도 이득은 동작점을 맞추면 줄며, BF@0.10을 잃는다 |

공통 문서: [`common/setup.md`](common/setup.md) · [`common/metrics.md`](common/metrics.md) ·
[`common/evaluation_protocol.md`](common/evaluation_protocol.md) · [`common/training_details.md`](common/training_details.md)

## 검증

- 90런 전부 epoch 100 완주, 재시도 0회. 결과 트리가 `runs/99_full_campaign` 밖을 가리키지 않음(순수성 검사).
- 확률맵 160개를 학습 로그와 별도 경로로 재채점, 실패 0(최대 차이 7.05e-4, 허용 1e-3).
- REPORT 넷의 소수 셋째 자리 이상 숫자 357개를 `tools/audit_paper_prose_numbers.py`로 CSV와 대조, 불일치 0.

## 재생성

```
conda activate bev-chamdog
bash tools/run_full_campaign_packages.sh     # 네 실험의 data/ provenance/ figures/
```
