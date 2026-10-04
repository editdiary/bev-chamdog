# 논문 패키지 — 실험 04 Edge Deployment (Jetson AGX Orin)

실험 1 고정 split 모델 5개를 Jetson AGX Orin에서 잰 결과다(2026-10-04). 노트북에 통째로 받아 **서버 접속 없이** 논문을 쓸 수 있게 모았다.

## 결과

- PyTorch fp32 **67.5 ms / 14.8 FPS**, 그중 인코더가 **81.8 %**
- 인코더만 TensorRT fp16: **23.6 ms / 42.3 FPS (2.86배)**, 정확도 차이 없음
- 인코더만 TensorRT INT8: **18.6 ms / 53.7 FPS (3.63배)**, IoU_free −0.0055 등 유의한 손실, 모델마다 손실 크기가 크게 다름
- 배포 권고는 fp16. 최적화 뒤 병목은 BEV 단계(12.4 ms)

## 읽는 순서

1. [`../common/setup.md`](../common/setup.md) — 과제·데이터·라벨·동결 설정
2. [`../common/metrics.md`](../common/metrics.md) — 지표 정의와 보고 규칙
3. [`REPORT.md`](REPORT.md) — 이 실험의 설계·수치·해석·논문 구성

## 폴더

| 경로 | 무엇 |
|---|---|
| `REPORT.md` | ★ 본문 |
| `make_package.py` | `data/`·`provenance/`의 생성 스크립트 (**정본** — CSV를 손으로 고치지 않는다) |
| `data/table_paper_edge.csv` | 논문 표 그대로 (지연·FPS + 7지표 mean ± SD, 유의한 칸에 `*`) |
| `data/table_paper_caption.txt` | 표 캡션 문구(측정 조건 + 통계) |
| `data/table1_speed.csv` | 구성별 지연 분해·FPS·가속·p99·엔진 메모리 |
| `data/table2_accuracy_paired.csv` | 지표별 짝지은 차이·95 % CI·p값 |
| `data/per_seed.csv`, `data/per_seed_diff_vs_fp32.csv` | 모델(시드)별 원자료와 차이 |
| `data/power.csv`, `data/sustained_run.csv`, `data/engine_layers.csv` | 전력·60초 연속 실행·층 정밀도 |
| `figures/make_latency_accuracy.py` | 그림 생성 스크립트 (정본) |
| `provenance/environment.json` | 보드 환경, 측정 프로토콜, 꾸러미 검증 결과 |

보드 측정 도구와 절차는 `tools/jetson/`(README 포함). 원자료(예측·tegrastats)는 `runs/jetson_bundle/results/`에 있다.
