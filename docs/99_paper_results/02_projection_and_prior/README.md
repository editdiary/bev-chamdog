# 논문 패키지 — 실험 02 Design Choices (투영 모델 · 초기화)

캠페인 v3(`runs/99_full_campaign`, 2026-10-02~03) 결과다. 노트북에 통째로 받아 **서버 접속 없이** 논문을 쓸 수 있게 모았다.

## 결과

- 어안을 핀홀로 펴면 `iou_free` −0.0074(150°) / −0.0128(120°), 커버리지 순서를 따른다
- SynWoodScape BEV 사전학습은 해롭다: `iou_free` −0.0060, BF@0.10 −0.0353

## 읽는 순서

1. [`../common/setup.md`](../common/setup.md) — 과제·데이터·라벨·동결 설정
2. [`../common/metrics.md`](../common/metrics.md) — 지표 정의와 보고 규칙
3. [`REPORT.md`](REPORT.md) — 이 실험의 설계·수치·해석·논문 구성

## 폴더

| 경로 | 무엇 |
|---|---|
| `REPORT.md` | ★ 본문 |
| `make_package.py` | `data/`·`provenance/`의 생성 스크립트 (**정본** — CSV를 손으로 고치지 않는다) |
| `data/table_paper_*.csv` | 논문 표 그대로 (mean ± SD, 유의한 칸에 `*`) — 실험 01은 `table1`·`table2` |
| `data/table_paper_caption.txt` | 표 캡션의 통계 문구 |
| `data/*.csv` | 그 밖의 원자료·짝지은 검정 결과 |
| `figures/make_*.py` | 그림 생성 스크립트 (정본, 02b에는 그림이 없다) |
| `provenance/` | 환경·설정, 확률맵 재채점 결과, 런 목록 |

가중치와 확률맵은 들어 있지 않다(`runs/99_full_campaign/`에 있다).
