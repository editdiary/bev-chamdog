# 논문 작성용 패키지

논문에 필요한 **숫자·그림·해석만** 모아 둔 폴더다. 체크포인트(약 57 GB)와 확률맵은 들어
있지 않다. 이 폴더를 통째로 노트북에 받아 두면 서버 접속 없이 논문을 쓸 수 있다.

> 서버 경로: `~/Desktop/bev-chamdog/docs/paper_package/`
> 전체 크기: 2 MB 미만

---

## 어디부터 읽나

1. 처음이면 **[`common/setup.md`](common/setup.md)** — 과제·데이터·라벨·동결 설정
2. 그 다음 **[`common/metrics.md`](common/metrics.md)** — 지표 정의와 보고 규칙
3. 그 다음 관심 있는 실험의 `REPORT.md`

논문의 **"Training and Implementation Details"**를 쓸 때는
**[`common/training_details.md`](common/training_details.md)** 하나면 된다 — optimizer·
schedule·증강·정밀도·환경·시간이 전부 있고, **`config.json`에 기록돼 있지만 실제로는
동작하지 않는 값 목록**과 고쳐 쓸 영문 초안이 붙어 있다. 그 문서 하나만 따로 떼어
넘겨도 자족적이다.

`common/`을 먼저 읽으라는 이유는, 실험 문서가 지표 정의와 보고 규칙을 반복하지 않고
여기를 가리키기 때문이다. **한 번만 읽으면 네 실험에 다 적용된다.**

## 실험 목록

| 실험 | 논문 역할 | 상태 | 핵심 결과 |
|---|---|---|---|
| [`01_overall`](01_overall/) | 최종 모델 대표 성능 + 시퀀스 일반화 | **완료** (2026-09-18, 40런) | 고정 split `iou_free` 0.812 ± 0.002 (margin **+0.294**) · LOSO macro 0.830 (margin **+0.294**) |
| [`02_projection_and_prior`](02_projection_and_prior/) | 카메라 모델과 source prior (설계 결정 둘의 방어) | **완료** (2026-09-21, 20런) | 어안 원본이 가상 핀홀을 이긴다 (**−0.0134** at 120°, **−0.0078** at 150°) · **커버리지가 해상도를 이긴다** · 합성 사전학습은 `iou_free` 불변, `f1@10cm` **−0.0169** |
| `03_boundary_uncertainty` | boundary-aware loss의 효과와 원인 | 원자료 존재, 재정리 대기 | — |
| `04_edge_deployment` | Jetson AGX Orin 지연 · FPS · 전력 · 메모리 | 측정 대기 | 기존 20.2 FPS는 `Y=1` 값이라 `Y=4` 재실측 필요 |

## 실험 폴더 규약

새 실험이 끝나면 아래 구조로 폴더를 하나 더 만든다. 구조를 맞춰 두면 논문 쓸 때 같은
자리에서 같은 것을 찾을 수 있다.

```
NN_이름/
├── REPORT.md          실험 설계 · 결과 수치 · 해석 · 논문 구성 제안
├── README.md          그 실험 폴더 안내와 "무엇을 보고 무엇을 그리나" 표
├── make_package.py    data/·provenance/ 생성 스크립트 (정본)
├── data/              논문 표 CSV · 그림 입력 CSV · 시드별 원자료
├── figures/           그림 생성 스크립트(정본) + 내보낸 PDF/PNG/SVG
└── provenance/        환경·설정·무결성·런 목록
```

**실험 폴더는 서로 독립이다.** 끝난 실험은 그대로 얼려 두고 인용한다. 반대로 **공통인
것은 `common/`에만** 둔다 — 지표 정의를 네 번 복사해 두면 하나만 고쳤을 때 나머지 셋이
조용히 틀려진다.

## 두 가지 원칙

**하나, 생성 스크립트가 정본이다.** `data/`의 CSV와 `figures/`의 PDF·SVG는 스크립트가
만든 결과물이다. **손으로 고치지 않는다.** 고치면 다음 재생성 때 사라지고, 그 사이에
문서와 데이터가 갈린다. 바꿀 것이 있으면 스크립트를 고치고 다시 돌린다.

**둘, 무결성 게이트를 통과하지 못한 결과는 넣지 않는다.** `make_package.py`는 무결성
검사가 통과하지 않은 bundle이면 거부하고 아무것도 쓰지 않는다.

## 서버에 남겨 둔 것

| 무엇 | 어디 | 왜 |
|---|---|---|
| 체크포인트 | `runs/paper_final/*/ckpt/` | 논문 작성에 불필요 |
| **threshold 전 확률맵** | `runs/paper_final/*/analysis/predictions/` | 새 문턱값·동작점 곡선을 **재학습 없이** 계산할 수 있다. 계산은 서버에서 하고 결과 CSV만 이 폴더에 추가한다 |
| TensorBoard 원본 | `runs/paper_final/*/logs/` | 필요한 스칼라는 각 실험 `data/val_curves.csv`에 뽑아 두었다 |

## 출처

- 캠페인 원장(정본): [`../paper_final_experiments.md`](../paper_final_experiments.md)
- 기계 판독 결과 원본: 서버 `runs/paper_final/*/analysis/RESULTS.json`
