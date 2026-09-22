# 논문 패키지 — 실험 01 Overall Performance

노트북에 통째로 받아서 **서버 접속 없이** 논문을 쓸 수 있게 모아 둔 폴더다.
가중치와 확률맵은 들어 있지 않다(§"여기 없는 것").

읽는 순서: 공통 사항 [`../common/setup.md`](../common/setup.md)(과제·데이터·라벨·동결
설정)과 [`../common/metrics.md`](../common/metrics.md)(지표 정의·보고 규칙)을 먼저 보고,
그 다음 [`REPORT.md`](REPORT.md)를 읽는다. `REPORT.md`에 이 실험의 설계, 전체 수치,
해석, 논문 표·그림 구성 제안, LaTeX 뼈대가 있다.

공통 문서를 분리한 이유는 실험이 넷이기 때문이다. 지표 정의를 네 번 복사해 두면 하나만
고쳤을 때 나머지 셋이 조용히 틀려진다.

---

## 세 줄 결과

- 고정 split `iou_free` **0.815 ± 0.001**, constant-map 기준선 0.518, **마진 +0.297**
- LOSO 7 fold macro **0.830**, **마진 +0.293**, fold 간 SD가 시드 SD의 **10.1배**
- **raw `iou_free` 순위와 마진 순위가 뒤집힌다** — 기준선이 통로 폭에 좌우되기 때문

---

## 폴더 구조

```
01_overall/
├── README.md                      이 파일
├── REPORT.md                      ★ 본문: 실험·지표·수치·해석·논문 구성
├── make_package.py                data/ 와 provenance/ 의 생성 스크립트(정본)
├── data/
│   ├── table1_fixed_split.csv       ★ 논문 Table 1 그대로
│   ├── table2_loso.csv              ★ 논문 Table 2 그대로
│   ├── figure_margin_inversion.csv  ★ Figure 1 입력 (7행)
│   ├── fixed_split_per_seed.csv     고정 split 시드별 원자료 (5행 × 9지표)
│   ├── fixed_split_summary.csv      고정 split mean/sd/n
│   ├── loso_per_fold_seed.csv       LOSO 원자료 (35행) — 부록용
│   ├── loso_fold_summary.csv        fold 7행: 요인·기준선·마진·지표
│   ├── loso_macro.csv               macro 집계와 fold 간 SD
│   └── val_curves.csv               40런 × 40 epoch × 3지표 검증 곡선 (4800행)
├── figures/
│   ├── make_margin_inversion.py     ★ Figure 1 생성 스크립트(정본)
│   └── fig1_margin_inversion.{pdf,png,svg}
└── provenance/
    ├── environment_and_config.json  환경·동결 설정·split 구성
    ├── integrity.json               확률맵 재채점 결과 (8건 전부 통과)
    └── run_manifest.csv             40런 목록
```

## 무엇을 보고 무엇을 그리나

| 하려는 것 | 읽을 파일 |
|---|---|
| 논문 Table 1 채우기 | `data/table1_fixed_split.csv` |
| 논문 Table 2 채우기 | `data/table2_loso.csv` |
| Figure 1 다시 그리기 | `figures/make_margin_inversion.py` → `data/figure_margin_inversion.csv` |
| 부록에 시드별 전체 값 싣기 | `data/fixed_split_per_seed.csv`, `data/loso_per_fold_seed.csv` |
| 수렴 곡선 그리기 | `data/val_curves.csv` (열: protocol, run, seed, epoch, metric, value) |
| 재현 절 쓰기 | `provenance/environment_and_config.json` |
| "검증했다" 문장 뒷받침 | `provenance/integrity.json` |

## 그림 다시 그리기

의존성은 **matplotlib 하나**다(pandas 불필요).

```bash
cd figures && python make_margin_inversion.py
```

`--formats=pdf --dpi=600` 같은 플래그를 쓰려면 `fire`가 추가로 필요하다. 없으면 인자 없이
기본값(pdf/png/svg, 400 dpi)으로 돈다 — 그림은 어느 쪽이든 나온다.

**PDF/SVG를 손으로 편집하지 않는다.** 스크립트가 정본이고, 편집하면 다음 재생성 때
사라진다. 색은 colorblind-safe 검사를 통과한 조합이므로 바꿀 때 주의한다.

## 데이터 다시 만들기 (서버에서만)

`make_package.py`는 서버의 결과 bundle을 읽으므로 노트북에서는 돌지 않는다.
숫자가 바뀌었을 때 서버에서:

```bash
cd docs/paper_package/01_overall
python make_package.py --root=../../../runs/paper_final_cumulative/01_overall
cd figures && python make_margin_inversion.py
```

무결성 검사가 통과하지 않은 bundle이면 스크립트가 거부하고 아무것도 쓰지 않는다.

## 여기 없는 것

| 무엇 | 어디에 | 왜 뺐나 |
|---|---|---|
| 40런 체크포인트 (약 38 GB) | 서버 `runs/paper_final_cumulative/01_overall/*/ckpt/` | 논문 작성에 불필요 |
| threshold 전 확률맵 80개 | 서버 `*/analysis/predictions/` | 새 문턱값·영역 지표 계산용. **재학습 없이** 쓸 수 있으나 서버에서 해야 한다 |
| TensorBoard 이벤트 원본 | 서버 `*/logs/` | 필요한 스칼라는 `val_curves.csv`에 뽑아 두었다 |

## 출처

- 공통 설정·지표: [`../common/setup.md`](../common/setup.md), [`../common/metrics.md`](../common/metrics.md)
- 캠페인 원장(정본): `docs/paper_final_experiments.md`
- 실행 계획: `docs/superpowers/plans/2026-09-18-paper-final-overall.md`
- 기계 판독 결과 원본: 서버 `runs/paper_final_cumulative/01_overall/*/analysis/RESULTS.json`
  (옛 `arc_huber` 캠페인은 `runs/paper_final/01_overall/`에 그대로 보존)
- 실행일 **2026-09-23 재학습**(`cumulative_l1`) · Python 3.11.15 / torch 2.7.0+cu128 / CUDA 12.8
  · NVIDIA RTX PRO 6000 Blackwell Max-Q
