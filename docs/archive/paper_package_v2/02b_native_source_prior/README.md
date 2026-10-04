# 논문 패키지 — 실험 02b source를 원본 기하 그대로 사전학습

노트북에 통째로 받아서 **서버 접속 없이** 논문을 쓸 수 있게 모아 둔 폴더다.
가중치와 확률맵은 들어 있지 않다(§"여기 없는 것").

읽는 순서: 공통 사항 [`../common/setup.md`](../common/setup.md)(과제·데이터·라벨·동결
설정)과 [`../common/metrics.md`](../common/metrics.md)(지표 정의·보고 규칙)을 먼저 보고,
그 다음 [`REPORT.md`](REPORT.md)를 읽는다.

**이 실험은 `02`의 축 B(source prior)를 확장한 것이다.** 먼저
[`../02_projection_and_prior/REPORT.md`](../02_projection_and_prior/REPORT.md) §4를 읽으면
맥락이 선다.

---

## 세 줄 결과

- **원본 기하 사전학습은 도움이 되지 않는다. 해롭다.** scratch 대비 `iou_free`
  **−0.0073**(|Δ|/SE 8.30, 5/5 시드). 9개 지표 중 어느 것도 사전학습 쪽이 낫지 않다
- **가공한 기하보다 더 해롭다 — 단 `iou_free`에 한해서다.** 직접 짝지으면 −0.0046(2.98),
  나머지 지표는 둘을 갈라 주지 못한다
- **두 사전학습이 주는 1에폭 prior가 서로 반대 방향이다.** 가공 기하는 자유공간을 더 넓게,
  원본 기하는 더 좁게 선언한다. **그런데 둘 다 최종 성능을 낮춘다**

대조군은 `01_overall`의 고정 split 5런을 **시드 1:1로 짝지어** 재사용했다.

---

## 폴더 구조

```
02b_native_source_prior/
├── README.md                              이 파일
├── REPORT.md                              ★ 본문: 설계·수치·해석·논문 구성
├── make_package.py                        data/ 와 provenance/ 의 생성 스크립트(정본)
├── data/
│   ├── table1_native_vs_adapted.csv         ★ 논문 표 그대로 (세 줄: scratch/가공/원본)
│   ├── table2_native_prior_all_metrics.csv  ★ 9지표 판정표 (scratch 대비)
│   ├── paired_differences.csv               2팔 × 9지표 짝지은 차이 + 시드별 차이
│   ├── per_seed_metrics.csv                 대조군 포함 3팔 × 5시드 × 9지표 원자료
│   ├── figure_early_epochs.csv              ★ 1에폭 prior 그림 입력 (3팔 × 40에폭)
│   ├── val_curves.csv                       15런 × 40에폭 × 3지표 검증 곡선
│   ├── source_pretrain_summary.csv          ★ 두 사전학습의 source 도메인 최종 성능
│   ├── source_pretrain_curves.csv           원본 기하 사전학습 5런의 source 도메인 곡선
│   ├── source_geometry.csv                  두 source 구성의 격자·셀·보정 여부
│   └── source_label_stats.csv               원본 기하 라벨 통계 (관측률·free 비율·ego)
└── provenance/
    ├── environment_and_config.json          환경·기하·프로토콜·판정 규칙·**유보 둘**
    ├── integrity.json                       확률맵 재채점 (80건 전부 통과)
    └── run_manifest.csv                     10런 목록 (보조항·초기화 포함)
```

**그림 폴더가 없다.** 이 실험은 표 하나와 1에폭 비교 하나로 충분해서 별도 그림을 만들지
않았다. 필요하면 `data/figure_early_epochs.csv`가 그대로 입력이 된다 — `02`의
`figures/make_source_prior_mechanism.py`가 같은 구조의 CSV를 읽는다.

## 무엇을 보고 무엇을 쓰나

| 하려는 것 | 읽을 파일 |
|---|---|
| 논문 표 채우기 (세 줄 비교) | `data/table1_native_vs_adapted.csv` |
| 9지표 판정표 쓰기 | `data/table2_native_prior_all_metrics.csv` |
| "두 prior가 반대 방향" 쓰기 | `data/figure_early_epochs.csv` (epoch 1~3) |
| "학습 실패가 아니다" 뒷받침 | `data/source_pretrain_summary.csv`, `data/source_pretrain_curves.csv` |
| source 설정 설명하기 | `data/source_geometry.csv`, `data/source_label_stats.csv` |
| 부록에 시드별 전체 값 싣기 | `data/per_seed_metrics.csv`, `data/paired_differences.csv` |
| 재현 절 쓰기 | `provenance/environment_and_config.json`, `provenance/run_manifest.csv` |
| "검증했다" 문장 뒷받침 | `provenance/integrity.json` |
| **유보 사항 캡션에 넣기** | `provenance/environment_and_config.json`의 `caveats` |

## ⚠ 반드시 같이 쓰는 유보 둘

1. **원본 기하 라벨에는 수동 보정이 없다.** 보정본은 ROI 8/4/±6에만 존재한다. 그래서
   두 사전학습 팔의 차이에는 **기하와 라벨 품질이 섞여 있다** — "공간 스케일이 원인"이라고
   쓸 수 없다.
2. **SynWoodScape의 BEV는 정사영이 아니다.** z=15 m 핀홀 뷰라 높이 `h`인 면이
   `15/(15−h)`배로 확대된다. 바닥은 정확하지만 장애물 경계가 위로 번진다.

## 데이터 다시 만들기 (서버에서만)

```bash
cd docs/archive/paper_package_v2/02b_native_source_prior
python make_package.py --root=../../../runs/paper_final_cumulative/02b_native_source_prior \
    --control=../../../runs/paper_final_cumulative/01_overall/fixed_split/logs \
    --adapted=../../../runs/paper_final_cumulative/02_projection_and_prior/source_prior/logs \
    --sigma_seed=0.0018
```

무결성 검사가 통과하지 않으면 스크립트가 거부하고 **아무것도 쓰지 않는다.**

## 이 실험을 다시 돌리려면 (서버)

```bash
python tools/build_synwoodscape_native_labels.py          # 라벨, 한 번만
python tools/run_paper_final_native_prior.py --gpu=0 --range_loss_mode=cumulative_l1
ROOT=runs/paper_final_cumulative/02b_native_source_prior \
  CONTROL=runs/paper_final_cumulative/01_overall/fixed_split/logs \
  bash configs/paper_final_native_prior_analysis.sh
```

## 여기 없는 것

| 무엇 | 어디에 | 왜 뺐나 |
|---|---|---|
| 10런 체크포인트 (약 9 GB) | 서버 `runs/paper_final_cumulative/02b_native_source_prior/*/ckpt/` | 논문 작성에 불필요 |
| threshold 전 확률맵 10개 | 서버 `*/analysis/predictions/` | 새 문턱값 계산용. 서버에서만 |
| 원본 기하 라벨 500개 | 서버 `dataset/synwoodscape_native_roi_15_15_15_h08/` | 통계는 `data/source_label_stats.csv`에 |
| TensorBoard 이벤트 원본 | 서버 `*/logs/` | 필요한 스칼라는 `data/val_curves.csv`에 |

## 출처

- 공통 설정·지표: [`../common/setup.md`](../common/setup.md), [`../common/metrics.md`](../common/metrics.md)
- 캠페인 원장(정본): `docs/paper_final_experiments.md` §6b·§8
- 격자 정의: `projects/bev_gt/grid.py`의 `SYNWOODSCAPE_NATIVE_GRID_SPEC`
- 라벨 생성: `tools/build_synwoodscape_native_labels.py`
- 실행일 2026-09-23 · Python 3.11.15 / torch 2.7.0+cu128 / CUDA 12.8
  · NVIDIA RTX PRO 6000 Blackwell Max-Q
