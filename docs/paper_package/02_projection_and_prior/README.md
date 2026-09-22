# 논문 패키지 — 실험 02 카메라 모델과 source prior

노트북에 통째로 받아서 **서버 접속 없이** 논문을 쓸 수 있게 모아 둔 폴더다.
가중치와 확률맵은 들어 있지 않다(§"여기 없는 것").

읽는 순서: 공통 사항 [`../common/setup.md`](../common/setup.md)(과제·데이터·라벨·동결
설정)과 [`../common/metrics.md`](../common/metrics.md)(지표 정의·보고 규칙)을 먼저 보고,
그 다음 [`REPORT.md`](REPORT.md)를 읽는다. `REPORT.md`에 설계, 전체 수치, 해석,
논문 표·그림 구성 제안, 논리 전개 순서, LaTeX 뼈대가 있다.

---

## 세 줄 결과

- **어안을 그대로 쓰는 것이 옳았다.** 가상 핀홀 두 팔이 모두 졌다 — 120°에서
  `iou_free` **−0.0134**, 150°에서 **−0.0078**. 9개 지표 전부, 5개 시드 전부 같은 방향
- **이긴 이유는 해상도가 아니라 커버리지다.** 핀홀 120°는 150°보다 정면 각해상도가
  **2.2배** 높은데도 더 나쁘다(−0.0055, 5/5 시드)
- **합성 사전학습은 경계를 흐리고, 자유공간의 양도 조금 깎는다.** `f1@10cm` **−0.0220**,
  `iou_free` **−0.0027**(둘 다 5/5). 40 cm 허용오차에서는 오히려 +0.0033로 뒤집힌다 —
  경계가 부서진 게 아니라 흐려진 것이다. 1에폭에 보이는 prior는 **3에폭이면 풀린다**

대조군은 `01_overall`의 고정 split 5런을 **시드 1:1로 짝지어** 재사용했다.

---

## 폴더 구조

```
02_projection_and_prior/
├── README.md                          이 파일
├── REPORT.md                          ★ 본문: 실험·수치·해석·논문 구성·논리 전개
├── make_package.py                    data/ 와 provenance/ 의 생성 스크립트(정본)
├── data/
│   ├── table1_projection.csv            ★ 논문 Table 3 그대로 (축 A)
│   ├── table2_source_prior.csv          ★ 논문 Table 5 그대로 (축 B, 9지표)
│   ├── figure_coverage_vs_accuracy.csv  ★ Figure 3 입력 (3행)
│   ├── figure_early_epochs.csv          ★ Figure 4 입력 (5시드 평균·표준편차)
│   ├── paired_differences.csv           3팔 × 9지표 짝지은 차이 전체 + 시드별 차이
│   ├── per_seed_metrics.csv             대조군 포함 4팔 × 5시드 × 9지표 원자료
│   ├── val_curves.csv                   20런 × 40에폭 × 3지표 검증 곡선
│   ├── projection_tradeoff.csv          화각 90~170°의 커버리지·각해상도 절충 (설계 근거)
│   ├── camera_coverage.csv              3-camera 대 front-only 커버리지 (축 C 기각 근거)
│   └── domain_prior.csv                 SynWoodScape 대 로봇의 라벨 밀도 (축 B 근거)
├── figures/
│   ├── make_coverage_vs_accuracy.py     ★ Figure 3 생성 스크립트(정본)
│   ├── fig1_coverage_vs_accuracy.{pdf,png,svg}
│   ├── make_source_prior_mechanism.py   ★ Figure 4 생성 스크립트(정본)
│   └── fig2_source_prior_mechanism.{pdf,png,svg}
└── provenance/
    ├── environment_and_config.json      환경·팔 설계값·프로토콜·판정 규칙
    ├── integrity.json                   확률맵 재채점 (240건 전부 통과)
    └── run_manifest.csv                 20런 목록 (투영·화각·초기화 포함)
```

## 무엇을 보고 무엇을 그리나

| 하려는 것 | 읽을 파일 |
|---|---|
| 카메라 모델 ablation 표 채우기 | `data/table1_projection.csv` |
| 안전·경계 지표 표 채우기 | `data/paired_differences.csv` (`arm=pinhole120` 행) |
| source prior 표 채우기 | `data/table2_source_prior.csv` |
| Figure 3 다시 그리기 | `figures/make_coverage_vs_accuracy.py` → `data/figure_coverage_vs_accuracy.csv` |
| Figure 4 다시 그리기 | `figures/make_source_prior_mechanism.py` → `data/figure_early_epochs.csv` |
| "왜 화각을 120°와 150°로 골랐나" 쓰기 | `data/projection_tradeoff.csv` |
| "왜 카메라 3대인가" setup 절에 쓰기 | `data/camera_coverage.csv` |
| "도메인 격차는 라벨 밀도다" 쓰기 | `data/domain_prior.csv` |
| 부록에 시드별 전체 값 싣기 | `data/per_seed_metrics.csv`, `data/paired_differences.csv` |
| 수렴 곡선 그리기 | `data/val_curves.csv` (열: arm, run, seed, epoch, metric, value) |
| 재현 절 쓰기 | `provenance/environment_and_config.json`, `provenance/run_manifest.csv` |
| "검증했다" 문장 뒷받침 | `provenance/integrity.json` |

## 그림 다시 그리기

의존성은 **matplotlib 하나**다(pandas 불필요).

```bash
cd figures
python make_coverage_vs_accuracy.py
python make_source_prior_mechanism.py
```

`--formats=pdf --dpi=600` 같은 플래그를 쓰려면 `fire`가 추가로 필요하다. 없으면 인자 없이
기본값(pdf/png/svg, 400 dpi)으로 돈다.

**PDF/SVG를 손으로 편집하지 않는다.** 스크립트가 정본이고, 편집하면 다음 재생성 때
사라진다. 재생성해도 바이트가 같도록 타임스탬프를 제거해 두었으므로, git이 변경으로
잡으면 그것은 **진짜 내용 변경**이다.

## 데이터 다시 만들기 (서버에서만)

```bash
cd docs/paper_package/02_projection_and_prior
python make_package.py --root=../../../runs/paper_final_cumulative/02_projection_and_prior \
    --control=../../../runs/paper_final_cumulative/01_overall/fixed_split/logs
cd figures && python make_coverage_vs_accuracy.py && python make_source_prior_mechanism.py
```

무결성 검사가 통과하지 않으면 스크립트가 거부하고 **아무것도 쓰지 않는다.**

## 이 실험을 다시 돌리려면 (서버)

```bash
python tools/run_paper_final_projection_prior.py --arms=all --gpu=0   # 완료 런은 건너뛴다
bash configs/paper_final_projection_prior_analysis.sh
```

## 여기 없는 것

| 무엇 | 어디에 | 왜 뺐나 |
|---|---|---|
| 20런 체크포인트 (약 19 GB) | 서버 `runs/paper_final_cumulative/02_projection_and_prior/*/ckpt/` | 논문 작성에 불필요 |
| threshold 전 확률맵 30개 | 서버 `*/analysis/predictions/` | 새 문턱값·영역 지표 계산용. **재학습 없이** 쓸 수 있으나 서버에서 해야 한다 |
| TensorBoard 이벤트 원본 | 서버 `*/logs/` | 필요한 스칼라는 `data/val_curves.csv`에 뽑아 두었다 |
| SynWoodScape 사전학습 로그 | 서버 `*/source_pretrain/logs/` | source 도메인 성능은 논문에 쓰지 않는다 |

## 출처

- 공통 설정·지표: [`../common/setup.md`](../common/setup.md), [`../common/metrics.md`](../common/metrics.md)
- 캠페인 원장(정본): `docs/paper_final_experiments.md` §6
- 설계 근거 측정 도구: `tools/measure_projection_tradeoff.py`, `tools/measure_domain_prior.py`
- 실행일 **2026-09-23 재학습**(`cumulative_l1`) · Python 3.11.15 / torch 2.7.0+cu128 / CUDA 12.8
  · NVIDIA RTX PRO 6000 Blackwell Max-Q
