# `03_boundary_uncertainty` — 폴더 안내

논문 Results 절 **"Effect of Boundary-Uncertainty-Aware Learning"**에 필요한 숫자·그림·해석만
모아 둔 폴더다. 체크포인트와 확률맵은 들어 있지 않다(서버 `runs/loss_effect/`).

**해석 정본은 [`REPORT.md`](REPORT.md)다.** 먼저 읽는다.

---

## 한 줄

**경계를 0/1로 감독한 것이 목적함수를 망가뜨리고 있었다.** soft target이 되올림을 95배
줄이고 목적함수를 품질과 다시 정렬시킨다. **정확도는 안 바뀌고 경계는 조금 번진다.**

## 무엇을 보고 무엇을 그리나

| 논문에 쓸 것 | 파일 | 한 줄 |
|---|---|---|
| **Table A** 사다리 정확도 | `data/table1_ladder_accuracy.csv` | 6조건 × 9지표, 평균±SD. **"거의 안 움직인다"가 요지** |
| **Table B** 목적함수 수렴 | `data/table2_objective_convergence.csv` | 되올림 % · val 최저 epoch · Δep · regret |
| **Table C** 경계 국소화 | `data/table3_boundary_localization.csv` | 대역/대역 밖 CE 상승과 대역의 몫 |
| **Table D** 보조항 (부록) | `data/appendix_aux_range_term.csv` | `C_soft` 대비 `D_range`·`E_cumulative` |
| **Table E** 동작점 (부록) | `data/table5_threshold_dispersion.csv` | 앵커 5개 × (τ\*, `FR`, `MOR`, `F1@10cm`, `IoU_free`) |
| **Figure 1** | `figures/fig1_rebound_localization.*` | 되올림이 경계에만 있다 · **본문 필수** |
| **Figure 2** | `figures/fig2_operating_point.*` | (a) 문턱 재현성 (b) **안전은 개선 아님** |
| **Figure 3** | `figures/fig3_boundary_precision_cost.*` | 경계가 번지는 비용 |

보조 자료: `data/table4_operating_point.csv`(단일 앵커 요약) ·
`data/paired_differences.csv`(짝지은 7쌍 × 9지표) · `data/per_seed_metrics.csv`(30런 원자료) ·
`data/val_curves.csv`(곡선 전체) · `data/figure_*.csv`(그림 입력).

## 조건 이름

| 코드 | 논문에서 부를 이름 | 손잡이 |
|---|---|---|
| `A_ce` | Weighted CE (baseline) | 역빈도 가중, hard target |
| `B_perset` | Per-set mean, no boundary term | 대역 감독 없음 |
| `C_hard` | Hard boundary target | λ_B=0.5, σ→0 |
| `C_soft` | **Soft boundary target (proposed)** | λ_B=0.5, σ=0.10 m |
| `D_range` | + auxiliary range term (`arc_huber`) | λ_R=0.3 — **논문에서 뺀다** |
| `E_cumulative` | + auxiliary range term (`cumulative_l1`) | λ_R=0.15 — **본문 마지막 칸** |

> **⚠ `D`와 `E`는 사다리의 다음 계단이 아니다.** 둘 다 `C_soft`에서 갈라지는 **대체 팔**이고,
> 다른 것은 보조항의 식뿐이다. 논문 본문은 `A_ce→B_perset→C_hard→C_soft→E_cumulative`
> 다섯 칸이고 `D_range`는 뺀다([2026-09-22 사용자 계획]). **보조항을 아예 빼고 넷으로 갈지는
> 검토 중이다** — 그래서 데이터는 여섯 칸을 다 담아 둔다.

## 재생성

```bash
conda activate bev-chamdog
cd docs/archive/paper_package_v2/03_boundary_uncertainty
python make_package.py                     # data/ + provenance/
cd figures && for f in make_*.py; do python "$f"; done
```

**생성 스크립트가 정본이다.** `data/`의 CSV와 `figures/`의 PDF·SVG를 손으로 고치지 않는다 —
다음 재생성 때 사라지고, 그 사이에 문서와 데이터가 갈린다.

`make_package.py`는 **무결성 게이트**를 먼저 통과해야 아무것이든 쓴다. 확률맵 재채점
480건 중 하나라도 실패하면 멈춘다.

## ⚠ 연휴 재학습 뒤 할 일

`cumulative_l1`로 실험 1·2를 다시 돌린 뒤에는 **이 폴더를 다시 생성하고 CSV를 diff한다.**
그러면 실제로 바뀐 칸만 뜨므로 논문에서 고칠 자리를 손으로 찾지 않아도 된다.

**다만 `A_ce`~`C_soft` 네 칸은 바뀌지 않는다** — 그 런들은 `λ_R = 0.0`이고 저장소가
그때 보조항을 계산조차 하지 않는다(로그에 `loss_range` 태그가 없는 것이 증거다).
바뀌는 것은 부록의 보조항 표와, 실험 1·2 패키지의 숫자다.

## 출처

- 해석 정본(**서버 전용** — 패키지 밖이다): `docs/archive/research/loss_effect_results.md`
- 지표 정의·집계·통계 규약: [`../common/evaluation_protocol.md`](../common/evaluation_protocol.md)
- 학습 설정: [`../common/training_details.md`](../common/training_details.md)
- 캠페인 원장(**서버 전용**): `docs/paper_final_experiments.md`
- 기계 판독 원본: 서버 `runs/loss_effect/analysis/RESULTS.json`
