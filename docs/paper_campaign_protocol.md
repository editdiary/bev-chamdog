# 논문 실험 캠페인 프로토콜 — 처음부터 다시 돌릴 때의 사용법

**이 문서는 "어떻게 돌리나"의 정본이다.** "무엇이 나왔나"는
[`paper_final_experiments.md`](paper_final_experiments.md)(원장)에 있고, "논문에 쓸 숫자"는
[`paper_package/`](paper_package/)에 있다. 셋을 섞지 않는다.

**[2026-09-22~23]에 이 절차로 70런을 사람 없이 완주했다.** watchdog 발동 0회, 무결성
실패 0건, 8시간 43분. 아래는 그때 실제로 한 것과 **그때 물린 것**을 합친 것이다.
다음에 설정을 바꿔 처음부터 다시 돌릴 때 이 문서만 보면 된다.

---

## 0. 불변식 — 이걸 어기면 나머지가 의미 없다

| 불변식 | 왜 |
|---|---|
| **conda 환경은 `bev-chamdog`** (Python 3.11.15 + torch 2.7.0+cu128) | `base`(3.14)에서는 numpy가 **256 KB 넘는 배열을 조용히 덮어쓴다.** 2026-09-01에 그 환경으로 20런을 통째로 날렸다. 학습 셸에 가드가 들어 있지만 분석·패키지 스크립트는 직접 확인한다 |
| **새 캠페인은 새 루트에 쓴다** (`runs/<이름>/`) | 덮어쓰면 "지금 논문에 들어가 있는 숫자가 어디서 나왔나"가 사라지고, 두 설정이 얼마나 달랐는지도 확인할 수 없다 |
| **대조군을 먼저 완성한다** | `02`·`02b`가 `01`의 고정 split을 **시드 1:1로 짝지어** 재사용한다. 그리고 판정 눈금 σ_seed도 거기서 나온다 |
| **사전학습 팔이 그 실험 안에서 먼저 돈다** | 미세조정 팔이 그 체크포인트에서 시작한다. 큐 러너가 순서를 보장하지만, 실험 순서는 사람이 정한다 |
| **`git stash`·`git checkout <다른 브랜치>`·`reset --hard` 금지** | 작업 트리를 다른 세션과 공유한다. 커밋은 `git add <경로>`로 **자기 파일만 명시**한다 |
| **푸시는 사용자가 한다** | 권한 분류기가 막는다. 명령어만 알려 준다 |
| **생성 스크립트가 정본이다** | CSV·SVG·PDF를 손으로 고치지 않는다. 다음 재생성에서 사라지고, 그 사이에 문서와 데이터가 갈린다 |

---

## 1. 전체 모양 — Phase 0~4

```
Phase 0  설정 동결 + 모든 손잡이를 밖으로 노출   (30분~2시간)
   ↓     ⚠ 노출 안 된 손잡이가 하나라도 있으면 "절반만 적용"된다
리허설   1 epoch × 팔마다 1런                    (10분)
   ↓     ⚠ 여기서 안 걸리는 건 8시간 뒤에 걸린다
Phase 1  본 학습                                 (실측 70런 = 8시간 43분)
   ↓     watchdog이 멈춤을 잡는다. 죽어도 같은 명령으로 이어진다
Phase 2  확률맵 내보내기 → 독립 재채점 → 게이트 → 짝지은 비교  (10분)
   ↓     ⚠ 하나라도 실패하면 패키지를 만들지 않는다
Phase 3  패키지 재생성 → 칸 단위 diff → (검토) → APPLY=1      (5분)
   ↓     ⚠ 기본은 덮어쓰지 않는다
Phase 4  원장 · AGENTS.md · 메모리 · **산문 숫자 감사**        (1~2시간)
```

**Phase 4가 가장 길다.** CSV는 스크립트가 갈아 주지만 `REPORT.md`·`README.md`에 박힌
소수 숫자 **400개 이상**은 사람이 고쳐야 한다. 감사기가 대상을 찍어 주지만 고치는 건 손이다.

---

## 2. Phase 0 — 설정 동결과 손잡이 노출

### 2.1 가장 흔한 사고: 절반만 적용

2026-09-22에 실제로 있던 상태다. `tools/train_synwoodscape.py`가 `range_loss_mode`를
받지 않아서, 캠페인을 새 loss로 옮기면 **미세조정만 바뀌고 사전학습 둘은 옛 형태로
남았다.** 로그에도 티가 나지 않는다.

**점검 절차.**

1. 바꾸려는 설정이 **학습 스크립트 둘 다**(`train_robot_bev.py`,
   `train_synwoodscape.py`)에서 인자로 받아지는가
2. **셸 둘 다**(`train_robot_bev_finetune.sh`, `train_synwoodscape_*_pretrain.sh`)가
   환경변수로 넘기는가. **변수명을 글자 그대로 같게** 둔다 — 다르면 한쪽만 바뀐 걸
   알아채지 못한다
3. **큐 러너 셋**이 그 값을 넘기는가. 값은 한 모듈에 모은다
   (`tools/paper_final_aux_loss.py`가 그 예다 — 세 러너가 import한다)
4. `config.json`에 **실제 값**이 적히는가. 하드코딩된 문자열이 남아 있지 않은가
5. 학습 시작 로그에 한 줄로 찍히는가 (`[loss] ...` — 나중에 grep으로 확인하려면 필요)

### 2.2 스모크 둘로 "실제로 바뀌었나"를 증명한다

**config.json만 보고 넘어가지 않는다.** 1 epoch씩 옛 설정과 새 설정으로 돌려서
**계산 경로가 실제로 갈렸는지**를 본다.

2026-09-22의 증거: `cumulative_l1`에서만 `range_cumulative_mae` 스칼라가 기록되고
그 값이 `loss_range`와 정확히 일치했다(0.9101). `arc_huber`에는 그 태그가 없었다.
train `loss_total`도 1.2022 대 0.8101로 갈렸다.

```bash
# 텐서보드 스칼라를 직접 읽는다 (torch.utils.tensorboard가 아니라 tensorboard 패키지)
python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("<로그폴더>/events.out.tfevents.*"); ea.Reload()
print(sorted(ea.Tags()["scalars"]))
PY
```

### 2.3 회귀 테스트를 건다

`tests/tools/test_paper_final_aux_loss.py`가 본보기다. **사전학습 런을 포함해** 모든 런이
같은 값을 받는지 고정하고, 런 이름 목록을 명시해 **러너가 하나 빠지면 걸리게** 한다.

---

## 3. 리허설 — 8시간 태우기 전 10분

```bash
R=runs/_rehearsal
python tools/run_paper_final_overall.py --protocol=fixed --seeds=0 --num_epochs=1 --gpu=0 \
    --output_root=$R/01_overall <새 설정 플래그>
python tools/run_paper_final_overall.py --protocol=loso --seeds=0 --folds=raws1 --num_epochs=1 ...
python tools/run_paper_final_projection_prior.py --seeds=0 --num_epochs=1 ...
python tools/run_paper_final_native_prior.py --seeds=0 --num_epochs=1 ...
# Phase 2도 관통시킨다 (SEEDS=0 FIXED_EPOCH=1 FOLDS=raws1)
```

**확인할 것 넷.**

1. 모든 런이 새 설정으로 만들어지는가 (manifest의 environment를 기계로 센다)
2. 경로가 새 루트 밖으로 새지 않는가
3. **체크포인트 의존 사슬** — `source_prior`가 **새 루트의** 사전학습을 무는가
   (전이 로그의 "loaded N tensors, skipped 0")
4. Phase 2가 rc=0으로 끝나는가. 무결성 **수치** 실패가 0인가
   (구조 검사 실패 "5런 기대, 1런 발견"은 1시드 리허설이라 정상이다)

**끝나면 지운다.** 1 epoch × 7런이 7.4 GB다.

---

## 4. Phase 1 — 학습

```bash
conda activate bev-chamdog
nohup bash tools/run_paper_final_cumulative_campaign.sh > <루트>/training_driver.log 2>&1 &
```

드라이버가 하는 일: 세 실험을 순서대로(`01` → `02` → `02b`), 한 실험이 실패해도
나머지는 계속. 큐는 **완료된 런을 건너뛰고 불완전한 런을 덮어쓰지 않는다**
(`blocked_incomplete`) — 중간에 죽어도 같은 명령을 다시 실행하면 이어진다.

### 4.1 멈춤 방지 — `tools/paper_final_run_command.py`

| 한도 | 값 | 무엇을 잡나 |
|---|---|---|
| **침묵** | 20분 무출력 | **주 감지기.** 가장 느린 런도 epoch이 30초 미만이다 |
| **총 시간** | 2시간 | 출력은 나오는데 안 끝나는 경우. 최장 런(19분)의 6배 |

**프로세스 그룹째 죽인다.** `bash`만 죽이면 `python`이 GPU를 붙든 채 고아로 남아
다음 런이 연쇄 실패한다. 회귀 테스트가 손자 프로세스까지 죽는지 잰다.

런마다 stdout이 `<루트>/<실험>/run_logs/<런>.log`에 남는다.

### 4.2 비용 (2026-09-23 실측, RTX PRO 6000 한 장)

| 런 종류 | 1런 | 비고 |
|---|---:|---|
| 로봇 미세조정 (01 고정·LOSO, 02 세 팔, 02b) | **5.5분** | 40 epoch, 8초/epoch |
| SynWoodScape 사전학습 (8/4/±6, 240×240) | **19분** | |
| SynWoodScape 사전학습 (원본 30 m, 200×200) | **17분** | |

70런 = **8시간 43분**. 디스크 **65 GB** — 런마다 체크포인트가 **둘**이다
(`model-000000040.pth` + `model_best-*.pth`, 각 470 MB). 하나로 세면 절반으로 틀린다.

### 4.3 감시

harness가 추적하는 백그라운드 작업 **둘**을 건다.

1. 완료/드라이버 사망 대기 — `until grep -q "CAMPAIGN_TRAINING_DONE" $LOG || ! kill -0 $PID`
2. 이상 징후만 알리는 감시 — watchdog 발동·Traceback·OOM·디스크 부족·런 실패

> **함정.** `bad=$(grep -c ... || echo 0)`은 매치 0일 때 `"0\n0"`이 되어
> `[ "$bad" -gt 0 ]`이 깨진다. `grep -c ... ; true` 쓰거나 `wc -l`로 센다.

### 4.4 끝났을 때 세는 법

**드라이버 로그의 "complete" 개수를 믿지 않는다** — 러너마다 완료 줄의 형식이 다르다.
manifest를 읽어 런마다 `config.json`(설정·epoch 수)과 **최종 체크포인트 존재**를 확인한다.

---

## 5. Phase 2 — 무결성과 분석

```bash
bash tools/run_paper_final_cumulative_analysis.sh
```

순서: `01` 분석 → **σ_seed 결정** → `02` 분석 → `02b` 분석 → 트리 순수성.

### 5.1 무결성이 무엇을 하나

threshold 적용 **전** 확률맵을 내보내고, **독립 경로로 지표를 처음부터 다시 계산해**
학습 로그와 대조한다. 허용치 **1e-3**. 2026-09-23 실측 최대 오차 **6.9e-4**.

**하나라도 실패하면 패키지를 만들지 않는다.** 게이트가 스크립트에 박혀 있다.

### 5.2 ⚠ 판정 눈금 σ_seed — 함부로 낮추지 않는다

**판정 규칙**: `|Δ̄| > σ_seed(대조군)` **그리고** `|Δ̄| ≥ 2·SE`, `SE = sd(Δ)/√5`.
n=5라 p값을 만들지 않는다. **둘 다** 넘어야 우열을 주장한다.

2026-09-23에 물린 것: 새 대조군의 고정 split σ_seed가 **0.0018 → 0.0007**로 보였다.
눈금이 작아지면 **같은 차이가 더 쉽게 유의가 되어 주장이 공짜로 강해진다.**

| 추정 방식 | 옛 | 새 | 자유도 |
|---|---:|---:|---:|
| 고정 split 5시드만 | 0.0018 | **0.0007** | 4 |
| LOSO fold별 평균 (7 fold) | 0.0024 | 0.0025 | — |
| **8군 묶음** (고정 + LOSO 7) | **0.0025** | **0.0026** | **32** |

LOSO fold별 추정치 14개가 **0.0010~0.0049**에 흩어진다 — **5표본 sd는 5배 범위로
흔들린다.** 묶음 추정으로는 변화가 없었다.

**결론: 사전 등록값을 유지하고, 아슬아슬한 칸은 더 보수적인 묶음값으로도 따져 둘 다
적는다.** 드라이버가 `SIGMA_SEED`를 기본 0.0018로 고정하고 측정값은 출력만 한다.

### 5.3 실행 중인 bash 스크립트를 편집하지 않는다

bash는 스크립트를 **증분으로 읽는다.** 도는 중에 고치면 동작이 깨진다.
고칠 게 생기면 메모해 두고 끝난 뒤에 고친다.

---

## 6. Phase 3 — 패키지

```bash
bash tools/run_paper_final_cumulative_packages.sh            # 대조만 (기본)
APPLY=1 bash tools/run_paper_final_cumulative_packages.sh    # 검토 뒤 반영
```

**기본은 덮어쓰지 않는다.** 임시 폴더에 만들고 칸 단위 대조만 출력한다.
`APPLY=1`이면 CSV·provenance를 복사하고 **그림 생성 스크립트도 같이 돌린다**
(CSV만 갈면 문서 안에서 숫자와 그림이 어긋나는데 그건 눈으로 안 잡힌다).

### 6.1 패키지 구조 계약

```
NN_이름/
├── README.md        노트북에서 혼자 읽을 수 있게. 세 줄 결과 + 폴더 구조 + "무엇을 보고 무엇을 쓰나"
├── REPORT.md        ★ 설계 · 전체 수치 · 해석 · 논문 구성 제안 · 쓰면 안 되는 문장
├── make_package.py  ★ data/ 와 provenance/ 의 정본. --root/--out_dir/--control을 받는다
├── data/            CSV만. 손으로 고치지 않는다
├── figures/         make_*.py(정본) + pdf/png/svg. 없어도 된다
└── provenance/      environment_and_config.json · integrity.json · run_manifest.csv
```

**패키지는 자족적이어야 한다** — 패키지 밖을 가리키는 링크 0건. 공통 사항은
`paper_package/common/{setup,metrics,training_details,evaluation_protocol}.md`에 있고
거기까지만 링크한다.

`make_package.py`는 **무결성이 통과하지 않으면 아무것도 쓰지 않고 멈춘다.**

### 6.2 무엇을 보나

| 도구 | 잡는 것 |
|---|---|
| `tools/diff_paper_package_csv.py` | CSV 칸 단위 변화. **판정 열**(`verdict`·`sign_agreement`·`better_direction`)은 따로 센다 |
| `tools/check_results_tree_purity.py` | **옛 결과 트리가 섞인 것.** 산출물의 경로 문자열을 훑는다 |

2026-09-23에 순수성 검사가 **실제로 잡은 것 둘**:
- `experiment_manifest.json`의 `control_arm`이 옛 트리 경로를 문자열로 박고 있었다
- README·REPORT의 **재현 명령과 "여기 없는 것" 표**가 옛 트리를 가리켰다 —
  그대로 따라 하면 옛 결과로 재생성된다 (11건)

---

## 7. Phase 4 — 기록

### 7.1 ⚠ 산문에 박힌 숫자 — 가장 많은 손이 드는 곳

패키지 넷의 `REPORT.md`·`README.md`에 소수 셋넷 자리 숫자가 **520개 이상** 있다.
CSV만 갈고 산문을 두면 **논문이 데이터와 어긋난 채로 작성된다.**

```bash
# 옛 CSV를 git에서 꺼내 임시 폴더에 두고, 새 패키지와 대조한다
python tools/audit_paper_prose_numbers.py --old=<옛패키지> --new=<새패키지>
```

- `stale` = 옛 CSV엔 있고 새 CSV엔 없다 → **반드시 고친다**
- `unsourced` = 어느 CSV에서도 못 찾았다 → 사람이 본다 (손계산·외부 출처·설계 상수)

2026-09-23 실적: 낡은 숫자 **81개**를 고쳤고, 최종 추적률 01 123/127 · 02 156/161 ·
02b 86/89 · 03 143/145. **감사기가 내 반올림 오차 둘도 잡았다**(차이에서 역산하다
마지막 자리가 어긋난 것).

> 감사기가 **의도한 인용도 `stale`로 찍는다** — 옛 캠페인 값을 이력으로 적은 경우다.
> 0으로 만들려 하지 말고, 남은 것 하나하나가 의도한 것인지 확인하고 넘어간다.

### 7.2 어디에 무엇을 적나

| 대상 | 무엇 |
|---|---|
| `docs/paper_final_experiments.md` §5 (진행 기록) | 바뀌지 않은 것 / **바뀐 결론** / 검증 결과 / 눈금 결정 |
| 같은 문서 §8 (실행 계획) | 완료 표시 + 실행 기록 표 + **계획에 없었는데 필요했던 것** |
| `AGENTS.md` 캠페인 블록 | **"논문 숫자의 정본은 패키지다. 원장 §4·§6 표는 이력이다"** + 바뀐 결론 + 읽는 법 |
| `docs/paper_package/README.md` | 실험 목록 표의 상태·핵심 결과 |
| 메모리 | 현재 지점, 뒤집힌 결론, 검증 도구 |

### 7.3 철회된 주장이 되살아나지 않았는지 확인한다

```bash
grep -rn "안전 개선\|안전성.*개선\|정확도.*개선\|재현성.*개선" docs/paper_package/*/{REPORT,README}.md AGENTS.md
```

걸린 것이 **철회를 명시한 문장**인지 확인한다. 되살아난 주장이면 고친다.

### 7.4 결론이 바뀌면 그것 자체가 보고 대상이다

**조용히 숫자만 갈아끼우지 않는다.** 2026-09-23에 셋이 바뀌었고 전부 적었다 —
판정이 선 칸, 논증의 근거가 좁아진 곳, 우연의 일치가 사라진 곳.

---

## 8. 지뢰 목록 — 실제로 물린 것만

| 지뢰 | 증상 | 대응 |
|---|---|---|
| **conda `base`** | numpy가 256 KB 넘는 배열을 조용히 덮어쓴다 | 모든 스크립트에서 `CONDA_DEFAULT_ENV` 확인 |
| **BEV 격자가 8로 안 나눠떨어짐** | decoder가 /8 후 skip 복원하다 26 대 25로 깨진다 | 100×100 ✗, 200×200 ✓ |
| **사전학습에 설정이 안 걸림** | 미세조정만 바뀐다. 로그에 티가 안 난다 | §2.1 점검 5단계 |
| **`constant_map_baseline.json`이 없다** | 패키지 생성이 멈춘다 | `tools/make_constant_map_baseline.py`. **학습과 무관한 값이라 재학습해도 같아야 한다**(1e-9 이내로 같았다 = 라벨·split 불변 확인) |
| **σ_seed가 작아 보임** | 판정이 공짜로 헐거워진다 | §5.2 묶음 추정 |
| **실행 중 bash 편집** | 동작이 깨진다 | 끝난 뒤에 고친다 |
| **옛 트리 경로가 산문에 남음** | 재현 명령이 옛 결과를 만든다 | 순수성 검사 |
| **체크포인트를 런당 1개로 셈** | 디스크 추정이 절반 | final + best = 941 MB/런 |
| **`pkill -f` 패턴** | 자기 셸을 죽인다 | 정확한 PID |
| **GPU 점유** | 지연 측정이 4.6배 틀어진다 | 재기 전 `nvidia-smi` |
| **`grep -c ... \|\| echo 0`** | `"0\n0"`이 되어 정수 비교가 깨진다 | `; true` 또는 `wc -l` |

---

## 9. 지금 쓰는 도구 목록

| 파일 | 역할 |
|---|---|
| `tools/run_paper_final_overall.py` | `01` 큐 (고정 split 5 + LOSO 35) |
| `tools/run_paper_final_projection_prior.py` | `02` 큐 (사전학습 5 + 세 팔 15). **큐 기계장치의 정본** |
| `tools/run_paper_final_native_prior.py` | `02b` 큐. 위 모듈의 `run_queue`를 재사용한다 |
| `tools/paper_final_aux_loss.py` | 세 러너가 공유하는 보조항 설정 (**값의 정본**) |
| `tools/paper_final_run_command.py` | watchdog (침묵 20분 / 총 2시간, 프로세스 그룹째) |
| `tools/run_paper_final_cumulative_campaign.sh` | Phase 1 드라이버 |
| `tools/run_paper_final_cumulative_analysis.sh` | Phase 2 드라이버 (σ_seed 고정 포함) |
| `tools/run_paper_final_cumulative_packages.sh` | Phase 3 드라이버 (`APPLY=1`이어야 덮어쓴다) |
| `configs/paper_final_overall_analysis.sh` | `01` 분석. `ROOT`·`SEEDS`·`FOLDS`·`FIXED_EPOCH` |
| `configs/paper_final_projection_prior_analysis.sh` | `02` 분석. `ARMS`·`PRETRAIN_ARM`·`CONTROL`·`SIGMA_SEED`로 **02b도 이걸 쓴다** |
| `configs/paper_final_native_prior_analysis.sh` | 위를 이름만 바꿔 부르는 얇은 래퍼 |
| `tools/make_constant_map_baseline.py` | constant-map 기준선 (`--expect`로 불변 확인) |
| `tools/diff_paper_package_csv.py` | 패키지 CSV 칸 단위 대조 |
| `tools/audit_paper_prose_numbers.py` | 산문에 박힌 숫자 감사 |
| `tools/check_results_tree_purity.py` | 옛 결과 트리 혼입 검사 |
| `tools/report_paired_arms.py` | 짝지은 비교 (`--sigma_seed`·`--baseline_path`) |
| `tools/build_overall_results_bundle.py` | `01`의 `RESULTS.json` |
| `tools/export_val_predictions.py` · `verify_val_predictions.py` | 확률맵 내보내기 · 독립 재채점 |

**테스트 503개.** 캠페인 관련은 `tests/tools/test_paper_final_*.py`,
`test_diff_paper_package_csv.py`, `test_check_results_tree_purity.py`,
`test_audit_paper_prose_numbers.py`, `test_run_paper_final_*.py`.

---

## 10. 새 캠페인을 열 때 고칠 곳

설정을 바꿔 **처음부터** 다시 돌린다면:

1. **루트 이름을 정한다** — `runs/<새이름>/`. 세 드라이버의 `ROOT` 기본값
2. **Phase 0** — 바뀌는 손잡이를 §2.1 다섯 단계로 노출. 값은 한 모듈에 모은다
3. **기본값을 과거 재현으로 둔다** — 러너를 그냥 돌리면 옛 캠페인이 나와야 한다.
   그래야 "이 매개변수화가 아무것도 바꾸지 않았다"를 manifest diff로 증명할 수 있다
4. **σ_seed 사전 등록값을 정하고 적는다** — 캠페인 시작 **전에**. 끝나고 정하면
   데이터를 보고 눈금을 고른 것이 된다
5. 리허설 → Phase 1~4

**대조군 설계가 바뀌면**(split·시드 수·epoch 수) `02`·`02b`의 "대조군 재사용"이 깨진다.
그 경우 세 실험을 전부 다시 돌려야 하고, `report_paired_arms.py`의 짝짓기도 확인한다.
