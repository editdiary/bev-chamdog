# Paper Final Overall Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 확정 설정의 고정 split 5런과 LOSO 7 folds × 5 seeds 35런을 GPU 0에서 실행하고, 재학습 없이 논문 표를 확인할 수 있는 검증된 결과 번들을 만든다.

**Architecture:** Python orchestrator가 40개 런의 완전한 환경변수와 split을 단일 출처로 생성하고 기존 fine-tuning 셸을 호출한다. 기존 scalar/prediction 도구를 재사용하되 LOSO는 fold별 라벨·예측 트리로 분리하고, LOSO 집계는 JSON도 출력한다. 전용 bundle builder가 fixed split과 LOSO를 별도 결과로 보존하고 상위 색인에서만 연결한다.

**Tech Stack:** Python 3.11, PyTorch 2.7.0+cu128, TensorBoard event accumulator, NumPy 1.26.4, Bash, pytest, Fire CLI.

**Spec:** `docs/paper_final_experiments.md`

## Global Constraints

- conda `bev-chamdog`, GPU `CUDA_VISIBLE_DEVICES=0`만 사용한다.
- 결과 루트는 `/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall`이다.
- 주 결과는 epoch 40이고 seeds는 두 프로토콜 모두 `0,1,2,3,4`다.
- LOSO folds는 `raws1,raws2,raws3,rawos1,rawos2,rawos3,rawos4`다.
- 최종 설정은 binary, ResNet-101 stride 8, target-only, native DS 3-camera, Y=4, `(δ,σ)=(0.30,0.10)`, `λ_B=0.5`, `λ_R=0.3`, `(δ_R,β)=(0.15,0.15)`, 40 epochs다.
- `runs/`는 Git에 추가하지 않으며 현재 사용자 작업공간은 건드리지 않는다.

---

### Task 1: 동결된 40런 매트릭스와 재시작 가능한 orchestrator

**Files:**
- Create: `tools/run_paper_final_overall.py`
- Create: `configs/paper_final_overall.sh`
- Create: `tests/tools/test_run_paper_final_overall.py`
- Modify: `docs/paper_final_experiments.md`

**Interfaces:**
- Produces: `RunSpec(protocol, run_name, seed, train_sequences, val_sequences, held_out)`.
- Produces: `build_run_specs(protocol, seeds, folds) -> list[RunSpec]`.
- Produces: `training_environment(spec, output_root, num_epochs, gpu) -> dict[str, str]`.
- Produces: `${OUT_ROOT}/experiment_manifest.json` before GPU execution.

- [x] **Step 1: Write the failing run-matrix tests**

```python
from tools.run_paper_final_overall import build_run_specs, training_environment


def test_all_protocol_builds_five_fixed_and_thirty_five_loso_runs():
    specs = build_run_specs("all", tuple(range(5)), ())
    assert len([s for s in specs if s.protocol == "fixed_split"]) == 5
    assert len([s for s in specs if s.protocol == "loso"]) == 35
    assert len({s.run_name for s in specs}) == 40


def test_each_loso_fold_has_six_train_sequences_and_one_held_out_sequence():
    for spec in build_run_specs("loso", tuple(range(5)), ()):
        assert len(spec.train_sequences) == 6
        assert spec.val_sequences == (spec.held_out,)
        assert spec.held_out not in spec.train_sequences


def test_training_environment_spells_out_the_frozen_configuration(tmp_path):
    spec = build_run_specs("fixed", (0,), ())[0]
    env = training_environment(spec, tmp_path, 40, "0")
    expected = {
        "FORMULATION": "binary", "LOSS": "soft_boundary",
        "DELTA_M": "0.30", "SIGMA_M": "0.10", "LAMBDA_B": "0.5",
        "LAMBDA_R": "0.3", "DELTA_R_M": "0.15", "HUBER_BETA_M": "0.15",
        "HEIGHT_BINS": "4", "HEIGHT_MIN_M": "-0.25", "HEIGHT_MAX_M": "1.75",
        "NUM_EPOCHS": "40", "SAVE_FREQ_EPOCHS": "40",
    }
    assert all(env[k] == v for k, v in expected.items())
```

- [x] **Step 2: Run tests and confirm import failure**

Run: `conda run -n bev-chamdog pytest tests/tools/test_run_paper_final_overall.py -q`

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement matrix, manifest, and queue**

CLI arguments are `--protocol=fixed|loso|all`, `--seeds`, `--folds`, `--num_epochs`, `--gpu`, `--output_root`, and `--plan_only`. Fixed names are `final_s{seed}`; LOSO names are `loso_{held}_s{seed}`. Explicitly pass every spec §3 value plus `AUGMENT=True`, `FLIP_AUGMENT=False`, `FREEZE_ENCODER=False`, `LABEL_SMOOTHING=0.0`, `BAND_KAPPA=1.0`, `LABEL_EPS=0.0`, `DELTA_R_OVER_M=None`, `LR=1e-4`, `WEIGHT_DECAY=1e-7`, and `MAX_CLASS_WEIGHT=20`.

The manifest records Git commit, environment versions, all RunSpecs, exact env, and timestamp. Skip only complete runs. Leave incomplete or failed data untouched, record failures in `queue_status.json`, and continue.

- [x] **Step 4: Add wrapper and verify**

Run:
```bash
conda run -n bev-chamdog pytest tests/tools/test_run_paper_final_overall.py -q
OUT_ROOT=/tmp/paper-final-plan PLAN_ONLY=True bash configs/paper_final_overall.sh
```

Expected: tests PASS; manifest has 40 unique runs and no training directories.

- [x] **Step 5: Update ledger and commit**

```bash
git add tools/run_paper_final_overall.py configs/paper_final_overall.sh tests/tools/test_run_paper_final_overall.py docs/paper_final_experiments.md
git commit -m "feat: add frozen paper overall run matrix"
```

---

### Task 2: 기계 판독 가능한 LOSO 집계

**Files:**
- Modify: `tools/report_loso.py`
- Create: `tests/tools/test_report_loso.py`

**Interfaces:**
- Produces: `summarize_rows(rows: list[dict]) -> dict`.
- Extends: `main(..., json_out=None)` without removing text output.
- JSON keys: `fixed_epoch`, `fold_order`, `folds`, `macro`, `warnings`, `interpretation`.

- [x] **Step 1: Write failing aggregation test**

```python
from tools.report_loso import summarize_rows


def test_loso_summary_averages_fold_means_not_seed_rows():
    rows = [
        {"fold": "raws1", "iou_free": (0.8, 0.01, 5), "fatal_rate": (0.1, 0.01, 5),
         "free_miss_rate": (0.2, 0.01, 5), "baseline": 0.6, "margin": 0.2,
         "se": 0.01, "n_val": 38, "n_seed": 5},
        {"fold": "rawos3", "iou_free": (0.7, 0.02, 5), "fatal_rate": (0.2, 0.02, 5),
         "free_miss_rate": (0.3, 0.02, 5), "baseline": 0.4, "margin": 0.3,
         "se": 0.02, "n_val": 37, "n_seed": 5},
    ]
    summary = summarize_rows(rows)
    assert summary["iou_free"]["mean"] == 0.75
    assert summary["margin"]["mean"] == 0.25
    assert summary["n_folds"] == 2
    assert summary["iou_free"]["mean_within_fold_seed_sd"] == 0.015
```

- [x] **Step 2: Run failing test**

Run: `conda run -n bev-chamdog pytest tests/tools/test_report_loso.py -q`

- [x] **Step 3: Extract aggregation and JSON output**

Expand every metric tuple to `{mean, sd, n}`. Each fold includes factor labels, `extrapolation`, baseline, margin, frame count, and seed values. Macro statistics use fold means and never flatten 35 runs or report `fold_sd/sqrt(7)` as an independent-sample SE.

- [x] **Step 4: Run regressions and commit**

```bash
conda run -n bev-chamdog pytest tests/tools/test_report_loso.py tests/tools/test_summarize_repeats.py -q
git add tools/report_loso.py tests/tools/test_report_loso.py
git commit -m "feat: export structured LOSO summaries"
```

---

### Task 3: Fold별 확률맵 보존과 무결성 검사

**Files:**
- Modify: `tools/verify_val_predictions.py`
- Create: `tests/tools/test_verify_val_predictions.py`
- Create: `configs/paper_final_overall_analysis.sh`

**Interfaces:**
- Extends: `verify_val_predictions.main(..., out_path=None)`.
- Produces: `loso/analysis/predictions/{fold}/labels.npz` and per-run prediction files.
- Produces: `loso/analysis/verify_predictions_{fold}.json`.

- [x] **Step 1: Write a failing custom-output test**

Use tiny matching NPZ fixtures and monkeypatch the event boundary. Assert `main(..., out_path=tmp_path / "custom.json")` writes only the custom path.

- [x] **Step 2: Run and confirm argument failure**

Run: `conda run -n bev-chamdog pytest tests/tools/test_verify_val_predictions.py -q`

- [x] **Step 3: Add backward-compatible output parameter**

```python
out = Path(out_path) if out_path else pred_dir.parent / "verify_predictions.json"
out.parent.mkdir(parents=True, exist_ok=True)
```

Do not change calculations or tolerances.

- [x] **Step 4: Implement analysis driver**

Enforce `bev-chamdog`, log commands with `tee`, and run: scalar export for both protocols; fixed prediction export/verification; seven fold-specific LOSO exports/verifications; fixed repeat summary; LOSO grouped summary; JSON LOSO report; Task 4 builder. Fail closed before bundle creation if any verification has failures.

- [x] **Step 5: Validate and commit**

```bash
conda run -n bev-chamdog pytest tests/tools/test_verify_val_predictions.py -q
bash -n configs/paper_final_overall_analysis.sh
git add tools/verify_val_predictions.py tests/tools/test_verify_val_predictions.py configs/paper_final_overall_analysis.sh
git commit -m "feat: verify paper predictions by LOSO fold"
```

---

### Task 4: Overall 결과 번들

**Files:**
- Create: `tools/build_overall_results_bundle.py`
- Create: `tests/tools/test_build_overall_results_bundle.py`

**Interfaces:**
- Consumes: both scalar/config/manifest sets, all verification JSON, and `report_loso.json`.
- Produces: per-protocol `analysis/RESULTS.json`, CSV, README.
- Produces: top-level `01_overall/analysis/RESULTS.json`, CSV, README without a joint mean.

- [ ] **Step 1: Write failing fixture-based tests**

```python
assert fixed["experiment"]["n_runs"] == 5
assert fixed["metrics_at_fixed_epoch"]["iou_free"]["n"] == 5
assert loso["experiment"]["n_runs"] == 35
assert loso["macro"]["n_folds"] == 7
assert combined["integrity"]["passed"] is True
assert "joint_mean" not in combined
```

- [ ] **Step 2: Run and confirm module absence**

Run: `conda run -n bev-chamdog pytest tests/tools/test_build_overall_results_bundle.py -q`

- [ ] **Step 3: Implement self-describing bundles**

Fixed output contains per-run values plus mean/SD/n at epoch 40 and best diagnostics. LOSO contains all seed values per fold, fold mean/SD, baseline, margin, factors, and macro summary. Include environment, Git commit, common/different config, split counts, integrity maximum delta, `_source`, and missing files. CSV scopes are `run`, `fixed_mean`, `fixed_sd`, `fold_seed`, `fold_mean`, `fold_sd`, and `macro`.

- [ ] **Step 4: Test and commit**

```bash
conda run -n bev-chamdog pytest tests/tools/test_build_overall_results_bundle.py -q
git add tools/build_overall_results_bundle.py tests/tools/test_build_overall_results_bundle.py
git commit -m "feat: build publication-ready overall result bundles"
```

---

### Task 5: Dry-run gates

**Files:**
- Modify: `docs/paper_final_experiments.md`
- Outputs: `/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall_dryrun/`

- [ ] **Step 1: Verify GPU and environment**

Check GPU 0, torch `2.7.0+cu128`, CUDA 12.8, numpy 1.26.4, and RTX PRO 6000.

- [ ] **Step 2: Run fixed seed 0 for one epoch**

Use `PROTOCOL=fixed SEEDS=0 NUM_EPOCHS=1` and the absolute dry-run root.

- [ ] **Step 3: Validate fixed artifacts**

Require frozen config except epoch 1, split counts 192/75, Y=4 metadata, and epoch-1 checkpoint.

- [ ] **Step 4: Run LOSO raws1/raws3 seed 0 for one epoch**

Use `PROTOCOL=loso FOLDS=raws1,raws3 SEEDS=0 NUM_EPOCHS=1`.

- [ ] **Step 5: Validate splits and predictions**

Require six train sequences, one held-out sequence, no overlap, correct frame counts, fold-specific last predictions, and zero verification failures.

- [ ] **Step 6: Run all tests, update ledger, commit**

```bash
conda run -n bev-chamdog pytest -q
git add docs/paper_final_experiments.md
git commit -m "docs: record paper overall dry-run validation"
```

---

### Task 6: 40런 execution, analysis, and ledger update

**Files:**
- Modify: `docs/paper_final_experiments.md`
- Outputs: `/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall/`

- [ ] **Step 1: Generate and audit launch manifest**

Require 40 unique runs, five seeds, seven folds, exact Git commit, and no config differences beyond identity/split.

- [ ] **Step 2: Launch GPU 0 queue**

```bash
CUDA_VISIBLE_DEVICES=0 OUT_ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall PROTOCOL=all SEEDS=0,1,2,3,4 NUM_EPOCHS=40 bash configs/paper_final_overall.sh
```

Monitor `queue_status.json`; preserve and diagnose failures before retrying.

- [ ] **Step 3: Gate analysis on complete artifacts**

Require 5 fixed and 35 LOSO runs at epoch 40, correct split counts, Y=4 metadata, uniform environment/config, and final checkpoints.

- [ ] **Step 4: Run analysis and publication gates**

```bash
CUDA_VISIBLE_DEVICES=0 ROOT=/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall bash configs/paper_final_overall_analysis.sh
```

Require all integrity checks to pass, fixed `n=5`, seven folds each `n_seed=5`, no missing files, and every fold's baseline margin.

- [ ] **Step 5: Update the single campaign ledger**

Record commit, environment, wall time, sizes, integrity maximum delta, fixed mean±SD, LOSO macro/fold SD, fold margins, extrema, and bundle paths. Separate facts from interpretation.

- [ ] **Step 6: Final verification and commit**

```bash
conda run -n bev-chamdog pytest -q
git status --short
git diff --check
git add docs/paper_final_experiments.md configs/paper_final_overall*.sh tools/run_paper_final_overall.py tools/report_loso.py tools/verify_val_predictions.py tools/build_overall_results_bundle.py tests/tools/test_run_paper_final_overall.py tests/tools/test_report_loso.py tests/tools/test_verify_val_predictions.py tests/tools/test_build_overall_results_bundle.py
git commit -m "exp: complete final overall performance evaluation"
```
