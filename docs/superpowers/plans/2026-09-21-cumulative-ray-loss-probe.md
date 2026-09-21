# Cumulative Soft-Target Ray Loss Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in cumulative soft-target ray L1 auxiliary loss, train one seed on GPU 1, and compare it with the existing `C_soft_s0` and `D_range_s0` results.

**Architecture:** Preserve the existing `arc_huber` implementation as the default. The new `cumulative_l1` mode builds exactly the same full-field target used by `L_cell`, gathers prediction and target along each ray, compares their cumulative radial integrals at every in-grid step with L1, and averages first within each ray and then over `RAY_OK` rays. A new one-run script writes only under `runs/cumulative_ray_loss/`; existing `runs/loss_effect/` artifacts remain read-only comparison inputs.

**Tech Stack:** Python 3.11, PyTorch 2.7.0+cu128, pytest, Fire CLI, Bash, TensorBoard event logs.

**Spec:** User-approved in-chat design on 2026-09-21; the existing baseline contract is `docs/loss_function_spec.md` §8.

## Global Constraints

- Use conda environment `bev-chamdog`; training must refuse other environments through the existing guard.
- Use GPU 1 only; GPU 0 is reserved for the running paper-final queue.
- Use seed 0 only and the exact final fixed-split config from `runs/loss_effect/` (`Y=4`, `delta=0.30`, `sigma=0.10`, `lambda_B=0.5`, res101, 40 epochs, batch 8, augmentation on).
- Do not modify `third_party/` or `mmdetection3d/`.
- Do not use a worktree; `AGENTS.md` explicitly prohibits it for this repository.
- Existing `arc_huber` behavior and defaults must remain bit-for-bit unchanged unless `range_loss_mode=cumulative_l1` is explicitly selected.
- Do not merge or push. Commit messages must be English.
- Store the new run under `runs/cumulative_ray_loss/`; never overwrite `runs/loss_effect/`.

## Review Focus

- A prediction equal to the composite soft target on supervised cells must produce exactly zero cumulative loss and zero gradient.
- A local overprediction followed by an equal underprediction must still incur loss before the cancellation point, proving the endpoint-cancellation hole is closed.
- Invalid and out-of-grid samples must contribute neither to cumulative sums nor to the profile-averaging denominator.
- `RAY_NO_FREE` and `RAY_CENSORED` rays must remain excluded by the existing hard-GT `RAY_OK` contract.
- Omitting the new mode flag must execute the old arc-Huber path with unchanged outputs and banner/config semantics.

---

### Task 1: Composite soft target and cumulative ray loss

**Files:**
- Modify: `projects/common/soft_boundary.py`
- Modify: `projects/common/range_loss.py`
- Modify: `tests/common/test_soft_boundary.py`
- Modify: `tests/common/test_range_loss.py`

**Interfaces:**
- Produces: `build_soft_boundary_target(d, valid, permanent_blind, delta, kind, sigma, alpha, kappa, eps) -> Tensor[B,1,H,W]`.
- Produces: `compute_cumulative_range_loss(prob_free, target_free, free_gt, valid, gather) -> tuple[Tensor, dict[str, Tensor]]`.
- Preserves: `compute_range_loss(...)` without behavior changes.

- [ ] **Step 1: Write the failing target-field test**

Add a test using a hand-constructed signed-distance tensor that asserts the returned target is `1-eps` in `omega_f`, `eps` in `omega_n` (including permanent blind), and the literal Gaussian soft-target values after `kappa` and `eps` in `omega_b`. With the final config's `eps=0`, these reduce to hard `1/0` outside the band. The production change that makes this test fail is any divergence between the target consumed by `L_cell` and the target consumed by the cumulative ray term.

- [ ] **Step 2: Run the target-field test and verify RED**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_soft_boundary.py -k composite_target
```

Expected: FAIL because `build_soft_boundary_target` does not exist.

- [ ] **Step 3: Implement and reuse the composite target builder**

Implement the builder from the existing `region_masks`, `soft_target`, `kappa`, and `eps` logic. Refactor `compute_soft_boundary_loss` to consume that builder so the cell and ray targets cannot silently diverge. Keep all existing loss values unchanged.

- [ ] **Step 4: Run the target-field test and existing soft-boundary tests**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_soft_boundary.py tests/common/test_soft_boundary_kappa.py
```

Expected: PASS.

- [ ] **Step 5: Write failing cumulative-loss contract tests**

Add real-tensor tests for these literal behaviors:

```python
loss, parts = compute_cumulative_range_loss(target, target, free_gt, valid, gather)
assert loss.item() == pytest.approx(0.0, abs=1e-9)

# Equal positive and negative residual masses have zero final arc error,
# but a non-zero cumulative profile between their positions.
assert parts["range_arc_mae"].item() == pytest.approx(0.0, abs=1e-6)
assert parts["range_cumulative_mae"].item() > 0.0
assert loss.item() == pytest.approx(parts["range_cumulative_mae"].item())
```

Also assert that changing prediction only where `valid=0` leaves the loss unchanged, and that the gradient is finite and non-zero on the two mismatched valid samples.

- [ ] **Step 6: Run cumulative tests and verify RED**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_range_loss.py -k cumulative
```

Expected: FAIL because `compute_cumulative_range_loss` does not exist.

- [ ] **Step 7: Implement the cumulative profile L1**

Use the supervision mask both in the integral and in the profile average:

```python
profile_mask = valid_ray * inside
residual = (gather(prob_free) - gather(target_free)) * profile_mask
cumulative_error = residual.cumsum(dim=-1) * gather.step_m
per_ray = (cumulative_error.abs() * profile_mask).sum(-1) / (profile_mask.sum(-1) + 1e-6)
loss = (per_ray * ok_f).sum() / (ok_f.sum() + 1e-6)
```

Compute `ok` from gathered hard `free_gt`, not from the soft target. Return the existing hard-GT endpoint diagnostics `range_arc_mae`, `range_arc_bias`, and `frac_rays_used` for direct comparability with old runs, plus `range_cumulative_mae`. The new loss itself is the profile error against the composite soft target; endpoint diagnostics do not define or alter its gradient.

- [ ] **Step 8: Run range-loss tests and verify GREEN**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_range_loss.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add projects/common/soft_boundary.py projects/common/range_loss.py tests/common/test_soft_boundary.py tests/common/test_range_loss.py
git commit -m "feat: add cumulative soft-target ray loss"
```

### Task 2: Opt-in training mode and regression-safe configuration

**Files:**
- Modify: `projects/common/binary_metrics.py`
- Modify: `tools/train_robot_bev.py`
- Modify: `configs/train_robot_bev_finetune.sh`
- Modify: `tools/measure_range_gradient.py`
- Test: `tests/common/test_range_loss.py`
- Test: `tests/tools/test_checkpoint_selection_integration.py`

**Interfaces:**
- Consumes: `build_soft_boundary_target(...)` and `compute_cumulative_range_loss(...)` from Task 1.
- Produces: `range_loss_mode: Literal["arc_huber", "cumulative_l1"]`, default `arc_huber`, threaded from shell to trainer to batch loss.
- Produces: gradient measurement for either mode using the same logits and BCE reference gradient.

- [ ] **Step 1: Write failing mode-selection tests**

Add one batch-level test proving `cumulative_l1` calls the cumulative implementation with the composite target and exposes `range_cumulative_mae`; add a default-mode regression assertion proving omitted mode returns the existing arc-Huber keys and values. Add validation coverage for an unknown mode raising `ValueError` before training.

- [ ] **Step 2: Run mode tests and verify RED**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_range_loss.py tests/tools/test_checkpoint_selection_integration.py -k 'range_loss_mode or cumulative'
```

Expected: FAIL because the mode argument is absent.

- [ ] **Step 3: Wire the mode through the training path**

Add `range_loss_mode="arc_huber"` to `train_robot_bev.main` and `run_batch_soft_boundary`. For `arc_huber`, execute the current function unchanged. For `cumulative_l1`, build the composite target from `d_bev_g`, `valid_bev_g`, and `permanent_blind`, then execute `compute_cumulative_range_loss`. Add `RANGE_LOSS_MODE=${RANGE_LOSS_MODE:-arc_huber}` to `configs/train_robot_bev_finetune.sh` and pass `--range_loss_mode`. Print `mode=cumulative_l1 | soft-target cumulative arc L1` without printing dead-zone/Huber parameters for that mode.

- [ ] **Step 4: Extend gradient calibration**

Add `range_loss_mode`, current final soft-target parameters (`delta_m=0.30`, `sigma_m=0.10`, `sigma_alpha=None`, `kappa=1`, `eps=0`), and the composite target to `tools/measure_range_gradient.py`. Its output must name the selected mode and print the measured `G_R/G_B` at `lambda_R=1` plus `lambda_R = target_ratio / measured_ratio`.

- [ ] **Step 5: Run targeted and trainer integration tests**

Run:

```bash
conda run -n bev-chamdog pytest -q tests/common/test_range_loss.py tests/common/test_soft_boundary.py tests/common/test_soft_boundary_kappa.py tests/tools/test_checkpoint_selection_integration.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add projects/common/binary_metrics.py tools/train_robot_bev.py configs/train_robot_bev_finetune.sh tools/measure_range_gradient.py tests/common/test_range_loss.py tests/tools/test_checkpoint_selection_integration.py
git commit -m "feat: expose cumulative ray loss mode"
```

### Task 3: One-seed experiment, verification, and comparison

**Files:**
- Create: `configs/probe_cumulative_ray_loss.sh`
- Create after measurement: `runs/cumulative_ray_loss/README.md` (ignored experimental record, not committed)
- Read only: `runs/loss_effect/analysis/RESULTS.json`
- Read only: `runs/loss_effect/logs/{C_soft_s0,D_range_s0}`

**Interfaces:**
- Consumes: `RANGE_LOSS_MODE=cumulative_l1` and the calibrated `LAMBDA_R`.
- Produces: `runs/cumulative_ray_loss/{logs,ckpt}/E_cumulative_s0` and a comparison record against the two existing seed-0 controls.

- [ ] **Step 1: Write the experiment script**

The script must set `CUDA_VISIBLE_DEVICES=1`, `SEED=0`, `NUM_EPOCHS=40`, `RANGE_LOSS_MODE=cumulative_l1`, and every other knob exactly as `configs/loss_effect.sh` uses for `D_range`. It must use `OUT_ROOT=${OUT_ROOT:-runs/cumulative_ray_loss}`, refuse to overwrite a non-empty run directory, and accept `LAMBDA_R` from the environment so the calibrated value is explicit in the run config.

- [ ] **Step 2: Validate the script without training**

Run `bash -n configs/probe_cumulative_ray_loss.sh` and a one-batch CPU/unit integration path; inspect the emitted config/banner to confirm GPU, seed, mode, and output root.

- [ ] **Step 3: Measure gradient scale on the existing `C_soft_s0` checkpoint**

Run the gradient tool on GPU 1 for both `arc_huber` and `cumulative_l1` using `runs/loss_effect/ckpt/C_soft_s0/model-000000040.pth` and identical batches. Set the probe's `LAMBDA_R` so the cumulative mode matches the old `lambda_R=0.3` arc-Huber gradient contribution at that checkpoint:

```text
lambda_new = 0.3 * ratio_arc_huber / ratio_cumulative_l1
```

Record both ratios and `lambda_new` in `runs/cumulative_ray_loss/README.md`. This is a probe calibration, not a new tuned hyperparameter sweep.

- [ ] **Step 4: Run the complete test suite before training**

Run:

```bash
conda run -n bev-chamdog pytest -q
```

Expected: all collected tests PASS. Report every pre-existing failure by name if the baseline is not green.

- [ ] **Step 5: Run the one-seed GPU experiment**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 LAMBDA_R=<calibrated> bash configs/probe_cumulative_ray_loss.sh
```

Expected: `E_cumulative_s0` reaches epoch 40, writes `model_best-*.pth` and `model-000000040.pth`, and logs `range_cumulative_mae` for train and validation.

- [ ] **Step 6: Compare against existing seed-0 controls**

Extract selected-checkpoint and epoch-40 values for `iou_free`, `f1@10cm`, `fatal_rate`, `free_miss_rate`, loss rebound, selected epoch, `range_arc_mae`, and range contribution. Compare `E_cumulative_s0` with `C_soft_s0` and `D_range_s0`. State explicitly that one seed cannot establish significance and judge only whether the result warrants an n=5 follow-up.

- [ ] **Step 7: Commit the experiment entry point and plan**

```bash
git add configs/probe_cumulative_ray_loss.sh docs/superpowers/plans/2026-09-21-cumulative-ray-loss-probe.md
git commit -m "exp: add cumulative ray loss probe"
```

- [ ] **Step 8: Verify branch state and artifacts**

Run:

```bash
git status --short --branch
find runs/cumulative_ray_loss -maxdepth 3 -type f | sort
```

Expected: only ignored run artifacts are outside Git; the branch has no uncommitted tracked changes.
