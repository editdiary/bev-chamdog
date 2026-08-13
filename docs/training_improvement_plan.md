# SynWoodScape Two-Head Training Improvement Plan

## Current Baseline

- Model: `TwoHeadSegnet` with occupancy and visibility heads.
- Dataset: `dataset/synwoodscape_2head_roi_8_4_6_h08`.
- ROI: front 8 m, rear 4 m, left/right 6 m, 0.05 m/cell.
- Best checkpoint from the first long run: epoch 41.
- Best selection metric: `(val_iou_drivable + val_iou_obstacle) / 2`.
- Observed issue: overall validation is reasonable, but obstacle IoU can drop sharply on samples where obstacle occupies only a small fraction of valid BEV cells.

## Improvement Principles

1. Measure before changing loss or architecture.
2. Separate true model errors from metric sensitivity caused by tiny obstacle regions.
3. Keep visibility as an auxiliary task unless visual inspection shows visibility-driven artifacts.
4. Prefer small, controlled experiments that can be compared with the current epoch-41 baseline.

## Step 1: More Diagnostic Occupancy Metrics

Status: implemented in `tools/train_synwoodscape.py`; next action is to run a full diagnostic baseline and interpret the new metrics.

Add metrics that explain why obstacle IoU changes:

- `obstacle_frac`: fraction of valid BEV cells that are non-drivable in GT.
- `false_obstacle`: fraction of valid drivable GT cells predicted as obstacle.
- `missed_obstacle`: fraction of valid obstacle GT cells predicted as drivable.
- obstacle IoU by GT obstacle fraction bins:
  - `empty`: `0.00 <= obstacle_frac <= 0.00`
  - `tiny`: `0.00 < obstacle_frac <= 0.01`
  - `small`: `0.01 < obstacle_frac <= 0.05`
  - `medium`: `0.05 < obstacle_frac <= 0.15`
  - `large`: `0.15 < obstacle_frac <= 1.00`

Expected benefit:

- Low obstacle IoU samples become interpretable: no-obstacle samples, tiny-obstacle samples, and genuinely hard obstacle-rich samples are separated.
- Future loss experiments can be judged by `missed_obstacle` and bin-wise obstacle IoU, not only global mean IoU.

Implemented TensorBoard keys include `occupancy_obstacle_fraction_epoch`, `occupancy_false_obstacle_epoch`, `occupancy_missed_obstacle_epoch`, and `occupancy_obstacle_iou_{empty,tiny,small,medium,large}_epoch` under both `train/` and `val/`.

## Step 2: Occupancy Loss Experiment

Status: proposed after Step 1 is logged.

Candidate:

- Keep current BCEWithLogits occupancy loss.
- Add Dice loss on obstacle/drivable occupancy inside `valid_bev_g`.
- First run with `lambda_vis=0.2` so visibility does not dominate once it has become easy.

Success criteria:

- `missed_obstacle` decreases without a large increase in `false_obstacle`.
- `small` and `medium` bin obstacle IoU improve over the current epoch-41 baseline.
- Visualized `pred occ x vis` does not become overly conservative.

## Step 3: Best Checkpoint Score Experiment

Status: proposed after Step 1 and Step 2.

Candidate best score:

```text
score = 0.3 * iou_drivable + 0.7 * iou_obstacle
```

Rationale:

- The current 0.5/0.5 score is reasonable for general pretraining.
- The target downstream task cares strongly about stable non-drivable detection.
- A 0.3/0.7 score emphasizes obstacle quality without completely ignoring drivable quality.

## Step 4: Photometric Augmentation

Status: proposed after metric/loss baselines are stable.

Allowed augmentations:

- brightness/contrast/saturation jitter
- gamma/exposure variation
- slight blur/noise

Avoid for now:

- crop/resize/rotation/flip that changes camera geometry without updating calibration.

Success criteria:

- Similar or better SynWoodScape validation metrics.
- Better visual robustness before fine-tuning on the custom fisheye dataset.

## Step 5: Worst-Case Visualization Workflow

Status: proposed after Step 1 metrics are available.

Add a visualization mode that selects validation samples by:

- lowest obstacle IoU
- highest `missed_obstacle`
- highest `false_obstacle`
- selected obstacle fraction bin

Expected benefit:

- Faster qualitative review of the exact failure modes that the new metrics reveal.
