# SynWoodScape Two-Head Training Improvement Plan

## Current Baseline

- Model: `TwoHeadSegnet` with occupancy and visibility heads.
- Dataset: `dataset/synwoodscape_2head_roi_8_4_6_h08`.
- ROI: front 8 m, rear 4 m, left/right 6 m, 0.05 m/cell.
- Current reference run: `twohead_pretrain_vis_fixed_..._260814_140932`, best epoch 47,
  val obstacle IoU 0.854, train 0.958.
- Best selection metric: `(val_iou_drivable + val_iou_obstacle) / 2`.
- Run-to-run noise floor: about 0.002 on val obstacle IoU.
- Observed issue: obstacle IoU is weak on samples with small obstacle regions
  (`tiny 0.566` / `small 0.637` vs `large 0.948`), misses outnumber false alarms 7:1, and there
  is a 0.10 train/val gap with no augmentation in the pipeline.

Metric conventions changed on 2026-08-14 (commits 45bec64, d0e089a). Numbers from runs before
that are not comparable — see `docs/next_session_synwoodscape_twohead.md`.

## Improvement Principles

1. Measure before changing loss or architecture.
2. Separate true model errors from metric sensitivity caused by tiny obstacle regions.
3. Keep visibility as an auxiliary task unless visual inspection shows visibility-driven artifacts.
4. Prefer small, controlled experiments; require a change larger than the noise floor.
5. Prefer changes that transfer to greenhouse fine-tuning over changes that only fit
   SynWoodScape's class ratio.

## Step 1: More Diagnostic Occupancy Metrics

Status: **done.** Implemented, run on a full baseline, and interpreted. The diagnostics did their
job immediately: the `empty` bin exposed that obstacle-free samples were being scored 0, and the
flat-zero visibility errors exposed the `valid`/`vis` conflation. Both were fixed; see
`docs/next_session_synwoodscape_twohead.md`.

Note that `occupancy_obstacle_iou_empty_epoch` no longer exists — that bin reports
`occupancy_empty_false_alarm_epoch` instead, because IoU on an obstacle-free sample is
structurally 0 and carries no information.

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

Status: proposed. Deprioritized below Step 4 — see "Priority after 2026-08-14" at the end.

Candidate:

- Keep current BCEWithLogits occupancy loss.
- Add Dice loss on occupancy inside `vis * valid`.
- `lambda_vis=0.2` is now a live parameter but is expected to matter little, since visibility is
  already accurate and does not compete with occupancy (see the A/B in the handoff doc).

**Make the Dice term class-symmetric, not obstacle-only.**

Pretraining has `obst_frac 0.125`, so obstacle is the minority class and an obstacle-only Dice
term helps it. The greenhouse fine-tuning target inverts this: obstacle-heavy, with drivable and
visible regions smaller. After that flip, an obstacle-only Dice term would be attached to the
easy majority class, where per-sample Dice saturates near 1 and contributes almost no gradient,
while the class actually needing help — drivable — gets nothing.

BCE does not have this failure mode because `compute_pos_weight` measures `neg/pos` on the train
split at runtime and re-balances automatically. A single-class Dice term does not.

Averaging drivable Dice and obstacle Dice makes whichever class is currently the minority the
dominant term, so the same loss definition keeps working across the distribution flip.

Success criteria:

- `missed_obstacle` decreases without a large increase in `false_obstacle`.
- `tiny` and `small` bin obstacle IoU improve by more than the 0.002 noise floor.
- `deploy_iou_obstacle` improves without `deploy_visible_coverage` dropping.
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

## Step 4: Augmentation

Status: proposed, and now the highest-value remaining step.

There is currently **no augmentation of any kind**. The dataset only resizes to 512x384 bilinear
and normalizes. With 400 training samples that leaves train obstacle IoU 0.958 against val 0.854.

### Safe: photometric

- brightness/contrast/saturation jitter
- gamma/exposure variation
- slight blur/noise

Geometry is untouched, so there is no calibration risk. This is also the augmentation that
targets the synthetic → real domain shift, which is what actually has to transfer to the
greenhouse dataset.

### Safe: BEV-space left-right flip

Simple-BEV's `Segnet.forward(x, bev_flip_indices=...)` flips the BEV feature map inside the
decoder (`third_party/models/simple_bev/nets/segnet.py:137-139`). The repo currently never
passes it. Because the flip happens *after* projection, calibration is not involved at all.
Flip `seg_bev_g`, `vis_bev_g` and `valid_bev_g` to match.

Two caveats:

- **Left-right only.** A mirrored feature map corresponds to a mirrored world, which is
  in-distribution only if the camera rig is symmetric about that axis. FV/MVL/MVR/RV is roughly
  left-right symmetric but not front-back, and the ROI is asymmetric (front 8 m, rear 4 m).
- It augments the decoder only — the encoder and the projection still see original data.

### Avoid: image-space geometric

Flip/crop/rotate on the camera image breaks the projection unless `pix_T_cams` is updated to
match. For fisheye `radial_poly` a horizontal flip is not even expressible as a simple intrinsic
change unless the distortion center sits exactly at the image center, so this is especially
risky here.

Success criteria:

- train/val obstacle IoU gap narrows from the current 0.10.
- val obstacle IoU does not regress beyond the 0.002 noise floor.
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

Note: samples with no obstacle in GT must be excluded from a "lowest obstacle IoU" ranking, or
they will fill the entire list. `tools/visualize_predictions.py` already labels them
`n/a (GT obstacle 없음)` rather than printing a misleading 0.000.

## Priority after 2026-08-14

Ordered by expected value for the greenhouse fine-tuning target, not for the SynWoodScape score:

1. **Step 4 augmentation** — photometric first, then BEV left-right flip. Zero calibration risk,
   directly attacks the 0.10 train/val gap, and encoder robustness is what actually transfers.
2. **Step 2 class-symmetric Dice** — targets the `tiny`/`small` bins and `missed_obstacle`.
3. **Step 5 worst-case visualization** — cheap, and makes every later experiment easier to judge.
4. **Step 3 checkpoint score reweighting** — leave until the loss and augmentation are settled;
   changing the selection rule while the loss is also changing makes runs hard to compare.

Rationale for putting augmentation ahead of loss shaping: the point of pretraining is to learn
geometry (image → BEV projection, where surfaces are), not the class prior of a dataset that
will be abandoned at fine-tuning. Tuning the pretrain loss to SynWoodScape's obstacle ratio is
work that fine-tuning then has to undo, whereas a more robust encoder carries over directly.
