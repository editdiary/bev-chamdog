# Next Session Handoff: SynWoodScape Two-Head Pretraining

Last updated: 2026-08-14

## One-Line State

SynWoodScape ROI 8/4/±6 m two-head pretraining runs end to end; three metric defects were found
and fixed on 2026-08-14, so occupancy numbers finally mean what they claim, and the remaining
open work is loss/augmentation experiments aimed at small obstacles and the train/val gap.

## Current Branch and Git State

```bash
git branch --show-current
# feat/synwoodscape-two-head-training
```

Recent commits:

```text
d0e089a Separate valid mask from visibility, add deployment metrics
45bec64 Exclude obstacle-free samples from occupancy IoU
ec5484e Document two-head training handoff
0294373 Add occupancy diagnostic training metrics
539b918 Add two-head prediction visualization
```

Do not merge or push without explicit user action.

## What Was Fixed on 2026-08-14 (read this before comparing any runs)

Three defects, all in metrics rather than in the model. None of them changed the occupancy
training signal, so earlier checkpoints are still usable — but earlier *numbers* are not
comparable with current ones.

### 1. Obstacle-free samples scored 0 in obstacle IoU (commit 45bec64)

`compute_iou` returns `intersection / (1e-4 + union)`. A sample whose GT has no obstacle cell
has `intersection = 0` always, so it scored 0 even when the model correctly predicted no
obstacle. That is 14 of 100 val samples and 81 of 400 train samples.

Proof it was an artifact: train `missed_obstacle` was 0.0003 and `false_obstacle` 0.0014 —
essentially perfect — while the empty bin still read 0.000.

Effect of the fix (same model, same weights, only measurement changed):

| | reported before | actual |
|---|---|---|
| train obstacle IoU | 0.777 | 0.969 |
| val obstacle IoU | 0.749 | 0.854 |

This also exposed a real 0.115 train/val gap that the artifact had been masking.

Obstacle IoU is now averaged over samples that contain obstacles. Obstacle-free samples are
reported as `empty_false_alarm` instead — the fraction of valid cells wrongly predicted as
obstacle on scenes that have none.

### 2. `valid_bev_g` was the visibility mask, which killed the visibility head (commit d0e089a)

The dataset returned the same array for `vis_bev_g` and `valid_bev_g`. `compute_two_head_loss`
weights the visibility term by `asymmetric_weight * valid`, so every invisible cell had weight
zero: the visibility head saw only positives and "predict visible everywhere" was the exact
global optimum.

Verified on real data:

```text
vis_neg_weight=   0.0 -> loss_vis=0.805862
vis_neg_weight=   3.0 -> loss_vis=0.805862     # parameter had no effect whatsoever
vis_neg_weight= 100.0 -> loss_vis=0.805862
always-visible logit -> loss_vis=0.00000000    # trivial solution is exactly optimal
(not gt_visible) and valid cell count: 0       # false_high denominator was empty
```

So `vis_neg_weight` was dead, `vis_false_high` was structurally 0.000 regardless of prediction,
and the near-zero visibility errors reported in the first three long runs mean nothing.

The mask contract is now:

| mask | meaning | used by |
|---|---|---|
| `valid_bev_g` | the cell has a label (ROI / annotatable range); all ones for SynWoodScape | outer bound of every loss and metric |
| `vis_bev_g` | the cell is actually observable | restricts where occupancy is judged |

Occupancy loss and occupancy metrics use `vis * valid`; visibility loss and metrics use `valid`.
`projects/models/simplebev_two_head.py` was already written for this contract and did not
change — only the dataset was violating it.

### 3. No deployment-condition metric existed (commit d0e089a)

`iou_obstacle` masks by GT visibility, which measures the occupancy head in isolation. A robot
has no GT and can only trust the region the model itself claims to see. `deploy_iou_*` evaluates
occupancy inside **predicted**-visible cells against full GT occupancy, so hallucinated
visibility is punished instead of masked away. Always read it together with
`deploy_visible_coverage` — a model that declares a small visible region gets an easy score.

## Run Inventory

| run | metric convention | best epoch | val_iou_mean | val obstacle IoU |
|---|---|---:|---:|---:|
| `twohead_pretrain_res101_..._260813_224942` | old | 41 | 0.870 | 0.751 |
| `twohead_pretrain_diag_baseline_..._260814_130110` | old | 52 | 0.869 | 0.749 |
| `twohead_pretrain_fixed_iou_..._260814_134525` | fix 1 only | 53 | 0.921 | 0.854 |
| `twohead_pretrain_vis_fixed_..._260814_140932` | fix 1+2+3 | 47 | 0.921 | 0.854 |
| `twohead_pretrain_photo_aug_..._260814_150556` | fix 1+2+3, `--augment=True` | 46 | 0.925 | 0.861 |

The first two are on the old convention and must not be compared numerically against the last
two. The jump from 0.749 to 0.854 is the measurement fix, not a model improvement.

**Run-to-run noise floor: about 0.002 on val obstacle IoU**, established from the first two runs
which had identical configuration. Any experiment must move the metric by more than that to
count as signal.

## Current Numbers (`photo_aug`, best epoch 46 — current best run)

```text
val_iou_drivable                 0.9889
val_iou_obstacle                 0.8607
train_iou_obstacle               0.9556
missed_obstacle                  0.0380
false_obstacle                   0.0056
empty_false_alarm                0.0001
vis_false_high                   0.0238
vis_false_low                    0.0029
deploy_iou_drivable              0.9885
deploy_iou_obstacle              0.8510
deploy_visible_coverage          0.9369
obstacle IoU by bin: tiny 0.594/n7  small 0.646/n10  medium 0.845/n20  large 0.949/n49
```

Best checkpoint:

```text
runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth
```

`vis_fixed` (best epoch 47) is the no-augmentation control for this run: val obstacle IoU
0.8544, train 0.9575, `tiny` 0.5661, `missed_obstacle` 0.0402, `deploy_iou_obstacle` 0.8540.

### A/B result: reviving the visibility head costs occupancy nothing

`fixed_iou` (visibility head degenerate) vs `vis_fixed` (visibility head actually trained), with
identical occupancy settings: val obstacle IoU 0.8539 vs 0.8544, inside the noise floor. The
train/val gap even narrowed slightly (0.116 → 0.103), so the revived visibility task acts as a
mild regularizer. There is no reason to drop the two-head structure.

### Deployment performance matches development performance

On `vis_fixed`, `deploy_iou_obstacle` 0.8540 vs GT-masked 0.8544, at coverage 0.9370 against a
true visible fraction of about 0.946. Predicted visibility is accurate enough that masking by
the model's own estimate loses nothing. This is the direct answer to "does this design serve the
actual goal" — and it does.

On `photo_aug` the two diverge slightly: GT-masked 0.8607 but deploy 0.8510 at the same
coverage. Since coverage and `vis_false_high` are unchanged, the difference sits in cells the
model wrongly believes it can see. The magnitude is small and the deploy metric's own noise
floor is not yet measured, so do not read anything into it before a repeat seed.

### A/B result: photometric augmentation helps the weakest bin

`vis_fixed` vs `photo_aug`, identical apart from `--augment=True`: val obstacle IoU 0.8544 →
0.8607, about 3x the noise floor, and the gain is concentrated in `tiny` (0.5661 → 0.5935).
`missed_obstacle` improved without `false_obstacle` worsening, so the model did not just become
more conservative. But the train/val gap moved only 0.103 → 0.095 — appearance variation is not
what the model was memorizing, so more of the same augmentation is unlikely to close it.

## Remaining Weaknesses

Numbers below are from `photo_aug`, the current best run.

1. **Small obstacles.** `tiny 0.594` / `small 0.646` vs `large 0.949`. Improved by augmentation
   but still the dominant weakness.
2. **Missed obstacles dominate.** `missed_obstacle` 0.038 vs `false_obstacle` 0.006, roughly 6x.
   This is the safety-critical direction.
3. **Overfitting persists.** train 0.956 / val 0.861. Photometric augmentation moved the gap
   only 0.103 → 0.095, so the remaining gap is not driven by appearance variation — most likely
   scene diversity at 400 training samples.

## Augmentation Status

Photometric augmentation is implemented in `projects/datasets/photometric.py`, enabled with
`--augment=True`, and applied to the **train split only**. Before this, the pipeline had no
augmentation at all.

Simple-BEV itself has no photometric augmentation to borrow. Its `data_aug_conf` only does
random resize + crop — with the intrinsics updated to match (`nuscenesdataset.py:773-784`,
a good illustration of why image-space geometric augmentation needs calibration bookkeeping) —
plus camera shuffling and camera dropout. `albumentations` and `kornia` are not installed;
`torchvision` already is, via the ResNet encoder, so no new dependency was added.

`TwoHeadSegnet` inherits Simple-BEV's `Segnet.forward(x, bev_flip_indices=None)` but the repo
never passes flip indices — and neither does any upstream Simple-BEV training script, so that
hook is untested in the reference implementation too.

### Which augmentations are safe here

| class | example | calibration risk | verdict |
|---|---|---|---|
| image-space geometric | flip/crop/rotate the camera image | **breaks projection** — `pix_T_cams` no longer matches the pixels, and for fisheye `radial_poly` a horizontal flip is not expressible as a simple intrinsic change | avoid |
| photometric | brightness, contrast, gamma, noise, blur | none — geometry untouched | safest and highest value, since SynWoodScape is synthetic and the greenhouse target is not |
| BEV-space geometric | flip the BEV feature map + all BEV GT together | none — the flip happens after projection | usable, see caveat |

BEV flip caveat: only **left-right** is valid. A mirrored feature map corresponds to a mirrored
world, which is in-distribution only if the camera rig is symmetric about that axis. FV/MVL/MVR/RV
is roughly left-right symmetric but not front-back, and the ROI itself is asymmetric (front 8 m,
rear 4 m). It also augments the decoder only, since the encoder and projection see original data.

## Dice Loss and the Fine-Tuning Distribution Flip

Pretraining has `obst_frac 0.125` — obstacle is the minority class. The greenhouse fine-tuning
target is the opposite: obstacle-heavy, with drivable and visible regions smaller.

This matters for how Dice is written:

- Dice's benefit comes from the imbalance itself. An obstacle-only Dice term helps the minority
  class now, but after the flip it would be attached to the **easy majority** class, where
  per-sample Dice saturates near 1 and the gradient vanishes.
- The class that would then need help is **drivable**.
- BCE does not have this problem: `compute_pos_weight` measures `neg/pos` on the train split at
  runtime, so it re-balances automatically when the distribution changes. A single-class Dice
  term does not.

**Therefore: if Dice is added, make it class-symmetric** (mean of drivable Dice and obstacle
Dice), so whichever class is the minority automatically dominates the term. That is the version
that survives pretrain → fine-tune transfer.

Broader point worth weighing before spending runs here: the job of pretraining is to learn
geometry (image → BEV projection, where surfaces are), not the class prior of a dataset that
will be abandoned. Over-tuning the pretrain loss to SynWoodScape's ratio is work that
fine-tuning has to undo. What actually transfers is encoder robustness — which argues for
augmentation ahead of loss shaping.

## Suggested Next Steps

Ordered by expected value for the real downstream task:

1. **Class-symmetric Dice** on occupancy, judged by `missed_obstacle` and the `tiny`/`small`
   bins rather than by global IoU. Now the top remaining lever, since small obstacles are still
   the dominant weakness.
2. **Repeat the `photo_aug` config with a different seed.** Two things need confirming: that the
   +0.006 obstacle IoU gain reproduces, and whether the -0.003 `deploy_iou_obstacle` move is
   noise. The deploy metric has only one prior run, so its noise floor is unmeasured.
3. **BEV left-right flip** using the existing `bev_flip_indices` hook, flipping
   `seg_bev_g`/`vis_bev_g`/`valid_bev_g` to match. Lower confidence than it looked: upstream
   never exercises the hook, and it augments the decoder only.
4. `lambda_vis` 0.5 → 0.2. Only now a live parameter, but visibility is already accurate and is
   not competing with occupancy, so expect little.

Note on the overfitting gap: photometric augmentation barely moved it, which is evidence that
appearance is not what the model is memorizing. Stronger augmentation of the same kind is
unlikely to help much; more scenes would.

## Commands

Training (edit `--exp_name` per experiment):

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n bev-chamdog \
  bash configs/train_synwoodscape_twohead_pretrain.sh 2>&1 | tee runs/synwoodscape_twohead/train_<name>.log
```

`conda run` buffers stdout unless `--no-capture-output` is passed, so without it the log file
stays empty until the run ends. A 60-epoch run takes about 24 minutes on GPU 1.

The script currently sets `--exp_name=twohead_pretrain_photo_aug` and `--augment=True`. Set
`--augment=False` to reproduce the no-augmentation control.

Visualization:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n bev-chamdog python tools/visualize_predictions.py \
  --ckpt_dir=runs/synwoodscape_twohead/ckpt/<RUN_NAME> \
  --model_name=model_best --step=<BEST_EPOCH> --num_samples=20 \
  --encoder_type=res101 --use_fisheye=True \
  --output_dir=runs/synwoodscape_twohead/viz_<RUN_NAME>_best
```

Latest preview: `runs/synwoodscape_twohead/viz_vis_fixed_best_epoch47/`

TensorBoard:

```bash
conda run -n bev-chamdog tensorboard --logdir runs/synwoodscape_twohead/logs --port 6006
```

Verification:

```bash
git status --short && git log --oneline -8
conda run -n bev-chamdog pytest -q          # 118 tests
```

GPU policy: use GPU 1. GPU 0 is often occupied by other work.

## Key Files

- `configs/train_synwoodscape_twohead_pretrain.sh`: main pretrain command.
- `tools/train_synwoodscape.py`: training loop, losses, logging, occupancy/deployment metrics.
- `tools/visualize_predictions.py`: checkpoint inference visualization.
- `projects/models/simplebev_two_head.py`: `TwoHeadSegnet`, two-head loss, visibility metrics.
- `projects/datasets/synwoodscape_simplebev.py`: mask contract for `seg`/`vis`/`valid`.
- `projects/bev_gt/grid.py`: ROI/grid spec.
- `docs/training_guide.md`: metric definitions and how to read them.
- `docs/training_improvement_plan.md`: ordered improvement plan.

Tests: `tests/tools/test_train_synwoodscape_logging.py`,
`tests/tools/test_visualize_predictions.py`, `tests/models/test_simplebev_two_head.py`,
`tests/datasets/test_synwoodscape_two_head_contract.py`.

## Known Caveats

- `runs/` and `dataset/` are local artifacts and ignored by git.
- Do not edit `third_party/` or `mmdetection3d/`.
- Fine-tuning may use a smaller ROI; pretraining uses a larger one because SynWoodScape has
  sparse near-ego obstacle coverage.
- `docs/BEV_loss_and_metrics_design.md` is fine-tuning-oriented and includes concepts not
  implemented in the pretrain yet, such as soft visibility.
- Merge and push are user-controlled actions.
