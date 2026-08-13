# Next Session Handoff: SynWoodScape Two-Head Pretraining

Last updated: 2026-08-14

## One-Line State

SynWoodScape ROI 8/4/±6 m labels are packaged, Simple-BEV is running as a two-head occupancy/visibility model, the first 60-epoch pretrain completed successfully, and diagnostic occupancy metrics were added for the next full run.

## Current Branch and Git State

Working branch:

```bash
git branch --show-current
# feat/synwoodscape-two-head-training
```

Important recent commits:

```text
0294373 Add occupancy diagnostic training metrics
539b918 Add two-head prediction visualization
4f44183 Improve SynWoodScape training logs
c4dc6a2 feat: add SynWoodScape two-head training
5e200cb feat: package SynWoodScape two-head labels
```

Do not merge or push without explicit user action.

## Key Files

Training/data:

- `configs/train_synwoodscape_twohead_pretrain.sh`: main long-run pretrain command.
- `tools/train_synwoodscape.py`: two-head training loop, losses, logging, diagnostic metrics.
- `tools/visualize_predictions.py`: two-head checkpoint inference visualization.
- `projects/models/simplebev_two_head.py`: `TwoHeadSegnet`, split logits, two-head loss, visibility metrics.
- `projects/datasets/synwoodscape_simplebev.py`: dataset contract for `seg_bev_g`, `vis_bev_g`, `valid_bev_g`.
- `projects/bev_gt/grid.py`: ROI/grid spec, including 8/4/±6 m two-head pretrain spec.

Documentation:

- `docs/training_guide.md`: current training command, metric interpretation, inference command.
- `docs/training_improvement_plan.md`: ordered improvement plan.
- `docs/BEV_loss_and_metrics_design.md`: fine-tuning-oriented loss/metric reference from earlier design discussion.
- `docs/dataset_analysis/synwoodscape_manual_labeling_roi_8_4_6.md`: label construction and manual review context.
- `docs/next_session_synwoodscape_twohead.md`: this handoff document.

Tests:

- `tests/tools/test_train_synwoodscape_logging.py`: log formatting and occupancy diagnostic metrics.
- `tests/tools/test_visualize_predictions.py`: occupancy/visibility image conversion.
- `tests/models/test_simplebev_two_head.py`: model/loss/visibility metric contract.
- `tests/datasets/test_synwoodscape_two_head_contract.py`: dataset shape/key contract.

## Dataset State

Final reviewed source labels:

```text
dataset/annotated_roi_8-4-6_semantic_crop/
```

Important subpaths:

- `final_rgb/`: 500 reviewed RGB label PNGs.
- `binary_non_drivable/`: binary non-drivable masks.
- `visibility_h08/visible/`: final H=0.8 visibility labels with ego excluded for training.
- `visibility_h08/raw_visible/`: raw visibility before ego exclusion.

Training-ready package:

```text
dataset/synwoodscape_2head_roi_8_4_6_h08/
```

Contents:

- `*_occupancy.npy`: 500 files, `uint8`, shape `240x240`, drivable `1`, obstacle `0`.
- `*_visible.npy`: 500 files, `bool`, shape `240x240`, H=0.8 visibility, ego excluded.
- `metadata.json`, `README.md`.
- total files: 1002.

## First Long Training Run

Command used:

```bash
CUDA_VISIBLE_DEVICES=1 bash configs/train_synwoodscape_twohead_pretrain.sh
```

Run name:

```text
twohead_pretrain_res101_bs16_lr3e-04_260813_224942
```

Outputs:

```text
runs/synwoodscape_twohead/logs/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/
runs/synwoodscape_twohead/ckpt/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/
```

Best checkpoint:

```text
runs/synwoodscape_twohead/ckpt/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/model_best-000000041.pth
```

Summary:

- best epoch: 41
- best `val_iou_mean`: about 0.870
- epoch 41 `val_drivable_iou`: about 0.989
- epoch 41 `val_obstacle_iou`: about 0.751
- epoch 60 remained close but slightly below best by score.
- val loss minimum occurred earlier than best IoU, so `model_best` should remain the default checkpoint.

## Inference Visualization State

Generated validation preview:

```text
runs/synwoodscape_twohead/viz_best_epoch41/
```

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n bev-chamdog python tools/visualize_predictions.py \
  --ckpt_dir=runs/synwoodscape_twohead/ckpt/twohead_pretrain_res101_bs16_lr3e-04_260813_224942 \
  --model_name=model_best \
  --step=41 \
  --num_samples=20 \
  --encoder_type=res101 \
  --use_fisheye=True \
  --output_dir=runs/synwoodscape_twohead/viz_best_epoch41
```

Each sample has:

- `*_compare.png`: RGB 4-cam + GT/pred occupancy + GT/pred visibility + pred occupancy masked by pred visibility.
- `*_pred_occupancy.png`
- `*_pred_visibility.png`
- `*_pred_combined.png`

Qualitative observation:

- Overall predictions are better than expected for a first full pretrain.
- Some low obstacle IoU samples have very small obstacle regions, so a few wrong cells can sharply reduce IoU.
- This motivated adding diagnostic metrics before changing loss.

## Current Metrics to Watch

Primary:

- `val/iou_obstacle_epoch`: main occupancy quality indicator.
- `val/iou_drivable_epoch`: useful but can be misleadingly high.
- `val_iou_mean`: current best checkpoint score, `0.5 * (drivable + obstacle)`.

Diagnostics added after first run:

- `val/occupancy_obstacle_fraction_epoch`: GT obstacle fraction in valid cells.
- `val/occupancy_false_obstacle_epoch`: drivable GT predicted as obstacle.
- `val/occupancy_missed_obstacle_epoch`: obstacle GT predicted as drivable.
- `val/occupancy_obstacle_iou_empty_epoch`
- `val/occupancy_obstacle_iou_tiny_epoch`
- `val/occupancy_obstacle_iou_small_epoch`
- `val/occupancy_obstacle_iou_medium_epoch`
- `val/occupancy_obstacle_iou_large_epoch`

Console bin format:

```text
val_obst_iou_bins empty:-/n0 tiny:.../nN small:.../nN medium:.../nN large:.../nN
```

Interpretation:

- `missed_obstacle` is the safety-critical error. Lower is better.
- `false_obstacle` is conservative error. Lower is better, but it is usually less dangerous than missed obstacles.
- `tiny`/`small` bins explain whether poor obstacle IoU is caused by very small obstacle regions.
- Empty bins are shown as `-/n0`; their IoU scalar is not written to TensorBoard to avoid NaN warnings.

Visibility:

- `vis_false_high`: invisible predicted visible; risky.
- `vis_false_low`: visible predicted invisible; conservative.
- In the first run both quickly went near zero, so visibility is probably easy with current binary labels.

## Recommended Next Run

Run the same baseline again first, now with the new diagnostic metrics:

```bash
CUDA_VISIBLE_DEVICES=1 bash configs/train_synwoodscape_twohead_pretrain.sh
```

Why repeat before changing loss:

- The first run did not have `false_obstacle`, `missed_obstacle`, or bin-wise obstacle IoU logged.
- A repeat run gives a comparable diagnostic baseline.
- After that, loss/score experiments can be judged by error direction, not only global IoU.

Suggested run name adjustment:

Edit `configs/train_synwoodscape_twohead_pretrain.sh` and change:

```bash
--exp_name=twohead_pretrain
```

to something like:

```bash
--exp_name=twohead_pretrain_diag_baseline
```

This is not required because timestamped run folders prevent overwrite, but it makes TensorBoard easier to read.

## TensorBoard Command

```bash
conda run -n bev-chamdog tensorboard --logdir runs/synwoodscape_twohead/logs --port 6006
```

Open:

```text
http://localhost:6006
```

If running remotely, use SSH port forwarding from your local machine.

## After Next Full Run

1. Identify the new run folder under `runs/synwoodscape_twohead/logs/` and `ckpt/`.
2. Compare against first baseline:
   - best epoch
   - `val_iou_obstacle_epoch`
   - `val/occupancy_missed_obstacle_epoch`
   - `val/occupancy_false_obstacle_epoch`
   - `val/occupancy_obstacle_iou_small_epoch`
   - `val/occupancy_obstacle_iou_medium_epoch`
3. Generate visualization from the new best checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n bev-chamdog python tools/visualize_predictions.py \
  --ckpt_dir=runs/synwoodscape_twohead/ckpt/<NEW_RUN_NAME> \
  --model_name=model_best \
  --step=<BEST_EPOCH> \
  --num_samples=20 \
  --encoder_type=res101 \
  --use_fisheye=True \
  --output_dir=runs/synwoodscape_twohead/viz_<NEW_RUN_NAME>_best
```

4. If diagnostics confirm the issue is missed obstacles, proceed to the next planned experiment:
   - lower `lambda_vis` from `0.5` to `0.2`
   - add occupancy Dice loss in a separate commit/experiment

## Useful Verification Commands

Before starting work:

```bash
git status --short
git log --oneline -8
```

Before/after code changes:

```bash
conda run -n bev-chamdog pytest -q
```

Training smoke test after changing training code:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n bev-chamdog python tools/train_synwoodscape.py \
  --exp_name=smoke \
  --num_epochs=1 \
  --batch_size=1 \
  --num_workers=0 \
  --max_samples=4 \
  --encoder_type=res101 \
  --use_fisheye=True \
  --val_freq_epochs=1 \
  --save_freq_epochs=1 \
  --log_dir=/tmp/bev_chamdog_smoke/logs \
  --ckpt_dir=/tmp/bev_chamdog_smoke/ckpt
```

## Suggested Prompt for the Next Session

Paste this to continue efficiently:

```text
지난 세션에서 SynWoodScape two-head pretraining을 정리했고, docs/next_session_synwoodscape_twohead.md에 핸드오프를 남겼어. 그 문서를 먼저 읽고 현재 git 상태를 확인한 뒤 이어서 작업해줘. 다음 목표는 새 diagnostic metric이 포함된 baseline full run을 돌리고, TensorBoard/로그 기준으로 first run과 비교한 다음, 필요하면 lambda_vis=0.2 또는 BCE+Dice loss 실험 설계를 진행하는 거야. GPU는 1번을 사용해줘.
```

## Known Caveats

- `runs/` and `dataset/` are local artifacts and ignored by git.
- Do not edit `third_party/` or `mmdetection3d/`.
- Current fine-tuning target may use a smaller ROI; pretraining uses larger ROI because SynWoodScape has sparse near-ego obstacle coverage.
- `docs/BEV_loss_and_metrics_design.md` is fine-tuning-oriented and includes concepts not fully implemented in SynWoodScape pretrain yet, such as soft visibility and greenhouse-specific metrics.
- Merge and push are user-controlled actions.
