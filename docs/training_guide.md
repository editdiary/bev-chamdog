# SynWoodScape Two-Head Simple-BEV Training Guide

이 문서는 현재 구현된 SynWoodScape 2-head pretraining을 실행하고 해석하는 기준을 정리한다. fine-tuning 환경을 기준으로 작성된 별도 설계 참고 문서는 `docs/BEV_loss_and_metrics_design.md`이고, 현재 pretrain 구현에서 실제로 쓰는 설정과 지표는 이 문서를 우선한다.

## Current Training Setup

- Model: `projects.models.simplebev_two_head.TwoHeadSegnet`
- Base network: standalone Simple-BEV wrapper, `third_party/` 직접 수정 없음
- Heads:
  - occupancy head: `drivable=1`, `obstacle=0`
  - visibility head: visible `1`, invisible `0`
- Dataset package: `dataset/synwoodscape_2head_roi_8_4_6_h08/`
- Label source:
  - reviewed manual occupancy labels: `dataset/annotated_roi_8-4-6_semantic_crop/final_rgb/`
  - visibility labels: H=0.8 gather-column visibility, ego excluded from training mask
- Grid/ROI: front 8 m, rear 4 m, left/right 6 m, `0.05 m/cell`, shape `240x240`
- Cameras: SynWoodScape 4-cam fisheye, `radial_poly`, `use_fisheye=True`
- GPU policy: use GPU 1 by default because GPU 0 may be occupied

Training-ready label package size:

- `500` occupancy `.npy`
- `500` visible `.npy`
- `metadata.json`, `README.md`
- total files: `1002`

## Main Training Command

Recommended script:

```bash
CUDA_VISIBLE_DEVICES=1 bash configs/train_synwoodscape_twohead_pretrain.sh
```

The script currently runs:

- `num_epochs=60`
- `batch_size=16`
- `lr=3e-4`
- `weight_decay=1e-7`
- `num_workers=8`
- `val_fraction=0.2` → train 400 / val 100
- `encoder_type=res101`
- `lambda_vis=0.5`
- `vis_neg_weight=3.0`
- logs: `runs/synwoodscape_twohead/logs/`
- checkpoints: `runs/synwoodscape_twohead/ckpt/`

Direct command equivalent:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n bev-chamdog python tools/train_synwoodscape.py \
  --exp_name=twohead_pretrain \
  --num_epochs=60 \
  --batch_size=16 \
  --lr=3e-4 \
  --weight_decay=1e-7 \
  --num_workers=8 \
  --val_fraction=0.2 \
  --split_seed=0 \
  --encoder_type=res101 \
  --use_fisheye=True \
  --lambda_vis=0.5 \
  --vis_neg_weight=3.0 \
  --val_freq_epochs=1 \
  --save_freq_epochs=5 \
  --log_dir=runs/synwoodscape_twohead/logs \
  --ckpt_dir=runs/synwoodscape_twohead/ckpt
```

## First Long Run Result

Completed run:

- run name: `twohead_pretrain_res101_bs16_lr3e-04_260813_224942`
- best checkpoint: `runs/synwoodscape_twohead/ckpt/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/model_best-000000041.pth`
- periodic checkpoints kept: epoch 50, 55, 60
- TensorBoard event: `runs/synwoodscape_twohead/logs/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/events.out.tfevents.1786628982.ubuntu`
- validation split file: `runs/synwoodscape_twohead/logs/twohead_pretrain_res101_bs16_lr3e-04_260813_224942/split_val_ids.txt`

Important scalar summary from the first long run:

| Epoch | val IoU mean | val drivable IoU | val obstacle IoU | note |
|---:|---:|---:|---:|---|
| 30 | about 0.865 | about 0.988 | about 0.741 | minimum val loss |
| 41 | about 0.870 | about 0.989 | about 0.751 | best checkpoint |
| 60 | about 0.869 | about 0.989 | about 0.749 | final checkpoint |

Interpretation:

- Training works and converges.
- Occupancy quality mostly saturates before the end of 60 epochs.
- Visibility is easy on this binary label and keeps improving, while occupancy validation loss starts worsening earlier.
- Use `model_best-000000041.pth` as the default pretrain checkpoint for now.

## Checkpoint Selection

Current best score:

```text
val_score = 0.5 * (val_iou_drivable + val_iou_obstacle)
```

This is not validation loss. `model_best-*.pth` is saved when this score improves.

Rationale:

- `drivable` dominates the map, so drivable IoU alone is misleading.
- `obstacle` is sparse but safety-critical.
- The current 0.5/0.5 mean keeps both classes visible during pretraining.

Future experiment documented in `docs/training_improvement_plan.md`:

```text
val_score = 0.3 * val_iou_drivable + 0.7 * val_iou_obstacle
```

Do not switch the score until the new diagnostic metrics have been observed on a real full run.

## Console Log Interpretation

Current epoch log shape:

```text
epoch 001/60 | time   23.1s | val_iou_mean↑ 0.753 | best_val_iou_mean↑ 0.753 | checkpoint: new best | val_obst_iou_bins empty:-/n0 tiny:.../nN small:.../nN medium:.../nN large:.../nN
  train | loss_total↓ ... | loss_occ↓ ... | loss_vis↓ ... | iou_drivable↑ ... | iou_obstacle↑ ... | vis_false_high↓ ... | vis_false_low↓ ... | obst_frac ... | false_obstacle↓ ... | missed_obstacle↓ ...
  val   | loss_total↓ ... | loss_occ↓ ... | loss_vis↓ ... | iou_drivable↑ ... | iou_obstacle↑ ... | vis_false_high↓ ... | vis_false_low↓ ... | obst_frac ... | false_obstacle↓ ... | missed_obstacle↓ ...
```

Direction markers:

- `loss_*↓`: lower is better.
- `iou_*↑`: higher is better.
- `vis_false_high↓`: lower is better. Invisible cells predicted as visible.
- `vis_false_low↓`: lower is better. Visible cells predicted as invisible.
- `false_obstacle↓`: lower is better. Drivable GT predicted as obstacle.
- `missed_obstacle↓`: lower is better. Obstacle GT predicted as drivable.

## Occupancy Metrics

Primary metrics:

- `iou_drivable`: IoU of predicted drivable cells.
- `iou_obstacle`: IoU of predicted obstacle cells.
- `val_iou_mean`: checkpoint score input, currently average of drivable/obstacle IoU.

Diagnostic metrics added after the first run:

- `obst_frac`: GT obstacle fraction inside `valid_bev_g`.
- `false_obstacle`: among GT drivable valid cells, fraction predicted as obstacle.
- `missed_obstacle`: among GT obstacle valid cells, fraction predicted as drivable.
- `val_obst_iou_bins`: obstacle IoU grouped by GT obstacle fraction:
  - `empty`: exactly 0 obstacle fraction
  - `tiny`: `0 < fraction <= 0.01`
  - `small`: `0.01 < fraction <= 0.05`
  - `medium`: `0.05 < fraction <= 0.15`
  - `large`: `0.15 < fraction <= 1.0`

How to read them:

- Low obstacle IoU with `tiny` obstacle fraction can be metric sensitivity, not necessarily a severe visual failure.
- High `missed_obstacle` is more safety-critical than high `false_obstacle`.
- High `false_obstacle` makes the model conservative and may block drivable space.
- Improvements should reduce `missed_obstacle` without causing a large `false_obstacle` increase.

TensorBoard scalar names:

- `train/occupancy_obstacle_fraction_epoch`
- `train/occupancy_false_obstacle_epoch`
- `train/occupancy_missed_obstacle_epoch`
- `train/occupancy_obstacle_iou_empty_epoch`
- `train/occupancy_obstacle_iou_tiny_epoch`
- `train/occupancy_obstacle_iou_small_epoch`
- `train/occupancy_obstacle_iou_medium_epoch`
- `train/occupancy_obstacle_iou_large_epoch`
- same keys under `val/`

Empty bins are shown as `-/n0` in console and skipped for IoU scalar writing to avoid TensorBoard NaN warnings.

## Visibility Metrics

Visibility is a separate head and is trained with asymmetric BCE:

- positive visible cells weight: 1
- invisible cells weight: `vis_neg_weight`, currently 3.0
- total loss: `loss_occ + lambda_vis * loss_vis`

Metrics:

- `vis_false_high`: invisible GT predicted visible. This is the risky visibility error.
- `vis_false_low`: visible GT predicted invisible. This is conservative/information loss.

The first long run showed `vis_false_high` and `vis_false_low` quickly near zero. That means visibility may be too easy on the current binary labels. Future experiments can lower `lambda_vis` from `0.5` to `0.2` to let occupancy dominate more.

## TensorBoard

Start TensorBoard:

```bash
conda run -n bev-chamdog tensorboard --logdir runs/synwoodscape_twohead/logs --port 6006
```

Watch first:

- `val/iou_obstacle_epoch`
- `val/occupancy_missed_obstacle_epoch`
- `val/occupancy_false_obstacle_epoch`
- `val/occupancy_obstacle_iou_small_epoch`
- `val/occupancy_obstacle_iou_medium_epoch`
- `val/loss_occ_epoch`
- `val/loss_vis_epoch`
- `train/loss_epoch` vs `val/loss_epoch`

## Inference Visualization

Generate 20 validation samples from the best checkpoint:

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

Each sample produces:

- `<sample_id>_compare.png`: RGB 4-cam + GT occupancy + pred occupancy + GT visibility + pred visibility + pred occ x vis
- `<sample_id>_pred_occupancy.png`: occupancy head only
- `<sample_id>_pred_visibility.png`: visibility head only
- `<sample_id>_pred_combined.png`: predicted occupancy masked by predicted visibility

The first generated review set has 20 samples and 80 PNG files under `runs/synwoodscape_twohead/viz_best_epoch41/`.

## Practical Signs During Training

Good signs:

- `loss_occ` decreases early, then validation occupancy metrics stabilize.
- `iou_obstacle` rises above trivial 0 and does not collapse.
- `missed_obstacle` trends down.
- `false_obstacle` does not explode while trying to reduce missed obstacles.
- `vis_false_high` remains near zero.

Warning signs:

- `iou_drivable` is high but `iou_obstacle` is near 0: trivial drivable solution.
- `missed_obstacle` stays high: unsafe occupancy behavior.
- `false_obstacle` rises sharply: model may become too conservative.
- `val/loss_occ_epoch` worsens while `train/loss_occ_epoch` keeps improving: overfit.
- GPU utilization low with long epoch time: DataLoader/I/O bottleneck; try `num_workers=8/12/16` and compare.

## Next Experiments

Follow `docs/training_improvement_plan.md` in order:

1. Run one full baseline again with the new diagnostic metrics, same hyperparameters.
2. Compare new metrics against the first epoch-41 baseline.
3. Try `lambda_vis=0.2` and BCE+Dice occupancy loss.
4. Consider changing best score to `0.3 * drivable + 0.7 * obstacle` only after metrics show the tradeoff clearly.
5. Add photometric-only augmentation after metric/loss experiments are stable.
