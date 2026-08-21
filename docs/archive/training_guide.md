# SynWoodScape Two-Head Simple-BEV Training Guide

> ## ⚠ [상태: 낡음 — 2-head 시절 문서다]
>
> **여기 적힌 플래그(`--lambda_vis`, `--vis_neg_weight`, `--head`)와 경로
> (`runs/synwoodscape_twohead/...`), 지표(`iou_drivable`/`iou_obstacle`)는 전부 존재하지 않는다.**
> 2026-08-17에 3-class 단일 head로 확정하며 2-head 코드를 제거했다
> (`docs/archive/free_space_metric_migration.md` §9).
>
> 지금 유효한 문서:
> - 실행 방법과 현재 상태: [`next_session_threeclass_training.md`](next_session_threeclass_training.md)
> - 코드 정독 가이드(데이터 → 텐서 → 모델 → loss → 지표): [`../training_pipeline_walkthrough.md`](../training_pipeline_walkthrough.md)
> - 지표 정의 정본: [`../BEV_loss_and_metrics_design.md`](../BEV_loss_and_metrics_design.md) §2.8
> - loss 설계 현황과 다음 단계: [`../finetune_overfitting_diagnosis.md`](../finetune_overfitting_diagnosis.md) §13
>
> 아래 내용은 당시의 튜닝 판단 근거로서만 남겨 둔다.


이 문서는 현재 구현된 SynWoodScape 2-head pretraining을 실행하고 해석하는 기준을 정리한다. fine-tuning 환경을 기준으로 작성된 별도 설계 참고 문서는 `docs/BEV_loss_and_metrics_design.md`이고, 현재 pretrain 구현에서 실제로 쓰는 설정과 지표는 이 문서를 우선한다.

> **이 문서는 pretraining(SynWoodScape) 전용이다.** 자체 데이터셋 fine-tuning은
> [`docs/finetuning_guide.md`](../finetuning_guide.md)를 본다 — 데이터셋·렌즈 모델·마스킹 규약이
> 달라서 실행 방법과 지표 해석 기준이 따로 있다. 다만 **지표 구현 자체는 두 경로가 공유하므로**
> (`projects/common/two_head_metrics.py`), 아래 "Occupancy Metrics" 이후의 해석은 양쪽에 그대로
> 적용된다.

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

Checkpoint score:

```text
val_score = val_iou_free
```

This is not validation loss. `model_best-*.pth` is saved when this score improves.

Rationale:

`iou_free` evaluates the deployable free space, combining occupancy and visibility. It penalizes
both calling occupied or unknown cells free and missing truly free cells, so an all-drivable or
otherwise degenerate map cannot win merely through a high legacy class IoU.

## Console Log Interpretation

Current epoch log shape:

```text
epoch 001/60 | time   23.1s | val_iou_free↑ 0.753 (+0.004) (+0.080 vs baseline) | best_val_iou_free↑ 0.753 | checkpoint: new best
  bins  | val_obst_iou_bins empty:fa .../nN tiny:.../nN small:.../nN medium:.../nN large:.../nN
  train | loss_total↓ ... | loss_occ↓ ... | loss_vis↓ ... | iou_drivable↑ ... | iou_obstacle↑ ... | vis_false_high↓ ... | vis_false_low↓ ... | obst_frac ... | false_obstacle↓ ... | missed_obstacle↓ ...
  val   | loss_total↓ ... | loss_occ↓ ... | loss_vis↓ ... | iou_drivable↑ ... | iou_obstacle↑ ... | vis_false_high↓ ... | vis_false_low↓ ... | obst_frac ... | false_obstacle↓ ... | missed_obstacle↓ ...
  deploy| (pred visibility 기준) iou_drivable↑ ... | iou_obstacle↑ ... | visible_coverage ...
```

`(+0.004)` next to `val_iou_free` is the change against the previous best (green when positive,
red when negative). It is omitted on the first scored epoch, where there is no previous best yet.
`(+0.080 vs baseline)` is the validation `iou_free` gap to the constant-map baseline; it is also
green when positive and red when negative.

The rows are colour-coded when the output is a terminal: metric names are dimmed so only the
numbers stand out, and the two IoU values are bold. Colour is disabled automatically when stdout
is redirected (`> train.log`, `| tee`) so log files stay free of escape sequences. Override with
`NO_COLOR=1` to force plain output, or `FORCE_COLOR=1` to keep colour through a pipe
(`FORCE_COLOR=1 bash configs/train_robot_bev_finetune.sh | less -R`).

Direction markers:

- `loss_*↓`: lower is better.
- `iou_*↑`: higher is better.
- `vis_false_high↓`: lower is better. Invisible cells predicted as visible.
- `vis_false_low↓`: lower is better. Visible cells predicted as invisible.
- `false_obstacle↓`: lower is better. Drivable GT predicted as obstacle.
- `missed_obstacle↓`: lower is better. Obstacle GT predicted as drivable.

## Occupancy Metrics

Primary metric:

- `val_iou_free`: checkpoint score input. It evaluates free space under the valid BEV mask.

Reference metrics:

- `iou_drivable`: IoU of predicted drivable cells.
- `iou_obstacle`: IoU of predicted obstacle cells, averaged over samples that actually contain
  obstacles in GT. See "Samples without obstacles" below. These legacy IoUs remain useful for
  diagnosis but do not select `model_best-*.pth`.

### Samples without obstacles

A sample whose GT has no obstacle cell always has `intersection = 0`, so its obstacle IoU is `0`
even when the model correctly predicts no obstacle at all. Such samples are therefore excluded
from `iou_obstacle` and from the bin IoUs, and their error is reported separately as a false
alarm rate. In the ROI 8/4/±6 m val split this is 14 of 100 samples, so including them dragged
the reported obstacle IoU down by roughly 0.10.

Because of this, legacy occupancy IoUs from runs before this change are **not** directly comparable
with the `iou_free` checkpoint score. Runs `twohead_pretrain_res101_..._260813_224942` and
`twohead_pretrain_diag_baseline_..._260814_130110` are on the old convention.

Diagnostic metrics added after the first run:

- `obst_frac`: GT obstacle fraction inside `valid_bev_g`.
- `false_obstacle`: among GT drivable valid cells, fraction predicted as obstacle.
- `missed_obstacle`: among GT obstacle valid cells, fraction predicted as drivable.
- `val_obst_iou_bins`: obstacle IoU grouped by GT obstacle fraction:
  - `empty`: exactly 0 obstacle fraction. Shows `fa <rate>` — the fraction of valid cells wrongly
    predicted as obstacle on these samples — instead of a meaningless IoU.
  - `tiny`: `0 < fraction <= 0.01`
  - `small`: `0.01 < fraction <= 0.05`
  - `medium`: `0.05 < fraction <= 0.15`
  - `large`: `0.15 < fraction <= 1.0`

How to read them:

- Low obstacle IoU with `tiny` obstacle fraction can be metric sensitivity, not necessarily a severe visual failure.
- `empty:fa` rising means the model invents obstacles on clean scenes; it is the false-alarm
  counterpart to `missed_obstacle`.
- High `missed_obstacle` is more safety-critical than high `false_obstacle`.
- High `false_obstacle` makes the model conservative and may block drivable space.
- Improvements should reduce `missed_obstacle` without causing a large `false_obstacle` increase.

TensorBoard scalar names:

- `train/occupancy_obstacle_fraction_epoch`
- `train/occupancy_false_obstacle_epoch`
- `train/occupancy_missed_obstacle_epoch`
- `train/occupancy_obstacle_iou_tiny_epoch`
- `train/occupancy_obstacle_iou_small_epoch`
- `train/occupancy_obstacle_iou_medium_epoch`
- `train/occupancy_obstacle_iou_large_epoch`
- `train/occupancy_empty_false_alarm_epoch`: false alarm rate on obstacle-free samples.
- `train/occupancy_empty_false_alarm_samples_epoch`: how many obstacle-free samples had any
  obstacle predicted.
- `train/occupancy_obstacle_count_<bin>_epoch`: sample count per bin.
- same keys under `val/`

There is no `occupancy_obstacle_iou_empty_epoch` scalar — that bin reports false alarm instead.
Bins with no samples are shown as `-/n0` in console and their IoU scalar is skipped to avoid
TensorBoard NaN warnings.

## Mask Contract: `valid` vs `vis`

These are different concepts and must not be the same array:

| mask | meaning | role |
|---|---|---|
| `valid_bev_g` | the cell has a label at all (ROI / annotatable range) | outer bound of every loss and metric |
| `vis_bev_g` | the cell is actually observable (H=0.8 visibility) | restricts *where occupancy is judged* |

- occupancy loss and occupancy metrics use `vis * valid`.
- visibility loss and visibility metrics use `valid` only.

SynWoodScape labels the whole ROI, so `valid_bev_g` is all ones.

**Why this matters.** Runs before 2026-08-14 had `valid_bev_g = vis_bev_g`. That made the
visibility loss weight `asymmetric_weight * valid` zero on every invisible cell, so the
visibility head only ever saw positives and "predict visible everywhere" was the exact global
optimum (`loss_vis = 0.0`). Two consequences, both verified on real data:

- `vis_neg_weight` had **no effect at all** — the loss was bit-identical for 0.0, 3.0 and 100.0.
- `vis_false_high`'s denominator `(~gt_visible) & valid` was the empty set, so it read `0.000`
  no matter what the model predicted.

So the near-zero visibility errors in the first two long runs mean nothing. Occupancy numbers
from those runs are still valid, because occupancy masking was correct.

## Visibility Metrics

Visibility is a separate head and is trained with asymmetric BCE:

- positive visible cells weight: 1
- invisible cells weight: `vis_neg_weight`, currently 3.0
- total loss: `loss_occ + lambda_vis * loss_vis`

Metrics:

- `vis_false_high`: invisible GT predicted visible. This is the risky visibility error.
- `vis_false_low`: visible GT predicted invisible. This is conservative/information loss.

Roughly 5% of ROI cells are invisible on average, and about 90% of those are obstacle cells in
the occupancy GT — the invisible region is essentially obstacles plus the shadow behind them.

## Deployment Metrics

`iou_obstacle` masks by **GT** visibility, which measures the occupancy head in isolation. At
deployment there is no GT: the robot can only trust the region the model itself claims to see.
The `deploy` console line and `*/deploy_*_epoch` scalars evaluate occupancy under exactly that
condition — masked by **predicted** visibility.

```text
  deploy| (pred visibility 기준) iou_drivable↑ 0.861 | iou_obstacle↑ 0.414 | visible_coverage 0.944
```

- `deploy_iou_drivable` / `deploy_iou_obstacle`: occupancy IoU inside predicted-visible cells,
  scored against full GT occupancy. Hallucinated visibility is punished here, because the wrong
  occupancy behind an occluder is no longer masked away.
- `deploy_visible_coverage`: fraction of valid cells the model claims to see.

**Always read the two together.** A model that declares a tiny visible region gets an easy
`deploy_iou_*`. Rising IoU with falling coverage is not an improvement.

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

**Pretraining is treated as done** (val obstacle IoU 0.861 / drivable 0.989). Its job was a usable
initialization for greenhouse fine-tuning, not a maximized SynWoodScape score, and the project has
moved to Phase 4 — see `docs/finetuning_guide.md`.

Of the original list: the diagnostic-metric baseline and photometric augmentation are **done**,
weight decay was swept and had **no effect**, and BCE+Dice is **ruled out** (the training objective
is already saturated at `missed_obstacle` 0.0008, so there is no training error left for Dice to
redistribute). Full results and reasoning: `docs/archive/synwoodscape_pretrain_experiment_log.md`.

If pretraining is revisited, `docs/archive/training_improvement_plan.md` keeps the remaining candidates —
worst-case visualization, `Segnet(rand_flip=True)` mirroring, and the 0.3/0.7 best-score reweighting.
