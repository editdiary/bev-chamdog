# SynWoodScape Manual Occupancy Labeling ROI 8/4/+-6

## Current Decision

Occupancy label generation is separated from visibility label generation.

- Occupancy ROI: front `8m`, rear `4m`, lateral `+-6m`
- Cell size: `0.05m`
- Output shape: `240 x 240`
- Manual labeling target: binary drivable / non-drivable occupancy
- Visibility target: generated automatically later with the same ROI/grid spec

The initial source crops were exported on 2026-08-13 to:

```text
dataset/synwoodscape_roi_8_4_6_semantic_crops/
```

## Exported Source Folders

```text
dataset/synwoodscape_roi_8_4_6_semantic_crops/
  gtLabels/      # single-channel class-id PNG crop from original BEV gtLabels
  rgbLabels/     # RGB semantic crop from original BEV rgbLabels
  palette_rgb/   # deterministic RGB rendering from gtLabels using local class colors
  class_legend.png
  README.md
```

There are 500 samples, named `00000.png` through `00499.png` in each crop folder.

## Manual Label Contract

The user plans to create RGB occupancy labels manually.

The later post-processing step should accept RGB label images with the same sample ids and shape:

```text
<manual_label_root>/<sample_id>.png
```

Expected semantic meaning:

- drivable area: user-selected RGB color, to be mapped to binary `1`
- non-drivable area: user-selected RGB color, to be mapped to binary `0`

The exact RGB colors are not fixed yet. After manual labeling is done, inspect a few label PNGs and define a robust RGB-to-binary conversion rule.

## Planned Post-Processing

After manual labels are ready:

1. Load manual RGB label.
2. Convert RGB label to binary occupancy.
3. Load original class-id crop from `gtLabels/`.
4. Overwrite core object classes as non-drivable even if the manual RGB label marked them drivable.

Core overwrite classes:

```text
4  pedestrian
10 four-wheeler vehicle
21 two-wheeler vehicle
24 ego vehicle
```

Candidate classes to decide later:

```text
22 static
23 dynamic
```

5 pole, 9 vegetation, 12 traffic sign, and 18 traffic light are intentionally not forced here because the user wants to manually control those ambiguous BEV artifacts.

## Manual Label Post-Processing Run

The user finished manual RGB labeling under:

```text
dataset/annotated_roi_8-4-6_semantic_crop/SegmentationClass/
```

The label map was:

```text
background:0,0,0
non-drivable:61,61,245
```

Interpretation used for post-processing:

- `(61, 61, 245)` = non-drivable
- every other RGB value = drivable

Object overwrite was first applied on 333 manually labeled samples, then rerun on all 500 source samples. Missing manual label files are treated as all-drivable before object overwrite. The script used was:

```text
work_dirs/apply_manual_occupancy_object_overwrite.py
```

Output:

```text
dataset/annotated_roi_8-4-6_semantic_crop_object_overwrite/
  final_rgb/              # RGB final label, black=drivable and blue=non-drivable
  binary_non_drivable/    # L-mode PNG, 0=drivable and 255=non-drivable
  review_compare/         # manual / object mask / final side-by-side images
  stats.txt
  README.md
```

Forced non-drivable overwrite classes:

```text
4  pedestrian
10 four-wheeler vehicle
21 two-wheeler vehicle
24 ego vehicle
```

Latest run summary:

```text
processed samples: 500
manual label files present: 333
all-drivable fallback samples: 167
overwrite_added_pixels: 1853345
```

## Refined Object Shape Run

After visual inspection, a small number of labels had holes where vegetation, traffic lights, or thin wires occluded ego/object semantic masks in the original BEV label.

A refined pass was generated with:

```text
work_dirs/refine_manual_occupancy_object_shapes.py
```

Output:

```text
dataset/annotated_roi_8-4-6_semantic_crop_object_overwrite_refined/
  final_rgb/
  binary_non_drivable/
  review_compare/
  stats.txt
  canonical_ego_mask.png
  README.md
```

Refinement rules:

- Missing manual label PNG = all-drivable before overwrite.
- Canonical ego silhouette from majority-voted class 24 masks is forced non-drivable for all samples.
- Pedestrian / four-wheeler / two-wheeler object masks are morphologically closed and hole-filled before overwrite.
- The review image marks newly added ego pixels in yellow and newly added object-fill pixels in orange.

Latest refined run summary:

```text
processed samples: 500
ego_added_pixels: 1020
object_added_pixels: 8449
```

## Visibility

Visibility is not manually labeled.

When final occupancy labels are built, generate `visible.npy` with the same `OccupancyGridSpec(front=8, rear=4, half_width=6, cell=0.05)`.

The previous `5/3/+-4` visibility masks must not be reused because their shape and cell coordinates differ.

## Visibility H<=0.8 Generation Run

Training visibility labels were generated with:

```text
work_dirs/build_visibility_h08_roi_8_4_6.py
```

Output:

```text
dataset/annotated_roi_8-4-6_semantic_crop/visibility_h08/
  raw_visible/       # npy, pure gather visibility before ego exclusion
  visible/           # npy, training loss mask = raw_visible & ~canonical_ego_mask
  raw_visible_png/
  visible_png/
  stats.txt
  README.md
```

Visibility rule:

- ROI: front 8m, rear 4m, half_width +-6m, cell 0.05m
- gather column visibility with max height `H=0.8m`
- one-sided occlusion test against the four camera depth maps
- ego region is excluded only from the training mask, not from `raw_visible`

Latest run summary:

```text
processed samples: 500
mean_raw_visible: 0.947505
mean_visible_for_loss: 0.937957
visible shape: 240x240 bool
ego cells in visible/: 0
```


## Training Package

The finalized occupancy and visibility labels were packaged into a single training-ready root with:

```text
tools/package_synwoodscape_2head_labels.py
```

Output:

```text
dataset/synwoodscape_2head_roi_8_4_6_h08/
  00000_occupancy.npy
  00000_visible.npy
  ...
  00499_occupancy.npy
  00499_visible.npy
  metadata.json
  README.md
```

Training package conventions:

- `*_occupancy.npy`: `uint8`, drivable `1`, non-drivable `0`
- `*_visible.npy`: `bool`, H=0.8 visibility/loss mask with ego excluded
- shape: `240 x 240`
- sample count: 500 occupancy files and 500 visible files

Latest package summary:

```text
processed samples: 500
mean_drivable_fraction: 0.829361
mean_visible_fraction: 0.937957
```

The previous temporary preview/prototype files under `work_dirs/` were deleted after packaging.
