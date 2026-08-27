# 학습 파이프라인 코드 정독 가이드

Last updated: 2026-08-18

> **목적**: "어떤 데이터를 어떻게 읽어서 → 어떤 텐서가 되어 → 어떤 loss로 학습되고 →
> 무엇으로 평가·선택되는가"를 실제 코드 위치와 함께 따라갈 수 있게 한다.
> 실행 방법(명령·플래그)은 [`AGENTS.md`](../AGENTS.md)와 각 `configs/*.sh`를 본다.
> loss의 미해결 항목은 [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7.

## 0. 전체 그림

```
데이터 파일 ─▶ Dataset.__getitem__ ─▶ batch dict ─▶ run_batch ─▶ Segnet ─▶ logits (B,3,H,W)
                                                       │                        │
                                                 라벨 3-class화            ┌────┴────┐
                                                       │                 loss    argmax
                                                       └─────────────────┘         │
                                                                                  지표
```

두 학습 경로는 **`Dataset`과 `vox_util`만 다르고 나머지는 같은 코드**다.

| | pretrain | fine-tuning |
|---|---|---|
| 스크립트 | `tools/train_synwoodscape.py` | `tools/train_robot_bev.py` |
| Dataset | `projects/datasets/synwoodscape_simplebev.py:84` | `projects/datasets/robot_simplebev.py:212` |
| lifting | `projects/models/fisheye_vox.py` (radial_poly) | `projects/models/double_sphere_vox.py` |
| 카메라 | 어안 4대 `FV/MVL/MVR/RV` | Double Sphere 3대 `front/left/right` |
| 이미지 | 512×384 | 512×288 |
| BEV 그리드 | 240×240 (전방8/후방4/±6 m) | 120×120 (전방4/후방2/±3 m) |
| 모델·loss·지표·로깅·체크포인트 선택 | **완전 공유** | ← |

---

## 1. 파일 → 텐서

### 1.1 pretrain (`synwoodscape_simplebev.py`)

```
rgb_images/{id}_{FV,MVL,MVR,RV}.png  → resize 512×384 → /255 → (4,3,384,512)
{id}_occupancy.npy                    → seg_bev_g   (1,240,240)
{id}_visible.npy                      → vis_bev_g   (1,240,240)
(파일 없음)                            → valid_bev_g = ones
```

라벨 루트는 `dataset/synwoodscape_2head_roi_8_4_6_h08`(500 샘플). 캘리브레이션 파생 텐서
(`pix_T_cams`, `cam0_T_camXs`)는 샘플과 무관하므로 `__init__:63-71`에서 한 번만 만든다.

### 1.2 fine-tuning (`robot_simplebev.py`)

```
rgb_images/{sample}/cam_{front,left,right}.jpg → resize 512×288 → (3,3,288,512)
occupancy_npy/{id}.npy   ┐
visibility_npy/{id}.npy  ├─ load_masked_labels(:156) → seg / vis / valid
common/*.png + calib.yaml┘
```

**여기가 두 경로의 가장 큰 차이다.** 로봇은 라벨을 그대로 쓰지 않고 마스크를 씌운다(`:156`):

```python
return occupancy, visible & ~permanent_blind, ~invalid
```

마스크는 `build_bev_masks(:137)`가 만든다:

| 마스크 | 출처 | 효과 |
|---|---|---|
| `permanent_blind` | `compute_camera_coverage_mask`(캘리브레이션) ∪ `self_mask.png` | **`vis=0`** |
| `invalid` | `rear_self_box.png` | **`valid=0`** |

**이 둘을 가르는 기준이 이 프로젝트의 핵심 계약이다** (모듈 docstring `:23-38`):

- `vis=0` = 배포 때도 존재하는 **물리적 가림**(화각 밖, ego 테이블) → `unknown`으로 학습된다.
- `valid=0` = **수집 아티팩트**(카트 손잡이, 미는 사람). 배포 때 없으므로 loss·지표 양쪽에서 제외.

둘을 합치면 "후방은 항상 가려져 있다"는 거짓을 학습한다. SynWoodScape 쪽에서 실제로 겪은 버그다.
라벨 계약의 정본은 [`finetuning_guide.md`](finetuning_guide.md) §1.3.

### 1.3 ⚠️ `occupancy`는 이름과 반대다

**`occ=1`이 drivable/free이고, `occ=0`이 장애물이다.** `free_space.py:33`의
`free = occ & vis & valid`가 그 정의다. 이걸 놓치면 코드가 전부 뒤집혀 보인다.

**이 규약이 해석되는 지점은 `projects/common/free_space.py:33-34` 한 곳뿐이다.** 나머지 코드는
이름만 통과시키고, `occ`/`vis`/`valid`를 조합해야 하는 곳은 전부 `decompose()`를 부른다
(2026-08-18에 통일: 그 전에는 두 trainer와 `class_weights_from_labels`,
`rescore_checkpoints`, `measure_label_geometry`가 `occ & vis & valid`를 각자 다시 쓰고 있었다).

`decompose()`는 **torch 텐서와 numpy 배열을 모두 받는다** -- 학습 루프는 텐서를 넘기고,
베이스라인·클래스 가중치를 실측하는 코드는 `.npy`를 읽은 직후의 numpy 배열을 넘긴다.
두 경로가 같은 값을 낸다는 것이 계약이며 `tests/common/test_free_space.py`가 고정한다.

> 이름(`occupancy`)이나 값(1=free) 자체를 바꾸는 것은 코드 243곳 + 디스크의 디렉터리/파일
> 이름까지 건드려야 해서 하지 않았다. 조합 지점을 한 곳으로 모으는 것으로 혼동 원인의
> 대부분이 사라진다고 판단했다.

---

## 2. 라벨 → 3-class 인덱스

`projects/common/free_space.py` (전체 62행) — 분해의 **단일 출처**. 학습·시각화·재채점 도구가
전부 여기를 쓴다. 각자 조합하게 두면 정의가 조용히 갈라지기 때문이다.

```python
free     = occ  & vis & valid      # 통과 가능하고 실제로 관측된 곳
occupied = ~occ & vis & valid      # 광선이 멈춘 표면 (두께 1셀이 정상)
unknown  = ~vis & valid            # 그 너머
                                   # valid=0은 셋 중 어디에도 속하지 않는다
```

`to_class_index(:39)`가 `(B,1,H,W)` long으로 만든다. `valid=0` 셀은 `UNKNOWN(0)`으로 떨어지지만
loss에서 `valid`를 곱해 지워지므로 무해하다.

`partition_defect_count(:57)`가 "세 마스크의 합 == valid" 불변식을 매 배치 검사하고, 깨지면
epoch 로그에 `[BUG] partition defect`가 찍힌다(`bev_occupancy_metrics.py:144`).

---

## 3. 모델

`projects/models/simplebev_three_class.py` (전체 75행). Simple-BEV 원본(`third_party/`,
submodule이라 **수정 금지**)을 상속해 **마지막 head 하나만** 교체한다.

```
(B,S,3,H,W) 이미지
  ├─ Encoder_res101                    segnet.py:159   stride 8, 128ch
  ├─ vox_util.unproject_image_to_mem   double_sphere_vox.py:117  ★ 어안 투영은 여기서 일어난다
  │     (pretrain은 fisheye_vox.py) 3D 복셀 중심 → 카메라 좌표 → 렌즈 모델 → 픽셀
  │     → grid_sample → (B,S,128,Z,4,X)
  ├─ reduce_masked_mean(dim=1)         segnet.py:427   카메라 축 평균 → (B,128,Z,4,X)
  ├─ bev_compressor                    segnet.py:346   Conv3×3 + InstanceNorm + GELU
  ├─ Decoder (resnet18 trunk)          segnet.py:58    stride2 conv → layer1~3 → UpsamplingAdd ×3
  └─ segmentation_head                 simplebev_three_class.py:22   ← 교체한 유일한 부분
        Conv3×3 → InstanceNorm → ReLU → Conv1×1(128→3)
  → logits (B,3,Z,X)
```

**알아둘 것:**

1. **`Y=4`다 (2026-08-27 채택, 그 전에는 `Y=1`이었다).** `Y`의 단일 출처는
   `projects/datasets/simplebev_vox.vox_dims(grid_spec, height_bins)`이고, 기본값이
   `DEFAULT_HEIGHT_BINS = 4` / 범위 −0.25~1.75 m다. **표본 높이는 지면 기준
   0, 0.5, 1.0, 1.5 m**이고 `bev_compressor` 입력이 `feat2d_dim*Y = 512`가 된다.

   `bev_compressor`가 하는 일이 바로 그 **높이 축을 채널로 접어 없애는 것**이다:
   `feat_mem`(B, C, Z, Y, X)을 permute+reshape으로 (B, C·Y, Z, X)로 만든 뒤
   `Conv2d(C·Y → C, 3×3)`로 줄인다. `Y=1`이던 때는 접을 것이 없어 사실상 3×3 conv
   어댑터 하나였다.

   **`height.json`이 없는 옛 체크포인트는 `LEGACY_HEIGHT_BINS = 1`로 읽힌다** --
   2026-08-27 이전 런이 전부 그렇다. 근거는 compendium §12.
2. **어안 투영은 모델이 아니라 `vox_util`에 있다.** submodule을 안 건드리려고 `Vox_util`을
   서브클래싱해 `unproject_image_to_mem` 하나만 오버라이드했다. `Segnet.forward`는 한 줄도
   바뀌지 않았고, 핀홀 전용 행렬 `pixB_T_camA`를 그냥 무시한다(§6.2 참고).
3. **`rand_flip=False`** — 두 그리드 모두 전후 비대칭이라 Simple-BEV의 Z축 flip 증강이
   물리적으로 성립하지 않는다.
4. **안 쓰는 head가 남아 있다** — `feat_head`, `instance_center_head`, `instance_offset_head`.
   loss에 안 들어가므로 gradient는 0이지만 forward 연산은 계속 된다(§6.1).

5. **어디서 공간이 섞이는가 -- 실측** (2026-08-27, 이 문서 §3.1). 정규화·활성을 선형화한
   뒤 출력 한 칸의 gradient가 닿는 입력 범위를 측정한 값이다.

   | 모듈 | 최대 도달 | 유효(gradient 질량 90 %) |
   |---|---|---|
   | lifting | **1 셀** (셀마다 완전 독립) | 1 셀 |
   | `bev_compressor` 3×3 | 3 셀 = 15 cm | 3 셀 = 15 cm |
   | `Decoder` U-Net | 118 셀 = 5.90 m (격자 폭 98 %) | **43 셀 = 2.15 m** |

   **BEV 좌표에서 공간 문맥을 만드는 것은 사실상 디코더뿐이다.** 다만 이것은 BEV 좌표에서만
   참이고, **이미지에서는 인코더가 이미 크게 섞었다** -- 특징맵 1픽셀의 유효 수용영역이
   179 px(입력 512 px 폭의 35 %)다. lifting이 못 하는 것은 "ego에서 이 셀까지의 광선" 같은
   **BEV 격자 구조를 따르는** 관계다.

   FLOPs 배분도 같이 재 두었다(batch 1, 카메라 3대): **인코더 254.62 G (90.3 %)**,
   `bev_compressor` 16.99 G, 디코더 10.45 G. **`bev_compressor` 하나가 디코더 전체보다
   비싸다** -- 512→128 3×3을 120×120 전 해상도에서 돌리기 때문이다.

`load_trunk_weights(:53)`는 shape이 맞는 키만 **명시적으로** 복사하고 나머지를 `skipped`로
리포트한다. `strict=False`로 조용히 넘기지 않으려는 설계이고, `skipped 0`이 곧
"출력 head까지 전이됐다"의 증거다(3-class pretrain → fine-tune에서 실측 확인: `loaded 677, skipped 0`).

---

## 4. 무엇이 학습되는가 (loss)

`projects/common/three_class_metrics.py:105` `run_batch`가 한 스텝의 전부다:

```python
rgb = batch["rgb_camXs"].to(device) - 0.5     # Segnet이 내부에서 +0.5 후 ImageNet 정규화
_, _, logits, _, _ = model(rgb, pix_T_cams, cam0_T_camXs, vox_util)
class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
loss, parts = compute_three_class_loss(logits, class_index, valid_bev_g, class_weights)
```

`compute_three_class_loss(:25)`:

```python
per_cell = F.cross_entropy(logits, class_index, weight=class_weights, reduction="none")
total    = (per_cell * valid).sum() / (valid.sum() + 1e-6)     # ← backward되는 것은 이것뿐
```

- **valid 셀에 한정한 가중 3-class cross-entropy.** auxiliary loss는 없다.
- `.mean()`이 아니라 **`valid.sum()`으로 정규화**한다. 프레임마다 유효 셀 수가 크게 달라
  (로봇은 20.9%) `.mean()`을 쓰면 batch 내 프레임 기여도가 불균등해진다.
- `class_weights`는 train split에서 역빈도로 계산된다(`:47`). 상한 `MAX_CLASS_WEIGHT=20`이
  실제로 작동 중이며 미해결 항목이다 → [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7.

### 4.1 `loss_unknown` / `loss_free` / `loss_occupied`는 total의 분해가 아니다

세 항은 **진단용 로깅**이고 total에 더해지지 않는다. 같은 `per_cell`을 쓰지만 **정규화 분모가
다르다** — total은 valid 전체로 나누고, 각 항은 **자기 클래스 셀 수로만** 나눈다(`:36-41`).
따라서 셋의 합 ≠ total이며, 셋은 "클래스별 평균 난이도"를 뜻한다.

따로 내는 이유: unknown이 셀의 79%(로봇)라 total만 보면 occupied 항이 언제 0으로 죽었는지
보이지 않는다.

### 4.2 최적화 (양쪽 동일)

`train_synwoodscape.py:233-238` / `train_robot_bev.py:271-276`

```python
AdamW(lr, weight_decay)
OneCycleLR(max_lr=lr, total_steps=num_epochs*steps_per_epoch+10,
           pct_start=0.05, anneal_strategy="linear", cycle_momentum=False)
clip_grad_norm_(5.0)
torch.manual_seed(0)   # 하드코딩 -- 다중 seed 비교에는 코드 변경이 필요하다
```

---

## 5. 무엇으로 평가되는가

지표의 출발점은 두 마스크뿐이다(`three_class_metrics.py:68`):

```python
gt   = decompose(seg_g, vis_g, valid_g)                        # gt["free"]
pred = decompose_from_class_index(logits.argmax(dim=1), valid_g)   # pred["free"]
```

**softmax argmax가 FREE인 셀**이 예측 free다. 임계값 튜닝은 없다.

### 5.1 매 배치 (train + val) — `free_space_metrics.py`

| 지표 | 정의 | 비고 |
|---|---|---|
| `iou_free` ↑ | **샘플별** IoU의 평균. union=0인 샘플은 평균에서 제외 | **체크포인트 선택 기준** |
| `fatal_rate` ↓ | `\|pred_free & ~gt_free\| / \|pred_free\|` = **1 − precision**. "갈 수 있다고 믿은 곳 중 틀린 비율" | |
| `free_miss_rate` ↓ | `\|~pred_free & gt_free\| / \|gt_free\|` = **1 − recall**. 보수성(정보 낭비) | |
| `iou_free_known` ↑ | `valid`를 GT 관측 영역(free ∪ occupied)으로 좁힌 `iou_free` | 보고 전용 |
| `iou_occupied` ↑ | occupied 면적 IoU | 보고 전용. **선택에 쓰지 않는다** |
| `iou_unknown` ↑ | unknown 면적 IoU | 보고 전용 |

전부 `free_metrics_from_masks` 한 곳에서 나온다. epoch 집계는 단순 평균이 아니라 **count 가중
평균**이고, IoU마다 count가 **다르다** — 그 클래스가 없는 샘플이 지표별로 다르게 빠지기 때문이다.

`fatal_rate`의 분모를 `|pred_free|`로 잡은 이유는 코드 주석에 있다 — `|~free_gt|`로 잡으면
분모가 격자의 83%라 값이 항상 작아 변별력이 없다(같은 체크포인트에서 0.0113 vs 0.0587).

`iou_occupied`는 **두께 1셀 표면에 면적 IoU를 씌운 것**이라 신뢰하면 안 된다. 로봇 2 epoch
실측에서 0.063이었고 같은 체크포인트의 `f1@20cm`은 0.665다. 문헌 비교용으로만 남겨 둔다.

### 5.2 val에서만 — `bev_occupancy_metrics.py` `evaluate_split`

train에서 안 하는 이유: 광선 루프와 거리변환이 CPU numpy라 매 스텝 돌리면 병목이 된다.

**M3 range error** (`free_space_metrics.py` `range_error` + `polar.py`)
720개 방위각(`DEFAULT_N_THETA=720`)으로 광선을 쏴 각 방향의 "첫 free 이후 첫 non-free까지의
거리" `r(θ)`를 구하고, `dr = r_pred − r_gt`를 모은다. **GT와 예측이 둘 다 `RAY_OK`인 광선만**
넣는다 — censored(격자 끝까지 free)를 r_max로 대체해 섞으면 통계가 그 상수에 눌린다.

- `mae`: `|dr|`의 평균 (m). 단위가 meter라 해석이 직접적이다
- `abs_p50` / `abs_p90`: `|dr|`의 50 / 90 백분위수. 실측 분포가 꼬리가 두꺼워
  (로봇에서 p50 0.150 대 p90 0.610) 평균과 백분위수를 **같이** 본다
- `bias`: `mean(dr)`. **`> 0`이면 장애물을 실제보다 멀다고 예측 = free 과대추정 = 위험한 쪽**
- `over_mean` / `under_mean`: 부호별 조건부 평균(둘 다 양수 크기)
- `missed_obstacle_rate`: **이 지표 없이 위 숫자들을 읽으면 안 된다.** 장애물을 완전히 놓친
  광선(`RAY_CENSORED` 예측)은 위 통계의 표본에서 빠지므로, 많이 놓치면 `mae`가 **좋아진다**.
  로봇 2 epoch에서 0.069 (834/12070 광선)

**거리별 `range_mae`** (`summarize_range_error_by_gt_range`) 광선을 그 광선의 **`r_gt`로**
층화한다. `r_pred`로 묶으면 구간 정의가 모델에 따라 움직여 run 간 비교가 안 된다. M4와 다른
것을 재는 이유: 광선 하나는 여러 링을 지나므로 셀 마스크로는 나눌 수 없다.
로봇 2 epoch 실측: 0.276 / 0.242 / **0.457** m — 원거리 링이 1.9배 나쁘다.

**M4 ring** (`metrics_per_ring`) 거리 링별 `iou_free`/`fatal_rate`. `valid`에 링 마스크를 곱해
같은 함수를 재사용한다. pretrain은 `(0,2,4,6,8) m`, 로봇은 `(0,1.5,3,4) m`.

**occupied `f1@τ`** (`occupied_metrics.py`) τ = 10/20/40 cm. 예측/GT occupied 각각에서 상대
집합까지의 거리변환(`scipy.ndimage.distance_transform_edt`, `sampling=cell_m`)을 만들고 τ
이내를 맞은 것으로 센다. **카운트를 모아 마지막에 한 번 나눈다**(micro-average) — 배치별 F1을
평균하면 프레임당 occupied 셀 수가 수십 배 차이 나 batch 크기에 값이 딸려 간다.
정의와 실측은 [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §2.8이 정본이다.

### 5.3 체크포인트 선택

```python
def select_checkpoint_score(free_metrics):
    return free_metrics["iou_free"]      # bev_occupancy_metrics.py:202
```

`iou_free` 하나뿐이다. 한 줄 함수를 남긴 이유는 주석에 있다 — 두 스크립트가 각자 dict를 직접
읽으면 한쪽만 조용히 바뀔 수 있어서, 테스트가 고정할 이음매를 만든 것이다.

`is_new_best = val_score > best_val_score`이고 `NaN > x`는 항상 False이므로 **val을 돌리지 않은
epoch은 자동으로 제외**된다.

### 5.4 베이스라인 (배너와 매 epoch에 병기)

이 프로젝트의 출발점이 "모델이 이미지를 안 보는 예측기에 지고 있었는데 아무도 몰랐다"이므로
(`archive/free_space_metric_migration.md` §1) 항상 둘을 같이 찍는다:

- **trivial**: "전부 drivable" IoU
- **constant-map**: train split의 셀별 다수결 free map을 val에 채점(`projects/common/baselines.py`)

실측: 로봇 0.398, SynWoodScape **0.866**. SynWoodScape 쪽 여유폭이 좁다는 것이
[`archive/next_session_threeclass_training.md`](archive/next_session_threeclass_training.md) §2.1(b)의 미해결 항목이다.

---

## 6. 읽다가 걸리는 것들

### 6.1 안 쓰는 decoder head는 **제거됐다** (2026-08-18)

원본 `Decoder.forward`(`segnet.py:110`)는 `feat_head` / `instance_center_head` /
`instance_offset_head`를 항상 계산하는데(nuScenes instance segmentation용) 3-class 학습은
`segmentation` 하나만 쓴다. loss에 안 들어가니 gradient는 0이지만 forward 연산은 매 스텝 돌았다.

`ThreeClassDecoder`가 `__init__`에서 세 head를 `delattr`하고 `forward`를 직접 구현한다
(`simplebev_three_class.py:31`). trunk는 원본과 동일하며,
`test_trunk_matches_the_upstream_decoder_after_removing_the_heads`가 원본과 같은 값을 내는지
대조한다. `Segnet.forward`가 5-튜플로 언패킹하므로 dict 키는 남기고 값만 `None`으로 둔다.

실측 효과(2026-08-18, RTX PRO 6000 공유 상태):

| | decoder forward | 제거 후 | 절감 |
|---|---|---|---|
| pretrain 240×240 bs16 | 68.3 ms | 38.3 ms | **43.9%** |
| fine-tune 120×120 bs8 | 10.0 ms | 6.7 ms | 33.1% |

- **파라미터 9개 텐서 / 459,267개(decoder의 12.0%) 제거.** fine-tuning의 weight transfer가
  `loaded 677` -> `loaded 668`로 줄어든 것이 그 확인이다(9 = 3 head × conv 2개+bias 1개).
- 학습 스텝 전체는 876 ms(bs16)이므로 학습 시간 절감은 약 3%. **주된 이득은 임베디드 배포**다.
- `__init__`이 `super().__init__()`으로 세 head를 만든 **뒤** 삭제하므로 파라미터 초기화
  RNG 소비량이 원본과 같다 -- 즉 이 변경이 초기 가중치를 바꾸지 않는다.
- `load_trunk_weights`의 skip 판정은 `unexpected_skips` / `head_was_transferred`로 옮겼다.
  제거 이전에 만든 체크포인트를 로드하면 그 9개가 `skipped`에 뜨는데, 그것은 정상적인 skip
  이므로 `RuntimeError`를 내지 않는다.

### 6.2 `pix_T_cams`는 실질적으로 쓰이지 않는다

`pix_T_cams`는 **카메라 좌표 → 픽셀 좌표 내부 파라미터(intrinsics) 4×4 행렬**이다. 핀홀
카메라에서 `pixel = K · point`를 하는 그 K다. `Segnet.forward`는 이것을 해상도에 맞춰
스케일한 뒤(`segnet.py:415`) 외부 파라미터와 곱해 `pixB_T_camA`로 `unproject_image_to_mem`에
넘긴다.

**그런데 어안 렌즈는 원근분할로 표현되지 않으므로 이 행렬로는 투영할 수 없다.** 그래서
`FisheyeVoxUtil`/`DoubleSphereVoxUtil`은 `pixB_T_camA`를 **무시하고**, 강체변환인
`camB_T_camA`(외부 파라미터)만 받아 렌즈 모델로 직접 픽셀 좌표를 계산한다
(`fisheye_vox.py:89-107`). 따라서:

- 로봇: `pix_T_cams = torch.eye(4)`(`robot_simplebev.py:202`). 형상만 맞춰 넘기는 자리표시자다.
- SynWoodScape: 핀홀 근사를 만들지만(`:64`) `use_fisheye=True`인 한 쓰이지 않는다.
  **단 `--use_fisheye=False`로 돌리면 원본 `Vox_util`이 이 값을 실제로 쓴다** — 그 비교 실험
  경로가 남아 있으므로 값을 지우면 안 된다.

`Segnet.forward`의 시그니처가 요구하므로 인자 자체를 없앨 수는 없다(submodule 수정 금지).

### 6.3 `range_abs_p50` / `abs_p90`의 batch-size 의존은 **고쳐졌다** (2026-08-18)

옛 `summarize_range_error`는 **배치별 백분위수를 `n_paired_rays`로 가중평균**했다. 배치별
중위수의 가중평균은 전체 분포의 중위수가 아니므로, 같은 모델·같은 데이터인데 batch를 어떻게
자르느냐로 숫자가 달라졌다(bs4 0.177 vs bs8 0.172, p90 0.701 vs 0.718).

이제 `range_error`가 `dr` 표본(`deltas`)을 그대로 실어 보내고,
`summarize_range_error`가 **전부 모은 뒤 한 번만** 통계를 낸다(`free_space_metrics.py`).
`over_mean`/`under_mean`도 같은 표본에서 직접 내므로, 부분집합 평균을 `over_count`로 가중해야
한다는 별도 규칙 자체가 없어졌다.

같은 체크포인트를 재채점한 실측(수정 후):

| batch_size | 4 | 8 | 16 |
|---|---|---|---|
| `abs_p50` | 0.175 | 0.175 | 0.175 |
| `abs_p90` | 0.700 | 0.700 | 0.700 |
| `iou_free` / `fatal` / `free_miss` / paired rays | 전부 일치 | 전부 일치 | 전부 일치 |
| `over_mean` | 0.416 | 0.417 | 0.417 |

**남은 `over_mean`의 0.001 차이는 지표가 아니라 모델 forward 탓이다.** 같은 체크포인트를
bs4와 bs8로 추론했을 때 argmax가 다른 셀이 532,800개 중 **11개(0.002%)** 있다 -- cuDNN이
batch size에 따라 다른 conv 알고리즘을 고르면서 생기는 부동소수 차이다. 집계 자체는
`test_summarize_range_error_does_not_depend_on_how_the_samples_are_batched`가 동일 표본에서
**정확한 일치**로 고정한다. 같은 batch size로는 재현된다(bs4 두 번 모두 0.416).

> 참고: 이 수정은 `iou_free` 정의를 건드리지 않으므로 재학습이 필요 없었다. 다만 §9.4의
> 옛 range 숫자(0.172 / 0.718)는 이제 새 정의로 재채점해야 비교 가능하다.

### 6.4 `bev_occupancy_metrics.py`라는 이름이 낡았다

지금 내용은 3-class 로깅·집계·체크포인트 선택이지 occupancy 지표가 아니다. Task 16에서 방금
개칭한 모듈을 하루 만에 또 바꾸면 git 이력에서 이름이 셋이 되므로 의도적으로 그대로 뒀다.

---

### 6.5 upstream `Decoder.forward` 주석의 해상도가 2배 틀렸다 (2026-08-27)

`segnet.py`의 주석은 `layer1` 다음을 `(H/4, W/4)`로 적어 두었지만, 이 디코더는
`first_conv`(stride 2) 뒤에 **`maxpool`을 쓰지 않는다.** 실제로는 `H/2`다.
skip 연결이 맞물리는 것으로 확인된다 -- 120 → 60 → 30 → 15 → 30 → 60 → 120.

실측 형상(입력 `(1,128,120,120)`):

```
first_conv (1,64,60,60)   layer1 (1,64,60,60)    layer2 (1,128,30,30)
layer3     (1,256,15,15)  up3_skip (1,128,30,30) up2_skip (1,64,60,60)
up1_skip   (1,128,120,120)                       segmentation_head (1,2,120,120)
```

**주석 대신 이 표를 쓴다.** `third_party/`는 수정 금지이므로 주석은 그대로 남아 있다.

### 6.6 카메라 병합 마스크는 복셀별이 아니라 **채널별**이다 (2026-08-27)

`Segnet.forward`가 `mask_mems = (torch.abs(feat_mems) > 0).float()`로 마스크를 만들고
`reduce_masked_mean(feat_mems, mask_mems, dim=1)`을 부른다. 마스크가 `feat`과 **같은 shape**
이므로 평균이 `(채널, 복셀)`마다 따로 계산된다. 실측:

```
cam0 = [1,2,3,4]   cam1 = [5,6,0,8]   cam2 = 화각 밖(전부 0)
결과 = [3.0, 4.0, 3.0, 6.0]
                    ↑ ch2에서만 cam1이 빠졌다 -- 값이 정확히 0이라서
```

즉 `abs(feat) > 0`은 **"화각 밖"과 "특징값이 우연히 0"을 구분하지 못한다.** 정확한 0은
드물어 실질적으로는 복셀별 카메라 수처럼 동작하지만, 계약상으로는 새는 구멍이다.
어느 카메라도 못 본 복셀은 `denom = EPS + 0`으로 나눠 **0이 되고 NaN은 아니다**.

**병합이 예외가 아니라 다수 경로다.** 캘리브레이션 기하로 세면 확정 config에서 그 셀을
보는 카메라가 2대인 경우가 **62 %**, 3대가 21 %, 1대가 15 %, 0대가 0.2~4.5 %다
(`tools/measure_height_bin_visibility.py --y_min=-0.25 --y_max=1.75 --n_bins=4`).
평균에 **가중이 없다** -- 비스듬히 스치는 카메라와 정면으로 보는 카메라가 같은 가중을 받는다.

## 7. 코드를 직접 따라갈 순서

한 번에 다 읽지 말고 이 순서로:

```
1. tools/train_synwoodscape.py:153-206          main 시작 ~ 배너 (무엇을 계산하고 시작하는지)
2. projects/datasets/synwoodscape_simplebev.py:84-110    __getitem__ = batch dict의 정의
   ↔ projects/datasets/robot_simplebev.py:137-166        마스킹 계약 (여기가 핵심)
3. projects/common/free_space.py                전체 62행. 3-class 정의의 단일 출처
4. projects/common/three_class_metrics.py:105-120   run_batch = 한 스텝 전부
   → :25-45                                     loss
5. projects/models/simplebev_three_class.py     전체 75행. 무엇만 바뀌었는지
   → third_party/models/simple_bev/nets/segnet.py:375-470   Segnet.forward (원본, 읽기만)
6. projects/models/fisheye_vox.py:80-123        투영이 실제로 일어나는 30줄
7. projects/common/free_space_metrics.py:27-93  M1/M2/M2b
8. projects/common/bev_occupancy_metrics.py:245-276   evaluate_split
9. tools/train_synwoodscape.py:272-330          학습 루프 (여기까지 오면 다 보인다)
```

`tools/train_robot_bev.py`는 3~9가 전부 같으므로 **`:144-269`(데이터 준비 + `init_checkpoint`)만**
따로 읽으면 된다.

---

## 8. 관련 문서

- loss 미해결 항목: [`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §1.7
- 지표 근거 정본: [`archive/free_space_metric_migration.md`](archive/free_space_metric_migration.md)
- 라벨 계약 정본: [`finetuning_guide.md`](finetuning_guide.md) §1.3
- 현재 작업 인수인계: [`archive/next_session_threeclass_training.md`](archive/next_session_threeclass_training.md)
