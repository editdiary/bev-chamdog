# Phase 4 Fine-Tuning 준비 문서

작성: 2026-08-14

> **[상태: 완료 — 기록용 문서다]**
>
> 이 문서는 데이터 실물을 받기 **전에** 쓴 계획서다. 여기서 "만들어야 한다"고 적은 것은
> 같은 날 전부 구현·검증됐고, §6의 미결 질문도 모두 답이 나왔다(§8 참고).
>
> **실제로 fine-tuning을 돌릴 때는 [`docs/finetuning_guide.md`](../finetuning_guide.md)를 본다.**
> 이 문서는 "왜 그렇게 만들었는지"의 근거로만 남긴다.

SynWoodScape pretraining(Phase 3)에서 자체 온실 데이터셋 fine-tuning(Phase 4)으로 넘어가기 위해
**무엇이 준비돼 있고, 무엇을 만들어야 하며, 무엇을 결정해야 하는지**를 정리한다.

관련 문서: `ROADMAP.md` Phase 4, `docs/BEV_loss_and_metrics_design.md`(fine-tuning 기준 loss/지표
설계), `docs/archive/synwoodscape_pretrain_experiment_log.md`(pretrain 실험 기록).

---

## 1. Pretrain이 넘기는 것

```text
runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth
```

| | 값 |
|---|---:|
| val obstacle IoU (GT visibility 마스킹) | 0.8607 |
| val drivable IoU | 0.9889 |
| `missed_obstacle` | 0.0380 |
| `deploy_iou_obstacle` @ coverage 0.9369 | 0.8510 |

Phase 3의 목적은 성능 최적화가 아니라 "어안 4-cam → BEV occupancy 파이프라인이 동작함을 증명"
하는 것이었고(`ROADMAP.md`), 그건 달성됐다. 남은 train/val 격차 0.095는 train 400장의 장면
다양성 한계이며 loss·정규화로는 풀리지 않는다는 것이 실험으로 확인됐다.

---

## 2. 체크포인트 이식성 — 검증 완료

**그리드 크기가 달라도 가중치는 그대로 로드된다.** 실측했다.

| | pretrain | robot (fine-tune) |
|---|---|---|
| ROI | 전방 8 / 후방 4 / 좌우 ±6 m | 전방 4 / 후방 2 / 좌우 ±3 m |
| 격자 | 240 × 240 | 120 × 120 |
| cell | 0.05 m | 0.05 m |

```text
checkpoint tensors: 680
robot grid 120x120 model params: 680
missing   : 0
unexpected: 0
shape mismatches: 0
```

이유: `Segnet`은 Z·X 축에 대해 완전 합성곱이다. Y축만 `bev_compressor`에서 채널로 접히는데
(`segnet.py:327-347`의 `Conv2d(feat2d_dim*Y, ...)`), 우리는 pretrain·fine-tune 모두 `Y=1`이다.
240과 120 모두 decoder의 8배 업샘플로 나눠떨어진다.

**카메라 수도 자유롭다.** Simple-BEV는 카메라별 전용 파라미터가 없고 `S`를 배치 차원으로 접는다.
4-cam → 3-cam 전환도 구조적으로 막히지 않는다.

> 확인 필요: 자체 데이터셋이 4-cam인지 3-cam인지. `AGENTS.md`와 설계 스펙은 **4-cam**이라고
> 하지만 `projects/datasets/synwoodscape_simplebev.py`의 docstring은 "fine-tune 3-cam"을
> 언급한다. §6 참고.

---

## 3. 지금 코드가 이미 fine-tuning 설계에 맞는 부분

`docs/BEV_loss_and_metrics_design.md`가 요구하는 계약과 현재 구현을 대조하면 이렇다.

| 설계 문서 요구사항 | 현재 상태 |
|---|---|
| occupancy는 **관측된 셀에서만** 학습 (`vis_gt`를 per-cell weight로) | ✅ `w_occ = vis_gt * valid` (`simplebev_two_head.py:84`) |
| visibility는 **전 격자에서** 학습 (마스킹하지 않음) | ✅ `w_vis = asymmetric_weight * valid` — 2026-08-14 수정으로 성립 |
| visibility 과대예측이 더 위험하므로 비대칭 loss | ✅ `vis_neg_weight` — 같은 수정으로 비로소 실제 동작 |
| "map 데이터 없음"을 loss에서 제외 | ✅ `valid_bev_g`가 그 역할. SynWoodScape는 전부 1 |
| visible-only ↔ amodal 전환이 플래그 하나 | ✅ occupancy 마스크를 `vis*valid` → `valid`로 바꾸면 됨 |

**2026-08-14의 `valid`/`vis` 분리가 결과적으로 fine-tuning 설계를 그대로 구현한 셈이다.**
설계 문서의 `occ_gt ∈ {0, 1, 255}`에서 `255`가 정확히 `valid=0`이고, `vis_gt ∈ [0,1]`이
`vis_bev_g`다. 이 수정 없이 fine-tuning에 들어갔다면 visibility head가 죽은 채로 온실
데이터에 옮겨갔을 것이다.

---

## 4. 만들어야 하는 것

### 4.1 데이터셋 어댑터 (필수)

`projects/datasets/robot_simplebev.py`. `SynWoodScapeSimpleBEVDataset`과 **같은 6개 키**를
반환하면 학습 루프는 그대로 쓸 수 있다.

| 키 | 형태 | dtype | 의미 |
|---|---|---|---|
| `rgb_camXs` | (S, 3, H, W) | float32 | `[0, 1]` 범위. 정규화는 학습 루프가 함 |
| `pix_T_cams` | (S, 4, 4) | float32 | 핀홀 근사용. `use_fisheye=True`면 투영에 미사용 |
| `cam0_T_camXs` | (S, 4, 4) | float32 | ref 프레임 기준 카메라 외부 파라미터 |
| `seg_bev_g` | (1, Z, X) | float32 | drivable=1 / obstacle=0 |
| `vis_bev_g` | (1, Z, X) | float32 | 관측 신뢰도. **soft 값 `[0,1]` 허용** |
| `valid_bev_g` | (1, Z, X) | float32 | 라벨이 존재하는 셀 |

설계 문서의 라벨에서 변환:

```python
seg_bev_g   = (occ_gt == 1)                    # 255는 여기서 drivable로 새면 안 됨
valid_bev_g = (occ_gt != 255) & ~ego_mask      # ego footprint 제외
vis_bev_g   = vis_gt                           # soft 값 그대로
```

`static_fov_mask`를 `vis_bev_g`에 곱할지 `valid_bev_g`에 곱할지는 결정 사항(§6).

### 4.2 soft visibility 대응 (필수)

현재 코드는 대부분 soft 값을 그대로 받지만 **두 군데가 binary를 전제**한다.

`tools/train_synwoodscape.py:128-149` — `compute_pos_weight`, `compute_trivial_baseline_iou`:

```python
occupancy = np.load(...).astype(bool)   # 255 -> True (drivable로 오인)
visible   = np.load(...).astype(bool)   # 0.01 -> True (거의 안 보이는 셀도 포함)
```

로봇 라벨에서는 둘 다 틀린다. `occ_gt == 1`과 `vis_gt` 가중합으로 바꿔야 한다. `pos_weight`가
틀리면 클래스 불균형 보정이 통째로 어긋나므로 **fine-tuning 전에 반드시 고쳐야 하는 항목**이다.

`projects/models/simplebev_two_head.py:95-99` — `asymmetric_weight`가 `vis_target > 0.5`로
이분한다. 동작은 하지만 soft 라벨의 중간값 구간을 계단으로 처리한다. 설계 문서가 말하는
"잎 커튼 뒤" 반투과 구간이 정확히 여기라, 연속 가중으로 바꿀지 결정이 필요하다(§6).

`visibility_error_rates`도 0.5로 이분한다. 설계 문서는 중간값 셀에 대한 MAE를 추가로 제안한다.

### 4.3 렌즈 모델 교체 가능화 (자체 카메라가 `radial_poly`가 아닐 때만)

`projects/models/fisheye_vox.py:102`가 `radial_poly_pixel_coords`를 직접 호출하고,
`set_camera_calibrations`(`:63-78`)가 `lens.coefficients[:4]`, `cx`, `cy`, `aspect_ratio`를
하드코딩한다. Kannala-Brandt/OpenCV fisheye 등이면 이 두 곳을 투영 함수 주입 형태로 바꿔야 한다.
`ROADMAP.md` Phase 4가 이미 예고한 항목이다.

**자체 카메라 캘리브레이션이 어떤 모델인지 확인이 먼저다**(§6). `radial_poly`면 이 작업은 불필요하다.

### 4.4 학습 스크립트 (필수)

`tools/train_robot_finetune.py` 또는 `tools/train_synwoodscape.py`에 데이터셋/그리드 선택
인자를 추가. 후자가 중복이 적지만 이름이 오해를 부른다. 지표·로깅·체크포인트 로직은 그대로
재사용 가능하다.

추가로 필요한 인자: `--init_ckpt`(pretrain 가중치 경로), `--freeze_encoder`(선택).

> **구현 결과:** 별도 스크립트 **`tools/train_robot_bev.py`**로 만들었다(이 문서가 제안한
> `train_robot_finetune.py`가 아니다). 중복을 피하려고 지표·로깅은
> `projects/common/two_head_metrics.py`로 추출해 양쪽이 공유한다 — 스크립트끼리 import하면
> pretrain 쪽을 손댈 때 fine-tune이 깨지기 때문이다.
> 인자는 `--init_checkpoint`로 이름이 바뀌었고, **`--freeze_encoder`는 아직 없다**
> (`docs/finetuning_guide.md` §9).

### 4.5 온실 기준 지표 (권장)

현재 지표는 대부분 그대로 유효하지만, **클래스 비율이 뒤집힌다**는 점을 감안해야 한다.
pretrain은 `obst_frac 0.125`(obstacle 소수), 온실은 반대다.

- `empty_false_alarm`은 "obstacle 없는 샘플"을 대상으로 하는데, 온실에서는 그런 샘플이 거의
  없을 것이다. 대신 **drivable이 없는 샘플**이 생길 수 있어 대칭 지표가 필요할 수 있다.
- bin 경계(`tiny ≤0.01`, `small ≤0.05`, `medium ≤0.15`, `large` 나머지)는 obstacle 비율
  기준이라 온실에서는 대부분 `large`에 몰린다. **drivable 비율 기준 bin으로 뒤집는 것**이
  정보량이 크다.
- 설계 문서는 거리대별 분리 평가를 강조한다("합쳐서 하나의 mIoU만 보면 근거리의 쉬운 성능이
  원거리 실패를 가린다"). 현재 없는 축이고, 온실에서는 추가 가치가 클 것으로 보인다.

---

## 5. 권장 진행 순서

1. **자체 데이터셋 실물 확인** — 포맷, 샘플 수, 캘리브레이션 모델, 카메라 수. §6의 질문들이
   여기서 대부분 해소된다.
2. **데이터셋 어댑터 + 계약 테스트** — 기존
   `tests/datasets/test_synwoodscape_two_head_contract.py`와 같은 방식으로, `valid`/`vis`가
   분리돼 있는지, 255가 새지 않는지를 테스트로 고정한다. pretrain에서 이 계약이 조용히
   깨졌던 전례가 있다.
3. **`pos_weight`/trivial baseline의 soft·255 대응 수정** — 어댑터와 같은 커밋.
4. **smoke run** — 소수 샘플로 shape·마스크·지표가 살아 있는지 확인. pretrain에서 쓰던
   `--max_samples` 방식 그대로.
5. **pretrain 가중치 로드 후 짧은 fine-tune** — 먼저 "스크래치 대비 나은가"만 확인. 여기서
   pretrain의 실제 가치가 처음으로 측정된다.
6. **지표 재설계** — 클래스 비율 역전과 거리대별 평가 반영.
7. **augmentation 결정** — 사용자가 논문·이론 검토 후. pretrain에서 확인된 사실:
   photometric은 안전하고 약한 이득, `rand_flip`은 미시도, 이미지 공간 기하 변환은
   calibration을 같이 고치지 않으면 위험.

**5번에서 pretrain 체크포인트의 가치를 반드시 대조군과 함께 재야 한다.** 스크래치 학습과
비교하지 않으면 pretraining이 실제로 도움이 됐는지 알 수 없다. Phase 3의 최종 검증이기도 하다.

---

## 6. 결정·확인이 필요한 사항

코드를 쓰기 전에 답이 필요한 것들이다.

**데이터 실물**
1. 자체 데이터셋의 실제 위치와 포맷은? (`dataset/`에 아직 없다)
2. 샘플 수는? pretrain은 500장이었고 400/100으로 나눴다.
3. 카메라 수 4개인가 3개인가? 문서가 엇갈린다(§2).
4. 캘리브레이션 렌즈 모델이 `radial_poly`인가? 아니면 §4.3 작업이 추가된다.

**라벨 규약**
5. `occ_gt`에 `255`(map 데이터 없음)가 실제로 존재하는가, 아니면 전 격자가 라벨돼 있는가?
6. `vis_gt`가 설계 문서대로 soft `[0,1]`인가, 아니면 binary인가?
7. `ego_mask`와 `static_fov_mask`가 산출물에 포함돼 있는가? `static_fov_mask`는 `vis`에
   곱할 것인가 `valid`에 곱할 것인가?
8. 온실 기준 drivable 정의는 확정됐는가? (`ROADMAP.md`가 "새로 정의한다"고만 적고 있다)

**학습 설계**
9. fine-tuning ROI를 `ROBOT_GRID_SPEC`(전방 4 / 후방 2 / ±3 m, 120×120) 그대로 갈 것인가?
10. encoder를 freeze할 것인가? 온실 데이터가 적다면 부분 freeze가 과적합을 줄일 수 있다.
11. soft visibility의 비대칭 가중을 계단(현행)으로 둘 것인가 연속으로 바꿀 것인가?

---

## 7. 지금 상태 요약

| 항목 | 상태 |
|---|---|
| pretrain 체크포인트 | ✅ 확보 |
| 120×120 그리드로 가중치 이식 | ✅ 실측 검증 (0 missing / 0 mismatch) |
| `valid`/`vis` 계약 | ✅ fine-tuning 설계와 일치 |
| soft visibility loss | ✅ 대부분 동작, 가중 방식만 결정 필요 |
| two-head 구조 유지 근거 | ✅ A/B로 확보 (occupancy 손해 0) |
| 지표 인프라 | ✅ 재사용 가능, 클래스 역전 대응만 필요 |
| 데이터셋 어댑터 | ✅ `projects/datasets/robot_simplebev.py` |
| `pos_weight` 마스킹 대응 | ✅ `compute_label_statistics` (마스킹 후 실측) |
| 렌즈 모델 교체 가능화 | ✅ DS 전용 모듈 신설 (아래 §8) |
| fine-tune 학습 스크립트 | ✅ `tools/train_robot_bev.py` |

---

## 8. §6 질문에 대한 답 (2026-08-14 확정)

데이터 실물(`dataset/sj_datasets/`)과 어노테이션 프로젝트 코드를 받아 전부 해소됐다.

| # | 질문 | 답 |
|---|---|---|
| 1 | 데이터 위치·포맷 | `dataset/sj_datasets/<시퀀스>/`, 공유 자산은 `common/` |
| 2 | 샘플 수 | raws1 38장. 시퀀스 단위로 계속 추가 예정 |
| 3 | 카메라 수 | **3개** (front/left/right). rear는 캘리에만 있고 라벨 생성에 안 쓰였다 |
| 4 | 렌즈 모델 | **Double Sphere.** `radial_poly`가 아니라 별도 모듈을 신설했다 |
| 5 | `occ_gt`에 255 | **없다.** 전 격자가 `{0,1}`로 라벨돼 있다 |
| 6 | `vis_gt` soft 여부 | **binary.** soft 대응 코드는 불필요했다 |
| 7 | `ego_mask` / `static_fov_mask` | 라벨에는 없다. `common/`의 두 PNG + 캘리브레이션에서 계산한 화각 원반으로 학습 코드가 만든다. **`vis`에 곱하는 것과 `valid`에 곱하는 것을 구분**했다(가이드 §3) |
| 8 | 온실 drivable 정의 | 수동 어노테이션 기준. `0=obstacle / 1=drivable` |
| 9 | fine-tuning ROI | `ROBOT_GRID_SPEC` 그대로. 라벨이 이미 120×120으로 같은 ROI다 |
| 10 | encoder freeze | **미결.** 옵션이 아직 없다(가이드 §9) |
| 11 | soft visibility 가중 | 해당 없음 — `vis_gt`가 binary라 논점이 사라졌다 |

가장 영향이 컸던 것은 4번과 7번이다. 4번 때문에 `projects/geometry/double_sphere.py`와
`projects/models/double_sphere_vox.py`를 새로 만들었고, 7번 때문에 마스킹을 두 종류로
나누는 설계가 나왔다.

이 문서를 쓸 때 예상하지 못했던 것도 있다: **클래스 비율이 예상과 반대**였다. 그리드 전체
obstacle 비율은 pretrain 17% → 자체 23%로 늘지만, loss가 실제로 보는 마스킹 영역
안에서는 12.4% → 5.4%로 **줄어든다**(온실 통로에서 raycast visibility가 장애물에 닿으며
멈춰 장애물 대부분이 경계 바깥에 놓인다). §4.2가 전제했던 "클래스 역전 대응"은 방향이
반대였던 셈이다.
