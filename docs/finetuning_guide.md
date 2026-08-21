# 자체 데이터셋 Fine-Tuning 실행 가이드

어노테이션이 끝난 뒤 실제로 fine-tuning을 돌릴 때 순서대로 따라가는 문서다.
설계 근거와 검증 과정은 `docs/archive/finetuning_preparation.md`(준비 단계 기록)와
`docs/archive/synwoodscape_pretrain_experiment_log.md`(pretrain 실험 기록)에 있다.

파이프라인은 2026-08-14에 raws1(38장)로 끝까지 검증했다 — 학습·평가·시각화가 모두 돌고,
캘리브레이션 체인은 어노테이션 프로젝트의 IPM 출력을 재현하는 것으로 확인했다. **남은 것은
데이터 물량뿐이다.**

---

## 1. 데이터셋 준비

### 1.1 디렉터리 구조

```
dataset/sj_datasets/
  common/                              <- 모든 시퀀스가 공유. 한 번만 만든다.
    calibration/
      calib.yaml                       <- DS intrinsic + extrinsic
      orientation.json                 <- cam<idx> -> 이름 매핑
      ds_model.py                      <- 원본 구현(테스트 대조용, 학습에는 안 쓰임)
    self_mask.png                      <- 120x120, ego 테이블이 가리는 영역
    rear_self_box.png                  <- 120x120, 카트 손잡이 + 미는 사람 영역
  raws1/                               <- 시퀀스 하나
    rgb_images/sample_000000/cam_front.jpg
                             cam_left.jpg
                             cam_right.jpg
    occupancy_npy/sample_000000.npy    <- uint8 (120,120), 0=obstacle 1=drivable
    visibility_npy/sample_000000.npy   <- uint8 (120,120), 0=unseen 1=visible
  raws2/                               <- 같은 구조로 계속 추가
  raws3/
```

시퀀스 폴더 이름은 자유지만 `occupancy_npy/`가 있어야 시퀀스로 인식된다.
`labels.csv`, `*_png/`, `review_png/`는 학습에 쓰이지 않는다(사람이 보는 용도).

### 1.2 새 시퀀스를 추가할 때 확인할 것

- [ ] `occupancy_npy`, `visibility_npy`, `rgb_images`의 sample 개수가 같은가
- [ ] `.npy`가 `(120, 120)` uint8이고 값이 `{0, 1}`뿐인가
- [ ] 각 `rgb_images/<sample_id>/`에 `cam_front/left/right.jpg` 세 장이 다 있는가
- [ ] **카메라 장착과 캘리브레이션이 raws1과 같은가.** 리그를 건드렸다면 §2의 재검증이
      필요하고, `common/`을 시퀀스별로 분리해야 한다
- [ ] 그리드 ROI가 같은가 (전방 4 m / 후방 2 m / 좌우 ±3 m, 0.05 m/cell)

빠르게 확인:

```bash
python -c "
import numpy as np, glob
from pathlib import Path
for seq in sorted(Path('dataset/sj_datasets').glob('raws*')):
    occ = sorted(glob.glob(f'{seq}/occupancy_npy/*.npy'))
    vis = sorted(glob.glob(f'{seq}/visibility_npy/*.npy'))
    rgb = sorted(glob.glob(f'{seq}/rgb_images/*/'))
    shapes = {np.load(f).shape for f in occ}
    values = set()
    for f in occ[:5]: values |= set(np.unique(np.load(f)).tolist())
    print(f'{seq.name}: occ {len(occ)} vis {len(vis)} rgb {len(rgb)} | shape {shapes} | values {sorted(values)}')
"
```

세 숫자가 같고 shape이 `{(120, 120)}`, values가 `[0, 1]`이면 된다.

### 1.3 라벨 규약

| | 의미 |
|---|---|
| `occupancy_npy` | `0` = obstacle, `1` = drivable. **unknown 클래스 없음** |
| `visibility_npy` | `0` = 가려짐, `1` = 관측 가능. BEV 2D raycast 기준 |
| `occupancy_3class_npy` | `0` = free, `1` = occupied, `2` = unknown. **검수 전용 — 학습 입력이 아니다**(아래) |

`visibility`가 카메라 화각을 담고 있지 않다는 점이 중요하다 — 그건 학습 코드가
`common/`의 마스크와 캘리브레이션으로 런타임에 보정한다(§3).

**정본은 `occupancy_npy` 하나다.** `visibility_npy`는 거기서 raycast로 파생한 것이고
(`raws1/README.md`), `occupancy_3class_npy`는 그 둘의 재조합이다. 실측으로 확인했다:
`np.where(vis, np.where(occ, free, occupied), unknown)`과 raws1 38/38 프레임이 완전히 일치한다.
정보가 늘지 않으므로 **파일이 늘어도 로더가 할 일은 줄지 않는다.**

#### `occupancy_3class_npy`를 학습 입력으로 쓰면 안 되는 이유

이 파일은 라벨의 원시 `visibility`만 쓰므로 §3의 영구 사각(`permanent_blind`, 5.6%)이
빠져 있다. 학습 코드가 실제로 쓰는 값과 대조하면 **raws1 38프레임 중 0프레임만 일치**하고
셀의 4.97%가 어긋난다 — 4.60%는 `free`로, 0.36%는 `occupied`로 찍혀 있지만 실제로는
`unknown`이어야 하는 셀이다. 그대로 학습하면 **카메라가 물리적으로 볼 수 없는 발밑 원반에서
"여기는 주행 가능"을 예측하라고 요구**하게 된다. 하필 로봇이 곧바로 진입하는 구간이다.

`permanent_blind`를 `unknown`으로 덮어 구우면 값은 학습 코드와 38/38 일치한다(실측).
그래도 정본으로 승격하지 않는 이유는 두 가지다:

1. **`valid`를 담을 수 없다.** `rear_self_box`(384셀) 중 `permanent_blind`와 겹치는 건
   126셀(32.8%)뿐이고, **나머지 258셀(67%)은 카메라가 실제로 담는 영역**이다. 배포 때
   카트와 사람이 없으면 관측 가능한 곳이라 `unknown`(학습하는 클래스)과 의미가 다르다.
   4번째 값을 넣는 순간 그건 3-class가 아니고, 배포에 없는 수집 아티팩트를 라벨에 굽는 셈이다.
2. **`permanent_blind`는 리그의 함수다.** 캘리브레이션·카메라 대수·그리드 스펙 중 하나만
   바뀌어도 전 시퀀스를 재생성해야 한다. 런타임 계산은 `Dataset.__init__`에서 한 번
   실행되므로(샘플마다가 아니다) 구워서 아끼는 런타임 비용은 0이다.

**권장 용법**: 검수용 파생 산출물로 두고, `occupancy_3class_png`를 어노테이션 QA에 쓴다.
단 지금 PNG는 발밑 원반을 `free`로 보여주는데 실제 학습되는 값은 `unknown`이라 눈으로 보는
것과 모델이 배우는 것이 다르다 — QA에 쓰려면 `permanent_blind`를 덮어 구운 버전이 낫다.
로더가 이 파일을 직접 읽게 하려면 정본에서 다시 계산해 일치하는지 assert하는 검증을 붙인다.
(2026-08-17 기준 `raws1`에만 있다. raws2·raws3·rawos1에는 없다.)

---

## 2. 좌표계와 캘리브레이션 — 검증 완료, 건드리지 말 것

리그를 바꾸지 않는 한 이 절은 읽기만 하면 된다.

**ego 프레임**: X 전방 / Y 좌측 / Z 상방, **원점은 지면**.
`projects/datasets/simplebev_vox.py`가 원점이 지면이라고 전제하며(`Y=1`이면 정확히
`ego z=0` 한 평면에서 이미지 특징을 뽑는다), pretrain 체크포인트도 그 규약으로 학습됐다.

**extrinsic 체인** (`projects/geometry/double_sphere.py`):

```
cam_T_lidar = T_cam_front @ T_front_lidar          # 둘 다 calib.yaml에 있다
ego_T_cam   = Translate(0, 0, +0.87) @ inv(cam_T_lidar)
```

`calib.yaml`의 extrinsic은 전부 front 카메라 기준이고, 라벨 파이프라인이 ego = body =
LiDAR를 쓰므로 `T_front_lidar`가 곧 ego extrinsic이다. LiDAR 원점이 지면 위 0.87 m라
z 평행이동 하나만 더 붙는다.

**`GROUND_Z_IN_LIDAR_M = -0.87`의 근거**: 어노테이션 프로젝트 안에서 `ipm.py`는 지면을
−0.8921로, `slab_label.py`는 −0.87로 잡아 2.2 cm 어긋나 있다. 배포된
`common/self_mask.png`를 두 값으로 역투영해 대조하면 −0.87 쪽이 맞는다(IoU 0.606 vs 0.498).
라벨과 같은 규약을 따라야 하므로 −0.87을 쓴다.

### 리그를 바꿨다면 — 재검증 방법

IPM을 렌더링해 어노테이션 프로젝트의 `review_png`와 눈으로 대조한다. 모델과 무관하게
캘리브레이션만 검증하는 방법이라 가장 확실하다.

```bash
python -c "
import numpy as np
from PIL import Image
from projects.bev_gt.grid import ROBOT_GRID_SPEC as G
from projects.bev_gt.ipm import render_ipm
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES, load_cameras, load_ego_T_cams
C='dataset/sj_datasets/common/calibration/calib.yaml'
S='dataset/sj_datasets/raws1'; sid='sample_000000'
imgs={n: np.asarray(Image.open(f'{S}/rgb_images/{sid}/cam_{n}.jpg').convert('RGB')) for n in FINETUNE_CAMERA_NAMES}
ipm=render_ipm(imgs, load_cameras(C), load_ego_T_cams(C), G)
occ=np.load(f'{S}/occupancy_npy/{sid}.npy'); o=ipm.copy()
o[occ==0]=(0.5*o[occ==0]+0.5*np.array([230,90,90])).astype(np.uint8)
Image.fromarray(np.concatenate([ipm,o],1)).resize((960,480), Image.NEAREST).save('/tmp/ipm_check.png')
print('wrote /tmp/ipm_check.png')
"
```

**GT obstacle(빨강)이 IPM에 보이는 실제 벽·작물 위에 얹히면 통과다.** 어긋나 있으면
extrinsic이나 지면 높이가 틀린 것이고, 그대로 학습하면 loss는 잘 내려가는데 IoU만 조용히
나빠져서 발견이 늦는다.

자동 회귀 테스트도 있다: `pytest tests/geometry/test_double_sphere.py tests/models/test_double_sphere_vox.py`

---

## 3. 마스킹 — 두 종류를 구분하는 이유

라벨의 `visibility`는 "다른 장애물에 가렸는가"만 담고 있다. 여기에 두 가지를 더 얹어야
학습이 성립한다. **둘은 성격이 달라서 처리도 다르다.**

| 영역 | 배포 때도 있나 | 처리 | 셀 수 | 근거 |
|---|---|---|---:|---|
| 카메라 광선이 안 닿는 원반 (r ≲ 0.5 m) | O | `vis=0` | 573 | 캘리브레이션에서 계산 |
| ego 테이블에 가린 영역 | O | `vis=0` | 293 | `common/self_mask.png` |
| 카트 손잡이 · 미는 사람 | **X** | `valid=0` | 384 | `common/rear_self_box.png` |

합쳐서 `vis=0`이 807셀(5.6%), `valid=0`이 384셀(2.7%)이다.

이 5.6%는 격자에 흩어져 있지 않고 **로봇 발밑에 뭉쳐 있다.** 원점 거리별 `vis=0` 비율:
`0–0.5 m` **100%** / `0.5–1.0 m` 51.8% / `1.0–2.0 m` 0.2% / `2.0 m` 이상 0%.
카메라 3대가 모두 수평 장착이라 발밑이 통째로 사각이다 — 인식으로 메울 수 없는 리그의
물리적 한계이므로, 접촉 방지는 근접 센서로 별도 보장한다
([`BEV_loss_and_metrics_design.md`](BEV_loss_and_metrics_design.md) §4 Phase 3).
카메라별 커버리지는 front 77.7% / left 61.5% / right 62.3%, 세 대 합집합 96.0%다.

**왜 다르게 처리하나:**

- `vis=0, valid=1` → occupancy loss에서 빠지지만 **visibility head는 "여기는 못 본다"를 학습**한다.
  배포 때도 그대로 있는 물리적 가림이므로 이게 정확한 라벨이고, 로봇이 발밑을 못 본다는 걸
  알아야 한다.
- `valid=0` → 두 head 모두 gradient를 안 받는다. 수집 아티팩트는 배포 때 없으므로
  `vis=0`으로 두면 "후방은 항상 가려져 있다"는 **거짓**을 영구히 학습한다(실제로는
  left/right 카메라 주변부가 후방을 덮는다). 잘못 배우느니 아무것도 안 배우는 게 낫다.
  실측이 이걸 뒷받침한다: `rear_self_box` 384셀 중 `permanent_blind`와 겹치는 건 126셀뿐이고
  **나머지 258셀(67%)은 카메라 커버리지 안**이다. 카트가 없으면 보이는 곳이다.
  (`rear_self_box.png`는 손잡이와 미는 사람을 하나로 통합한 영역이다. 사람 위치는 프레임마다
  달라지지만 마스크는 고정 합집합이라 보수적으로 잡혀 있고, 그게 맞다 — 몇 셀의 학습 신호를
  잃는 대신 사람을 장애물로 학습하는 사고를 확실히 막는다.)

> **기준 한 줄: 배포에도 남는 가림은 `vis=0`, 수집 과정에만 있는 것은 `valid=0`.**

`self_mask.png`만 쓰면 안 된다 — 이미지 픽셀을 지면에 투영해 만든 마스크라 **광선이 아예
안 닿는 안쪽 원반에는 구멍이 뚫려 있다**(573셀 중 514셀이 마스크 밖). 그래서
`build_bev_masks()`가 캘리브레이션에서 원반을 계산해 합친다.

부수 효과로 로봇 자체 footprint는 이 원반 안에 통째로 들어가 자동으로 처리된다.

### 마스킹의 대가

occupancy loss가 덮는 셀이 줄어든다. raws1 전체 기준:

| 단계 | 커버리지 |
|---|---:|
| 라벨 raycast visibility 그대로 | 23.64% |
| + 영구 사각 `vis=0` | 18.67% |
| + 아티팩트 `valid=0` | **16.95%** |

3분의 1이 빠지지만 빠지는 쪽이 전부 잘못된 신호다.

---

## 4. 학습 실행

### 4.1 명령어

```bash
# 정상 경로 -- split은 config 기본값이 이미 아래와 같으므로 체크포인트만 넘기면 된다.
#   train = raws2,raws3,rawos1,rawos2,rawos4  (192 frames)
#   val   = raws1,rawos3                      (75 frames)
INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \
  bash configs/train_robot_bev_finetune.sh

# split을 바꿔 볼 때만 명시한다
TRAIN_SEQUENCES=raws2,raws3,rawos1,rawos2,rawos4 VAL_SEQUENCES=raws1,rawos3 \
  INIT_CHECKPOINT=<...>.pth bash configs/train_robot_bev_finetune.sh

# 한 값만 바꾸는 스윕 -- config 파일을 복사하지 말 것.
# 나머지 인자가 한 곳에만 있어야 두 런이 정말 같은 조건인지 확인할 수 있다.
EXP_NAME=ft_lr3e-5 LR=3e-5   bash configs/train_robot_bev_finetune.sh
EXP_NAME=ft_aug   AUGMENT=True bash configs/train_robot_bev_finetune.sh
```

GPU는 `CUDA_VISIBLE_DEVICES`로 지정한다(config 기본값 0번).

### 4.2 주요 인자

| 환경변수 | 기본값 | 언제 바꾸나 |
|---|---|---|
| `TRAIN_SEQUENCES` | `raws2,raws3,rawos1,rawos2,rawos4` (192) | 시퀀스 추가할 때마다 |
| `VAL_SEQUENCES` | `raws1,rawos3` (75) | split을 바꿀 때만 |
| `VAL_TAIL_FRACTION` | `0.2` | `VAL_SEQUENCES`가 있으면 무시된다 |
| `LR` | `1e-4` | 과적합이 심하면 `3e-5` |
| `AUGMENT` | `False` | 광도 증강. train/val 격차가 클 때 후보 |
| `INIT_CHECKPOINT` | **없음 (필수)** | 안 넘기면 스크립트가 즉시 에러를 낸다 |

### 4.3 split 전략 — 가장 중요한 결정

**시퀀스 단위로 나눈다. 프레임을 섞지 않는다.**

한 시퀀스는 연속 주행을 거리 기반으로 샘플링한 것이라 프레임끼리 장면이 크게 겹친다.
무작위로 나누면 val 프레임 바로 옆 프레임이 train에 들어가서 val 숫자가 실제보다
훨씬 좋게 나온다.

`VAL_TAIL_FRACTION`(시퀀스 뒤쪽 연속 구간)은 시퀀스가 하나뿐일 때의 임시방편이다.
**이 숫자로 실험 A/B를 판정하면 안 된다.** 두 가지 이유가 있다:

1. 경계에 걸친 두 프레임이 여전히 인접하다
2. **한 시퀀스 안에서도 구간마다 장면 성격이 크게 다르다.** raws1 실측:

   | | 앞 30장 (train) | 뒤 8장 (val) |
   |---|---:|---:|
   | occupancy loss 커버리지 | 11.73% | **36.53%** |
   | 그 안의 obstacle 비율 | 7.34% | **3.11%** |

   3배 차이다. 주행 후반이 훨씬 트인 구역이라 val이 다른 종류의 장면이 된다.

학습 배너가 train/val 분포를 나란히 찍고 차이가 크면 `<- train과 크게 다르다`를 붙인다.
**이 경고가 뜬 런의 val 숫자는 모델 성능이 아니라 분포 불일치를 보고 있는 것이다.**

---

## 5. 지표 읽는 법

### 5.1 우선순위

**1위 — `val iou_obstacle`. 사실상 유일한 주 지표다.**

(옛 `iou_drivable`은 제거됐다 -- 마스킹된 영역에서 drivable이 92~95%라 "항상 drivable"이
그 값을 그대로 받는다. 배너의 `trivial 'always drivable' baseline IoU`가 그 기준선이고,
모델이 그보다 낮으면 drivable 쪽은 아무것도 배우지 못한 것이다.

**2위 — train과 val의 obstacle IoU 격차.** 데이터가 충분해지고 있는지의 신호다.
시퀀스를 추가할 때마다 이것부터 본다.

**3위 — `false_obstacle` vs `missed_obstacle`.** 같은 IoU라도 의미가 정반대다.

| | 의미 | 로봇에게 |
|---|---|---|
| `missed_obstacle`↑ | 장애물을 주행가능으로 봄 | **위험** — 들이받는다 |
| `false_obstacle`↑ | 통로를 장애물로 봄 | 소심 — 못 지나간다 |

**4위 — obstacle 크기 bin** (`tiny/small/medium/large`). 실패가 어느 크기대에 사는지.
샘플이 적으면 대부분 빈칸이라 물량이 쌓인 뒤에 의미가 생긴다.

### 5.2 함정 세 가지

**① `deploy_visible_coverage`를 같이 안 보면 속는다.**

`deploy_iou_obstacle`은 **예측된** visibility로 마스킹한 IoU다. 모델이 "대부분 안 보인다"고
선언하면 평가 면적이 줄어 IoU가 좋아 보인다. 실측 런에서 coverage가 0.801 → 0.248로
떨어지는 동안 deploy IoU는 거의 유지됐다.

> **규칙: coverage가 떨어지면서 좋아진 deploy IoU는 인정하지 않는다.**

`vis_false_low`가 같이 오르고 있으면 이 현상이 맞다(실측 0.058 → 0.503).

**② GT에 obstacle이 없는 샘플의 IoU는 구조적으로 0이다.**

완벽히 맞혀도 0이라 평균을 끌어내린다. 학습 코드는 이런 샘플을 obstacle IoU 평균에서
제외하고(`obstacle_count`로 가중), 시각화는 `n/a (no GT obstacle)`로 표시하며,
`--sort_by=obstacle_iou` 정렬에서도 뺀다. **직접 IoU를 계산할 일이 있으면 같은 처리를 할 것.**

**③ val이 작으면 노이즈가 지배한다.**

pretrain은 val 100장에서 노이즈 바닥이 0.002(GT 마스킹 기준), 0.01(deploy 기준)이었다.
val이 8장이면 그보다 훨씬 크다.

> **판정 기준을 먼저 만들어라.** 같은 설정으로 seed만 바꿔 두 번 돌리고, 그 차이보다
> 작은 변화는 없는 것으로 취급한다. 이 값을 재기 전에는 어떤 실험도 판정할 수 없다.

### 5.3 TensorBoard

```bash
tensorboard --logdir runs/robot_bev/logs
```

`train/`과 `val/` 아래에 `iou_obstacle_epoch`, `occupancy_false_obstacle_epoch`,
`occupancy_missed_obstacle_epoch`, `deploy_iou_obstacle_epoch`,
`deploy_visible_coverage_epoch`, `occupancy_obstacle_iou_{tiny,small,medium,large}_epoch`가 있다.

---

## 6. 시각화

정량 지표가 얇은 동안에는 **이쪽이 더 믿을 만한 판정 수단이다.**

```bash
# 전체 샘플
python tools/visualize_robot_predictions.py \
    --ckpt=runs/robot_bev/ckpt/<run>/model_best-<step>.pth \
    --sequences=raws1 --sort_by=index --limit=0 \
    --out_dir=runs/robot_bev/viz/<run>_all

# 최악 샘플부터
python tools/visualize_robot_predictions.py --ckpt=<...>.pth \
    --sort_by=false_obstacle --limit=8 --out_dir=runs/robot_bev/viz/<run>_worst_fa
python tools/visualize_robot_predictions.py --ckpt=<...>.pth \
    --sort_by=missed_obstacle --limit=8 --out_dir=runs/robot_bev/viz/<run>_worst_missed
```

`--sort_by`: `index | obstacle_iou | missed_obstacle | false_obstacle`.
`index` 외에는 전부 나쁜 순이다.

### 패널 읽는 법

왼쪽부터 **카메라 3장 → IPM(실제 장면) → GT occupancy → pred occupancy → GT visibility →
pred visibility**.

- 초록 = 주행가능, 빨강 = 장애물, 파랑 = 관측가능
- **회색 = loss에서 제외된 셀** (영구 사각지대 + 수집 아티팩트)

IPM을 넣은 이유가 여기 있다. GT와 예측만 보면 "이 셀이 왜 장애물인지" 알 수 없는데,
같은 좌표계에 실제 장면을 깔면 예측이 통로를 따라가는지 바로 보인다.

### 무엇을 확인하나

- 예측 통로가 IPM의 실제 통로와 **겹치는가** (어긋나면 캘리브레이션 의심 → §2 재검증)
- 회색 영역이 의도한 자리인가 (원점 주변 원반 + 후방 박스)
- 장애물 경계가 벽·작물 위에 있는가, 아니면 통로 한가운데 덩어리로 뜨는가

---

## 7. 증상별 대응

| 증상 | 원인 | 조치 |
|---|---|---|
| val obstacle IoU가 train과 크게 벌어짐 | 데이터 부족 | **시퀀스 추가.** 다른 레버는 효과 없다 |
| val이 아예 안 오르고 best가 epoch 1 | 위와 같음 (심함) | 시퀀스 추가 전까지 튜닝 무의미 |
| 배너에 `train과 크게 다르다` 경고 | split 분포 불일치 | `VAL_SEQUENCES`로 시퀀스 단위 분리 |
| `false_obstacle`이 높음 | 통로를 장애물로 봄 | 시각화로 위치 확인. pretrain 도메인 갭이면 학습으로 해소됨 |
| `missed_obstacle`이 높음 | **위험한 방향** | `pos_weight`를 자동값보다 낮춰 obstacle 가중 ↑ |
| `share u/f/o`가 한쪽으로 쏠림 | 클래스 가중치가 학습을 지배 | `MAX_CLASS_WEIGHT` 조정 (진단 §12) |
| `f1@20cm`은 높은데 `iou_occupied`가 낮음 | 정상이다 -- 예측이 몇 셀 두껍다는 뜻 (진단 §9) |  |
| `missed_obstacle_rate` 상승 | 장애물을 아예 놓치는 광선이 늘었다 | `range_mae`와 **함께** 읽는다 |
| 예측이 IPM 통로와 어긋남 | 캘리브레이션 | §2 재검증. 학습 문제가 아니다 |
| loss는 내려가는데 IoU가 안 오름 | 마스크/좌표 규약 | §2, §3 확인 |

**과적합이 확인됐을 때의 순서** (효과가 큰 것부터):

1. 시퀀스 추가 — 다른 모든 것보다 우선한다
2. `AUGMENT=True` (광도 증강. pretrain에서 +0.006, tiny bin +0.027)
3. `LR=3e-5`
4. encoder 동결 — **아직 구현 안 됨** (§9)

---

## 8. 실측 기준선 (2026-08-14, raws1 38장)

나중에 "좋아진 건가"를 판단할 기준점이다. 조건: 3-cam, 512×288, `res101`,
`lr=1e-4`, `batch_size=8`, augment 없음, pretrain best에서 초기화.

**zero-shot (pretrain 그대로, fine-tuning 없음)** — `vis*valid` 영역 기준:

| | |
|---|---:|
| drivable IoU | 0.532 |
| obstacle IoU | 0.051 |

**25 epoch fine-tuning** (train 30 / val tail 8):

| | epoch 1 | epoch 25 |
|---|---:|---:|
| train obstacle IoU | 0.134 | **0.398** |
| val obstacle IoU | 0.058 | **0.043** |
| val `false_obstacle` | 0.272 | 0.444 |
| `deploy_visible_coverage` | 0.801 | 0.248 |

**best checkpoint가 epoch 1**, 즉 pretrain 초기값 그 자체였다. 30장으로는 학습할수록
val이 나빠진다. 하이퍼파라미터 문제가 아니라 데이터 문제이므로, 물량이 쌓이기 전에는
이 숫자를 개선하려는 시도가 의미 없다.

시각화 결과는 `runs/robot_bev/viz/`에 남겨뒀다(`ep01_pretrain_init`, `ep25_all`,
`ep25_worst_false_alarm`, `ep25_worst_missed`). pretrain 초기값이 **온실 바닥을 통째로
장애물로 예측**하는 게 보인다 — SynWoodScape의 drivable은 아스팔트인데 흰색 온실 바닥은
그렇게 보이지 않기 때문이고, 이게 도메인 갭의 정체다.

---

## 9. 아직 안 만든 것 / 알려진 한계

- **encoder 동결 옵션이 없다.** 샘플이 적을 때 유용한데 인자가 없다.
  `tools/train_robot_bev.py`의 모델 생성 뒤에 `requires_grad_(False)`를 붙이면 된다.
- **`Y=1`이라 이미지 특징을 `ego z=0` 한 평면에서만 뽑는다.** 자동 라벨 파이프라인
  (`slab_label.py`)의 occupancy는 지상 0.87~1.67 m 슬래브 기준인데, raws1은 수동
  어노테이션이라 실제 기준이 무엇인지 확정하지 못했다. 성능이 정체하면 `vox_bounds`의
  높이 범위나 `Y`를 늘리는 게 실험 후보다. 다만 **pretrain이 지면 높이에서 학습됐으므로
  바꾸면 전이가 깨진다** — 물량이 충분해진 뒤에 시도할 것.
- **`rand_flip=False`로 고정.** ROI가 전후 비대칭(전방 4 m / 후방 2 m)이라 Simple-BEV의
  Z축 flip 증강이 물리적으로 성립하지 않는다. X축만 뒤집도록 오버라이드하면 쓸 수 있다
  (`docs/archive/training_improvement_plan.md` Step 4 참고).
- **후방은 화각 최외곽으로만 덮인다.** front/left/right 각 175°라 방위각 360°가 기하학적으로
  채워지긴 하지만, 후방은 렌즈 주변부라 해상도가 매우 낮다. 후방 성능 기대치는 낮게 잡는다.
- **`common/`이 전역 공유다.** 리그를 바꾼 시퀀스가 생기면 시퀀스별 `common/`으로
  분리해야 한다(현재 `RobotBEVDataset(common_root=...)`로 인자화는 되어 있다).

---

## 관련 문서

- `docs/archive/finetuning_preparation.md` — 준비 단계 기록(무엇을 만들어야 했는지, 검증 근거)
- `docs/archive/synwoodscape_pretrain_experiment_log.md` — pretrain 실험과 방법론 교훈
- `docs/archive/training_improvement_plan.md` — 개선안 목록과 기각된 것들
- `docs/BEV_loss_and_metrics_design.md` — loss/지표 설계 원안
