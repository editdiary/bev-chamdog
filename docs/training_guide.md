# SynWoodScape Simple-BEV 학습 가이드 (Phase 3.3)

`tools/train_synwoodscape.py`를 실행하면서 **어떤 값을 보고, 무엇을 조정할지** 정리한
문서다. 스크립트 자체의 인자 목록은 `configs/train_synwoodscape_baseline.sh`를 참고.

숫자(loss/IoU)만으로 판단하기 부족하면 **`tools/visualize_predictions.py`**로 예측 BEV를
GT와 나란히 그린 PNG를 볼 수 있다:

```bash
python tools/visualize_predictions.py \
    --ckpt_dir=work_dirs/checkpoints_synwoodscape/<run_name> \
    --output_dir=work_dirs/viz_<run_name>
```

`<run_name>`은 학습 종료 로그의 `saved a checkpoint: work_dirs/checkpoints_synwoodscape/<run_name>/...`
줄에서 그대로 가져오면 된다. 기본은 `model_best-*.pth`(val 점수 최고 시점)를 불러오고, val
sample 중 앞 8개(`--num_samples`로 조절)에 대해 `<sample_id>_compare.png`를 만든다 — 4-cam
썸네일 위에 GT/예측 BEV 지도를 `dataset/synwoodscape_occupancy_gt/*_combined.png`와 같은
색(회색=미관측, 초록=drivable, 빨강=obstacle)으로 나란히 그린다. `--encoder_type`/`--use_fisheye`는
그 체크포인트를 만들 때 쓴 값과 반드시 같아야 한다(모델 구조가 다르면 로드가 깨진다).

## 1. 가장 먼저 알아야 할 것 — trivial baseline이 매우 높다

실제 500 samples(train 449 / val 51, `synwoodscape_split.py` 클러스터 분할, seed=0) 기준:

| 값 | train | val |
|---|---:|---:|
| drivable 비율(=`pos_weight` 계산에 쓰는 클래스 균형) | 92.98% | 93.61% |
| "항상 drivable로 예측"의 drivable IoU | 0.930 | 0.936 |
| "항상 drivable로 예측"의 obstacle(비주행가능) IoU | 0 | 0 |

즉 **아무것도 학습하지 않고 전부 drivable이라고만 찍어도 drivable IoU가 0.93을 넘는다.**
그래서 이 스크립트는 drivable IoU 하나만 보지 않고 **obstacle IoU를 항상 같이 찍는다**
(`train/iou_obstacle_epoch`, `val/iou_obstacle_epoch`). 체크포인트도 두 IoU의 평균
(`(drivable_iou + obstacle_iou)/2`)이 최고일 때 `model_best-*.pth`로 저장한다.

**따라서 학습이 잘 되고 있는지 판단할 때는 반드시 obstacle IoU를 먼저 봐라.** drivable
IoU가 0.93 근처에 머무르고 obstacle IoU가 0에 가깝게 정체돼 있으면, 모델이 실제로는
"항상 drivable"이라는 trivial 해로 수렴한 것이다 — loss가 줄어들고 있어도 마찬가지다.

## 2. 매 epoch 로그에서 보는 것

콘솔 출력 예(터미널에서는 epoch/train/val/new best 부분이 각각 색으로 구분되어 보인다):
```
epoch 012/60 | time  8.3s | train loss 0.0512 drivable 0.941 obstacle 0.301 | val loss 0.0498 drivable 0.938 obstacle 0.276  * new best
```
`* new best`는 이번 epoch에서 val (drivable+obstacle)/2 IoU가 지금까지 중 최고를 찍어
`model_best-*.pth`를 새로 저장했다는 표시다.

- **obstacle IoU (train/val)** — §1 이유로 가장 먼저 볼 값. 0.93 근처의 drivable IoU보다
  이 값이 실제로 뭔가를 배우고 있는지 알려준다.
- **train vs val obstacle IoU 격차** — train만 계속 오르고 val은 정체·하락하면 overfitting.
  500장뿐인 데이터셋이라 이 격차는 예상보다 빨리 나타날 수 있다.
- **loss가 NaN/inf** — 학습률이 너무 크거나 어딘가 수치 문제. 즉시 중단하고 §4의 lr 항목 확인.
- tensorboard(`work_dirs/logs_synwoodscape/`)에는 위 값 전부(`train/*_epoch`, `val/*_epoch`,
  `train/loss_step`, `train/lr`)가 곡선으로 남는다. `tensorboard --logdir work_dirs/logs_synwoodscape`.

## 3. 증상 → 원인 → 조치

| 증상 | 가능한 원인 | 조치 |
|---|---|---|
| train loss가 여러 epoch째 거의 안 줄어듦 | lr이 너무 낮음 / 학습 자체가 너무 이름(초기 epoch) | `--lr`을 2~3배 올려보기(예: 3e-4→1e-3). 최소 10~20 epoch은 지켜본 뒤 판단 |
| train loss가 발산(NaN, 갑자기 폭증) | lr이 너무 높음 | `--lr`을 1/3~1/10로 낮추기. `clip_grad_norm_(..., 5.0)`은 이미 걸려 있음 |
| obstacle IoU가 0 근처에 정체, drivable IoU는 §1의 trivial 값 근처 | 모델이 trivial 해(항상 drivable)로 수렴 | `--pos_weight`를 자동 계산값(약 0.075)보다 **낮춰서**(예: 0.03~0.05) obstacle class 가중치를 상대적으로 높이거나, lr을 낮춰 학습을 더 안정화. `pos_weight`는 `BCEWithLogitsLoss`의 **양성(=drivable) class** 가중치라 값을 낮출수록 obstacle 쪽에 더 민감해진다 |
| train obstacle IoU는 계속 오르는데 val obstacle IoU는 정체/하락 | overfitting (500장은 매우 작은 데이터셋) | epoch 수를 줄이거나(`model_best-*.pth`가 이미 최적 시점을 저장해 둠) `--weight_decay`를 1e-7→1e-4~1e-3으로 올리기. `--encoder_type=res50` 또는 `effb0`로 모델 용량을 줄여보는 것도 방법 |
| GPU OOM | batch_size나 encoder가 지금 여유 메모리에 비해 큼 | `--batch_size`를 절반으로, 그래도 안 되면 `--encoder_type=res50` 또는 `effb0`로. §5 참고 |
| epoch 하나가 너무 오래 걸림, `nvidia-smi`로 보면 GPU 활용률이 낮음 | 데이터 로딩이 병목(디스크 I/O) | `--num_workers`를 8~16 범위에서 올려보기(CLAUDE.md 권장치) |
| `--split_seed`를 바꾸니 val IoU가 크게 달라짐 | val이 51장뿐이라 분산이 큼(클러스터 단위라 사실상 몇 개 안 되는 "장면") | 한 seed의 값을 과신하지 말 것. 2~3개 seed로 돌려 평균/분산을 같이 보고하는 걸 권장 |

## 4. 하이퍼파라미터별 권장 시작값과 조정 방향

| 인자 | 기본값 | 의미 / 조정 기준 |
|---|---|---|
| `lr` | `3e-4` | Simple-BEV 원본(`train_nuscenes.py`) 기본값. OneCycleLR로 warmup 5%→선형 감소. loss가 안 줄면 올리고, 발산하면 내림(§3) |
| `weight_decay` | `1e-7` | 원본 기본값(거의 규제 없음). 500장으로 overfitting이 보이면 1e-4~1e-3까지 올려볼 것 |
| `batch_size` | `4` | GPU 메모리와 직결(§5). 그래디언트가 너무 노이즈 많다 싶으면(loss가 들쭉날쭉) 늘리고, OOM이면 줄임 |
| `num_epochs` | `60` | 500장 기준 시작값. val obstacle IoU 곡선이 평평해지거나 하락하기 시작하는 지점을 보고 늘리거나 줄임 — `model_best-*.pth`가 매 시점 최고 기록을 저장하므로 "너무 오래 돌렸다"의 리스크는 낮음 |
| `pos_weight` | 자동 계산(≈0.075, train split 실측 neg/pos) | `None`이면 매 실행 시 실제 train split에서 다시 계산됨(스크립트 시작 시 값이 출력됨). obstacle에 더 민감하게 하려면 이 자동값보다 낮춰서 직접 지정 |
| `encoder_type` | `res101` | `res101`(가장 큼, 논문 기본) → `res50` → `effb0`(가장 가벼움) 순. §5 메모리 표 참고 |
| `val_fraction` | `0.1` | val 51장이 이미 적은 편이라 더 낮추는 건 권장하지 않음. 데이터가 늘어나면 조정 |
| `use_fisheye` | `True` | Phase 3.2에서 검증된 실제 `radial_poly` 투영. `False`로 두면 Phase 3.1의 임시 핀홀 근사로 돌아간다 — 디버깅/속도 비교 외에는 끌 이유가 없음 |

## 5. GPU 메모리(다른 연구실과 GPU를 공유 중일 때)

`tools/smoke_test_fisheye_segnet.py`로 실측한 값(batch_size=1, res101, 4-cam, 512x384):
**peak GPU 메모리 2.73 GiB.** 배치 크기에는 대략 선형으로 비례한다고 보고 시작하되,
실제 실행 시 첫 몇 epoch 동안 `nvidia-smi`로 직접 확인해서 맞추는 걸 권장한다(인코더별
파라미터 수 차이 때문에 정확히 선형은 아님):

- `--batch_size`를 낮추는 게 메모리를 가장 확실하게 줄이는 방법이다.
- `--encoder_type=res50`(res101보다 파라미터 적음) 또는 `effb0`(EfficientNet-B0, 가장 가벼움)로
  바꾸면 배치 크기를 유지하면서 메모리를 아낄 수 있다.
- 학습 시작 직후 `nvidia-smi`로 실제 사용량을 한 번 확인하고, 다른 연구실의 job에 영향이
  없는 수준인지 판단할 것. 스크립트를 강제 종료해도 프로세스 종료 시 CUDA 메모리는 즉시
  반환된다(확인 완료).

## 6. 알아두면 좋은 제약/설계 결정

- **`rand_flip`은 껐다(고정, CLI로 켤 수 없음).** Simple-BEV의 `rand_flip`은 BEV grid를
  전후(Z축)로도 무작위로 뒤집는데, `SYNWOODSCAPE_PRETRAIN_GRID_SPEC`은 전방5m/후방3m로
  **비대칭**이라 전후 반전이 물리적으로 성립하지 않는다(뒤집으면 카메라 배치와 안 맞는
  장면이 된다). 좌우(X축) 반전은 그리드 자체는 대칭이지만, 원본 코드가 GT/카메라 좌우
  교체를 같이 처리해주지 않아 추가 구현이 필요하다 — 지금 범위 밖.
- **center/offset head는 쓰지 않는다.** `Segnet`은 인스턴스 center/offset도 같이 출력하지만
  (3D 검출/추적용, 이후 확장 과제) 우리 loss는 segmentation(=occupancy) 출력만 본다.
  `model.ce_weight`/`center_weight`/`offset_weight`(Simple-BEV의 uncertainty-weighting 파라미터)는
  모델에 존재하지만 우리 loss가 참조하지 않아 그래디언트가 없다 — 학습에 영향 없이 무시하면 된다.
- **4-cam(FV/MVL/MVR/RV) 그대로 학습한다.** 최종 자체 로봇 fine-tuning은 3-cam(후면 제외)이지만,
  Simple-BEV는 카메라별 전용 파라미터가 없어(encoder 공유, masked-mean fusion) pretrain
  4-cam → fine-tune 3-cam 전환이 구조적으로 문제 없다(`ROADMAP.md` Phase 3.1 참고).
- **어떤 이미지가 train/val인지 직접 확인하려면**: split은 폴더로 나뉘어 있지 않고
  `projects/datasets/synwoodscape_split.py`가 `vehicle_data`의 ego 위치로 계산한다
  (`--split_seed`가 같으면 항상 완전히 같은 51장이 val이 된다 — 결정적 함수). 학습을 실행하면
  그 실행에서 실제로 쓰인 목록이 `work_dirs/logs_synwoodscape/<run_name>/split_train_ids.txt`
  / `split_val_ids.txt`에 그대로 저장되니, 그 파일을 열어보면 된다. (체크포인트 폴더가 아니라
  로그 폴더에 둔다 — `saverloader.load()`가 체크포인트 폴더의 파일 전체를
  `{model_name}-{step}.pth` 패턴으로 가정하고 훑어서, 다른 파일이 섞여 있으면 죽는다.)
- **재실행마다 다른 폴더에 저장된다**: `run_name`에 실행 시각(`YYMMDD_HHMMSS`)이 붙어서
  `{exp_name}_{encoder_type}_bs{batch_size}_lr{lr}_{시각}` 형식이 된다. 이게 없으면 같은
  `--exp_name`으로 다시 실행할 때 epoch 번호가 1부터 다시 시작되면서 이전 실행의 체크포인트
  파일명과 그대로 겹쳐 덮어써진다(`model_best`는 `keep_latest=1`이라 새 실행 첫 저장에서
  즉시 삭제됨). 그래서 실험을 구분하고 싶으면 `--exp_name`만 잘 지어주면 되고, 폴더 충돌을
  직접 신경 쓸 필요는 없다.
- **체크포인트 위치**: `work_dirs/checkpoints_synwoodscape/<run_name>/model-*.pth`(주기적
  저장, `--save_freq_epochs`) 와 `model_best-*.pth`(val (drivable+obstacle)/2 IoU 최고 시점,
  1개만 유지). `saverloader.load(ckpt_dir, model, model_name="model_best")`(simple_bev
  submodule)로 다시 불러올 수 있다 — `tools/visualize_predictions.py`가 이렇게 한다.
