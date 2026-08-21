# 인수인계 -- binary 전환과 파이프라인 검증 (2026-08-19)

> **[2026-08-21] 진입점이 아니다.** 서사 정본은 [`experiment_history.md`](experiment_history.md)로
> 옮겼다. 이 문서는 **운영 메모**로 남긴다 -- 도구 목록·실행 명령·`cam0..3` 매핑 함정처럼
> 다른 곳에 중복되지 않은 실무 정보가 여기 있다. 아래 §0의 "논의 중"은 종결됐다(진단 §22–§23).

근거 정본은 [`finetune_overfitting_diagnosis.md`](finetune_overfitting_diagnosis.md) **§15–§23**.

브랜치 `feat/bev-surface-aware-loss`, 워킹트리 clean, 테스트 **300 passed**.
커밋 `fe6899b`(유도) → `9f124e7`(동영상 로더)까지 9개.

---

## 0. [종결됨] 예측 영상의 "예상치 못한 결과"

**결론: 정량 지표(`iou_free` 0.80 천장)와 정성 품질이 어긋난 이유는 라벨이 정의한 task 안에
이미지에서 유도할 수 없는 부분이 있었기 때문이다.** 근거와 분해는 진단 §22–§23,
서사는 [`experiment_history.md`](experiment_history.md) §4에 있다.
아래 표의 `*_bin19_*.mp4` 두 편은 2026-08-21에 삭제됐고 `*_noos1_ep35_*` 3편이 대체한다.

| 산출물 | 내용 |
|---|---|
| `runs/robot_bev/viz/raws1_fullseq_bin19_30fps.mp4` | val 시퀀스 raws1 전체 4069프레임, 30 fps, 136초, 321 MB |
| `runs/robot_bev/viz/rawos3_fullseq_bin19_30fps.mp4` | val 시퀀스 rawos3 전체 4850프레임 (렌더링 중이었다면 완료 확인 필요) |

만든 명령:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/render_prediction_video.py \
    --ckpt=runs/robot_bev/ckpt/ft_binary_res101_bs8_lr1e-04_260819_123332/model_best-000000019.pth \
    --formulation=binary --sequence_root=dataset/sj_datasets_full-seq/val/<seq> \
    --out=runs/robot_bev/viz/<seq>_fullseq_bin19_30fps.mp4 --fps=30 --batch_size=8
```

- 프레임당 약 0.11초(4069프레임 ≈ 8분). `mp4v`만 되는 opencv 빌드라 H.264보다 크다.
- **`dataset/sj_datasets_full-seq/`는 라벨 없는 원본 전체 시퀀스다.** `val/`에 raws1, rawos3이
  있고 `train/`은 아직 비어 있다(사용자가 옮기는 중).
- 파일명이 `cam0..cam3.jpg`이고 **매핑이 인덱스 순서가 아니다**:
  `cam0=front, cam1=right, cam2=rear, cam3=left` (`calibration/orientation.json`).
  순서대로 붙이면 좌우가 뒤바뀐 채 그럴듯한 그림이 나온다. 테스트가 못박고 있다.

---

## 1. 확정된 것

### 1.1 (D) binary 정식화 -- 동률이고, 병리는 사라졌다 (§15, §16)

`occupied`를 예측 클래스에서 빼고 **예측 free의 ego 기준 경계에서 유도**한다.

- **전제 확인(§15)**: GT free를 광선에 되돌리면 GT `occupied`가 `f1@20cm` **0.984**로 복원되고
  유도 표면의 92.6 %가 실제 GT occupied다. 즉 `occupied`는 `free`의 잉여다.
- **결과(§16)**: `iou_free` 0.8003 대 3-class 0.7979 — **동률**(노이즈 대역의 1/8).
  `f1@10cm`은 0.557 대 0.618로 **−0.058 손해**(유도 자체 −0.024 + 감독 상실 −0.034).
- **순이득**: 클래스 가중치가 `[1.0, 3.909]`가 되어 `MAX_CLASS_WEIGHT`가 아예 안 걸린다.
  §12가 "상한을 고르는 문제가 아니다"라고 적은 교착이 정식화로 소멸했다.
- 코드: `--formulation=three_class|binary` (config `FORMULATION`). loss 본체는
  `projects/common/segmentation_loss.py`에서 두 경로가 공유하고, 갈리는 것은
  `tools/train_robot_bev.py`의 `_FORMULATIONS` dict 한 곳뿐이다.

### 1.2 과적합의 정체 -- 오류율이 아니라 확신이다 (§17)

`loss_<class>`를 클래스 가중치로 나누면 raw nats이고 `exp(-nats)`가 정답 확률의 기하평균이다.

- ep2에 train 0.885 = val 0.888 → **ep3부터 격차가 벌어져 한 번도 멈추지 않는다.**
  ep60에 train 0.995 / val 0.620.
- `iou_free`는 오히려 오르는데 loss는 4배 → **argmax 결정은 멀쩡하고 확률 보정만 무너진다.**
- `loss_not_free`가 도는 **ep19**가 "일반화 가능한 신호의 고갈점"이고 `iou_free` best와 일치한다.

### 1.3 과적합 손잡이 셋 -- 전부 `iou_free`를 못 올렸다 (§17.1)

| | binary(res101) | res50 | freeze_encoder | label_smoothing 0.05 |
|---|---|---|---|---|
| `iou_free` ↑ | **0.8003** | 0.7900 | 0.7120 | 0.7968 |
| train `iou_free` @30 | 0.9721 | **0.9756** | 0.8828 | 0.9696 |
| val loss 증가배수 | 1.59배 | 1.36배 | 1.12배 | **1.09배** |

- **"모델이 커서 외운다"는 기각.** res50이 encoder를 반토막 내는데 train이 오히려 더 높다.
- **encoder 동결은 기각.** val −0.088, train도 0.883으로 과소적합. encoder는 적응해야 한다.
- **label smoothing은 증상만 정확히 없앤다.** 발산이 멎고 지표는 그대로. 확률을 쓸 계획이
  생기면(시간축 융합 등) 공짜로 켠다.

### 1.4 좌우 반전 증강 -- 기하는 정확하지만 채택하지 않는다 (§17.2, §19)

- 기하 검증 통과: lifting 등변성 오차 **6.5e-05** (값 표준편차 0.5555 대비 네 자릿수 작다).
  두 단계 중 하나만 하면 4.9로 깨진다 — 둘 다 테스트가 고정.
- 구현은 캘리브레이션을 건드리지 않고 `DoubleSphereVoxUtil.set_mirror_x`에서 (1) ref 프레임
  질의점 반전 + (2) 표본 좌표 `(W-1) - x`. `cx' = W - cx` 방식은 상수가 특징맵 해상도·stride에
  얽혀 위험하다.
- **성능은 나빠진다**: `iou_free` −0.028, `f1@10cm` −0.065. 원인은 **장면이 좌우 대칭이 아님**
  (프레임별 `IoU(free, mirror(free))` 평균 **0.49**, `permanent_blind` 자기 거울상 IoU 0.856).
- 플래그 `--flip_augment`/`FLIP_AUGMENT`로 남겨 뒀다. 기본 꺼짐.

### 1.5 조기 종료 -- 상시 적용 (사용자 결정)

`NUM_EPOCHS` 기본값을 60 → **30**으로 내렸다. val이 ep19에 끝나므로 실험 한 번이 13분 → 6분.

---

## 2. 파이프라인 검증 현황 (§18)

### 2.1 [수정됨] 시각화가 학습과 다른 입력을 넣고 있었다 (§18.1)

`tools/visualize_robot_predictions.py`가 **`- 0.5` 정규화를 빠뜨렸다.** 같은 train 샘플이
`iou_free` 0.731 → **0.987**로 바뀌었다.

> **§9의 정성 판정("occupied를 두꺼운 띠로 예측", "통로가 옆으로 넓다", "허위 visibility")은
> 전부 이 버그 위에서 내려졌다. 재검토 대상이다.** 수정 후 그림에서 "두꺼운 띠"는 안 보인다.

### 2.2 [확인됨] 정상인 것

| 항목 | 결과 |
|---|---|
| 학습 자체 | train 샘플 `iou_free` 0.987, 오차 지도 거의 빔. val 0.902 |
| BatchNorm train/eval | eval 0.9662 / train모드 0.9657 / 통계 재추정 0.9665 → **불일치 없음** |
| 이미지 의존성 | 섞으면 0.79 → 0.37 |
| IPM(우리 캘리브) 대 GT 라벨 횡방향 정렬 | 267프레임 평균 **+0.56셀 = +2.8 cm** → 큰 오정렬 없음 |
| lifting 배관 | 독립 numpy 재구현과 대조(unit), 반전 등변성 6.5e-05 |

### 2.3 [미해결] 특징맵 표본 좌표가 어긋난다 (§18.3)

`normalize_grid2d`(픽셀 **인덱스** 규약)로 정규화한 뒤 `grid_sample(align_corners=False)`
(픽셀 **가장자리** 규약)로 표본한다. 실측(특징맵 W=64, native 환산 ×20):

| 목표 x | 현재 | align=True |
|---|---|---|
| 8 | 7.63 (**−7.5 native px**) | 8.00 |
| 32 | 32.01 (+0.2) | 32.00 |
| 48 | 48.26 (+5.2) | 48.00 |
| 63 (마지막 열) | **31.50** (절반이 zero-pad와 섞임) | 63.00 |

중심 0, 가장자리로 갈수록 커지는 계통 오차. upstream Simple-BEV도 같다(`utils/vox.py:337`).
**단순히 `align_corners=True`가 정답이라고 단정할 수 없다** — 좌표 스케일과 stride 8 수용영역
중심까지 고려하면 총 오프셋이 0.5~1 특징픽셀(10~20 native px) 규모다.

> **제안: `--pixel_offset`을 넣어 {−1.0, −0.5, 0, +0.5} 특징픽셀을 스윕한다**(각 6분).
> 평평하면 무시, 최적점이 있으면 캘리브레이션급 개선. `f1@10cm`(=2셀)가 0.56에서 막힌 것과
> 크기가 맞물린다.

### 2.4 [미검증] 라벨의 절대 기하는 어노테이션 도구가 정했다 (§18.4)

`raws1/README.md`: 라벨은 **BEV 수동 어노테이션**이고 `visibility`는 거기서 raycast로 파생.
어노테이션 프로젝트 안에서도 지면 높이가 −0.8921/−0.87로 갈려 있었고 우리는 `self_mask`
역투영 IoU 0.606 대 0.498로 −0.87을 골랐다. §2.2의 IPM 대조가 큰 오정렬을 배제했지만
색 기반 근사이고 횡방향 중심만 본다(스케일·회전·전후는 미검증).

### 2.5 남은 반증 실험 (전부 6분 이하, 사용자 요청으로 중단 상태)

1. **방위각별 range 오차 프로파일** (학습 불필요). 720방향마다 `r_pred - r_gt`를 val 전체로
   평균해 **각도에 대한 그래프**를 그린다. 들쭉날쭉=인식 오차 / 일정한 값=스케일 오차 /
   사인파=평행이동 / 밀린 사인파=yaw. 기존 `range_error`가 이미 광선별 delta를 계산한다.
2. **좌우 extrinsic 교환** -- val이 크게 나빠져야 정상. 안 나빠지면 기하가 안 쓰이고 있다.
3. **지면 높이 ±5 cm 섭동** -- 민감도가 곧 −0.87의 정확도 요구 수준.
4. **`--pixel_offset` 스윕** (§2.3).

> `traj_recall`(설계 문서 §2.7)이 라벨과 독립적으로 검증하는 유일한 축인데 **데이터셋에
> pose가 없다**(`labels.csv`는 셀 카운트만). 지금은 불가능하다.

---

## 3. 다음 갈래 (§16.3)

병목은 **free 경계의 위치 정확도**이고 그 위에 192장 과적합이 얹혀 있다. 유도 상한이
`f1@10cm` 0.976인데 실측이 0.56이다.

- **(a) 거리 aux 항** -- occupied 감독을 거리 형태로 되돌린다. binary에는 `p_occupied`가 없어
  §13.2의 `p_occupied × d`를 못 쓰므로 (i) 방위각별 first-non-free 거리 회귀(M3의 미분 가능한
  짝) 또는 (ii) per-cell 거리장 회귀(`_distance_field_m` 재사용) 중 선택.
- **(b) 과적합** -- 광도 증강·weight decay·용량 축소·반전 증강이 모두 기각됐다. 남은 것은
  기하 증강의 다른 형태, 시간축 정보, 데이터 추가다.
- **(c) 시퀀스 단위 교차검증(LOSO 7-fold)** -- **사용자 결정으로 "학습이 잘 되게 된 뒤"로
  미뤘다.** 근거: 시퀀스 간 난이도 편차가 크다(leave-one-out constant-map `iou_free`
  0.385~0.685, 표준편차 0.131). 지금 비교하는 0.002~0.02 차이는 이 split에서만 유효한 잠정치다.
  한 설정당 42분. **뚜렷한 차이(예: frozen −0.088)만 확정으로 다룬다.**

### 시퀀스 이질성 실측 (참고)

| seq | frames | free% | 프레임 간 free 표준편차 | LOO constant-map `iou_free` |
|---|---|---|---|---|
| raws1 | 38 | 16.5% | 12.2% | 0.630 ← val |
| raws2 | 41 | 17.6% | 11.7% | 0.671 |
| rawos1 | 39 | 15.1% | 5.6% | 0.685 |
| rawos2 | 40 | 17.2% | 11.9% | 0.607 |
| raws3 | 36 | 25.8% | 3.0% | **0.385** |
| rawos3 | 37 | 24.5% | 5.4% | **0.391** ← val |
| rawos4 | 36 | 27.5% | 7.7% | **0.388** |

좁은 통로 4개 / 넓은 곳 3개로 갈린다. **현재 val은 각 그룹 하나씩(raws1+rawos3)이라 잘
고른 split이다.** 문제는 비율이 아니라 2개 시퀀스로는 작은 차이를 판정할 수 없다는 것.

---

## 4. 이 세션에 추가된 코드

| 파일 | 내용 |
|---|---|
| `projects/common/segmentation_loss.py` | 마스킹 가중 CE + 역빈도 가중치. 두 정식화 공용. `label_smoothing` 인자 |
| `projects/common/binary_metrics.py` | binary loss·가중치·`predicted_parts`(유도 포함)·`run_batch` |
| `projects/common/polar.py` | `frontier_cells` -- 광선이 멈춘 셀 = 유도된 occupied |
| `projects/common/occupied_metrics.py` | `derive_occupied` -- 배치 텐서 래퍼 |
| `projects/models/double_sphere_vox.py` | `set_mirror_x`, `build_..._util(mirror_x=)` |
| `projects/models/simplebev_three_class.py` | `num_classes` 인자 (기본 3) |
| `tools/measure_derived_occupied.py` | GT 상한 실측 + 광선 수 스윕 + 형태학적 대안 기각 |
| `tools/render_prediction_video.py` | 라벨 없는 시퀀스 → mp4. `cam0..3` / `cam_<name>` 자동 판별 |
| `tools/rescore_checkpoints.py` | head 대 derived occupied를 나란히 출력 |
| `tools/analyze_error_structure.py` | 프레임별 분포 + 방위각 프로파일 + 품질별 대조 (§22) |
| `tools/visualize_robot_predictions.py` | `--formulation=binary`, **`- 0.5` 버그 수정** |
| `configs/train_robot_bev_finetune.sh` | `FORMULATION`, `ENCODER_TYPE`, `FREEZE_ENCODER`, `LABEL_SMOOTHING`, `FLIP_AUGMENT`, `NUM_EPOCHS` 기본 30 |

### 런 목록 (전부 binary·scratch·augment, val=`raws1,rawos3`)

| 런 | 설정 | best `iou_free` |
|---|---|---|
| `ft_binary_..._123332` | 기준, 60 epoch | **0.8003 @ep19** ← 현재 최고, 시각화·영상에 쓴 체크포인트 |
| `ft_bin_res50_...` | `ENCODER_TYPE=res50`, 30 ep | 0.7900 |
| `ft_bin_frozen_...` | `FREEZE_ENCODER=True` | 0.7120 |
| `ft_bin_ls05_...` | `LABEL_SMOOTHING=0.05` | 0.7968 |
| `ft_bin_flip_...` | `FLIP_AUGMENT=True` | 0.7725 |
| `ft_bin_noos1_..._171526` | train에서 `rawos1` 제거 (153장), 60 ep | 0.7969 @ep35 (동률, §20) |
| `ft_scratch_..._191705` | **3-class** 비교군(§11의 승자) | 0.7979 |

체크포인트는 `model_best`만 남기고 주기 저장분은 `tools/prune_runs.py`로 정리했다(§14).
