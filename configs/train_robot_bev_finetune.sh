#!/bin/bash
# 자체 수집 데이터셋 -> 단일 head Simple-BEV fine-tuning 설정 (`FORMULATION` 참고).
#
# pretrain 설정(`configs/train_synwoodscape_threeclass_pretrain.sh`)과 같은 관례를 따른다:
# Simple-BEV에는 config 파일 체계가 없고 Fire 키워드 인자 + 셸 스크립트로 값을 남긴다.
# 한 값만 바꾸는 스윕은 파일을 복사하지 말고 환경변수로 덮어쓴다:
#   EXP_NAME=ft_lr3e-5 LR=3e-5 bash configs/train_robot_bev_finetune.sh
#
# 실행: bash configs/train_robot_bev_finetune.sh
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXP_NAME="${EXP_NAME:-robot_finetune}"
LR="${LR:-1e-4}"
AUGMENT="${AUGMENT:-False}"

# 아래 둘은 과적합 스윕 대상이라 환경변수로 노출한다
# (`docs/finetune_overfitting_diagnosis.md` §2). 1e-7은 Simple-BEV 기본값이고 실질적으로
# 정규화가 없다 -- 192장 로봇 데이터에서 val loss가 epoch 4부터 올라간다.
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-7}"
# **기본값을 60 -> 30으로 내렸다 (2026-08-19, 사용자 결정).** val이 ep19 근처에서 끝나고
# 그 뒤 41 epoch은 train을 외우기만 한다(§16, §17). 30이면 best 지점을 여유 있게 덮으면서
# 실험 한 번이 13분 -> 6분이 된다. 옛 60 epoch 런과 곡선을 비교할 때만 60으로 되돌린다.
NUM_EPOCHS="${NUM_EPOCHS:-30}"
# 역빈도 가중치의 상한. 1이면 가중치 없음. 20이 기존 기본값이고 근거가 없다
# (`docs/finetune_overfitting_diagnosis.md` §3, §12).
MAX_CLASS_WEIGHT="${MAX_CLASS_WEIGHT:-20}"

# 정식화. `three_class`(free/occupied/unknown) 또는 `binary`(free/not-free).
# binary는 `occupied`를 예측하지 않고 예측 free의 경계에서 유도해 보고한다 -- 근거와 상한
# 실측은 `docs/finetune_overfitting_diagnosis.md` §15. 지표 집합과 체크포인트 선택 기준은
# 두 정식화가 완전히 같으므로 두 런을 한 표에 놓고 비교할 수 있다.
# binary에서는 순수 역빈도가 free 3.94라 MAX_CLASS_WEIGHT가 아예 걸리지 않는다.
FORMULATION="${FORMULATION:-three_class}"

# 과적합 손잡이 세 개 (`docs/finetune_overfitting_diagnosis.md` §16.3(b), §17).
# train이 192장인데 res101은 41.0 M 파라미터(그중 encoder 37.0 M = 90 %)다. Y=4 기준.
ENCODER_TYPE="${ENCODER_TYPE:-res101}"      # res101 / res50 / res18 -- 용량 자체를 줄인다
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"   # ImageNet 특징 고정, BEV decoder만 학습(3.5 M)
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"   # 과신 억제. 0.05~0.1이 통상값
# 좌우 반전 증강. 광도 증강과 달리 기하 다양성을 실제로 늘리는 유일한 수단이다.
# ROI가 좌우 대칭(±3 m)이라 성립하고, 기하는 tests/models/test_double_sphere_vox.py가 고정한다.
FLIP_AUGMENT="${FLIP_AUGMENT:-False}"

# 난수 시드. **반복 실험의 전제다** -- 같은 config를 시드만 바꿔 여러 번 돌려 σ를 실측하고,
# 그 σ보다 작은 차이는 주장하지 않는다. 집계는 `tools/summarize_repeats.py`.
SEED="${SEED:-0}"

# loss 종류: `weighted_ce`(현행 역빈도 가중 CE, 대조군) 또는 `soft_boundary`.
# 설계는 docs/soft_boundary_loss_design.md. 아래 넷은 soft_boundary에서만 쓰인다.
#   SIGMA_M      **라벨 경계 불확실성 [m]. 여기에 사전 지식이 들어간다.** 확정 0.10
#   DELTA_M      절대 상한 [m] -- "오차가 이걸 넘는 일은 없다"는 단정. 확정 0.30
#   LAMBDA_B     경계 항 가중치. 0.0으로 두면 "soft 항이 일을 하는가" ablation
#   SOFT_TARGET  linear 또는 gaussian(확정)
#   SIGMA_ALPHA  옛 손잡이 alpha=sigma/delta. **SIGMA_M과 동시 지정 금지.**
#
# **`k = DELTA_M / SIGMA_M >= 3`을 지킨다.** k가 작으면 절단이 sigma를 갉아먹어서 지정한
# 값이 전달되지 않는다(k=1이면 54 %만 남는다). 옛 확정값 delta=0.15/alpha=1.0이 k=1이라
# "sigma=15cm"라고 써 놓고 실제로는 8cm를 뜻하고 있었다. 근거: 명세 §6.2, 설계 문서 §21.
LOSS="${LOSS:-weighted_ce}"
DELTA_M="${DELTA_M:-0.30}"
LAMBDA_B="${LAMBDA_B:-0.5}"
SOFT_TARGET="${SOFT_TARGET:-gaussian}"
SIGMA_ALPHA="${SIGMA_ALPHA:-None}"
# `SIGMA_M`의 기본값은 확정값 0.10이지만, **`SIGMA_ALPHA`를 명시한 호출은 옛 손잡이를 쓰겠다는
# 뜻**이므로 그때는 `None`으로 둔다. 둘을 같이 넘기면 `resolve_sigma`가 거부한다 -- 그 거부는
# 옳지만, 그것 때문에 2026-08-27 이전 스윕 스크립트 19개가 통째로 깨지면 안 된다.
if [ "${SIGMA_ALPHA}" != "None" ]; then
    SIGMA_M="${SIGMA_M:-None}"
else
    SIGMA_M="${SIGMA_M:-0.10}"
fi

# 방위각 자유거리 보조항 `L_range` (docs/soft_boundary_loss_design.md §13).
#   LAMBDA_R      보조항 가중치. **0이면 항이 계산되지 않는다.** 추측하지 말고
#                 `python tools/measure_range_gradient.py --lambda_r=1.0`로 캘리브레이션한다
#                 (목표: gradient 비 G_R/G_B ~ 0.1)
#   DELTA_R_M     허용 반폭 [m]. 이 안에서 gradient가 0이 된다. 0이면 dead zone ablation
#   DELTA_R_OVER_M  **과대예측(fatal 방향) 쪽 관용만** [m]. None이면 DELTA_R_M과 같아 대칭.
#                 대칭이면 "벽 안쪽 δ_R까지 free 예측"이 무벌점이 되는데 그게 주 판정 축이다
#   HUBER_BETA_M  Huber 전환점 [m]. 미터로 둔다 (정규화하면 L2로 퇴화한다)
LAMBDA_R="${LAMBDA_R:-0.0}"
DELTA_R_M="${DELTA_R_M:-0.15}"
DELTA_R_OVER_M="${DELTA_R_OVER_M:-None}"
HUBER_BETA_M="${HUBER_BETA_M:-0.15}"

#   BAND_KAPPA  대역 target을 0.5 쪽으로 섞는 계수. `y' = k*y + (1-k)/2`. **1.0이면 전과 같다.**
#               `delta`(대역 폭)·`alpha`(모양)와 직교하는 셋째 손잡이이고, 대역 폭을 안
#               건드리므로 좁은 통로에서 hard `Omega_F`를 먹지 않는다. 근거는
#               `projects/common/soft_boundary.py`의 `DEFAULT_KAPPA` 주석.
BAND_KAPPA="${BAND_KAPPA:-1.0}"

#   LABEL_EPS   세 영역 **전부**에 걸리는 균일 label smoothing. Omega_F는 `1-eps`,
#               Omega_N은 `eps`, 대역은 `eps + (1-2eps)*y`. **0.0이면 전과 같다.**
#               `delta`와 달리 기하를 안 건드리고(Omega_F가 셀을 안 잃는다), `kappa`와 달리
#               대역 밖까지 간다. 근거는 `soft_boundary.py`의 `DEFAULT_EPS` 주석.
LABEL_EPS="${LABEL_EPS:-0.0}"

# lifting의 **높이 축** (`projects/datasets/simplebev_vox.py`의 `vox_bounds` docstring).
#   HEIGHT_BINS   `Segnet(Z, Y, X)`의 Y. **2026-08-27에 기본값을 1 -> 4로 옮겼다.**
#   HEIGHT_MIN_M / HEIGHT_MAX_M   높이 범위 [m, 지면=0]. 채택값 -0.25 ~ 1.75
#                 (표본 높이 0, 0.5, 1.0, 1.5 m). None 둘이면 ±0.25 대칭 슬래브 = 옛 설정.
#
# **이 시점 이후의 런은 그 이전 런과 lifting 기하가 다르다.** 옛 런과 한 표에 세우려면
# `HEIGHT_BINS=1 HEIGHT_MIN_M=None HEIGHT_MAX_M=None`을 명시한다.
#
# **왜 바꿨나.** 라벨 파이프라인(`dataset/.../slab_label.py`)의 occupancy는
# *"슬래브 [z_ref, z_ref+0.8]은 로봇이 통과해야 하는 높이 구간"*, 즉 **지상 0.87~1.67 m의
# 기둥 질의**다. 그런데 `Y=1`은 `ego z=0` 한 평면에서만 이미지 특징을 뽑았다.
# n=3 실험(`configs/height_bins_arms.sh`) 결과 `iou_free` 0.7950 -> 0.8104,
# `f1@10cm` 0.5437 -> 0.6085(중·원거리에 집중), 지연 +3.2 %, 재현성 불변.
# **다만 예측했던 메커니즘(장애물 검출 개선)은 확인되지 않았다** -- 실제로 좋아진 것은
# 중·원거리 경계 정밀도다. 자세한 것은 `configs/height_bins_arms.sh` 머리말.
#
# **bin 중심을 반드시 확인할 것** -- `Vox_util`은 복셀 중심에서 표본하므로 범위와 Y의 조합에
# 따라 기존 z=0 평면이 사라질 수 있다. `[-0.125, 1.875]` + Y=8이면 중심이 0, 0.25, ..., 1.75가
# 되어 bin 0이 기존 평면과 정확히 일치한다. 학습 배너가 실제 표본 높이를 찍는다.
HEIGHT_BINS="${HEIGHT_BINS:-4}"
HEIGHT_MIN_M="${HEIGHT_MIN_M:--0.25}"
HEIGHT_MAX_M="${HEIGHT_MAX_M:-1.75}"

# 특징맵 표본 좌표의 기하 (`docs/finetune_overfitting_diagnosis.md` §18.3).
#   PIXEL_CONVENTION  pixel_center(기본·옳은 규약) 또는 legacy_index(옛 동작).
#                     legacy_index는 정규화(픽셀 인덱스)와 grid_sample(픽셀 가장자리) 규약이
#                     섞여 표본 위치가 `x*W/(W-1) - 0.5`로 어긋난다 -- 배율 오차라 오프셋으로
#                     못 없앤다. `tests/models/test_pixel_grid.py`가 둘 다 고정한다.
#   PIXEL_OFFSET      규약을 고친 뒤 남는 상수 편이 [특징픽셀]. 1 = native 20 px.
#                     **스윕 결과 0으로 확정**(§18.3.5) -- 네 칸 전부 노이즈, 최적점 없음.
#
# **2026-08-25에 기본값을 pixel_center로 옮겼다(사용자 승인).** 성능이 근거가 아니라
# correctness가 근거다 -- 15런 스윕에서 품질·안전 지표는 전부 노이즈였다.
# **이 시점 이후의 런은 runs/ablation·설계 문서 §15·§16의 숫자와 기하가 다르다.**
PIXEL_CONVENTION="${PIXEL_CONVENTION:-pixel_center}"
PIXEL_OFFSET="${PIXEL_OFFSET:-0.0}"

# 산출물 위치. 반복 실험은 본 실험 폴더를 어지럽히지 않도록 별도 트리에 쌓는다:
#   LOG_DIR=runs/robot_bev_cv/seeds/logs CKPT_DIR=runs/robot_bev_cv/seeds/ckpt
LOG_DIR="${LOG_DIR:-runs/robot_bev/logs}"
CKPT_DIR="${CKPT_DIR:-runs/robot_bev/ckpt}"

# 산출물 디렉터리 이름을 직접 정한다. 비우면 자동 이름
# (`{EXP_NAME}_res101_bs8_lr1e-04_s{SEED}_{타임스탬프}`)이 쓰인다. ablation처럼 셀 구성이
# 미리 정해진 실험에서는 `A_ce_s0`처럼 직접 준다 -- 자동 이름은 폴더에서 읽을 수 없다.
# **같은 이름이 이미 있으면 trainer가 실패한다**(tfevents가 섞이는 것을 막는다).
RUN_NAME="${RUN_NAME:-None}"

# 시퀀스가 늘어나면 여기에 콤마로 추가한다. VAL_SEQUENCES는 **train에 없는 시퀀스**여야
# 한다 -- 한 시퀀스는 연속 주행을 거리 기반으로 샘플링한 것이라 프레임을 섞어 나누면
# val이 train을 그대로 들여다본다.
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-raws2,raws3,rawos1,rawos2,rawos4}"   # 192 frames
VAL_SEQUENCES="${VAL_SEQUENCES:-raws1,rawos3}"                 # 75 frames

# VAL_SEQUENCES가 비어 있을 때만 쓰이는 임시 holdout -- 각 시퀀스의 뒤쪽 연속 구간을 뗀다.
# 경계 프레임이 인접해 있어 숫자가 낙관적이므로 sanity check 용도로만 본다.
VAL_TAIL_FRACTION="${VAL_TAIL_FRACTION:-0.2}"

# **프로브 전용** 프레임 단위 무작위 split (진단 문서 §28.4). 0이면 끈다.
# 0보다 크면 VAL_SEQUENCES를 무시하고 7시퀀스 프레임 전체를 섞어 나눈다. **누출이 설계상
# 있으므로 여기서 나온 숫자는 성능으로 보고하지 않는다** -- 묻는 것은 "원거리 10 cm 정밀도가
# 입력에 있나"이고 성능이 아니다. FRAME_SPLIT_SEED는 학습 SEED와 **다른 인자**이고,
# 시드 여러 개가 같은 split을 보도록 고정한다.
FRAME_SPLIT_FRACTION="${FRAME_SPLIT_FRACTION:-0.0}"
FRAME_SPLIT_SEED="${FRAME_SPLIT_SEED:-0}"
# 무작위 split의 개선판: 시퀀스마다 연속 블록 하나를 val로 떼고 양쪽 GAP프레임을 버린다.
# 0보다 크면 FRAME_SPLIT_FRACTION보다 우선한다. 이웃 프레임 누출을 줄이는 쪽이다.
FRAME_BLOCK_LEN="${FRAME_BLOCK_LEN:-0}"
FRAME_SPLIT_GAP="${FRAME_SPLIT_GAP:-3}"

# pretrain에서 나온 best 체크포인트. 240x240 -> 120x120, 4-cam -> 3-cam 모두 그대로 로드된다
# (Segnet은 (Z, X)에 대해 완전 합성곱이고 카메라별 전용 파라미터가 없다).
#
# **기본값은 없다 -- 반드시 넘겨야 한다.** 예전 기본값은 옛 2-head pretrain 체크포인트를
# 가리켰는데 그 산출물이 아카이브로 옮겨진 뒤 삭제되어 죽은 경로가 됐다. 죽은 경로를
# 기본값으로 두면 실행이 즉시 실패하는 대신 (스크립트에 따라) 조용히 랜덤 초기화로
# 시작할 위험이 있어, 아예 비워 두고 없으면 에러를 내게 한다.
#
#   INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \\
#     bash configs/train_robot_bev_finetune.sh
#
# 3-class pretrain 체크포인트를 넘기면 출력 head까지 전이돼 배너에 `skipped 0`이 찍힌다
# (실측 `loaded 668 tensors, skipped 0`). Phase 3 A/B의 가장 큰 교란이었던 head 전이
# 비대칭(`docs/archive/free_space_metric_migration.md` §8.4)이 그래서 사라진다.
# `INIT_CHECKPOINT=none`은 "pretrain 없이"를 뜻한다 (trainer의 `from_scratch`가 해석한다).
if [ -z "${INIT_CHECKPOINT:-}" ]; then
    echo "ERROR: INIT_CHECKPOINT를 지정해야 한다. 예:" >&2
    echo "  INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \\" >&2
    echo "    bash configs/train_robot_bev_finetune.sh" >&2
    exit 1
fi

python tools/train_robot_bev.py \
    --exp_name="${EXP_NAME}" \
    --run_name="${RUN_NAME}" \
    --train_sequences="${TRAIN_SEQUENCES}" \
    --val_sequences="${VAL_SEQUENCES}" \
    --val_tail_fraction="${VAL_TAIL_FRACTION}" \
    --frame_split_fraction="${FRAME_SPLIT_FRACTION}" \
    --frame_split_seed="${FRAME_SPLIT_SEED}" \
    --frame_block_len="${FRAME_BLOCK_LEN}" \
    --frame_split_gap="${FRAME_SPLIT_GAP}" \
    --init_checkpoint="${INIT_CHECKPOINT}" \
    --num_epochs="${NUM_EPOCHS}" \
    --batch_size=8 \
    --lr="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --max_class_weight="${MAX_CLASS_WEIGHT}" \
    --formulation="${FORMULATION}" \
    --num_workers=8 \
    --encoder_type="${ENCODER_TYPE}" \
    --freeze_encoder="${FREEZE_ENCODER}" \
    --label_smoothing="${LABEL_SMOOTHING}" \
    --flip_augment="${FLIP_AUGMENT}" \
    --seed="${SEED}" \
    --loss="${LOSS}" \
    --delta_m="${DELTA_M}" \
    --lambda_b="${LAMBDA_B}" \
    --soft_target="${SOFT_TARGET}" \
    --sigma_alpha="${SIGMA_ALPHA}" \
    --sigma_m="${SIGMA_M}" \
    --lambda_r="${LAMBDA_R}" \
    --delta_r_m="${DELTA_R_M}" \
    --delta_r_over_m="${DELTA_R_OVER_M}" \
    --huber_beta_m="${HUBER_BETA_M}" \
    --band_kappa="${BAND_KAPPA}" \
    --label_eps="${LABEL_EPS}" \
    --height_bins="${HEIGHT_BINS}" \
    --height_min_m="${HEIGHT_MIN_M}" \
    --height_max_m="${HEIGHT_MAX_M}" \
    --pixel_convention="${PIXEL_CONVENTION}" \
    --pixel_offset="${PIXEL_OFFSET}" \
    --augment="${AUGMENT}" \
    --val_freq_epochs=1 \
    --save_freq_epochs=10 \
    --log_dir="${LOG_DIR}" \
    --ckpt_dir="${CKPT_DIR}"
