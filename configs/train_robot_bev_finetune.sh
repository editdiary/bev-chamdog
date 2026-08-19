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
# train이 192장인데 res101은 40.6 M 파라미터(그중 encoder 37.0 M)다.
ENCODER_TYPE="${ENCODER_TYPE:-res101}"      # res101 / res50 / res18 -- 용량 자체를 줄인다
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"   # ImageNet 특징 고정, BEV decoder만 학습(3.5 M)
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"   # 과신 억제. 0.05~0.1이 통상값
# 좌우 반전 증강. 광도 증강과 달리 기하 다양성을 실제로 늘리는 유일한 수단이다.
# ROI가 좌우 대칭(±3 m)이라 성립하고, 기하는 tests/models/test_double_sphere_vox.py가 고정한다.
FLIP_AUGMENT="${FLIP_AUGMENT:-False}"

# 시퀀스가 늘어나면 여기에 콤마로 추가한다. VAL_SEQUENCES는 **train에 없는 시퀀스**여야
# 한다 -- 한 시퀀스는 연속 주행을 거리 기반으로 샘플링한 것이라 프레임을 섞어 나누면
# val이 train을 그대로 들여다본다.
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-raws2,raws3,rawos1,rawos2,rawos4}"   # 192 frames
VAL_SEQUENCES="${VAL_SEQUENCES:-raws1,rawos3}"                 # 75 frames

# VAL_SEQUENCES가 비어 있을 때만 쓰이는 임시 holdout -- 각 시퀀스의 뒤쪽 연속 구간을 뗀다.
# 경계 프레임이 인접해 있어 숫자가 낙관적이므로 sanity check 용도로만 본다.
VAL_TAIL_FRACTION="${VAL_TAIL_FRACTION:-0.2}"

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
# 비대칭(`docs/free_space_metric_migration.md` §8.4)이 그래서 사라진다.
# `INIT_CHECKPOINT=none`은 "pretrain 없이"를 뜻한다 (trainer의 `from_scratch`가 해석한다).
if [ -z "${INIT_CHECKPOINT:-}" ]; then
    echo "ERROR: INIT_CHECKPOINT를 지정해야 한다. 예:" >&2
    echo "  INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \\" >&2
    echo "    bash configs/train_robot_bev_finetune.sh" >&2
    exit 1
fi

python tools/train_robot_bev.py \
    --exp_name="${EXP_NAME}" \
    --train_sequences="${TRAIN_SEQUENCES}" \
    --val_sequences="${VAL_SEQUENCES}" \
    --val_tail_fraction="${VAL_TAIL_FRACTION}" \
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
    --augment="${AUGMENT}" \
    --val_freq_epochs=1 \
    --save_freq_epochs=10 \
    --log_dir=runs/robot_bev/logs \
    --ckpt_dir=runs/robot_bev/ckpt
