#!/bin/bash
# 자체 수집 데이터셋 -> 3-class 단일 head Simple-BEV fine-tuning 설정.
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
# 가리켰는데 `runs/`가 `runs/_archive_2-head/`로 옮겨져 죽은 경로가 됐다. 죽은 경로를
# 기본값으로 두면 실행이 즉시 실패하는 대신 (스크립트에 따라) 조용히 랜덤 초기화로
# 시작할 위험이 있어, 아예 비워 두고 없으면 에러를 내게 한다.
#
#   INIT_CHECKPOINT=runs/synwoodscape_threeclass/ckpt/<run>/model_best-<step>.pth \\
#     bash configs/train_robot_bev_finetune.sh
#
# 3-class pretrain 체크포인트를 넘기면 출력 head까지 전이돼 배너에 `skipped 0`이 찍힌다
# (실측 `loaded 668 tensors, skipped 0`). Phase 3 A/B의 가장 큰 교란이었던 head 전이
# 비대칭(`docs/free_space_metric_migration.md` §8.4)이 그래서 사라진다.
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
    --num_epochs=60 \
    --batch_size=8 \
    --lr="${LR}" \
    --weight_decay=1e-7 \
    --num_workers=8 \
    --encoder_type=res101 \
    --augment="${AUGMENT}" \
    --val_freq_epochs=1 \
    --save_freq_epochs=10 \
    --log_dir=runs/robot_bev/logs \
    --ckpt_dir=runs/robot_bev/ckpt
