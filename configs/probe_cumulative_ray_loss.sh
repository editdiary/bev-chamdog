#!/usr/bin/env bash
# soft-target cumulative arc L1을 기존 loss_effect 대조군과 비교하는 probe.
#
# seed 0은 2026-09-21에 완료됐다(`runs/cumulative_ray_loss/README.md`). 반복 실험은 `SEEDS`로
# 준다 -- run 이름은 seed에서 `E_cumulative_s{N}`으로 유도되므로 따로 주지 않는다.
#
#   LAMBDA_R=0.1526 SEEDS="1 2 3 4" bash configs/probe_cumulative_ray_loss.sh
#
# `LAMBDA_R`은 seed마다 재보정하지 않는다 -- 0.1526은 `C_soft_s0` ep40에서 `arc_huber`와
# gradient 기여를 맞춘 값 하나이고(명세 §9.1), 이 반복 실험에서는 상수다.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
OUT_ROOT="${OUT_ROOT:-runs/cumulative_ray_loss}"
RUN_PREFIX="E_cumulative"
SEEDS="${SEEDS:-0}"
: "${LAMBDA_R:?gradient 보정값을 LAMBDA_R로 명시해야 한다}"

# 덮어쓰기 검사는 **학습을 시작하기 전에 seed 전부에 대해** 한다 -- 큐 중간에서 멈추면
# 그때까지 쓴 GPU 시간이 버려진다.
for seed in ${SEEDS}; do
    for path in "${OUT_ROOT}/logs/${RUN_PREFIX}_s${seed}" "${OUT_ROOT}/ckpt/${RUN_PREFIX}_s${seed}"; do
        if [ -d "${path}" ] && [ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            echo "ERROR: 기존 run을 덮어쓸 수 없다: ${path}" >&2
            exit 1
        fi
    done
done
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

echo "=== cumulative ray probe ==="
echo "GPU=${CUDA_VISIBLE_DEVICES} seeds=${SEEDS} epochs=40 mode=cumulative_l1 lambda_r=${LAMBDA_R}"
echo "output=${OUT_ROOT} run_prefix=${RUN_PREFIX}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

for seed in ${SEEDS}; do
    run="${RUN_PREFIX}_s${seed}"
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    EXP_NAME=E_cumulative RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS=40 \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.30 SIGMA_M=0.10 SIGMA_ALPHA=None LAMBDA_B=0.5 \
    BAND_KAPPA=1.0 LABEL_EPS=0.0 \
    RANGE_LOSS_MODE=cumulative_l1 LAMBDA_R="${LAMBDA_R}" \
    DELTA_R_M=0.15 DELTA_R_OVER_M=None HUBER_BETA_M=0.15 \
    HEIGHT_BINS=4 HEIGHT_MIN_M=-0.25 HEIGHT_MAX_M=1.75 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    LR=1e-4 WEIGHT_DECAY=1e-7 MAX_CLASS_WEIGHT=20 \
    LABEL_SMOOTHING=0.0 FLIP_AUGMENT=False FREEZE_ENCODER=False \
    PROJECTION=ds_native PINHOLE_HFOV_DEG=None \
    PIXEL_CONVENTION=pixel_center PIXEL_OFFSET=0.0 \
    SAVE_FREQ_EPOCHS=40 \
    FRAME_SPLIT_FRACTION=0.0 FRAME_BLOCK_LEN=0 \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done
echo "CUMULATIVE_PROBE_DONE ($(date +%H:%M:%S))"
