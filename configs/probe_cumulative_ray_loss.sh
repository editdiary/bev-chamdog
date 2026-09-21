#!/usr/bin/env bash
# soft-target cumulative arc L1을 기존 loss_effect seed-0 대조군과 비교하는 단일 probe.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
OUT_ROOT="${OUT_ROOT:-runs/cumulative_ray_loss}"
RUN_NAME="E_cumulative_s0"
: "${LAMBDA_R:?gradient 보정값을 LAMBDA_R로 명시해야 한다}"

for path in "${OUT_ROOT}/logs/${RUN_NAME}" "${OUT_ROOT}/ckpt/${RUN_NAME}"; do
    if [ -d "${path}" ] && [ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "ERROR: 기존 run을 덮어쓸 수 없다: ${path}" >&2
        exit 1
    fi
done
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

echo "=== cumulative ray probe ==="
echo "GPU=${CUDA_VISIBLE_DEVICES} seed=0 epochs=40 mode=cumulative_l1 lambda_r=${LAMBDA_R}"
echo "output=${OUT_ROOT} run=${RUN_NAME}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

EXP_NAME=E_cumulative RUN_NAME="${RUN_NAME}" SEED=0 NUM_EPOCHS=40 \
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
TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
VAL_SEQUENCES="raws1,rawos3" \
LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
bash configs/train_robot_bev_finetune.sh
