#!/usr/bin/env bash
# **탐색용.** 누적형 ray loss를 보조항이 아니라 셀 loss와 **대등한 정식 loss**로 올렸을 때
# 무엇이 달라지는지 본다 (사용자 요청, 2026-09-21).
#
# ## 무엇이 다른가
#
# `configs/probe_cumulative_ray_loss.sh`의 동결 설정과 **`LAMBDA_R` 하나만** 다르다. 그래야
# 관찰된 차이를 가중치에 귀속할 수 있다.
#
# `λ_R`은 숫자로 고르지 않고 gradient 비로 정한다(명세 §9). `tools/measure_range_gradient.py`로
# `λ_R=1`의 비 `G_R/G_cell`을 재면:
#
#   C_soft_s0 ep40 체크포인트에서      4.0236  -> 비 1.0을 만드는 λ_R = 0.2485
#   E_cumulative_s0 ep40 체크포인트에서 3.0197  -> 비 1.0을 만드는 λ_R = 0.3312
#
# 두 측정이 다른 이유는 잰 모델이 다르기 때문이다 -- 앞쪽은 range 항을 한 번도 본 적 없어
# arc 오차가 크고, 뒤쪽은 누적항으로 학습돼 이미 줄어 있다. 그 사이를 취해 **0.33**을 쓴다.
#
# **`λ_R=0.33`이 옛 `arc_huber`의 0.3과 비슷한 것은 우연이다.** 두 식이 다르므로 같은 숫자가
# 같은 세기를 뜻하지 않는다 -- 반드시 위의 비로 읽는다.
#
# 괄호로 `λ_R=1.0`(비 ≈ 3, 누적항이 셀 loss를 밀어내는 영역)도 한 점 잡아 방향을 본다.
#
# ## 이 산출물의 지위
#
# **n=1 탐색이고 판정이 아니다.** `iou_free`의 시드 간 표준편차가 0.0013 수준이므로 그보다
# 작은 차이는 이 런들로 갈리지 않는다. 확정 config(`RANGE_LOSS_MODE=arc_huber`)와
# 동결된 n=5 probe(`runs/cumulative_ray_loss/`)는 이 스크립트가 건드리지 않는다.
#
#   LAMBDA_R=0.33 bash configs/probe_cumulative_primary.sh
#   LAMBDA_R=1.0  bash configs/probe_cumulative_primary.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
OUT_ROOT="${OUT_ROOT:-runs/cumulative_primary}"
SEED="${SEED:-0}"
: "${LAMBDA_R:?gradient 보정값을 LAMBDA_R로 명시해야 한다}"

# run 이름에 `λ_R`을 박는다 -- 이 가지의 런은 가중치로만 구별되므로 이름에서 읽혀야 한다.
TAG="$(printf 'lr%s' "${LAMBDA_R}" | tr -d '.')"
RUN_NAME="F_cumprimary_${TAG}_s${SEED}"

for path in "${OUT_ROOT}/logs/${RUN_NAME}" "${OUT_ROOT}/ckpt/${RUN_NAME}"; do
    if [ -d "${path}" ] && [ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "ERROR: 기존 run을 덮어쓸 수 없다: ${path}" >&2
        exit 1
    fi
done
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

echo "=== cumulative primary-loss probe (탐색, n=1) ==="
echo "GPU=${CUDA_VISIBLE_DEVICES} seed=${SEED} epochs=40 mode=cumulative_l1 lambda_r=${LAMBDA_R}"
echo "output=${OUT_ROOT} run=${RUN_NAME}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

echo "=== ${RUN_NAME} 시작 ($(date +%H:%M:%S)) ==="
EXP_NAME=F_cumprimary RUN_NAME="${RUN_NAME}" SEED="${SEED}" NUM_EPOCHS=40 \
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
echo "=== ${RUN_NAME} 완료 ($(date +%H:%M:%S)) ==="
