#!/usr/bin/env bash
# `delta`만 격리한 ablation -- 나머지는 전부 원래 loss 그대로 (2026-08-27).
#
# ## 왜 이것만 따로 돌리나
#
# `delta = 0.45`가 `L_B`의 val 발산을 눈에 띄게 줄인 유일한 개입이었다. 그런데 그 관찰은
# 한 점(0.15 대 0.45)이고, 그 사이에 `kappa`·`eps` 같은 새 손잡이를 얹은 런들이 섞여 있어
# **`delta` 자체의 곡선을 본 적이 없다.** 여기서는 `kappa = 1.0`, `eps = 0.0`으로 **새 손잡이를
# 전부 끄고** `delta`만 움직인다.
#
# `alpha = 1.0` 고정. 대조군 `runs/alpha_y4/logs/a100_s0`(delta=0.15)와
# `runs/loss_convergence/logs/d045_s0`(delta=0.45)가 이미 그 alpha에 있으므로, 이 4런을
# 더하면 **delta만 다른 6점 계열**(0.10 ~ 0.45)이 된다. alpha는 계열 내내 상수라 delta
# 효과를 오염시키지 않는다.
#
# ## 반드시 공통 눈금으로 읽어야 한다
#
# **로그의 `kl_boundary`는 delta끼리 비교할 수 없다.** delta가 커지면 `Omega_B`가 넓어지면서
# 경계에서 먼 **쉬운** 셀(delta=0.45에서 벽 40 cm 안쪽 target이 0.958)이 평균에 섞이고,
# 동시에 target 자체가 평평해져 본질적으로 맞히기 쉬워진다. 실측에서 delta=0.45의
# `kl_boundary` 0.0870 중
#
#     0.0870 -> 0.1052 (셀 집합을 |d|<=0.15로 통일) -> 0.2691 (target도 공통으로 통일)
#
# 이었다. 즉 **로그가 보여준 이점의 3분의 2 이상이 "더 쉬운 것을 재고 있었던 것"이다.**
# 대조군은 같은 눈금에서 0.4780이므로 실제 개선은 96 %가 아니라 44 %다.
#
# 집계는 `tools/report_boundary_calibration.py`로 한다 -- 고정 셀(`|d| <= 0.15`)에서
# **공통 target**에 대해 채점하므로 delta끼리 비교가 성립한다.
#
# 실행: CUDA_VISIBLE_DEVICES=1 bash configs/sweep_delta_only.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/loss_convergence}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# 0.15는 `runs/alpha_y4/logs/a100_s0`, 0.45는 `d045_s0`이 이미 같은 config로 있다.
DELTAS=(0.10 0.20 0.25 0.30)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for delta in "${DELTAS[@]}"; do
    run="d$(printf '%03d' "$(python3 -c "print(int(round(${delta}*100)))")")_s${SEED}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (delta=${delta}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="d${delta}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M="${delta}" LAMBDA_B=0.5 SIGMA_ALPHA=1.0 BAND_KAPPA=1.0 LABEL_EPS=0.0 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
    HEIGHT_BINS=4 HEIGHT_MIN_M=-0.25 HEIGHT_MAX_M=1.75 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    LR=1e-4 \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done

echo "집계: python tools/report_boundary_calibration.py \\"
echo "        --runs=runs/alpha_y4/logs/a100_s0,${OUT_ROOT}/logs/d010_s0,${OUT_ROOT}/logs/d020_s0,\\"
echo "${OUT_ROOT}/logs/d025_s0,${OUT_ROOT}/logs/d030_s0,${OUT_ROOT}/logs/d045_s0"
