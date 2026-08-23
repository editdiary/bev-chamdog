#!/usr/bin/env bash
# **같은 config·같은 시드를 반복해 런 간 노이즈 바닥을 잰다.**
#
# 왜 필요한가: §9~§11의 모든 판정이 n=1이고 σ를 `ce` config의 §24 값(f1@10cm 0.0017)에서
# 빌려 쓰고 있다. 그런데 §13.6에서 §10.3의 베이스라인(α=0.5, λ_B=0.5)을 같은 시드로 다시
# 돌렸더니 `f1@10cm`이 플래토 전체에서 **−0.0041 ± 0.0014** 낮게 나왔다 -- 빌린 σ의 2.4배다.
#
# 같은 시드인데도 갈리는 이유: Simple-BEV의 lifting이 `grid_sample`을 쓰고 그 backward가
# atomicAdd라 GPU 축약 순서가 런마다 다르다. `cudnn.benchmark`는 켜져 있지 않다.
#
# 여기서 재는 것은 **σ_run**(같은 시드, 런 간)이고, 이것이 스윕 표의 두 런을 비교할 때
# 걸리는 노이즈다. §24가 잰 σ_seed(시드 간)와는 다른 성분이고, σ_seed는 이것을 포함한다.
# **soft loss config에서는 둘 다 측정된 적이 없다**(§9의 명시적 미해결 항목).
#
# 판정: 스윕에서 이 σ_run의 2배보다 작은 `f1@10cm` 차이는 주장하지 않는다.
#
# 실행: bash configs/measure_run_noise.sh
# 집계: python tools/report_convergence.py --log_root=runs/robot_bev_cv/run_noise/logs
set -euo pipefail

SEED="${SEED:-0}"
REPEATS="${REPEATS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/run_noise}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for i in $(seq 1 "${REPEATS}"); do
    name="noise_r${i}"
    echo "=== ${name} 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 LAMBDA_R=0.0 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh

    for dir in "${OUT_ROOT}/ckpt/${name}_"*; do
        [ -d "${dir}" ] && rm -rf "${dir}"
    done
    echo "=== ${name} 완료 ($(date +%H:%M:%S)) ==="
done
echo "RUN_NOISE_DONE"
