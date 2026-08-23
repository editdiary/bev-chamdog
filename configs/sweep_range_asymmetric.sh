#!/usr/bin/env bash
# 4차 스윕 -- **비대칭 dead zone.** 과대예측(fatal 방향) 쪽 관용만 좁힌다 (§13.8).
#
# 왜: §13.7이 두 가지를 같이 실측했다.
#   (1) `arc_bias`가 전 런에서 +0.063 ~ +0.086 m -- 모델이 일관되게 자유공간을 과대예측한다
#   (2) `λ_R=0.1` 런은 `f1` +2.2σ를 사려고 `fatal`을 +7.8σ 팔았다
# 그런데 §14가 주 판정 지표를 `fatal_rate`·`missed_obstacle`로 옮겼다. 즉 대칭 dead zone은
# **판정하기로 한 축을 δ_R까지 무료로 허용하고 있었다** -- "벽 안쪽 20 cm까지 free 예측"이
# 무벌점이다.
#
# 보수 방향(δ_R⁻)은 0.20으로 넓게 유지한다. 그쪽 오차는 라벨 불확실성과 구별되지 않고
# 비용도 낮다(§4.5: "안전한 방향의 드리프트를 멈추는 것 자체는 목표가 아니다").
#
# 베이스는 `sb_r30`(δ_R 양방향 0.20)이고 §13.6.3에 이미 있다 -- 대조군이 공짜다.
# 판정은 §14의 프로토콜: `fatal_rate`(σ_run 0.0005)·`missed_obstacle`(0.0006)은 **단일 런으로
# 2σ 판정**된다. `free_miss_rate`를 반드시 같이 읽는다(교환 관계다).
#
# 실행: bash configs/sweep_range_asymmetric.sh
# 집계: python tools/report_convergence.py --log_root=runs/robot_bev_cv/loss_sweep/logs
set -euo pipefail

SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/loss_sweep}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|DELTA_R_OVER_M   (δ_R⁻는 0.20 고정)
CONFIGS=(
    "sb_r30_ov10|0.10"
    "sb_r30_ov00|0.0"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name over <<< "${entry}"
    echo "=== ${name} (δ_R⁺=${over}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M="${over}" HUBER_BETA_M=0.10 \
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
echo "ASYM_SWEEP_DONE"
