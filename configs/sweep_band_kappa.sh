#!/usr/bin/env bash
# 대역 target 수축 계수 `kappa`를 스윕한다 -- `L_B`를 수렴시키는 셋째 손잡이 (2026-08-27).
#
# ## 왜 이 손잡이인가
#
# `L_B`의 val KL은 **어떤 `alpha`에서도 수렴하지 않는다**(`runs/alpha_y4`). epoch 1~3에
# 최저를 찍고 끝까지 오르기만 하며 가장 나은 `alpha=1.0`에서도 +48 %다. 원인은 target이
# 입력이 담고 있는 것보다 날카롭다는 것이다 -- BCE의 최소화자는 `E[y|image]`인데 실측
# 특징맵 해상도가 1~2 m에서 10.5 cm, 3~4 m에서 20~24 cm이므로 Bayes 최적 예측기조차 그만큼
# 뭉개져 있다.
#
# **target을 뭉개는 기존 손잡이 둘이 다 막혀 있다.**
#
# - `alpha`는 포화한다. 함의 표준편차 상한이 `delta/sqrt(3)` = 8.7 cm이고 `alpha=1.0`에서
#   이미 8.1 cm(상한의 93 %)다. 무한대로 키워도 0.6 cm를 더 못 넓힌다.
# - `delta`는 기하를 먹는다. 이 로봇은 좁은 통로를 다니고 **자유 셀의 절반이 벽에서 32 cm
#   이내**다(val 75프레임). `delta=0.45`면 자유공간의 69 %가 hard `Omega_F` 자격을 잃고
#   `delta=0.60`이면 프레임의 28 %가 hard free 셀을 하나도 못 가진다.
#
# `kappa`는 `y' = k*y + (1-k)/2`로 **대역 폭을 안 건드리면서** 상한 없이 평평하게 만든다.
# `Omega_F`도 통로도 그대로다. 뜻도 라벨 쪽이다 -- "경계 위치와 무관하게 (1-k)의 확률로 이
# 라벨은 정보가 없다"는 평평한 라벨 잡음이고, 이 데이터셋 라벨이 LiDAR/SLAM 위에 사람 손
# 보정이 얹힌 것이므로 그런 성분 자체는 자연스럽다. **다만 보정 절차가 기록돼 있지 않아
# `kappa`를 데이터에서 추정할 수 없다** -- `delta`와 마찬가지로 스윕으로 고른다.
#
# ## 무엇을 보나
#
# 주 지표는 **`kl_boundary`의 val 상승폭**이다. `kappa`가 낮을수록 줄어야 한다.
#
# **그런데 그것만 보면 반드시 속는다.** `kappa -> 0`이면 대역 target이 전부 0.5가 되어
# 경계 감독이 사라지고, 그러면 KL은 당연히 안 오른다 -- 배울 것이 없으니까. 그래서
# **target에 의존하지 않는 지표**(`iou_free`·`fatal_rate`·`f1@10cm`)가 같이 안 나빠져야만
# 개선이다. `kappa=0.2`를 넣는 이유가 그 붕괴점을 보기 위해서다.
#
# `kappa=1.0`은 `runs/alpha_y4/logs/a100_s0`(alpha=1.0, delta=0.15)가 이미 대조군이므로
# 여기서 다시 돌리지 않는다.
#
# 실행: CUDA_VISIBLE_DEVICES=1 bash configs/sweep_band_kappa.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/loss_convergence}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|BAND_KAPPA
KAPPAS=(
    "k080|0.8"
    "k060|0.6"
    "k040|0.4"
    "k020|0.2"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${KAPPAS[@]}"; do
    IFS='|' read -r name kappa <<< "${entry}"
    run="${name}_s${SEED}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (kappa=${kappa}) 시작 ($(date +%H:%M:%S)) ==="
    # 대조군 `a100_s0`의 config를 그대로 복제하고 **kappa만 바꾼다.**
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.15 LAMBDA_B=0.5 SIGMA_ALPHA=1.0 BAND_KAPPA="${kappa}" \
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

echo "집계: python tools/report_alpha_convergence.py --log_root=${OUT_ROOT} \\"
echo "        --reference=runs/alpha_y4/logs/a100_s0"
