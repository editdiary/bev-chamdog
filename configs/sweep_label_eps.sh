#!/usr/bin/env bash
# 균일 label smoothing `eps`를 스윕한다 -- "확신은 대역 밖이 만든다" 가설의 판정 (2026-08-27).
#
# ## 가설
#
# `L_B`의 val KL 발산은 target이 날카로워서가 아니라 **모델이 전역적으로 확신에 차 있어서**이고,
# 그 확신은 **대역 밖 88 %의 hard 감독**이 만든다. 근거는 셋이다.
#
# 1. `kappa`(대역만 완화)는 확신을 실제로 줄였는데(대역 확신 예측 50.7 % -> 21 %) KL 상승은
#    안 줄었고 `kappa <= 0.4`에서 더 나빠졌다.
# 2. target 평탄도가 사실상 같은 두 config에서 모델이 따라오는 정도가 다르다 --
#    `kappa=0.4`(target H 0.652)는 격차 -0.142, `delta=0.45`(target H 0.658)는 -0.065.
#    그리고 `delta` 쪽만 수렴한다.
# 3. `Omega_F` 예측 엔트로피가 `kappa`로는 0.079 -> 0.109인데 `delta=0.45`에서는 0.260이다.
#
# 즉 `delta=0.45`가 수렴한 것은 target을 평평하게 해서가 아니라 hard 감독의 비중을 바꿔
# (`Omega_F` 15.3 % -> 6.3 %) **전역 확신을 낮췄기 때문**이다. `eps`는 그 전역 효과를
# **기하를 안 건드리고** 얻는다 -- `delta`는 0.15로 두므로 좁은 통로가 안전하다.
#
# ## 두 번째 질문 -- 경계 구조가 아직 일을 하는가
#
# `eps`가 수렴을 준다면 그다음 질문은 "그래도 경계가 특별한가"다. 대조군이 코드 변경 없이
# 만들어진다 -- **`DELTA_M`을 0에 가깝게** 주면 `Omega_B`가 비고 모든 셀이 hard(=eps로
# 완화된) target을 받는다. 즉 **경계 구조만 정확히 제거한 판본**이다.
# 여기서 차이가 안 나면 이 loss의 정직한 이름은 "per-set 평균 + label smoothing"이다.
#
# ## 읽을 때의 함정
#
# **`eps`는 `Omega_F`/`Omega_N`에도 상수 하한 `H(eps)`을 만든다.** eps=0.10이면 0.325이므로
# `loss_free`가 0.07 -> 0.40으로 뛰는 것이 성능 붕괴처럼 보인다. `kl_free`/`kl_not_free`가
# 그것을 뺀 값이고 집계 도구가 그쪽을 쓴다.
#
# **안전 지표를 반드시 같이 본다.** `Omega_N`은 93.3 %가 vis=0(벽 뒤)이고 사람이 라벨한
# obstacle은 0.0 %다. 거기서 `eps`는 "벽 뒤가 eps 확률로 free다"라는 거짓이고 `fatal`
# 방향의 주장이 된다. 대칭으로 두는 것은 가설 검정을 위해서이고(확신의 76.6 %가 거기 있다),
# `fatal_rate`/`missed_obstacle`이 나빠지면 `Omega_N`용 eps를 따로 두는 것이 다음 수다.
#
# 실행: CUDA_VISIBLE_DEVICES=1 bash configs/sweep_label_eps.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/loss_convergence}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|LABEL_EPS|DELTA_M
# 마지막 칸이 **경계 구조 제거 대조군**이다 -- delta -> 0이면 Omega_B가 빈다.
CONFIGS=(
    "e002|0.02|0.15"
    "e005|0.05|0.15"
    "e010|0.10|0.15"
    "e020|0.20|0.15"
    "e010_nob|0.10|0.001"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name eps delta <<< "${entry}"
    run="${name}_s${SEED}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (eps=${eps}, delta=${delta}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M="${delta}" LAMBDA_B=0.5 SIGMA_ALPHA=1.0 BAND_KAPPA=1.0 LABEL_EPS="${eps}" \
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
