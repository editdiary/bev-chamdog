#!/usr/bin/env bash
# `delta` x `alpha` 격자와 긴 스케줄 -- 두 손잡이가 하나인지 가른다 (2026-08-27).
#
# ## 왜 이 5런인가
#
# `runs/loss_convergence`의 delta ablation(alpha=1.0 고정)에서 두 가지가 남았다.
#
# 1. **긴 스케줄.** delta=0.30의 val loss가 40 epoch 끝에서도 내려가는 것처럼 보인다.
#    실측 기울기는 -0.00008/epoch(10 epoch에 0.0008)이라 사실상 0이지만, **그 평평함의
#    원인이 데이터 소진이 아니라 `OneCycleLR`이다.** 스케줄이 `num_epochs`에 묶여 있어
#    (`train_robot_bev.py:613`, linear anneal, pct_start=0.05) 40 epoch 런은 끝에서 lr이
#    0으로 죽는다. 그래서 **80 epoch 런은 "40 epoch를 더 돌린 것"이 아니라 다른 실험이다**
#    -- epoch 2부터 lr 궤적이 갈린다. 그러니 `d015_e80`을 같이 돌린다. 이게 없으면
#    "80이 더 좋다"가 delta 때문인지 스케줄 때문인지 영원히 못 가른다.
#
# 2. **alpha=0.5.** delta와 alpha는 target의 10->90 % 전이 폭이라는 **한 축**을 공유한다.
#    실측 폭(m): (0.20,0.5)=23.7cm, (0.25,0.5)=29.6cm, (0.30,0.5)=35.5cm이고
#    alpha=1.0 계열은 (0.15)=22.5, (0.20)=30.0, (0.25)=37.5cm다. 즉 alpha=0.5는
#    alpha=1.0을 delta의 0.79배로 옮긴 것에 가깝다.
#
#    **폭 등가가 예측 못 하는 것이 하나 있다: `Omega_B`의 셀 집합은 delta 혼자 정한다.**
#    이 3런이 그것만 격리한다.
#
# ## 사전 등록한 예측 -- 결과를 보기 전에 적는다
#
# alpha=1.0 계열의 `공통 kl` 대 전이폭 곡선(15.0cm->0.5644, 22.5->0.4780, 30.0->0.3862,
# 37.5->0.3299, 44.9->0.3019, 67.4->0.2691)에 선형 보간하면
#
#     d020_a050 -> 0.463    d025_a050 -> 0.391    d030_a050 -> 0.345
#
# **예측대로면 두 손잡이는 하나이고 논문에는 delta만 쓴다.** `d030_a050`이 0.345를
# 뚜렷이(>2 sigma) 이기면 넓힌 셀 집합이 독립적으로 일한다는 뜻이고 그건 새 결과다.
#
# ## 읽는 법
#
# **로그의 `kl_boundary`로 config끼리 비교하면 반드시 속는다**(설계 문서 §20.1).
# `loss_boundary`의 상수 하한 `H_bar`가 target이 평평할수록 커져서, delta=0.30에서는
# 경계 loss의 71 %가 학습으로 못 움직이는 상수다. 총 loss 커브가 평평해 보이는 이유의
# 절반이 그것이다. 판정은 `tools/report_boundary_calibration.py`의 `공통 kl`로 한다.
#
# 실행: CUDA_VISIBLE_DEVICES=0 bash configs/sweep_delta_alpha.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/delta_alpha}"
SEED="${SEED:-0}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|DELTA_M|SIGMA_ALPHA|NUM_EPOCHS
RUNS=(
    "d030_e80|0.30|1.0|80"
    "d015_e80|0.15|1.0|80"
    "d020_e80|0.20|1.0|80"
    "d025_e80|0.25|1.0|80"
    "d020_a050|0.20|0.5|40"
    "d025_a050|0.25|0.5|40"
    "d030_a050|0.30|0.5|40"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${RUNS[@]}"; do
    IFS='|' read -r name delta alpha epochs <<< "${entry}"
    run="${name}_s${SEED}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (delta=${delta} alpha=${alpha} epochs=${epochs}) 시작 ($(date +%H:%M:%S)) ==="
    # 대조군 `runs/alpha_y4/logs/a100_s0`의 config를 복제하고 위 셋만 바꾼다.
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${epochs}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M="${delta}" LAMBDA_B=0.5 SIGMA_ALPHA="${alpha}" SIGMA_M=None BAND_KAPPA=1.0 LABEL_EPS=0.0 \
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

echo "집계:"
echo "  python tools/report_alpha_convergence.py --log_root=${OUT_ROOT} \\"
echo "        --reference=runs/alpha_y4/logs/a100_s0"
echo "  python tools/report_boundary_calibration.py --runs=runs/alpha_y4/logs/a100_s0,\\"
echo "runs/loss_convergence/logs/d020_s0,runs/loss_convergence/logs/d025_s0,runs/loss_convergence/logs/d030_s0,\\"
echo "${OUT_ROOT}/logs/d020_a050_s0,${OUT_ROOT}/logs/d025_a050_s0,${OUT_ROOT}/logs/d030_a050_s0,\\"
echo "${OUT_ROOT}/logs/d015_e80_s0,${OUT_ROOT}/logs/d030_e80_s0"
