#!/usr/bin/env bash
# `Y=4`에서 soft target의 모양 매개변수 `α = σ/δ`를 다시 스윕한다 (2026-08-27).
#
# ## 왜 다시 하나
#
# α = 0.5는 설계 문서 §10.3의 1차 스윕에서 `f1@10cm` 0.5491 > 0.5470(α=1.0)을 근거로
# 채택됐다. **그 근거는 §13.6.1이 σ_run(`f1@10cm`) = 0.0037을 실측하면서 철회됐다** --
# 차이 0.0021은 0.57σ다. 같은 표에서 `val kl`(경계 항의 줄일 수 있는 부분)은 α에 대해
# 단조로 좋아진다(1.103 → 0.593). 즉 **α = 0.5를 α = 1.0보다 고른 근거가 지금은 없다.**
#
# 그리고 그 사이에 두 가지가 바뀌었다.
#
# 1. `Y = 1 → 4`가 채택됐다(AGENTS.md). 1차 스윕은 전부 `Y = 1`이다.
# 2. **val loss 발산의 유일한 원인이 경계 항이라는 것이 확인됐다** -- `y4_s0`에서 ep3 최저
#    0.4956 → ep39 0.5408(+0.0452)인데 가중 기여로 경계 +0.0952, 나머지 셋은 전부 음수다.
#    α는 그 항의 target confidence를 직접 정하는 유일한 손잡이다.
#
# ## 무엇을 보나
#
# **각 항이 수렴하는가**다. 특히 `L_B`(경계 항) -- train은 내려가는데 val이 조기 최저를
# 지나 계속 오르면 그 target이 일반화되지 않는다는 뜻이다. `L_F`·`L_N`·`L_range`도 같이
# 봐야 한다. 한 항을 평평하게 만드느라 다른 항이 나빠지면 교환이지 개선이 아니다.
#
# **`loss_boundary`는 α끼리 직접 비교하면 안 된다** -- target 엔트로피 `H̄`가 α마다 달라
# 상수 하한이 다르다(α=0.25에서 0.110, 선형에서 0.362). `kl_boundary = loss_boundary − H̄`가
# 비교 가능한 쪽이고, 그것조차 평평한 target이 본질적으로 맞히기 쉬우므로 **판정은 target에
# 의존하지 않는 지표**(`iou_free`·`fatal_rate`·`f1@10cm`)와 같이 읽어야 한다.
#
# 시드는 1개다. **그래서 이 스윕은 순위를 확정하지 못한다** -- σ_run(`f1@10cm`) = 0.0037,
# σ_run(`iou_free`)도 같은 규모이므로 품질 지표의 작은 차이는 읽으면 안 된다. 여기서 읽을
# 수 있는 것은 **수렴 곡선의 모양**(단조 하락인가, 조기 최저 후 발산인가)과 그 크기다.
#
# 실행: CUDA_VISIBLE_DEVICES=1 bash configs/sweep_soft_alpha_y4.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/alpha_y4}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|SOFT_TARGET|SIGMA_ALPHA
# `linear`는 α → ∞ 극한이고(§6.3), 이 계열에서 **가장 평평한** target이다. α는 무시된다.
CONFIGS=(
    "a025|gaussian|0.25"
    "a050|gaussian|0.5"
    "a100|gaussian|1.0"
    "a300|gaussian|3.0"
    "a500|gaussian|5.0"
    "linear|linear|None"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name target alpha <<< "${entry}"
    run="${name}_s${SEED}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    # `y4_s0`의 config를 그대로 복제하고 **α만 바꾼다**. 하나라도 다르면 이 스윕이
    # `runs/height_bins`와 나란히 읽히지 않는다.
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET="${target}" \
    DELTA_M=0.15 LAMBDA_B=0.5 SIGMA_ALPHA="${alpha}" \
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

echo "집계: python tools/report_alpha_convergence.py --log_root=${OUT_ROOT}"
