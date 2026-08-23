#!/usr/bin/env bash
# **loss 항 ablation -- 한 계단에 한 가지만 추가한다.** 산출물은 `runs/ablation/`.
#
# 왜 새로 돌리나(2026-08-23 사용자 요청): 지금까지의 스윕 18개 런이
# `runs/robot_bev_cv/loss_sweep/logs` 한 폴더에 섞여 있고, 자동 런 이름
# (`sb_a050_re_res101_bs8_lr1e-04_s0_260823_150800`)으로는 무엇을 바꾼 런인지 읽을 수 없다.
# 게다가 각 셀이 n=1인데 σ_run(같은 config·같은 시드 반복)이 f1@10cm에서 0.0037이라
# (§13.6.1) 정확도 지표의 단일 런 차이는 노이즈와 구별되지 않는다. CE 대조군은 8/21,
# soft+range는 8/23에 돌아 2일 떨어져 있기도 하다.
#
# **사다리.** 인접한 두 셀은 손잡이 하나만 다르므로 그 차이가 그 항의 효과다.
#
#   A_ce      weighted_ce                          (대조군: 역빈도 가중 CE)
#   B_perset  soft_boundary, λ_B = 0               A -> B: per-set 평균(½L_F + ½L_N).
#                                                  경계 대역 ±δ의 셀은 감독에서 빠진다
#   C_soft    + λ_B = 0.5, gaussian α = 0.5        B -> C: **soft 경계 항** L_B
#   D_range   + λ_R = 0.3, δ_R = 0.20, β = 0.10    C -> D: **보조항** L_range
#
# 즉 A->B->C가 "CE 대비 L_soft-BCE의 효과", C->D가 "L_range 추가의 효과"다.
#
# **시드 3개.** 이제 설정이 다 정해졌으므로 반복 평균이 필요한 시점이다(§14 규칙 3).
# 시드를 바깥 루프에 둔다 -- 중간에 멈춰도 그 시점까지 **네 셀이 같은 n**을 갖는다.
#
# 12런 x 약 10분 = 약 2시간.
#
# 실행: bash configs/ablation_loss.sh
# 집계: python tools/report_ablation.py
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/ablation}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|LOSS|LAMBDA_B|LAMBDA_R
CELLS=(
    "A_ce|weighted_ce|0.0|0.0"
    "B_perset|soft_boundary|0.0|0.0"
    "C_soft|soft_boundary|0.5|0.0"
    "D_range|soft_boundary|0.5|0.3"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    for entry in "${CELLS[@]}"; do
        IFS='|' read -r name loss lambda_b lambda_r <<< "${entry}"
        run="${name}_s${seed}"
        if [ -d "${OUT_ROOT}/logs/${run}" ]; then
            echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
            continue
        fi
        echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        EXP_NAME="${name}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS="${loss}" SOFT_TARGET=gaussian \
        DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B="${lambda_b}" \
        LAMBDA_R="${lambda_r}" DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
        FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
        VAL_SEQUENCES="raws1,rawos3" \
        LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
        bash configs/train_robot_bev_finetune.sh

        # 주기 저장분만 지우고 `model_best`는 남긴다 -- 이전 스윕은 폴더를 통째로 지워서
        # 나중에 이긴 config의 체크포인트를 다시 학습해야 했다.
        find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
        echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
    done
done
echo "ABLATION_DONE"
