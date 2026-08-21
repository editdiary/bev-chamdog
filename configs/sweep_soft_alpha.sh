#!/usr/bin/env bash
# 2차 스윕 -- gaussian target의 **모양 축 alpha = sigma/delta**. delta는 0.15로 고정한다.
#
# 왜 alpha인가: 정규화된 target은 `u = d/delta`로 쓰면 alpha만의 함수라 delta가 식에서
# 사라진다. 즉 delta는 폭(gradient 희석 -> 수렴)만, alpha는 모양(경계 정밀도)만 정한다.
# sigma를 미터로 주면 delta를 바꿀 때 모양이 조용히 딸려간다
# (docs/soft_boundary_loss_design.md §10).
#
# 이미 아는 두 점: alpha=0.5 -> f1@10cm 0.5491 / 되올림 +16.5%
#                 alpha=inf(=linear) -> 0.5430 / +14.6%
# 즉 **alpha를 줄이는 방향이 좋았다.** 그래서 아래는 작은 alpha에 무게를 둔다.
#
# alpha=1.0은 이미 아는 두 점 사이의 보간이라 정보량이 낮지만, "alpha를 줄이는 것이
# 단조로 좋다"를 확인하는 값싼 앵커로 하나 둔다. 단조가 확인되면 큰 alpha는 더 안 본다.
#
# 마지막 하나는 다른 축이다: alpha=0.5를 고정하고 **lambda_B만** 0.5 -> 0.25로 내린다.
# 경계 항의 gradient 집중(축소 가능 gradient의 68%가 셀의 10%에)을 delta를 건드리지 않고
# 희석하는 유일한 손잡이다. 기준점(alpha=0.5, lambda=0.5)이 이미 있어 단일 손잡이 대조가 된다.
#
# **주의: 이 루프가 도는 동안 이 스크립트나 train_robot_bev_finetune.sh를 편집하지 말 것.**
# bash가 파일을 게으르게 읽어 실행 중 수정하면 죽는다.
#
# 실행: bash configs/sweep_soft_alpha.sh
# 집계: python tools/report_convergence.py --log_root=runs/robot_bev_cv/loss_sweep/logs
set -euo pipefail

SEEDS="${SEEDS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/loss_sweep}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|DELTA_M|LAMBDA_B|SIGMA_ALPHA
CONFIGS=(
    "sb_a025|0.15|0.5|0.25"
    "sb_a033|0.15|0.5|0.3333"
    "sb_a100|0.15|0.5|1.0"
    "sb_a050_lb25|0.15|0.25|0.5"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r name delta lambda_b alpha <<< "${entry}"
        echo "=== seed ${seed} / ${name} 시작 ($(date +%H:%M:%S)) ==="
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        EXP_NAME="${name}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS=soft_boundary SOFT_TARGET=gaussian \
        DELTA_M="${delta}" LAMBDA_B="${lambda_b}" SIGMA_ALPHA="${alpha}" \
        FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
        VAL_SEQUENCES="raws1,rawos3" \
        LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
        bash configs/train_robot_bev_finetune.sh

        # 체크포인트는 지운다 -- 집계는 로그만으로 되고, 승자는 확정 후 한 번 더 돌려 만든다.
        for dir in "${OUT_ROOT}/ckpt/${name}_"*"_s${seed}_"*; do
            [ -d "${dir}" ] && rm -rf "${dir}"
        done
        echo "=== seed ${seed} / ${name} 완료 ($(date +%H:%M:%S)) ==="
    done
done
echo "ALPHA_SWEEP_DONE"
