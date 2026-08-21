#!/usr/bin/env bash
# soft-boundary loss의 1차 판정 스윕. 설계는 docs/soft_boundary_loss_design.md.
#
# 무엇을 답하는가:
#   (1) soft-boundary가 현행 역빈도 가중 CE보다 나은가          -> `ce` 대 `sb_*`
#   (2) soft 항이 실제로 일을 하는가                            -> `lb00` 대 `lb50`
#   (3) 대역 폭 delta는 0.10인가 0.15인가                       -> `d10` 대 `d15`
#   (4) target 모양이 중요한가 (탐색용 1점)                     -> `linear` 대 `gaussian`
#
# **`lb00`도 delta에 의존한다.** lambda_B=0이면 경계 대역이 loss에서 통째로 빠지므로
# delta가 "얼마나 빼는가"를 정한다. 그래서 두 delta 모두에 대해 ablation을 돌린다.
#
# **시드가 바깥 루프다.** 중간에 죽어도 완결된 시드까지는 전체 config의 표가 나온다
# (config이 바깥이면 앞쪽 config만 n=3이고 뒤쪽은 0이 되어 비교가 안 된다).
#
# 시드 3개를 처음부터 도는 이유: sigma_seed = 0.0015(iou_free) / 0.0017(f1@10cm)이라
# 단일 런으로는 0.005급 차이도 판정할 수 없다(진단 §24). 시드 없이 판정했다가 되돌린
# 것이 이 프로젝트가 이미 한 번 밟은 함정이다.
#
# 체크포인트는 런마다 지운다 -- 18런이면 34 GB가 쌓이고, 집계는 로그만으로 된다.
# 승자 config은 확정 후 한 번 더 돌려 체크포인트를 만든다.
#
# **주의: 이 루프가 도는 동안 이 스크립트나 `train_robot_bev_finetune.sh`를 편집하지 말 것.**
# bash가 스크립트를 게으르게 읽어 실행 중 파일이 바뀌면 바이트 오프셋이 어긋나 죽는다
# (2026-08-21에 실제로 겪었다 -- 시드 반복 중 문서 정리가 config 스크립트를 고쳐 죽었다).
#
# 실행:
#     bash configs/sweep_soft_boundary.sh
#     SEEDS="0" bash configs/sweep_soft_boundary.sh          # 스크리닝만
#
# 집계:
#     python tools/summarize_repeats.py \
#         --log_root=runs/robot_bev_cv/loss_sweep/logs --fixed_epoch=40 --group_by=exp_name
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/loss_sweep}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|LOSS|DELTA_M|LAMBDA_B|SOFT_TARGET|SIGMA_M
# gaussian의 sigma=0.075는 delta/2다 -- 대역 끝이 +-2 sigma라 S자가 실제로 드러나는 값이다.
# sigma가 delta보다 훨씬 크면 정규화된 Phi가 선형에 수렴하므로 대조가 무의미해진다.
CONFIGS=(
    "ce|weighted_ce|0.15|0.5|linear|None"
    "sb_d10_lb00|soft_boundary|0.10|0.0|linear|None"
    "sb_d10_lb50|soft_boundary|0.10|0.5|linear|None"
    "sb_d15_lb00|soft_boundary|0.15|0.0|linear|None"
    "sb_d15_lb50|soft_boundary|0.15|0.5|linear|None"
    "sb_d15_lb50_g|soft_boundary|0.15|0.5|gaussian|0.075"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r name loss delta lambda_b target sigma <<< "${entry}"
        echo "=== seed ${seed} / ${name} 시작 ($(date +%H:%M:%S)) ==="
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        EXP_NAME="${name}" \
        SEED="${seed}" \
        NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS="${loss}" \
        DELTA_M="${delta}" \
        LAMBDA_B="${lambda_b}" \
        SOFT_TARGET="${target}" \
        SIGMA_M="${sigma}" \
        FORMULATION=binary \
        ENCODER_TYPE=res101 \
        AUGMENT=True \
        INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
        VAL_SEQUENCES="raws1,rawos3" \
        LOG_DIR="${OUT_ROOT}/logs" \
        CKPT_DIR="${OUT_ROOT}/ckpt" \
        bash configs/train_robot_bev_finetune.sh

        for dir in "${OUT_ROOT}/ckpt/${name}_"*"_s${seed}_"*; do
            [ -d "${dir}" ] && rm -rf "${dir}"
        done
        echo "=== seed ${seed} / ${name} 완료 ($(date +%H:%M:%S)) ==="
    done
    echo "SEED_${seed}_DONE"
done

echo "ALL_DONE"
