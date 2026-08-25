#!/usr/bin/env bash
# **같은 장면 일반화 프로브 -- 이웃 누출을 막은 판.** 진단 문서 §28.4. 산출물 `runs/frame_blocks/`.
#
# `configs/probe_frame_split.sh`(프레임 단위 무작위)의 개선판이다. 외부 검토(2026-08-25b)가
# 지적한 것: **프레임 간격이 약 0.5~0.8 m이라 val 프레임의 바로 옆 프레임은 거의 같은
# 이미지다.** 그것이 train에 있으면 "못 본 시점에서도 10 cm가 나온다"가 아니라 **"옆 프레임을
# 봤다"**를 재게 된다.
#
# 그래서 시퀀스마다 **연속 블록 하나**를 val로 떼고 **양쪽 3프레임을 버린다**.
#
#     train train train [버림 x3] VAL x5 [버림 x3] train train train
#
# 버린 프레임은 train도 val도 아니다. 실측: **train 190 / val 35 / 버림 42**이고, val 프레임과
# 가장 가까운 train 프레임의 인덱스 거리가 **4**(약 2~3 m)다. train 크기가 시퀀스
# holdout(192)과 거의 같아 **데이터 양이 교란되지 않는다.**
#
# ## 읽는 법 (외부 검토 수용 -- 2치가 아니라 위치로 읽는다)
#
# 세 값 사이의 **어디에 놓이는지**를 본다. 기준은 같은 런 안의 train 프레임과,
# `runs/ablation`의 시퀀스 holdout val이다.
#
#   3~4 m `f1@10cm`  ~0.9        -> 표본 간격이 정밀도를 직접 구속하지 않는다. **강한 증거.**
#                    ~0.6~0.8    -> 표현과 일반화가 **둘 다** 관여한다.
#                    ~0.35       -> 공간 표현 한계가 후보로 **다시 올라온다** -> stride 8→4로 간다.
#
# 그리고 높게 나와도 **"라벨 절대 오차가 작다"로는 읽지 않는다** -- 인접 프레임의 어노테이션이
# 같은 방향으로 치우쳐 있으면(한 시퀀스의 벽을 통째로 20 cm 밀어 그었다면) 이 점수는 높으면서
# 절대 오차는 클 수 있다. 이 프로브가 말하는 것은 **프레임 간 일관성**이다.
#
# 3런 x 약 7분 = 약 20분. **이미 있는 런 폴더는 건너뛴다.**
#
# 실행: bash configs/probe_frame_blocks.sh
# 집계: bash configs/probe_frame_blocks.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/frame_blocks}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
BLOCK_LEN="${BLOCK_LEN:-5}"
GAP="${GAP:-3}"
# `VAL_SEQUENCES=""`를 넘기면 안 된다 -- 셸의 `${VAL_SEQUENCES:-raws1,rawos3}`가 빈
# 문자열도 기본값으로 바꿔서 그 두 시퀀스가 두 번 들어간다(2026-08-25에 실제로 겪었다).
# 프레임 split 경로는 어차피 `VAL_SEQUENCES`를 무시하므로 표준 5+2를 그대로 넘긴다.
TRAIN_SEQS="raws2,raws3,rawos1,rawos2,rawos4"
VAL_SEQS="raws1,rawos3"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

if [ "${1:-}" = "--report" ]; then
    for tag in val train; do
        echo "=== 블록 프로브 ${tag} 프레임 ==="
        python tools/report_f1_by_range.py \
            --log_root="${OUT_ROOT}" --cells=probe --seeds="$(echo ${SEEDS} | tr ' ' ',')" \
            --split_file="${OUT_ROOT}/logs/probe_s0/split_${tag}_samples.txt"
    done
    exit 0
fi

for seed in ${SEEDS}; do
    run="probe_s${seed}"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME=probe RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    TRAIN_SEQUENCES="${TRAIN_SEQS}" VAL_SEQUENCES="${VAL_SEQS}" \
    FRAME_BLOCK_LEN="${BLOCK_LEN}" FRAME_SPLIT_GAP="${GAP}" FRAME_SPLIT_SEED="${SPLIT_SEED}" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh

    find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done
echo "FRAME_BLOCKS_PROBE_DONE"
