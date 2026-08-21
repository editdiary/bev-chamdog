#!/usr/bin/env bash
# 실험 1 -- 대표 split을 시드만 바꿔 반복한다. **σ_seed를 측정값으로 만드는 실행이다.**
#
# 왜 이 split이 대표인가: val = `raws1`(햇빛 O, 좁은 통로) + `rawos3`(가림막, 넓은 통로)로
# 2x2 요인표의 대각을 덮고, train에는 네 칸이 모두 남는다. 사용자가 이 조합을 고른 이유가
# 그대로 근거다.
#
# 무엇을 보고할 것인가: **고정 epoch(선언한 값)이 주 숫자**이고 best epoch은 부지표다.
# best는 max 연산이라 위로 편향된다 -- 실측 plateau 폭이 0.003~0.008이었다.
#
# 체크포인트는 seed 0만 남긴다. 나머지는 로그(런당 약 300 KB)만 있으면 집계가 되고,
# 남기면 런당 1.9 GB가 쌓인다.
#
# 실행:
#     bash configs/repeat_seeds.sh              # SEEDS 기본값 "0 1 2"
#     SEEDS="0 1 2 3 4" bash configs/repeat_seeds.sh
#     SEEDS="1 2" bash configs/repeat_seeds.sh  # 이미 끝난 시드는 빼고 이어서
#
# **주의: 이 루프가 도는 동안 이 스크립트나 `train_robot_bev_finetune.sh`를 편집하지 말 것.**
# bash는 스크립트를 파일에서 게으르게 읽으므로, 실행 중에 파일이 바뀌면 저장해 둔 바이트
# 오프셋이 어긋나 토큰 중간에서 재개하고 `unexpected EOF while looking for matching '"'`로
# 죽는다. 2026-08-21에 실제로 겪었다 -- seed 0이 정상 완료된 직후 문서 정리 작업이
# config 스크립트의 문서 경로를 고쳐서 seed 1·2가 시작되지 못했다.
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/seeds}"
KEEP_CKPT_SEED="${KEEP_CKPT_SEED:-0}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    echo "=== seed ${seed} 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="rep" \
    SEED="${seed}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    FORMULATION=binary \
    ENCODER_TYPE=res101 \
    AUGMENT=True \
    INIT_CHECKPOINT=none \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" \
    CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh

    # seed 0을 뺀 나머지의 체크포인트는 바로 버린다 -- 그림·영상용으로 한 개면 충분하다.
    if [ "${seed}" != "${KEEP_CKPT_SEED}" ]; then
        for dir in "${OUT_ROOT}/ckpt/rep_"*"_s${seed}_"*; do
            [ -d "${dir}" ] && rm -rf "${dir}" && echo "  ckpt 삭제: ${dir}"
        done
    fi
    echo "=== seed ${seed} 완료 ($(date +%H:%M:%S)) ==="
done

echo "ALL_DONE"
echo "집계: python tools/summarize_repeats.py --log_root=${OUT_ROOT}/logs --fixed_epoch=${NUM_EPOCHS}"
