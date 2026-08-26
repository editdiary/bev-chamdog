#!/usr/bin/env bash
# **stride 8 -> 4 팔 (두 split x 시드 3 = 6런).** 사전 선언은 진단 문서 **§29**. 산출물 `runs/stride4/`.
#
# 바꾸는 것은 **`ENCODER_TYPE=res101_s4` 하나뿐**이다. lifting이 표본하는 2D 특징맵이
# 64x36 -> 128x72가 되고, 격자(5 cm)·loss·데이터·epoch·split은 전부 그대로다.
#
# ## 대조군은 재실행하지 않는다 (§29.4)
#
# 둘 다 `pixel_center` / offset 0 / D_range 확정 config / 40 epoch / scratch로 규약이 일치한다.
#
#   arm  split                        stride-8 대조군                        3~4 m f1@10cm
#   seq  시퀀스 holdout(raws1+rawos3) runs/pixel_offset/logs/off_0.00_s{0,1,2}   0.356
#   blk  블록 프레임 프로브           runs/frame_blocks/logs/probe_s{0,1,2}      0.501 (본 프레임 0.845)
#
# 블록 split 재현 파라미터는 `BLOCK_LEN=5 GAP=3 SPLIT_SEED=0`이고 **대조군과 같은 값이어야
# 한다** -- 다르면 val 프레임 집합이 달라져 짝 비교가 아니게 된다.
#
# ## 읽는 법은 이미 정해져 있다 -- 결과를 보고 바꾸지 않는다
#
# 가설 검정은 §29.5의 **거리 x 허용오차 무늬**다: 근거리 `f1@10cm`은 거의 불변, 중거리
# (1.5~3 m)와 원거리(3~4 m)가 올라가고, `f1@40cm`은 거의 불변. **전역 `f1@10cm` 하나만
# 오르고 이 무늬가 안 따라오면 가설의 확인이 아니다.** 링 무늬가 전혀 안 움직이면 표본 밀도
# 가설을 **기각**한다(어느 쪽이든 결론이 강해진다).
#
# **채택 판정은 링 표로 하지 않는다**(§29.6). `iou_free` + `fatal_rate`가 노이즈 바닥
# (σ_seed = 0.0015, n=3)을 넘느냐, 그리고 Orin 예산 안이냐로 한다. 진단 지표로 아키텍처를
# 채택하면 `f1@10cm`을 주 판정에서 강등한 이유를 되돌리는 것이 된다.
#
# **교란 하나를 미리 밝혀 둔다**(§29.7): stride 4로 가면 §18.3.3의 두 특징 격자 0.5픽셀
# 어긋남이 절대 크기로 절반이 된다. stride-4가 좋아지면 원인이 "표본 밀도"와 "내부 정렬"
# 둘로 섞인다 -- 판정을 뒤집지는 않지만 해석에 반드시 적는다.
#
# ## 채점할 때 반드시 넘겨야 하는 것 (§29.3)
#
# 재채점 도구는 `encoder_type` 기본값이 `res101`이라 **`--encoder_type=res101_s4`를 안 넘기면
# `load_state_dict(strict=True)`가 즉시 실패한다.** 조용히 틀리지는 않는다.
#
# 6런 x 약 8분 = 약 45분(다른 작업과 GPU를 나눠 쓰면 더 걸린다). **이미 있는 런 폴더는 건너뛴다.**
#
# 실행: bash configs/stride4_arms.sh
# 집계: bash configs/stride4_arms.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/stride4}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
ENCODER="${ENCODER:-res101_s4}"
# 블록 split 재현 파라미터 -- `runs/frame_blocks` 대조군과 같은 값이다. 바꾸지 말 것.
SPLIT_SEED="${SPLIT_SEED:-0}"
BLOCK_LEN="${BLOCK_LEN:-5}"
GAP="${GAP:-3}"
# 프레임 split 경로는 `VAL_SEQUENCES`를 무시하지만, 빈 문자열을 넘기면 셸의 `:-` 기본값이
# 되살아나 두 시퀀스가 두 번 들어간다(2026-08-25에 실제로 겪었다). 표준 5+2를 그대로 넘긴다.
TRAIN_SEQS="raws2,raws3,rawos1,rawos2,rawos4"
VAL_SEQS="raws1,rawos3"
# **GPU를 다른 작업과 나눠 쓸 때만 의미가 있는 설정이다.** 실측(2026-08-26): stride-4 학습
# 피크가 17.4 GB인데 그때 GPU 1의 여유가 약 20 GB뿐이라, 기본 할당자가 순간 조각화로
# `CUDACachingAllocator ... OOM` 재시도 경고를 1 epoch에 4번 냈다(stride-8은 0번). 죽지는
# 않지만 40 epoch x 6런에서는 진짜 OOM이 될 수 있다. `expandable_segments`로 0번이 된다.
# **계산은 전혀 바뀌지 않는다** -- 할당자 설정이라 배치 크기도 config도 그대로다.
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

if [ "${1:-}" = "--report" ]; then
    seeds_csv="$(echo ${SEEDS} | tr ' ' ',')"
    echo "=== [seq] 시퀀스 holdout -- stride-4 (대조군: runs/pixel_offset off_0.00, 0.356) ==="
    python tools/report_f1_by_range.py \
        --log_root="${OUT_ROOT}" --cells=seq --seeds="${seeds_csv}" \
        --encoder_type="${ENCODER}"
    for tag in val train; do
        echo "=== [blk] 블록 프로브 ${tag} 프레임 -- stride-4 (대조군: runs/frame_blocks, val 0.501 / train 0.845) ==="
        python tools/report_f1_by_range.py \
            --log_root="${OUT_ROOT}" --cells=blk --seeds="${seeds_csv}" \
            --encoder_type="${ENCODER}" \
            --split_file="${OUT_ROOT}/logs/blk_s0/split_${tag}_samples.txt"
    done
    exit 0
fi

# 공통 config -- 두 팔이 공유한다. 대조군과 다른 것은 ENCODER_TYPE 하나뿐이다.
run_one() {
    local run="$1"; shift
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        return
    fi
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
        EXP_NAME="${run%_s*}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS=soft_boundary SOFT_TARGET=gaussian \
        DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
        LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
        PIXEL_CONVENTION=pixel_center PIXEL_OFFSET=0.0 \
        FORMULATION=binary ENCODER_TYPE="${ENCODER}" AUGMENT=True INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="${TRAIN_SEQS}" VAL_SEQUENCES="${VAL_SEQS}" \
        LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
        "$@" \
        bash configs/train_robot_bev_finetune.sh

    find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
}

for SEED in ${SEEDS}; do
    # arm 1: 시퀀스 holdout. `runs/pixel_offset/logs/off_0.00_s*`와 짝이다.
    run_one "seq_s${SEED}"
    # arm 2: 블록 프레임 split. `runs/frame_blocks/logs/probe_s*`와 짝이다.
    run_one "blk_s${SEED}" \
        FRAME_BLOCK_LEN="${BLOCK_LEN}" FRAME_SPLIT_GAP="${GAP}" FRAME_SPLIT_SEED="${SPLIT_SEED}"
done
echo "STRIDE4_ARMS_DONE"
