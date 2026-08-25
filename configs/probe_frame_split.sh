#!/usr/bin/env bash
# **같은 장면 일반화 프로브** -- 진단 문서 §28.4. 산출물은 `runs/frame_split/`.
#
# ## 이건 성능 실험이 아니다
#
# 프레임 단위 무작위 split은 **누출이 설계상 있다** -- 한 시퀀스는 연속 주행을 샘플링한
# 것이라 val 프레임 바로 옆 프레임이 train에 들어간다. **여기서 나온 숫자를 성능으로
# 보고하면 안 된다.** 그러면 왜 돌리는가.
#
# ## 답해야 하는 질문
#
# §28.2가 `f1@10cm`을 train 프레임에서 재서 3~4 m에서도 **0.938**을 얻었다(시퀀스 holdout
# val은 0.355). 표본 간격이 20~24 cm인 구간이므로, 그 간격이 원거리 정밀도의 바닥이라는
# §27의 해석이 반증 쪽으로 기울었다. **남은 구멍이 하나다: 모델이 train 191프레임을
# 외웠을 수 있다.** 외운 것이면 0.938은 계산이 아니라 회수이고 §27의 해석이 살아난다.
#
# 필요한 것은 **못 본 프레임(외울 수 없다)이면서 같은 장면(새 장면 일반화를 요구하지 않는다)**
# 인 프레임이고, 그것을 만드는 유일한 방법이 이 split이다.
#
#   원거리 `f1@10cm`이 train 수준(~0.9)  -> 정보가 입력에 있다. **표본 밀도 가설 기각.**
#                                          병목은 장면 다양성(7시퀀스 / 267프레임)이다.
#   val 수준(~0.35)                      -> 새 시점 자체가 어렵다. 가설이 **살아난다**.
#
# **한 방향으로만 결정적이다**(높게 나오는 쪽). 낮게 나오면 일반화 실패와 라벨 잡음이
# 섞여 아무것도 못 가른다.
#
# ## 판정을 런 내부에서 한다
#
# 채점을 **같은 런의 train 프레임과 val 프레임 양쪽에서** 한다. 그러면 "본 프레임 대 못 본
# 프레임"이 같은 체크포인트·같은 기하 안에서 비교되므로, 2026-08-25의 표본 규약 전환
# (`legacy_index` -> `pixel_center`)을 건너뛰는 비교를 하지 않아도 된다. `runs/ablation`의
# 숫자는 보조 참조로만 쓴다.
#
# ## 고정한 것
#
# - loss·epoch·encoder·augment는 확정 config(`D_range`) 그대로. **움직이는 손잡이는 split 하나.**
# - `FRAME_SPLIT_SEED`는 학습 `SEED`와 **다른 인자**다. 시드 3개가 **같은 split**을 봐야
#   시드 분산에 split 분산이 섞이지 않는다.
# - val 비율 0.28 -> 267프레임 중 75프레임. **시퀀스 holdout(192/75)과 같은 크기**라야
#   train 데이터 양이 비교 가능하다.
#
# 3런 x 약 7분 = 약 20분. **이미 있는 런 폴더는 건너뛴다.**
#
# 실행: bash configs/probe_frame_split.sh
# 집계: bash configs/probe_frame_split.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/frame_split}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.28}"
# `VAL_SEQUENCES=""`를 넘기면 안 된다 -- 셸의 `${VAL_SEQUENCES:-raws1,rawos3}`가 빈
# 문자열도 기본값으로 바꿔서 그 두 시퀀스가 두 번 들어간다(2026-08-25에 실제로 겪었다).
# 프레임 split 경로는 어차피 `VAL_SEQUENCES`를 무시하므로 표준 5+2를 그대로 넘긴다.
TRAIN_SEQS="raws2,raws3,rawos1,rawos2,rawos4"
VAL_SEQS="raws1,rawos3"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

if [ "${1:-}" = "--report" ]; then
    # val(못 본 프레임)과 train(본 프레임)을 나란히 낸다. split은 학습이 남긴 목록을
    # 그대로 읽는다 -- 재유도하면 조용히 다른 집합을 채점할 수 있다.
    for tag in val train; do
        echo "=== 프로브 ${tag} 프레임 ==="
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
    FRAME_SPLIT_FRACTION="${VAL_FRACTION}" FRAME_SPLIT_SEED="${SPLIT_SEED}" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh

    find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done
echo "FRAME_SPLIT_PROBE_DONE"
