#!/usr/bin/env bash
# **LOSO 7-fold x 시드 3 = 21런.** 설계는 진단 문서 §25.3, **읽는 법은 §25.6**. 산출물 `runs/robot_bev_cv/loso/`.
#
# ## 이 실험이 답하는 질문 하나
#
# 지금까지의 **모든 숫자가 `raws1`+`rawos3` 한 split에 조건부**다. LOSO가 그것을 푼다 --
# "**처음 보는 시퀀스** 하나에서 얼마나 하나"에 답한다. 새 데이터가 아니라 **있는 데이터의
# 정직한 재사용**이다.
#
# ## 실행 전에 못박는 것 (§25.2) -- 이 선언이 CV를 정직하게 만든다
#
# **config는 동결됐다.** loss 연구가 설계 문서 §16.8로 닫힌 뒤 아무것도 바뀌지 않았고,
# stride 8→4는 §29.9에서 기각돼 아키텍처도 그대로다. 아래 config는 그 동결된 값이다.
#
#   binary / soft_boundary(gaussian, δ=0.15, α=0.5, λ_B=0.5) + L_range(λ_R=0.3, δ_R=0.20, β=0.10)
#   res101(stride 8) / scratch / photometric augment / bs8 / lr 1e-4 / 40 epoch / pixel_center
#
# > **CV는 단 한 번 실행하고, 결과가 나쁘게 나와도 config를 재조정하지 않는다.**
# > 결과를 보고 손잡이를 돌리면 선택 편향이 CV 안으로 들어가고, 그러면 이 21런이
# > 답하려던 질문에 답하지 못하게 된다. **[사용자 방침] 새 가지는 열지 않는다.**
#
# ## 왜 val이 시퀀스 1개인가 (§25.6)
#
# LOSO의 표본 단위는 프레임이 아니라 **시퀀스**이고 우리에게 시퀀스는 7개뿐이다.
# val을 2개로 늘리면 표본이 7개에서 3~4개로 줄고 train도 줄어든다.
# **1개가 최대 표본 수이자 최대 train 크기다.**
#
# ## 고정 epoch으로 보고한다 -- val이 1개일 때의 유일한 실질적 위험을 없앤다
#
# val이 시퀀스 1개인데 그 시퀀스로 체크포인트를 고르면 **그 fold에 과적합한다.**
# 고정 epoch(40)이 그것을 원천 제거하고, 비용은 실측 0.0006(0.4σ)뿐이다(진단 §24.3).
# 집계 도구가 `--fixed_epoch=40`을 기본으로 쓴다.
#
# ## 체크포인트를 남기지 않는다
#
# 21런 x 약 465 MB다. 판정에 필요한 것은 전부 tfevents에 있고, `split_*_samples.txt`와
# `config.json`도 남는다. 재채점이 필요해지면 그 fold만 다시 돌린다.
#
# 21런 x 약 7분 = **약 2.5시간.** **이미 있는 런 폴더는 건너뛴다**(중단 후 재실행 안전).
#
# 실행: bash configs/loso_folds.sh
# 집계: python tools/report_loso.py
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_SEQUENCES="raws1 raws2 raws3 rawos1 rawos2 rawos3 rawos4"
FOLDS="${FOLDS:-$ALL_SEQUENCES}"
SEEDS="${SEEDS:-0 1 2}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/loso}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
# GPU를 다른 작업과 나눠 쓸 때만 의미가 있다. 계산은 바뀌지 않는다(`configs/stride4_arms.sh` 참고).
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

# fold 이름 -> 그 시퀀스를 뺀 나머지 6개를 콤마로.
train_sequences_without() {
    local held="$1" out=""
    for s in ${ALL_SEQUENCES}; do
        [ "$s" = "$held" ] && continue
        out="${out:+${out},}${s}"
    done
    echo "$out"
}

if [ "${1:-}" = "--report" ]; then
    python tools/report_loso.py --log_root="${OUT_ROOT}/logs"
    exit 0
fi

echo "=== LOSO 7-fold x 시드 ${SEEDS} 시작 ($(date +%H:%M:%S)) ==="
for held in ${FOLDS}; do
    train_seqs="$(train_sequences_without "${held}")"
    for seed in ${SEEDS}; do
        run="loso_${held}_s${seed}"
        if [ -d "${OUT_ROOT}/logs/${run}" ]; then
            echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
            continue
        fi
        echo "=== ${run} 시작 ($(date +%H:%M:%S)) | val=${held} | train=${train_seqs} ==="
        env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
            PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
            EXP_NAME="loso_${held}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
            LOSS=soft_boundary SOFT_TARGET=gaussian \
            DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
            LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
            PIXEL_CONVENTION=pixel_center PIXEL_OFFSET=0.0 \
            FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
            TRAIN_SEQUENCES="${train_seqs}" VAL_SEQUENCES="${held}" \
            LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
            bash configs/train_robot_bev_finetune.sh

        # 체크포인트는 남기지 않는다 -- 21런 x 465 MB이고 판정은 tfevents로 한다.
        rm -rf "${OUT_ROOT}/ckpt/${run}" 2>/dev/null || true
        echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
    done
done
echo "LOSO_DONE ($(date +%H:%M:%S))"
