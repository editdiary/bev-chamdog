#!/usr/bin/env bash
# 논문 최종 Overall Performance: fixed split 5 seeds + LOSO 7 folds x 5 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=(
    --protocol "${PROTOCOL:-all}"
    --seeds "${SEEDS:-0,1,2,3,4}"
    --folds "${FOLDS:-raws1,raws2,raws3,rawos1,rawos2,rawos3,rawos4}"
    --num_epochs "${NUM_EPOCHS:-40}"
    --gpu "${CUDA_VISIBLE_DEVICES:-0}"
    --output_root "${OUT_ROOT:-runs/paper_final/01_overall}"
)
if [ "${PLAN_ONLY:-False}" = "True" ]; then
    ARGS+=(--plan_only)
fi

conda run --no-capture-output -n bev-chamdog \
    python tools/run_paper_final_overall.py "${ARGS[@]}"
