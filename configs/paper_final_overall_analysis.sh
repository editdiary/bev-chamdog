#!/usr/bin/env bash
# `01_overall`의 원시 보존 -> 무결성 검사 -> 집계 -> 결과 번들.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    exec conda run --no-capture-output -n bev-chamdog bash "$0" "$@"
fi

ROOT="${ROOT:-runs/paper_final/01_overall}"
FIXED="${ROOT}/fixed_split"
LOSO="${ROOT}/loso"
FIXED_OUT="${FIXED}/analysis"
LOSO_OUT="${LOSO}/analysis"
SEEDS="${SEEDS:-0,1,2,3,4}"
# LOSO fold 목록. 부분 재실행(한 fold만 다시 채점)과 리허설을 위해 밖에서 바꿀 수 있다.
FOLDS="${FOLDS:-raws1 raws2 raws3 rawos1 rawos2 rawos3 rawos4}"
FIXED_EPOCH="${FIXED_EPOCH:-40}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${FIXED_OUT}" "${LOSO_OUT}/predictions" "${ROOT}/analysis"

step() {
    local out="$1"
    shift
    echo ""
    echo "############ $(basename "${out}" .txt) ($(date +%H:%M:%S)) ############"
    "$@" 2>&1 | tee "${out}"
}

train_sequences_without() {
    local held="$1" out=""
    local all="raws1 raws2 raws3 rawos1 rawos2 rawos3 rawos4"
    for sequence in ${all}; do
        [ "${sequence}" = "${held}" ] && continue
        out="${out:+${out},}${sequence}"
    done
    echo "${out}"
}

echo "=== paper_final overall analysis ($(date +%F' '%H:%M:%S)) ==="
echo "root=${ROOT} seeds=${SEEDS} fixed_epoch=${FIXED_EPOCH} gpu=${CUDA_VISIBLE_DEVICES}"

step "${FIXED_OUT}/export_run_scalars.txt" \
    python tools/export_run_scalars.py --log_root="${FIXED}/logs" --out_dir="${FIXED_OUT}"
step "${LOSO_OUT}/export_run_scalars.txt" \
    python tools/export_run_scalars.py --log_root="${LOSO}/logs" --out_dir="${LOSO_OUT}"

step "${FIXED_OUT}/export_val_predictions.txt" \
    python tools/export_val_predictions.py --log_root="${FIXED}" \
        --cells=final --seeds="${SEEDS}" --which=best,last \
        --out_dir="${FIXED_OUT}/predictions"
step "${FIXED_OUT}/verify_val_predictions.txt" \
    python tools/verify_val_predictions.py --root="${FIXED}" \
        --pred_dir="${FIXED_OUT}/predictions" \
        --out_path="${FIXED_OUT}/verify_predictions.json"

for held in ${FOLDS}; do
    pred_dir="${LOSO_OUT}/predictions/${held}"
    train_sequences="$(train_sequences_without "${held}")"
    mkdir -p "${pred_dir}"
    step "${LOSO_OUT}/export_predictions_${held}.txt" \
        python tools/export_val_predictions.py --log_root="${LOSO}" \
            --cells="loso_${held}" --seeds="${SEEDS}" --which=best,last \
            --out_dir="${pred_dir}" --train_sequences="${train_sequences}" \
            --val_sequences="${held}"
    step "${LOSO_OUT}/verify_predictions_${held}.txt" \
        python tools/verify_val_predictions.py --root="${LOSO}" \
            --pred_dir="${pred_dir}" \
            --out_path="${LOSO_OUT}/verify_predictions_${held}.json"
done

step "${FIXED_OUT}/summarize_repeats.txt" \
    python tools/summarize_repeats.py --log_root="${FIXED}/logs" \
        --fixed_epoch="${FIXED_EPOCH}" --group_by=exp_name
step "${LOSO_OUT}/summarize_repeats.txt" \
    python tools/summarize_repeats.py --log_root="${LOSO}/logs" \
        --fixed_epoch="${FIXED_EPOCH}" --group_by=val_sequences
step "${LOSO_OUT}/report_loso.txt" \
    python tools/report_loso.py --log_root="${LOSO}/logs" \
        --fixed_epoch="${FIXED_EPOCH}" --json_out="${LOSO_OUT}/report_loso.json"

# 하나라도 불일치하면 논문용 결과 묶음을 만들지 않는다.
python - "${FIXED_OUT}/verify_predictions.json" "${LOSO_OUT}"/verify_predictions_*.json <<'PY'
import json
import sys
from pathlib import Path

failed = []
for name in sys.argv[1:]:
    path = Path(name)
    payload = json.loads(path.read_text())
    if payload.get("failures"):
        failed.append(str(path))
if failed:
    raise SystemExit("무결성 실패: " + ", ".join(failed))
print(f"무결성 게이트 통과: {len(sys.argv) - 1} files")
PY

step "${ROOT}/analysis/build_overall_results_bundle.txt" \
    python tools/build_overall_results_bundle.py --root="${ROOT}" \
        --fixed_epoch="${FIXED_EPOCH}"

echo "=== PAPER_FINAL_OVERALL_ANALYSIS_DONE ($(date +%F' '%H:%M:%S)) ==="
