#!/usr/bin/env bash
# `02_projection_and_prior`의 원시 보존 -> 무결성 검사 -> 짝지은 비교.
#
# `01_overall`과 같은 관례다(`configs/paper_final_overall_analysis.sh`). 다른 점은 둘이다:
#   - 대조군을 다시 돌리지 않고 `01`의 고정 split 5런을 재사용한다(원장 §6.0).
#   - 주 통계가 평균 비교가 아니라 **시드끼리 짝지은 차이**다 -- 그래야 런 간 산포가 상쇄된다.
#
# 투영 모델은 각 런의 `config.json`에서 되찾으므로 여기서 넘기지 않는다
# (`projects/models/virtual_pinhole.projection_for_run_dirs`).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    exec conda run --no-capture-output -n bev-chamdog bash "$0" "$@"
fi

ROOT="${ROOT:-runs/paper_final/02_projection_and_prior}"
CONTROL="${CONTROL:-runs/paper_final/01_overall/fixed_split/logs}"
ARMS="${ARMS:-source_prior pinhole120 pinhole150}"
# 사전학습 팔의 이름. `02b`가 이 스크립트를 그대로 재사용하므로 밖에서 바꿀 수 있어야
# 한다 -- 같은 로직을 복사하면 한쪽만 고쳐졌을 때 조용히 갈린다.
PRETRAIN_ARM="${PRETRAIN_ARM:-source_pretrain}"
LABEL="${LABEL:-02 projection & prior}"
DONE_TAG="${DONE_TAG:-PAPER_FINAL_02_ANALYSIS_DONE}"
SEEDS="${SEEDS:-0,1,2,3,4}"
FIXED_EPOCH="${FIXED_EPOCH:-100}"   # 학습 길이와 같다 (tools/paper_final_epochs.py)
# 판정 눈금. **대조군을 다시 학습했으면 그쪽에서 다시 잰 값을 넘겨야 한다** --
# 비워 두면 도구의 기본값(2026-09-18 캠페인 실측 0.0018)을 쓴다.
SIGMA_SEED="${SIGMA_SEED:-}"
# constant-map baseline. 라벨과 val split만으로 정해지므로 보통 그대로 둔다.
BASELINE_PATH="${BASELINE_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${ROOT}/analysis"

step() {
    local out="$1"
    shift
    echo ""
    echo "############ $(basename "${out}" .txt) ($(date +%H:%M:%S)) ############"
    "$@" 2>&1 | tee "${out}"
}

echo "=== paper_final ${LABEL} analysis ($(date +%F' '%H:%M:%S)) ==="
echo "root=${ROOT} arms=${ARMS} pretrain=${PRETRAIN_ARM} control=${CONTROL} seeds=${SEEDS} epoch=${FIXED_EPOCH}"

verify_json=()
arm_specs=""
for arm in ${ARMS}; do
    arm_root="${ROOT}/${arm}"
    arm_out="${arm_root}/analysis"
    mkdir -p "${arm_out}/predictions"

    step "${arm_out}/export_run_scalars.txt" \
        python tools/export_run_scalars.py --log_root="${arm_root}/logs" --out_dir="${arm_out}"
    step "${arm_out}/export_val_predictions.txt" \
        python tools/export_val_predictions.py --log_root="${arm_root}" \
            --cells="${arm}" --seeds="${SEEDS}" --which=best,last \
            --out_dir="${arm_out}/predictions"
    step "${arm_out}/verify_val_predictions.txt" \
        python tools/verify_val_predictions.py --root="${arm_root}" \
            --pred_dir="${arm_out}/predictions" \
            --out_path="${arm_out}/verify_predictions.json"
    step "${arm_out}/summarize_repeats.txt" \
        python tools/summarize_repeats.py --log_root="${arm_root}/logs" \
            --fixed_epoch="${FIXED_EPOCH}" --group_by=exp_name

    verify_json+=("${arm_out}/verify_predictions.json")
    arm_specs="${arm_specs:+${arm_specs},}${arm}=${arm_root}/logs"
done

# 사전학습 팔은 source 도메인 런이라 target 확률맵이 없다. scalar만 보존한다.
if [ -d "${ROOT}/${PRETRAIN_ARM}/logs" ]; then
    mkdir -p "${ROOT}/${PRETRAIN_ARM}/analysis"
    step "${ROOT}/${PRETRAIN_ARM}/analysis/export_run_scalars.txt" \
        python tools/export_run_scalars.py \
            --log_root="${ROOT}/${PRETRAIN_ARM}/logs" \
            --out_dir="${ROOT}/${PRETRAIN_ARM}/analysis"
fi

# 하나라도 불일치하면 논문용 결과를 만들지 않는다 -- `01`과 같은 게이트다.
python - "${verify_json[@]}" <<'PY'
import json
import sys
from pathlib import Path

failed = [name for name in sys.argv[1:]
          if json.loads(Path(name).read_text()).get("failures")]
if failed:
    raise SystemExit("무결성 실패: " + ", ".join(failed))
print(f"무결성 게이트 통과: {len(sys.argv) - 1} files")
PY

paired_extra=()
[ -n "${SIGMA_SEED}" ] && paired_extra+=(--sigma_seed="${SIGMA_SEED}")
[ -n "${BASELINE_PATH}" ] && paired_extra+=(--baseline_path="${BASELINE_PATH}")

step "${ROOT}/analysis/report_paired_arms.txt" \
    python tools/report_paired_arms.py --control="${CONTROL}" --arms="${arm_specs}" \
        --fixed_epoch="${FIXED_EPOCH}" "${paired_extra[@]}" \
        --json_out="${ROOT}/analysis/paired_arms.json"

echo "=== ${DONE_TAG} ($(date +%F' '%H:%M:%S)) ==="
