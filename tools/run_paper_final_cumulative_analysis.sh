#!/usr/bin/env bash
# 재학습한 캠페인의 **Phase 2** -- 원시 보존 → 무결성 검사 → 짝지은 비교.
#
# 학습 드라이버(`tools/run_paper_final_cumulative_campaign.sh`)가 끝난 뒤에 돌린다.
#
# ## 순서가 의미를 갖는다
#
# `01`이 먼저다. 이유는 둘이다.
#   1. `02`·`02b`의 **대조군**이 `01`의 고정 split이다.
#   2. 판정 눈금 **σ_seed를 `01`에서 다시 잰다.** 대조군을 다시 학습했으므로 예전 값
#      (0.0018)을 그대로 쓰면 "차이 없음" 경계가 옛 런의 산포로 정해진다.
#
# **무결성 게이트가 실패하면 거기서 멈춘다**(`set -e`). 원장 §8: "하나라도 실패하면
# 패키지를 만들지 않는다."
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- 'bev-chamdog'가 필요하다." >&2
    exit 1
fi

ROOT="${ROOT:-$PWD/runs/paper_final_cumulative}"
FIXED_EPOCH="${FIXED_EPOCH:-40}"
SEEDS="${SEEDS:-0,1,2,3,4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== 재학습 캠페인 분석 시작 $(date '+%F %T') ==="
echo "  루트=${ROOT}  epoch=${FIXED_EPOCH}  seeds=${SEEDS}  GPU=${CUDA_VISIBLE_DEVICES}"

ROOT="${ROOT}/01_overall" SEEDS="${SEEDS}" FIXED_EPOCH="${FIXED_EPOCH}" \
    bash configs/paper_final_overall_analysis.sh
echo "PHASE2_DONE 01_overall"

# **σ_seed를 새 대조군에서 다시 잰다.** 고정 split 5시드의 `iou_free` 표준편차다.
SIGMA_SEED="$(python - "${ROOT}/01_overall/fixed_split/analysis/RESULTS.json" <<'PY'
import json, sys
sd = json.load(open(sys.argv[1]))["metrics_at_fixed_epoch"]["iou_free"]["sd"]
if sd is None:
    raise SystemExit("고정 split의 iou_free sd가 없다 -- 5시드가 다 끝났나 확인해라")
print(f"{float(sd):.6f}")
PY
)"
CONTROL="${ROOT}/01_overall/fixed_split/logs"
BASELINE_PATH="${ROOT}/01_overall/fixed_split/analysis/constant_map_baseline.json"
echo ""
echo "### 새 대조군에서 다시 잰 판정 눈금: σ_seed = ${SIGMA_SEED}  (기존 캠페인 0.0018)"
echo "### 대조군 로그: ${CONTROL}"

ROOT="${ROOT}/02_projection_and_prior" CONTROL="${CONTROL}" \
    SIGMA_SEED="${SIGMA_SEED}" BASELINE_PATH="${BASELINE_PATH}" \
    SEEDS="${SEEDS}" FIXED_EPOCH="${FIXED_EPOCH}" \
    bash configs/paper_final_projection_prior_analysis.sh
echo "PHASE2_DONE 02_projection_and_prior"

ROOT="${ROOT}/02b_native_source_prior" CONTROL="${CONTROL}" \
    SIGMA_SEED="${SIGMA_SEED}" BASELINE_PATH="${BASELINE_PATH}" \
    SEEDS="${SEEDS}" FIXED_EPOCH="${FIXED_EPOCH}" \
    bash configs/paper_final_native_prior_analysis.sh
echo "PHASE2_DONE 02b_native_source_prior"

# **옛 결과 트리가 섞이지 않았는지 확인한다.** 분석·패키지 도구의 기본값은 전부 옛 루트를
# 가리키므로, 덮어쓰기를 하나라도 빠뜨리면 옛 런과 새 런이 섞인 표가 나온다.
echo ""
echo "############ 결과 트리 순수성 ############"
python tools/check_results_tree_purity.py --targets="${ROOT}" \
    --expect="runs/$(basename "${ROOT}")"

echo ""
echo "=== 재학습 캠페인 분석 종료 $(date '+%F %T') ==="
echo "CAMPAIGN_ANALYSIS_DONE sigma_seed=${SIGMA_SEED}"
