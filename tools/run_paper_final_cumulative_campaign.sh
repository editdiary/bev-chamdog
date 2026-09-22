#!/bin/bash
# 캠페인 전체를 **보조항 `cumulative_l1`로** 다시 돌린다 (원장 §8, 연휴 실행 계획).
#
# 기존 `runs/paper_final/`은 **건드리지 않는다.** 새 루트에 쓴다 -- 덮어쓰면 지금 논문에
# 들어가 있는 숫자의 출처가 사라지고, 두 형태가 얼마나 달랐는지도 확인할 수 없다.
#
# 순서가 중요하다. 사전학습이 먼저여야 그것에서 시작하는 팔이 `blocked_missing_input`으로
# 건너뛰어지지 않는다. 큐 러너가 팔 순서를 보장하므로 여기서는 실험 순서만 정한다.
#
#   1) 01_overall  (대조군. 다른 모든 비교의 기준이라 먼저 완성한다)
#   2) 02_projection_and_prior
#   3) 02b_native_source_prior
#
# **중간에 죽어도 같은 명령을 다시 실행하면 이어진다** -- 러너가 완료된 런을 건너뛰고
# 불완전한 런은 덮어쓰지 않는다(`blocked_incomplete`).
#
# 실행: bash tools/run_paper_final_cumulative_campaign.sh
set -u
cd "$(dirname "$0")/.."

REQUIRED_CONDA_ENV="${REQUIRED_CONDA_ENV:-bev-chamdog}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${REQUIRED_CONDA_ENV}" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- '${REQUIRED_CONDA_ENV}'가 필요하다." >&2
    echo "    (base(3.14)에서는 numpy가 256 KB 넘는 배열을 조용히 덮어쓴다)" >&2
    exit 1
fi

GPU="${GPU:-0}"
MODE="${MODE:-cumulative_l1}"
ROOT="${ROOT:-runs/paper_final_cumulative}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"

mkdir -p "${ROOT}"
echo "=== 캠페인 재학습 시작 $(date '+%F %T') ==="
echo "  루트=${ROOT}  보조항=${MODE}  GPU=${GPU}  epochs=${NUM_EPOCHS}"
echo "  커밋=$(git rev-parse --short HEAD)  환경=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h . | tail -1
echo

run_phase () {
    local label="$1"; shift
    echo ""
    echo "##################################################################"
    echo "## ${label}   시작 $(date '+%F %T')"
    echo "##################################################################"
    "$@"
    local rc=$?
    echo "## ${label} 종료 $(date '+%F %T')  rc=${rc}"
    echo "PHASE_DONE ${label} rc=${rc}"
    return 0   # 한 실험이 실패해도 나머지는 계속 돌린다 -- 사람이 없다
}

run_phase "01_overall" \
    python tools/run_paper_final_overall.py \
        --protocol=all --num_epochs="${NUM_EPOCHS}" --gpu="${GPU}" \
        --range_loss_mode="${MODE}" --output_root="${ROOT}/01_overall"

run_phase "02_projection_and_prior" \
    python tools/run_paper_final_projection_prior.py \
        --num_epochs="${NUM_EPOCHS}" --gpu="${GPU}" \
        --range_loss_mode="${MODE}" --output_root="${ROOT}/02_projection_and_prior"

run_phase "02b_native_source_prior" \
    python tools/run_paper_final_native_prior.py \
        --num_epochs="${NUM_EPOCHS}" --gpu="${GPU}" \
        --range_loss_mode="${MODE}" --output_root="${ROOT}/02b_native_source_prior"

echo ""
echo "=== 캠페인 재학습 종료 $(date '+%F %T') ==="
df -h . | tail -1
echo "CAMPAIGN_TRAINING_DONE"
