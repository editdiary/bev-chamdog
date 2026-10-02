#!/usr/bin/env bash
# **캠페인 v3 본 실험 드라이버** -- 학습 -> 끊긴 런 정리·재시도 -> 그 실험의 분석, 실험마다.
#
# 계획: `docs/paper_campaign_v3_plan.md` (2026-10-02 사용자 승인). 절차 정본:
# `docs/paper_campaign_protocol.md`.
#
#   01_overall (고정 split 5 -> LOSO 35)  ->  02_projection_and_prior (20)
#   ->  02b_native_source_prior (10)  ->  03_boundary_uncertainty (20)        = 90런, 약 27 h
#
# **사람이 없어도 끝까지 간다.** 한 단계가 실패해도 다음으로 넘어가고, 단계마다 `PHASE_DONE`
# 표지와 rc를 남긴다. 학습이 멈추면 watchdog(큐 러너: 침묵 20분 / 2시간, 03: 런당 1시간)이 죽인다.
# 끊긴 런은 `tools/clean_incomplete_runs.py`로 지우고 러너를 **한 번 더** 돌린다(완료 런은 건너뛴다).
#
# **분리 세션에서 돌린다** -- 셸 도구의 시간 상한에 학습이 같이 죽은 적이 있다(원장 §5 2026-10-02 (3)):
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash tools/run_full_campaign.sh \
#       > runs/99_full_campaign/campaign.log 2>&1 < /dev/null &
#
# 리허설(1 epoch, 시드 0, LOSO 1 fold):
#   ROOT=runs/99_rehearsal NUM_EPOCHS=1 PRETRAIN_EPOCHS=1 SEEDS=0 FOLDS=raws1 bash tools/run_full_campaign.sh
#
# 일부만: PHASES="01_fixed 01_loso 01_analysis 02 02b 03" 중 고른다(공백 구분).
set -u
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- 'bev-chamdog'가 필요하다." >&2
    echo "    (base(3.14)에서는 numpy가 256 KB 넘는 배열을 조용히 덮어쓴다)" >&2
    exit 1
fi

ROOT="${ROOT:-runs/99_full_campaign}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
SEEDS="${SEEDS:-0,1,2,3,4}"
FOLDS="${FOLDS:-raws1,raws2,raws3,rawos1,rawos2,rawos3,rawos4}"
PHASES="${PHASES:-01_fixed 01_loso 01_analysis 02 02b 03}"
export CUDA_VISIBLE_DEVICES="${GPU}"
SEEDS_SPACE="${SEEDS//,/ }"
FOLDS_SPACE="${FOLDS//,/ }"
DELETED="${ROOT}/deleted_incomplete_runs.txt"

mkdir -p "${ROOT}"
echo "=== 캠페인 v3 시작 $(date '+%F %T') ==="
echo "  루트=${ROOT}  epochs=${NUM_EPOCHS}  pretrain=${PRETRAIN_EPOCHS}  seeds=${SEEDS}  folds=${FOLDS}"
echo "  phases=${PHASES}  GPU=${GPU}  커밋=$(git rev-parse --short HEAD)  환경=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h . | tail -1

want() { [[ " ${PHASES} " == *" $1 "* ]]; }

run_phase() {   # run_phase <이름> <명령...>
    local label="$1"; shift
    echo ""
    echo "##################################################################"
    echo "## ${label}   시작 $(date '+%F %T')"
    echo "##################################################################"
    "$@"
    local rc=$?
    echo "## ${label} 종료 $(date '+%F %T')  rc=${rc}"
    echo "PHASE_DONE ${label} rc=${rc}"
    return 0   # 한 단계가 실패해도 나머지는 계속 돌린다 -- 사람이 없다
}

train_with_retry() {   # train_with_retry <이름> <실험 루트> <명령...>
    local label="$1" root="$2"; shift 2
    run_phase "${label}" "$@"
    # 끊긴 런을 지우고 한 번 더. 완료 런은 러너가 건너뛰므로 남은 것만 다시 돈다.
    local n
    n="$(python tools/clean_incomplete_runs.py --root="${root}" --num_epochs="${NUM_EPOCHS}" \
            --record="${DELETED}" | grep -c '불완전 -> 삭제' ; true)"
    if [ "${n}" != "0" ]; then
        echo "!!! ${label}: 끊긴 런 ${n}개를 지우고 한 번 더 돌린다 (${DELETED})"
        run_phase "${label}_retry" "$@"
        python tools/clean_incomplete_runs.py --root="${root}" --num_epochs="${NUM_EPOCHS}" --dry_run
    fi
}

if want 01_fixed; then
    train_with_retry "01_fixed_split" "${ROOT}/01_overall/fixed_split" \
        python tools/run_paper_final_overall.py --protocol=fixed --seeds="${SEEDS}" \
            --num_epochs="${NUM_EPOCHS}" --gpu="${GPU}" --output_root="${ROOT}/01_overall"
fi
if want 01_loso; then
    train_with_retry "01_loso" "${ROOT}/01_overall/loso" \
        python tools/run_paper_final_overall.py --protocol=loso --seeds="${SEEDS}" \
            --folds="${FOLDS}" --num_epochs="${NUM_EPOCHS}" --gpu="${GPU}" \
            --output_root="${ROOT}/01_overall"
fi
if want 01_analysis; then
    run_phase "01_analysis" env ROOT="${ROOT}/01_overall" SEEDS="${SEEDS}" FOLDS="${FOLDS_SPACE}" \
        FIXED_EPOCH="${NUM_EPOCHS}" bash configs/paper_final_overall_analysis.sh
fi

CONTROL="${ROOT}/01_overall/fixed_split/logs"
BASELINE_PATH="${ROOT}/01_overall/fixed_split/analysis/constant_map_baseline.json"

if want 02; then
    train_with_retry "02_projection_and_prior" "${ROOT}/02_projection_and_prior" \
        python tools/run_paper_final_projection_prior.py --seeds="${SEEDS}" \
            --num_epochs="${NUM_EPOCHS}" --pretrain_epochs="${PRETRAIN_EPOCHS}" --gpu="${GPU}" \
            --output_root="${ROOT}/02_projection_and_prior"
    run_phase "02_analysis" env ROOT="${ROOT}/02_projection_and_prior" CONTROL="${CONTROL}" \
        BASELINE_PATH="${BASELINE_PATH}" SEEDS="${SEEDS}" FIXED_EPOCH="${NUM_EPOCHS}" \
        bash configs/paper_final_projection_prior_analysis.sh
fi
if want 02b; then
    train_with_retry "02b_native_source_prior" "${ROOT}/02b_native_source_prior" \
        python tools/run_paper_final_native_prior.py --seeds="${SEEDS}" \
            --num_epochs="${NUM_EPOCHS}" --pretrain_epochs="${PRETRAIN_EPOCHS}" --gpu="${GPU}" \
            --output_root="${ROOT}/02b_native_source_prior"
    run_phase "02b_analysis" env ROOT="${ROOT}/02b_native_source_prior" CONTROL="${CONTROL}" \
        BASELINE_PATH="${BASELINE_PATH}" SEEDS="${SEEDS}" FIXED_EPOCH="${NUM_EPOCHS}" \
        bash configs/paper_final_native_prior_analysis.sh
fi
if want 03; then
    train_with_retry "03_boundary_uncertainty" "${ROOT}/03_boundary_uncertainty" \
        env OUT_ROOT="${ROOT}/03_boundary_uncertainty" NUM_EPOCHS="${NUM_EPOCHS}" \
            SEEDS="${SEEDS_SPACE}" bash configs/loss_effect.sh
    run_phase "03_analysis" env ROOT="${ROOT}/03_boundary_uncertainty" SEEDS="${SEEDS}" \
        FIXED_EPOCH="${NUM_EPOCHS}" bash configs/loss_effect_analysis.sh
fi

echo ""
echo "############ 결과 트리 순수성 ############"
python tools/check_results_tree_purity.py --targets="${ROOT}" --expect="runs/$(basename "${ROOT}")"
echo ""
echo "############ 완료 런 수 ############"
for ckpt in $(find "${ROOT}" -type d -name ckpt | sort); do
    total=$(find "${ckpt}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    done_n=$(find "${ckpt}" -mindepth 2 -maxdepth 2 -name 'model-0*.pth' | wc -l)
    echo "  ${ckpt}: 런 ${total}, 마지막 체크포인트 ${done_n}"
done
echo ""
echo "=== 캠페인 v3 종료 $(date '+%F %T') ==="
df -h . | tail -1
echo "CAMPAIGN_V3_DONE"
