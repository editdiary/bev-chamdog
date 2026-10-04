#!/usr/bin/env bash
# **캠페인 v3 Phase 3** -- 논문 패키지 `docs/99_paper_results/` 생성 + 그림 재생성.
#
# 새 폴더라 덮어쓸 옛 것이 없다(옛 캠페인은 `docs/archive/paper_package_v2/`에 그대로 있다). 생성기마다
# 무결성 게이트가 있어 하나라도 실패하면 그 실험은 아무것도 쓰지 않는다.
#
# 리허설: ROOT=runs/99_rehearsal OUT=<임시 폴더> SEEDS=0 FIXED_EPOCH=1 REHEARSAL=1 bash tools/run_full_campaign_packages.sh
set -u
cd "$(dirname "$0")/.."
if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- 'bev-chamdog'가 필요하다." >&2
    exit 1
fi
REPO="$PWD"
ROOT="$(cd "${ROOT:-runs/99_full_campaign}" && pwd)"
PKG="${REPO}/docs/99_paper_results"
OUT="${OUT:-${PKG}}"
SEEDS="${SEEDS:-0,1,2,3,4}"
FIXED_EPOCH="${FIXED_EPOCH:-100}"
REHEARSAL="${REHEARSAL:-0}"
CONTROL="${ROOT}/01_overall/fixed_split/logs"
BASELINE="${ROOT}/01_overall/fixed_split/analysis/constant_map_baseline.json"
echo "=== 패키지 생성 $(date '+%F %T') | 루트=${ROOT} -> ${OUT} | seeds=${SEEDS} epoch=${FIXED_EPOCH} ==="

fail=0
gen() {   # gen <실험> <make_package 인자...>
    local name="$1"; shift
    mkdir -p "${OUT}/${name}"
    echo ""
    echo "############ ${name} ############"
    ( cd "${PKG}/${name}" && python make_package.py --out_dir="${OUT}/${name}" "$@" )
    local rc=$?
    echo "PACKAGE_DONE ${name} rc=${rc}"
    [ "${rc}" -ne 0 ] && fail=1
    return 0
}

gen 01_overall --root="${ROOT}/01_overall" --rehearsal="$([ "${REHEARSAL}" = 1 ] && echo True || echo False)"
gen 02_projection_and_prior --root="${ROOT}/02_projection_and_prior" --control="${CONTROL}" \
    --baseline_path="${BASELINE}" --seeds="${SEEDS}" --fixed_epoch="${FIXED_EPOCH}"
gen 02b_native_source_prior --root="${ROOT}/02b_native_source_prior" --control="${CONTROL}" \
    --adapted="${ROOT}/02_projection_and_prior/source_prior/logs" --baseline_path="${BASELINE}" \
    --seeds="${SEEDS}" --fixed_epoch="${FIXED_EPOCH}"
gen 03_boundary_uncertainty --root="${ROOT}/03_boundary_uncertainty" --rerun_twin="${CONTROL}" \
    --seeds="${SEEDS}" --fixed_epoch="${FIXED_EPOCH}"

# 그림: 생성기 옆 figures/의 스크립트가 `../data/`를 읽는다. 리허설은 OUT 쪽에 복사해서 돌린다.
echo ""
echo "############ 그림 ############"
for name in 01_overall 02_projection_and_prior 03_boundary_uncertainty; do
    [ -d "${PKG}/${name}/figures" ] || continue
    if [ "${OUT}" != "${PKG}" ]; then
        mkdir -p "${OUT}/${name}/figures"
        cp "${PKG}/${name}/figures/"make_*.py "${OUT}/${name}/figures/"
    fi
    for script in "${OUT}/${name}/figures/"make_*.py; do
        ( cd "${OUT}/${name}/figures" && python "$(basename "${script}")" )
        rc=$?
        echo "FIGURE_DONE ${name}/$(basename "${script}") rc=${rc}"
        [ "${rc}" -ne 0 ] && fail=1
    done
done

echo ""
echo "=== 패키지 생성 종료 $(date '+%F %T') fail=${fail} ==="
echo "PACKAGES_DONE fail=${fail}"
exit "${fail}"
