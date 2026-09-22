#!/usr/bin/env bash
# 재학습한 캠페인의 **Phase 3** -- 패키지를 다시 만들고 **덮어쓰기 전에 대조한다**.
#
# ## 기본은 덮어쓰지 않는다
#
# 임시 폴더에 만들고 `tools/diff_paper_package_csv.py`로 칸 단위 대조만 출력한다.
# 바뀐 칸을 읽고 납득한 뒤에 `APPLY=1`로 다시 돌려야 실제 패키지가 바뀐다.
# 이유: **판정이 뒤집히는 칸은 그 자체가 보고 대상**이라서 조용히 갈아끼우면 안 된다(원장 §8).
#
# ## 03은 손대지 않는다
#
# `runs/loss_effect`는 재학습 대상이 아니다. `A_ce`~`C_soft`는 λ_R=0이라 보조항이
# 관여하지 않았고, `E_cumulative`는 처음부터 `cumulative_l1`이다.
#
# 실행:
#   bash tools/run_paper_final_cumulative_packages.sh            # 대조만
#   APPLY=1 bash tools/run_paper_final_cumulative_packages.sh    # 실제 반영
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- 'bev-chamdog'가 필요하다." >&2
    exit 1
fi

ROOT="${ROOT:-$PWD/runs/paper_final_cumulative}"
STAGE="${STAGE:-$PWD/runs/paper_final_cumulative/_package_stage}"
APPLY="${APPLY:-0}"
PKG="$PWD/docs/paper_package"
CONTROL="${ROOT}/01_overall/fixed_split/logs"
BASELINE="${ROOT}/01_overall/fixed_split/analysis/constant_map_baseline.json"
REL_TOL="${REL_TOL:-0.02}"
# 판정 눈금. **사전 등록값을 유지한다** -- 새 대조군에서 다시 재면 0.0007이 나오지만
# 그건 5표본 sd의 흔들림이다(원장 §8). 넘기지 않으면 02b 생성기가 대조군에서 직접
# 재 버리므로 여기서 명시한다.
SIGMA_SEED="${SIGMA_SEED:-0.0018}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
echo "=== Phase 3: 패키지 재생성 $(date '+%F %T') ==="
echo "  결과 루트=${ROOT}"
echo "  임시 폴더=${STAGE}"
echo "  APPLY=${APPLY}  (0이면 대조만 하고 아무것도 덮어쓰지 않는다)"

echo ""
echo "############ 01_overall ############"
( cd "${PKG}/01_overall" && python make_package.py \
    --root="${ROOT}/01_overall" --out_dir="${STAGE}/01_overall" )
python tools/diff_paper_package_csv.py --old="${PKG}/01_overall" \
    --new="${STAGE}/01_overall" --rel_tol="${REL_TOL}"

echo ""
echo "############ 02_projection_and_prior ############"
( cd "${PKG}/02_projection_and_prior" && python make_package.py \
    --root="${ROOT}/02_projection_and_prior" --out_dir="${STAGE}/02_projection_and_prior" \
    --control="${CONTROL}" --baseline_path="${BASELINE}" )
python tools/diff_paper_package_csv.py --old="${PKG}/02_projection_and_prior" \
    --new="${STAGE}/02_projection_and_prior" --rel_tol="${REL_TOL}"

echo ""
echo "############ 02b_native_source_prior (신규) ############"
# 대조할 기존 패키지가 없다. 바로 제자리에 만든다 -- 덮어쓸 것이 없으므로 안전하다.
( cd "${PKG}/02b_native_source_prior" && python make_package.py \
    --root="${ROOT}/02b_native_source_prior" \
    --control="${CONTROL}" \
    --adapted="${ROOT}/02_projection_and_prior/source_prior/logs" \
    --baseline_path="${BASELINE}" --sigma_seed="${SIGMA_SEED}" )

if [ "${APPLY}" = "1" ]; then
    echo ""
    echo "############ 반영 ############"
    for name in 01_overall 02_projection_and_prior; do
        cp -v "${STAGE}/${name}/data/"*.csv "${PKG}/${name}/data/"
        cp -v "${STAGE}/${name}/provenance/"* "${PKG}/${name}/provenance/"
    done

    # **그림도 같이 다시 만든다.** CSV만 갈고 그림을 두면 문서 안에서 숫자와 그림이
    # 어긋나는데, 그건 눈으로 잡히지 않는다. 생성 스크립트가 그림의 정본이다.
    echo ""
    echo "############ 그림 재생성 ############"
    for name in 01_overall 02_projection_and_prior 02b_native_source_prior; do
        for script in "${PKG}/${name}/figures/"make_*.py; do
            [ -e "${script}" ] || continue
            echo "  ${name}/$(basename "${script}")"
            ( cd "${PKG}/${name}/figures" && python "$(basename "${script}")" )
        done
    done
    echo "반영 완료."
else
    echo ""
    echo "대조만 했다. 바뀐 칸을 확인한 뒤 APPLY=1로 다시 돌려라."
fi

# **패키지 산출물에도 옛 트리가 섞이지 않았는지 확인한다.**
# `03`은 일부러 `runs/loss_effect`를 가리키므로 무시 목록에 넣는다.
echo ""
echo "############ 결과 트리 순수성 ############"
python tools/check_results_tree_purity.py \
    --targets="${STAGE}/01_overall,${STAGE}/02_projection_and_prior,${PKG}/02b_native_source_prior" \
    --expect="runs/$(basename "${ROOT}")" --ignore="runs/loss_effect"

echo ""
echo "=== Phase 3 종료 $(date '+%F %T') ==="
echo "CAMPAIGN_PACKAGES_DONE apply=${APPLY}"
