#!/usr/bin/env bash
# `02b_native_source_prior`의 원시 보존 -> 무결성 검사 -> 짝지은 비교.
#
# **로직을 복사하지 않는다.** `02`의 분석 스크립트를 그대로 실행하고 이름만 바꿔 넘긴다 --
# 같은 절차를 두 벌 두면 한쪽만 고쳐졌을 때 두 실험의 채점 방식이 조용히 갈린다.
#
# 다른 점은 팔 이름 둘뿐이다:
#   - 미세조정 팔  `source_prior`        -> `source_prior_native`
#   - 사전학습 팔  `source_pretrain`     -> `source_pretrain_native`
#
# 대조군은 `02`와 **같은** `01_overall`의 고정 split이다(시드 1:1로 짝지어진다).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-runs/paper_final/02b_native_source_prior}" \
CONTROL="${CONTROL:-runs/paper_final/01_overall/fixed_split/logs}" \
ARMS="${ARMS:-source_prior_native}" \
PRETRAIN_ARM="${PRETRAIN_ARM:-source_pretrain_native}" \
LABEL="${LABEL:-02b native source prior}" \
DONE_TAG="${DONE_TAG:-PAPER_FINAL_02B_ANALYSIS_DONE}" \
exec bash configs/paper_final_projection_prior_analysis.sh "$@"
