#!/usr/bin/env bash
# `runs/loss_effect`의 20런을 **원시 데이터 보존 -> 집계** 순서로 전부 훑는다.
#
# 계획 정본: `docs/temp/loss_stability_evaluation_plan_for_claude_2026-09-01.md`.
# 학습 스크립트: `configs/loss_effect.sh`.
#
# ## 순서에 이유가 있다
#
#   1. 원시 보존 -- `scalars.csv`(모든 epoch x 모든 태그)와 `predictions/`(val 확률맵 원본)
#   2. 집계     -- 다섯 도구. 전부 stdout을 `.txt`로 남기고 가능한 것은 CSV/JSON도 낸다
#
# **보존을 먼저 하는 이유**: 집계 도구는 각자 필요한 것만 골라 읽으므로, 나중에 다른 질문이
# 생기면 체크포인트를 다시 올려야 한다. 확률맵을 먼저 떨어뜨려 두면 체크포인트 없이도
# 어떤 τ·어떤 영역·어떤 지표든 다시 계산할 수 있다(런당 약 2 MB, 체크포인트의 1/235).
#
# ## 산출물 (전부 `runs/loss_effect/analysis/`)
#
#   scalars.csv                 런 x 태그 x epoch 전부 (long format)
#   configs.json                런마다 config.json 전체
#   manifest.json               런마다 epoch 수·태그 목록·완주 여부
#   predictions/labels.npz      GT free/occupied/unknown/valid + 부호 거리장 d
#   predictions/{run}__best.npz 선택 체크포인트의 val p(free) (float16)
#   predictions/{run}__last.npz 마지막 epoch의 val p(free)
#   verify_predictions.json     확률맵 vs 학습 로그 대조 결과 (무결성)
#   seed_jitter/*.npz           광선별 R̂와 RAY_OK (도구가 자동 캐시)
#   *.txt                       각 도구의 출력 그대로
#   per_sequence.json           시퀀스별·시드별 지표
#   RESULTS.json                ★ 핵심 결과 전부를 담은 단일 파일
#   RESULTS.csv                 ★ 같은 내용의 긴 형식 표
#   README.md                   폴더 안내(어느 파일이 무엇인지)
#   threshold_sweep_rows.csv    셀 x 시드 x τ 의 모든 지표
#   seed_jitter.json            광선 산포 요약
#   decision_disagreement.json  결정 불일치 요약
#
# 실행: CUDA_VISIBLE_DEVICES=0 bash configs/loss_effect_analysis.sh
set -uo pipefail          # `-e`를 뺀다 -- 도구 하나가 죽어도 나머지는 돌아야 한다.
cd "$(dirname "$0")/.."

# **환경을 강제한다** -- 학습 스크립트와 같은 이유다(`configs/train_robot_bev_finetune.sh`의
# 주석). 분석을 다른 환경에서 돌리면 지표 계산 경로가 달라지고, 실제로 `base`(Python 3.14)의
# numpy는 큰 배열을 조용히 덮어썼다(`projects/common/npsafe.py`).
REQUIRED_CONDA_ENV="${REQUIRED_CONDA_ENV:-bev-chamdog}"
if [ "${ALLOW_ANY_ENV:-0}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "${REQUIRED_CONDA_ENV}" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- '${REQUIRED_CONDA_ENV}'가 필요하다." >&2
    echo "    conda activate ${REQUIRED_CONDA_ENV}   (또는 의도한 것이면 ALLOW_ANY_ENV=1)" >&2
    exit 1
fi

ROOT="${ROOT:-runs/loss_effect}"
CELLS="${CELLS:-A_ce,B_perset,C_hard,C_soft,D_range}"
SEEDS="${SEEDS:-0,1,2,3,4}"
OUT="${ROOT}/analysis"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT}"

step() {   # step <이름> <명령...>
    local name="$1"; shift
    echo ""
    echo "############ ${name}  ($(date +%H:%M:%S)) ############"
    # `tee`로 화면과 파일에 동시에 남긴다 -- 사람이 지켜보지 않아도 전문이 보존된다.
    "$@" 2>&1 | tee "${OUT}/${name}.txt"
    local rc="${PIPESTATUS[0]}"
    [ "${rc}" -ne 0 ] && echo "!!! ${name} 실패 (exit ${rc}) -- 계속 진행한다"
    return 0
}

echo "=== loss_effect 분석 시작 ($(date +%F' '%H:%M:%S)) ==="
echo "    root=${ROOT}  cells=${CELLS}  seeds=${SEEDS}  GPU=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# ── 1. 원시 보존 ────────────────────────────────────────────────────────────────
step export_run_scalars \
    python tools/export_run_scalars.py --log_root="${ROOT}/logs" --out_dir="${OUT}"

step export_val_predictions \
    python tools/export_val_predictions.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --which=best,last

# **무결성 검사 -- 여기서 실패하면 아래 집계를 믿으면 안 된다.**
# 저장된 확률맵이 학습 루프가 그 epoch에 기록한 val 지표를 재현하는지 대조한다. 전혀 다른
# 코드 경로로 잰 두 값이 맞으면 행 순서·기하·체크포인트 선택이 전부 맞은 것이다.
step verify_val_predictions \
    python tools/verify_val_predictions.py --root="${ROOT}"

# ── 2. 집계 ─────────────────────────────────────────────────────────────────────
# (2-1) 계획 §2: 수렴·되올림·Δep·regret + §3.1 품질 + 재현성 F 검정
# `--sigma_from=None`: 노이즈 바닥을 **이 실험 자신의 pooled 표준편차**(4셀 x n=5, df=16)로
# 잡는다. 기본값은 `runs/archive/`의 옛 σ_run 런들인데 그것은 `Y=1`·옛 loss config라
# 여기 가져오면 다른 기하의 노이즈를 기준으로 삼게 된다.
step report_ablation \
    python tools/report_ablation.py --log_root="${ROOT}/logs" --sigma_from=None

# (2-2) 계획 §3.1: 고정 epoch과 best epoch을 나란히. **논문 주 숫자는 고정 epoch 쪽이다.**
# `--group_by=exp_name`이 곧 셀별 묶음이다(`EXP_NAME`이 셀 이름으로 들어간다).
step summarize_repeats_fixed40 \
    python tools/summarize_repeats.py --log_root="${ROOT}/logs" --fixed_epoch=40 \
        --group_by=exp_name

# (2-3) 계획 §3.2/§4: 전역 편향 산포 / 광선별 산포 / 편향 제거 후 국소 산포
step report_seed_jitter \
    python tools/report_seed_jitter.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" \
        --json_out="${OUT}/seed_jitter.json"

# (2-4) 계획 §6: 같은 free_miss에서의 비교 + 목표 동작점을 맞추는 τ의 시드 산포
#
# **τ 격자를 0.1~0.9로 넓힌다.** 동작점을 맞춘 비교는 네 칸의 `free_miss` 곡선이 **동시에
# 덮는 구간**에서만 가능한데, `B_perset`은 경계 감독이 없어 곡선이 다른 자리에 놓일 수 있다.
# 격자가 좁으면 그 칸 하나 때문에 겹치는 구간이 비고, 그러면 **판정 표 전체가 사라진다.**
WIDE_TAUS="0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90"

step report_threshold_sweep \
    python tools/report_threshold_sweep.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --taus="${WIDE_TAUS}" \
        --csv_out="${OUT}/threshold_sweep_rows.csv"

# **주 비교쌍만 따로 한 번 더 돌린다.** 위가 네 칸의 공통 구간에 묶이는 반면 이쪽은
# `A_ce`와 `D_range`만 겹치면 되므로 구간이 넓고, 그래서 **판정 표가 살아남는다.**
# 이 실험의 주 질문("기존 BCE 대 튜닝한 loss")에 답하는 표는 이쪽이다.
step report_threshold_sweep_AD \
    python tools/report_threshold_sweep.py --log_root="${ROOT}" \
        --cells=A_ce,D_range --seeds="${SEEDS}" --taus="${WIDE_TAUS}" \
        --csv_out="${OUT}/threshold_sweep_rows_AD.csv"

# (2-5) 계획 §5: 셀 단위 free<->non-free 뒤집힘. 고정 τ와 동작점 정합 둘 다.
step report_decision_disagreement \
    python tools/report_decision_disagreement.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" \
        --json_out="${OUT}/decision_disagreement.json"

# 고정 epoch 체크포인트에서도 같은 것을 잰다 -- **재현성 주장은 두 기준에서 모두 성립해야
# 한다**(선택 epoch은 argmax라 선택 규칙의 성질이 섞인다).
step report_decision_disagreement_last \
    python tools/report_decision_disagreement.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --which=last \
        --json_out="${OUT}/decision_disagreement_last.json"

step report_decision_disagreement_AD \
    python tools/report_decision_disagreement.py --log_root="${ROOT}" \
        --cells=A_ce,D_range --seeds="${SEEDS}" \
        --json_out="${OUT}/decision_disagreement_AD.json"

# (2-6) 계획 §7 + §11-1: 시퀀스별 분해와 시드별 표. 확률맵에서 재므로 CPU만 쓴다.
step report_per_sequence \
    python tools/report_per_sequence.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --which=best \
        --json_out="${OUT}/per_sequence.json"

# ── 3. 결과를 파일 하나로 묶는다 ────────────────────────────────────────────────
# 위 산출물은 도구마다 형식이 다르다. 여기서 `RESULTS.json`(단일 결과 파일) ·
# `RESULTS.csv`(긴 형식 표) · `README.md`(폴더 안내)로 한 겹 얹는다.
step build_results_bundle \
    python tools/build_results_bundle.py --root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}"

# ── 4. 목록 ─────────────────────────────────────────────────────────────────────
echo ""
echo "############ 산출물 ($(date +%H:%M:%S)) ############"
find "${OUT}" -type f -printf '%10s  %p\n' | sort -k2
du -sh "${OUT}"
echo "=== LOSS_EFFECT_ANALYSIS_DONE ($(date +%F' '%H:%M:%S)) ==="
