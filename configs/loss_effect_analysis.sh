#!/usr/bin/env bash
# `runs/loss_effect`(또는 `ROOT`)의 사다리 런을 **원시 데이터 보존 -> 집계** 순서로 전부 훑는다.
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
#   *.txt                       각 도구의 출력 그대로
#   RESULTS.json                ★ 핵심 결과 전부를 담은 단일 파일
#   RESULTS.csv                 ★ 같은 내용의 긴 형식 표
#   README.md                   폴더 안내(어느 파일이 무엇인지)
#   threshold_sweep_rows.csv    셀 x 시드 x τ 의 모든 지표
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
# **[2026-10-02] 광선 보조항을 뺐다 -- 사다리는 `C_soft`까지 넷이다**(사용자 결정).
# 옛 `D_range`·`E_cumulative` 런을 다시 훑을 때는 `CELLS`로 넘기되, 아래 도구들은 새 지표
# tag(`projects/common/metric_spec.py`)를 요구하므로 옛 런에는 맞지 않는다.
CELLS="${CELLS:-A_ce,B_perset,C_hard,C_soft}"
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
# **[2026-10-02] 패키지(`docs/paper_package/03_boundary_uncertainty`)가 실제로 읽는 것만
# 남겼다**: 고정 epoch 집계와 τ 스윕. 아래 도구들은 옛 지표(`fatal_rate`·`free_miss_rate`·
# `range_*`·`f1@τ` micro)로 짜여 있어 파이프라인에서 뺐다 -- 코드는 옛 런 재분석용으로 남는다.
#   report_ablation              (σ_run 대조 -- 옛 지표 tag)
#   report_seed_jitter           (광선 거리 산포 = 방위각 거리 지표, 사용자가 뺐다)
#   report_cumulative_probe      (보조항 팔 전용 -- 보조항을 뺐다)
#   report_decision_disagreement (목표 free_miss 정합 -- recall로 옮기지 않았다)
#   report_per_sequence          (옛 지표 dict)
# 이들이 만들던 RESULTS.json의 축 3·4와 `per_sequence`는 비고(`_missing`에 적힌다), 패키지는
# 그 축을 쓰지 않는다.

# (2-1) 계획 §3.1: 고정 epoch과 best epoch을 나란히. **논문 주 숫자는 고정 epoch 쪽이다.**
# `--group_by=exp_name`이 곧 셀별 묶음이다(`EXP_NAME`이 셀 이름으로 들어간다).
step summarize_repeats_fixed40 \
    python tools/summarize_repeats.py --log_root="${ROOT}/logs" --fixed_epoch="${FIXED_EPOCH:-100}" \
        --group_by=exp_name

# (2-2) 계획 §6: 같은 recall에서의 precision 비교 + 목표 동작점을 맞추는 τ의 시드 산포
#
# **τ 격자를 0.1~0.9로 넓힌다.** 동작점을 맞춘 비교는 네 칸의 recall 곡선이 **동시에
# 덮는 구간**에서만 가능한데, `B_perset`은 경계 감독이 없어 곡선이 다른 자리에 놓일 수 있다.
# 격자가 좁으면 그 칸 하나 때문에 겹치는 구간이 비고, 그러면 **판정 표 전체가 사라진다.**
WIDE_TAUS="0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90"

step report_threshold_sweep \
    python tools/report_threshold_sweep.py --log_root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --taus="${WIDE_TAUS}" \
        --csv_out="${OUT}/threshold_sweep_rows.csv"

# ── 3. 결과를 파일 하나로 묶는다 ────────────────────────────────────────────────
# 위 산출물은 도구마다 형식이 다르다. 여기서 `RESULTS.json`(단일 결과 파일) ·
# `RESULTS.csv`(긴 형식 표) · `README.md`(폴더 안내)로 한 겹 얹는다.
step build_results_bundle \
    python tools/build_results_bundle.py --root="${ROOT}" \
        --cells="${CELLS}" --seeds="${SEEDS}" --fixed_epoch="${FIXED_EPOCH:-100}"

# ── 4. 목록 ─────────────────────────────────────────────────────────────────────
echo ""
echo "############ 산출물 ($(date +%H:%M:%S)) ############"
find "${OUT}" -type f -printf '%10s  %p\n' | sort -k2
du -sh "${OUT}"
echo "=== LOSS_EFFECT_ANALYSIS_DONE ($(date +%F' '%H:%M:%S)) ==="
