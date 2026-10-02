#!/usr/bin/env bash
# **epoch 길이 사전 실험** (2026-10-02 사용자 요청) -- 40 epoch 대 100 epoch에서 학습 곡선의 모양.
#
# 묻는 것: 40 epoch은 모델이 튜닝되기 전에 관습적으로 정한 값이다. 100 epoch으로 늘리면 곡선이
# 어떻게 되는가. **기존 결과와 비교하지 않는다** -- 이 실험만으로 새로 기록한다(사용자 결정).
#
#   칸      A_ce   = 일반(가중) BCE, 대조군        λ_B = 0, λ_R = 0
#           C_soft = 제안 soft-BCE                 δ = 0.30, σ = 0.10, λ_B = 0.5, λ_R = 0
#   epoch   40 / 100        시드 0 하나 (반복 없음)        = 4런
#
# 설정은 `configs/loss_effect.sh`의 칸 표에서 그대로 온다(이 스크립트는 칸과 epoch만 고른다).
# 지표는 2026-10-02 개편본(`projects/common/metric_spec.py`), 광선 보조항은 꺼져 있다.
#
# **읽을 때 주의.** LR 스케줄이 `OneCycleLR`이라 **스케줄 길이가 `NUM_EPOCHS`에 묶여 있다.**
# 100 epoch 런의 epoch 40은 40 epoch 런의 끝과 **같은 상태가 아니다**(그 시점의 LR이 다르다).
# 그래서 40 epoch을 따로 돌린다.
#
# 실행:   CUDA_VISIBLE_DEVICES=0 bash configs/epoch_length_probe.sh
# 결과:   runs/99_epoch_exp/{ep040,ep100}/{logs,ckpt}/{A_ce,C_soft}_s0
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- bev-chamdog가 필요하다" >&2
    exit 1
fi

ROOT="${ROOT:-runs/99_epoch_exp}"
CELLS="${CELLS:-A_ce C_soft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${ROOT}"

echo "=== epoch 길이 사전 실험 시작 ($(date +%F' '%H:%M:%S)) | GPU ${CUDA_VISIBLE_DEVICES} | 칸 ${CELLS} ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# epoch|상한(초). 실측 약 8.1 s/epoch x 2런 + 시작 준비의 약 2.5배를 상한으로 둔다 --
# 학습이 멈추면 이 시간 뒤에 프로세스가 죽고 다음 묶음으로 넘어간다(무한 대기 방지).
for spec in "40|2400" "100|5400"; do
    IFS='|' read -r epochs limit <<< "${spec}"
    tag="$(printf 'ep%03d' "${epochs}")"
    echo ""
    echo "############ ${tag} ($(date +%H:%M:%S), 상한 ${limit}s) ############"
    OUT_ROOT="${ROOT}/${tag}" NUM_EPOCHS="${epochs}" SEEDS=0 ONLY_CELLS="${CELLS}" \
        timeout --kill-after=120 "${limit}" bash configs/loss_effect.sh
    rc=$?
    [ "${rc}" -eq 124 ] && echo "!!! ${tag} 시간 상한 초과 -- 강제 종료했다"
    echo "=== ${tag} 종료 rc=${rc} ($(date +%H:%M:%S)) ==="
done

echo ""
echo "=== 완료 런 ==="
for d in "${ROOT}"/ep*/ckpt/*; do
    [ -d "${d}" ] && echo "  ${d}: $(ls "${d}" | tr '\n' ' ')"
done
echo "EPOCH_PROBE_DONE ($(date +%F' '%H:%M:%S))"
