#!/usr/bin/env bash
# **높이 범위 사전 실험** (2026-10-02 사용자 요청) -- BEV feature를 만들 때 표본하는 높이 범위를
# 바꾸면 결과가 달라지는가.
#
#   설정 (모두 bin 간격 0.5 m)              표본 높이 [m, 지면 = 0]
#   y4_-0.25_1.75   Y=4 [-0.25, 1.75] (확정값)  0, 0.5, 1.0, 1.5
#   y8_-2.00_2.00   Y=8 [-2, 2]                -1.75 ... 1.75
#   y8_-1.00_3.00   Y=8 [-1, 3]                -0.75 ... 2.75
#
# 칸은 `C_soft`(제안 soft-BCE) 하나, 시드 0, 100 epoch(`tools/paper_final_epochs.py`). 나머지 설정은
# `configs/loss_effect.sh`의 칸 표 그대로다 -- 이 스크립트는 높이 셋만 바꾼다(`PROBE_HEIGHT_*`).
# **기존 결과와 비교하지 않는다**(사용자 결정).
#
# 학습은 셸 도구가 아니라 분리된 세션에서 돌린다(원장 §5 "2026-10-02 (3)"의 운영 메모):
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash configs/height_range_probe.sh \
#       > runs/99_height_exp/driver.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "${CONDA_DEFAULT_ENV:-}" != "bev-chamdog" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- bev-chamdog가 필요하다" >&2
    exit 1
fi

ROOT="${ROOT:-runs/99_height_exp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${ROOT}"
echo "=== 높이 범위 사전 실험 시작 ($(date +%F' '%H:%M:%S)) | GPU ${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 이름|Y|최소|최대|상한(초). 상한은 Y=4 실측(100 epoch 약 14분)의 4배 이상이다 -- Y=8은 3D 볼륨이
# 두 배라 더 걸린다. 학습이 멈추면 이 시간 뒤에 죽고 다음으로 넘어간다(무한 대기 방지).
for spec in "y4_-0.25_1.75|4|-0.25|1.75|3600" \
            "y8_-2.00_2.00|8|-2.0|2.0|5400" \
            "y8_-1.00_3.00|8|-1.0|3.0|5400"; do
    IFS='|' read -r tag bins hmin hmax limit <<< "${spec}"
    echo ""
    echo "############ ${tag} ($(date +%H:%M:%S), 상한 ${limit}s) ############"
    OUT_ROOT="${ROOT}/${tag}" SEEDS=0 ONLY_CELLS=C_soft \
        PROBE_HEIGHT_BINS="${bins}" PROBE_HEIGHT_MIN_M="${hmin}" PROBE_HEIGHT_MAX_M="${hmax}" \
        timeout --kill-after=120 "${limit}" bash configs/loss_effect.sh
    rc=$?
    [ "${rc}" -eq 124 ] && echo "!!! ${tag} 시간 상한 초과 -- 강제 종료했다"
    echo "=== ${tag} 종료 rc=${rc} ($(date +%H:%M:%S)) ==="
done

echo ""
echo "=== 완료 런 ==="
for d in "${ROOT}"/*/ckpt/*; do
    [ -d "${d}" ] && echo "  ${d}: $(ls "${d}" | tr '\n' ' ')"
done
echo "HEIGHT_PROBE_DONE ($(date +%F' '%H:%M:%S))"
