#!/usr/bin/env bash
# **특징맵 표본 좌표 스윕** -- 진단 문서 §18.3의 결함을 고친 뒤 남는 상수 편이를 실측한다.
# 산출물은 `runs/pixel_offset/`.
#
# ## 무엇을 재는가
#
# 결함은 둘로 나뉜다(`projects/models/pixel_grid.py`).
#
#   (A) **정규화 규약 불일치** -- `normalize_grid2d`(픽셀 인덱스)와
#       `grid_sample(align_corners=False)`(픽셀 가장자리)를 섞어 써서 표본 위치가
#       `x*W/(W-1) - 0.5`가 된다. **배율 오차라 오프셋으로 못 없앤다.** 이건 정답이 있는
#       correctness 문제이고 `pixel_center`가 그 정답이다
#       (`tests/models/test_pixel_grid.py`가 `x' = x`를 정확히 고정한다).
#
#   (B) **남는 상수 편이** -- native -> 512x288 리사이즈 규약, stride 8 특징맵의 수용영역
#       중심, `Encoder_res101.upsampling_layer`(stride16 -> stride8 보간)가 각각 자기 규약을
#       더한다. 두 극단을 계산하면 **-0.475**(리사이즈 half-pixel + 수용영역 중심 8j+3.5)와
#       **-0.0375**(중심 8j)로 갈린다. **유도로 확정되지 않으므로 스윕으로 잰다.**
#
# 이 스윕은 (A)를 고정하고 (B)를 잰다. 사전 예측: 최적점이 음수 쪽 [-0.5, 0]에 있다.
#
# ## 대조군
#
# `runs/ablation/D_range_s{0,1,2}`가 그대로 대조군이다(`legacy_index`, offset 0).
# **같은 코드다** -- 학습 경로(`tools/train_robot_bev.py`, `projects/`,
# `configs/train_robot_bev_finetune.sh`)의 마지막 변경이 그 ablation을 만든 커밋
# 23c0660이고, 이번 규약 작업은 기본값을 건드리지 않았다(단위 테스트가 `legacy_index`의
# 비트 단위 동일성을 고정한다). 그래도 refactor가 end-to-end로 무해한지 보려고
# `legacy_s0` 한 런을 smoke check로 같이 돌린다 -- `D_range_s0`과 σ_run 안에서 같아야 한다.
#
# ## 왜 시드 3개인가
#
# n=1로는 못 읽는다. `D_range`의 시드 간 σ가 `f1@10cm` 0.0067, `iou_free` 0.0010이라
# (`runs/ablation/report_n3.txt`) 단일 런 차이는 노이즈와 구별되지 않는다 -- §15의 교훈이
# 그것이고, 판정은 반드시 실측 노이즈 바닥에 대고 한다. 시드를 **바깥 루프**에 둬서
# 중간에 멈춰도 네 칸이 같은 n을 갖게 한다.
#
# 13런 x 약 10분 = 약 2시간 10분.
#
# 실행: bash configs/sweep_pixel_offset.sh
# 집계: python tools/report_pixel_offset.py
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/pixel_offset}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|PIXEL_CONVENTION|PIXEL_OFFSET
CELLS=(
    "off_m1.00|pixel_center|-1.0"
    "off_m0.50|pixel_center|-0.5"
    "off_0.00|pixel_center|0.0"
    "off_p0.50|pixel_center|0.5"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

# loss는 확정 config(D_range) 고정 -- 이 스윕이 움직이는 손잡이는 기하 하나뿐이다.
run_one() {
    local run="$1" convention="$2" offset="$3" seed="$4"
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        return 0
    fi
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${run%_s*}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
    PIXEL_CONVENTION="${convention}" PIXEL_OFFSET="${offset}" \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh

    # 주기 저장분만 지우고 `model_best`는 남긴다 (ablation과 같은 규약).
    find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
}

# refactor smoke check를 **먼저** 돌린다 -- 여기서 깨지면 나머지 12런이 통째로 무의미하다.
run_one "legacy_s0" "legacy_index" "0.0" "0"

for seed in ${SEEDS}; do
    for entry in "${CELLS[@]}"; do
        IFS='|' read -r name convention offset <<< "${entry}"
        run_one "${name}_s${seed}" "${convention}" "${offset}" "${seed}"
    done
done
echo "PIXEL_OFFSET_SWEEP_DONE"
