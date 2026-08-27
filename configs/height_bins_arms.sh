#!/usr/bin/env bash
# **lifting 높이 축 Y=1 -> 8 팔 (시드 3 = 3런).** 산출물 `runs/height_bins/`.
#
# 바꾸는 것은 **높이 설정 셋뿐**이다: `HEIGHT_BINS=8`, `HEIGHT_MIN_M=-0.125`,
# `HEIGHT_MAX_M=1.875`. 격자(5 cm)·encoder·loss·데이터·epoch·split은 전부 그대로다.
#
# ## 가설 (실행 전에 적었다, 2026-08-27)
#
# 라벨 파이프라인 `dataset/sj_datasets/common/temp/slab_label.py`의 occupancy는 지면
# occupancy가 **아니다**. 그 파일이 스스로 이렇게 적고 있다:
#
#   "슬래브 [z_ref, z_ref+thick] 은 **로봇이 통과해야 하는 높이 구간**을 뜻한다"
#   "2D **기둥 누적**이면 충분하다 -- 슬래브 자르기가 flatten 앞에 오므로 로봇보다 높은
#    장애물(열린 문·상인방·천장)은 이미 배제돼 있다"
#
# `z_ref`는 LiDAR 수평면(지상 약 0.87 m), `thick=0.8`이므로 **라벨이 판정하는 구간은 지상
# 0.87~1.67 m**다. 그런데 `Y=1`은 `ego z=0` **한 평면**에서만 이미지 특징을 뽑는다
# (`Vox_util`이 복셀 중심에서 표본하고 기본 슬래브가 ±0.25 m라 중심이 정확히 0).
# **모델은 발치를 보고 가슴 높이의 통과 가능성을 맞춰야 한다.**
#
# 지금도 어느 정도 되는 이유는 encoder 수용영역이 넓어 발치 feature에 몸통 정보가 이미
# 들어 있기 때문이다. 구멍은 **접지선이 없거나 가려진 물체**(테이블 상판, 선반, 늘어진
# 가지)이고, 그것이 정확히 라벨이 obstacle로 찍는 것들이다.
#
# ## 높이 범위를 [-0.125, 1.875] / Y=8로 고른 이유
#
# bin 중심이 `0, 0.25, 0.50, ..., 1.75`가 되어 **bin 0이 기존 `Y=1`의 표본 평면과 정확히
# 일치**한다(`tests/datasets/test_simplebev_vox.py`가 `Vox_util.Mem2Ref`로 직접 확인).
# 즉 **정보가 순증한다** -- 나빠지면 원인이 "정보 부족"이 아니라 "용량/과적합"으로 좁혀진다.
# `[0, 1.7]`처럼 잡으면 최하단 중심이 0.106 m가 되어 기존 평면이 사라지므로 쓰지 않는다.
#
# ## 사전 확인: 정보가 실제로 들어오는가 (실행 전에 쟀다)
#
# `python tools/measure_height_bin_visibility.py` -- 화각 밖이면 `valid_mem=0`이라 그 bin은
# 항상 0이 되고, `bev_compressor`가 무시하도록 학습된다. 그러면 "변화 없음"이 가설 기각이
# 아니라 **"정보가 애초에 안 들어옴"**이 되어 실험이 헛돈다. 실측:
#
#   높이 [m]   전체    0~1m    1~2m 이상
#    +0.000   95.5%   49.1%   ~100%     <- 현재 Y=1의 표본 평면
#    +0.750   99.8%   98.3%   ~100%
#    +1.000   99.8%   97.2%   ~100%     <- 라벨 대역
#    +1.250   98.9%   87.8%   ~100%     <- 라벨 대역
#    +1.500   97.6%   72.5%   ~100%     <- 라벨 대역
#    +1.750   95.6%   50.4%   ~100%
#
# **라벨 대역 3개 bin의 평균 유효 비율이 98.8%다 -- 정보는 들어온다.** 그러므로 "변화 없음"이
# 나오면 그것은 가설 기각으로 읽어도 된다.
#
# 부수적으로 나온 것: **근거리(0~1 m)에서 가장 안 보이는 높이가 z=0(49.1%)이고 가장 잘
# 보이는 높이가 z=0.75(98.3%)다.** 카메라가 지상 0.87 m라 코앞 지면은 화각 주변부다.
# 다만 그 셀들은 대부분 이미 `permanent_blind`(0~1 m 대역의 63.8%)라 loss에서 빠져 있다.
# **`permanent_blind`는 이 실험에서 재계산하지 않는다** -- 마스크를 바꾸면 task 정의가 바뀌어
# 과거 숫자 전부와 비교 불가가 된다. 한 번에 하나만 바꾼다.
#
# ## 읽는 법은 이미 정해져 있다 -- 결과를 보고 바꾸지 않는다
#
# 가설이 예측하는 무늬는 **장애물 쪽 지표가 움직이는 것**이다:
#
#   `missed_obstacle_rate` 하락 + `iou_occupied` 상승  -> 가설의 확인
#   `iou_free`만 오르고 위 둘이 안 움직임              -> 용량 효과이지 가설의 확인이 아니다
#   셋 다 시드 σ 안                                    -> **가설 기각** (위 사전 확인이
#                                                        "정보가 안 들어왔다"를 이미 배제했다)
#
# **채택 판정은 위 진단 지표로 하지 않는다**(§29.6과 같은 규율). `iou_free` + `fatal_rate`가
# 시드 노이즈 바닥을 넘느냐, 그리고 Orin 예산 안이냐로 한다. 진단 지표로 아키텍처를 채택하면
# `f1@10cm`을 주 판정에서 강등한 이유를 되돌리는 것이 된다.
#
# **σ는 대조군 3시드에서 실측한 값을 쓴다.** `σ_run`(같은 config·같은 시드)은 하한이라
# 서로 다른 config 비교에 쓸 수 없다(§15.4 철회).
#
# ## 대조군은 재실행하지 않는다
#
# `runs/pixel_offset/{logs,ckpt}/off_0.00_s{0,1,2}`가 **이 팔과 완전히 같은 config**다
# (40 epoch / soft_boundary gaussian δ=0.15 α=0.5 λ_B=0.5 / λ_R=0.3 δ_R=0.20 β=0.10 /
# pixel_center offset 0 / binary / augment / scratch / 시퀀스 holdout). config.json으로
# 대조 확인함. stride-4 실험(§29.4)이 같은 대조군을 썼다.
#
# ## 실측 비용 (RTX PRO 6000, batch 8, 512x288)
#
#   파라미터 40.56M -> 41.59M (+2.5%, 전부 bev_compressor)
#   fwd+bwd 피크 메모리 19.8 GiB -> 26.0 GiB
#   fwd+bwd 시간은 사실상 동일 (lifting이 encoder에 비해 싸다)
#
# 3런 x 약 8분 = 약 25분. **이미 있는 런 폴더는 건너뛴다.**
#
# 실행: bash configs/height_bins_arms.sh
# 집계: bash configs/height_bins_arms.sh --report
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/height_bins}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
# 대조군. 재실행하지 않고 집계에서만 참조한다.
CONTROL_ROOT="${CONTROL_ROOT:-runs/pixel_offset}"
CONTROL_CELL="${CONTROL_CELL:-off_0.00}"

HEIGHT_BINS="${HEIGHT_BINS:-8}"
HEIGHT_MIN_M="${HEIGHT_MIN_M:--0.125}"
HEIGHT_MAX_M="${HEIGHT_MAX_M:-1.875}"
# 런 이름 접두사. **`HEIGHT_BINS`를 바꾸면 이것도 같이 바꾼다** -- 안 바꾸면 폴더 이름이
# 실제 설정과 어긋나 나중에 오독을 부른다(`runs/height_bins_regress/README.md`가 그 사례).
ARM="${ARM:-y${HEIGHT_BINS}}"

TRAIN_SEQS="raws2,raws3,rawos1,rawos2,rawos4"
VAL_SEQS="raws1,rawos3"
# 피크 26 GiB다. GPU를 나눠 쓸 때 조각화로 OOM 재시도가 뜨는 것을 막는다 (§stride4와 같은 이유).
# **계산은 전혀 바뀌지 않는다.**
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

if [ "${1:-}" = "--report" ]; then
    seeds_csv="$(echo ${SEEDS} | tr ' ' ',')"
    echo "=== [Y=${HEIGHT_BINS}] ${OUT_ROOT} -- 고정 epoch ${NUM_EPOCHS} ==="
    python tools/summarize_repeats.py --log_root="${OUT_ROOT}/logs" \
        --fixed_epoch="${NUM_EPOCHS}" --pattern="${ARM}_s*"
    echo
    echo "=== [Y=1 대조군] ${CONTROL_ROOT}/logs/${CONTROL_CELL}_s* -- 같은 config ==="
    python tools/summarize_repeats.py --log_root="${CONTROL_ROOT}/logs" \
        --fixed_epoch="${NUM_EPOCHS}" --pattern="${CONTROL_CELL}_s*"
    exit 0
fi

run_one() {
    local run="$1"; shift
    if [ -d "${OUT_ROOT}/logs/${run}" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        return
    fi
    echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
        EXP_NAME="${run%_s*}" RUN_NAME="${run}" SEED="${SEED}" NUM_EPOCHS="${NUM_EPOCHS}" \
        HEIGHT_BINS="${HEIGHT_BINS}" \
        HEIGHT_MIN_M="${HEIGHT_MIN_M}" HEIGHT_MAX_M="${HEIGHT_MAX_M}" \
        LOSS=soft_boundary SOFT_TARGET=gaussian \
        DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B=0.5 \
        LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
        PIXEL_CONVENTION=pixel_center PIXEL_OFFSET=0.0 \
        FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="${TRAIN_SEQS}" VAL_SEQUENCES="${VAL_SEQS}" \
        LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
        "$@" \
        bash configs/train_robot_bev_finetune.sh

    find "${OUT_ROOT}/ckpt/${run}" -name 'model-*.pth' -delete 2>/dev/null || true
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
}

for SEED in ${SEEDS}; do
    run_one "${ARM}_s${SEED}"
done
echo "HEIGHT_BINS_ARMS_DONE"
