#!/usr/bin/env bash
# `delta` 후보 셋의 **시드 간 재현성**을 잰다 (2026-08-27).
#
# ## 왜 재현성인가
#
# `delta` ablation에서 안정도로 쓸 수 있는 지표가 전부 `delta`에 대해 **단조**였다 --
# val loss 상승폭, 암기 격차, 경계 KL 상승폭 모두 `delta`가 클수록 좋아지고 멈출 지점을
# 못 준다. 게다가 val loss의 최저점은 **덜 학습된 모델**을 가리킨다(delta=0.15에서
# 경계 KL 최저가 epoch 2, 거기 `f1@10cm` 0.5169 대 끝 0.5974). 그러니 커브 모양은
# 안정도의 근거가 못 된다.
#
# 남는 진짜 안정도는 **같은 config를 다시 학습하면 같은 지도를 주는가**다. 이 프로젝트가
# 이미 그 축에 주장을 갖고 있고(재현성 5~22배, 설계 문서 §15), 물리 단위로 읽힌다 --
# "시드를 바꿔 재학습하면 자유공간 경계가 median X cm 움직인다".
#
# ## config
#
# `alpha = 1.0`, 40 epoch. **시드 0이 이미 있는 것과 맞춘 값이다** -- delta 계열 전체가
# 그 조건에서 측정됐으므로 여기서 바꾸면 시드 0을 못 쓴다. 확정 config의 `alpha = 0.5`와
# 다르다는 점은 알고 두는 것이고, `alpha`는 `delta`와 같은 축(전이폭)이라 순위를 안 바꾼다.
#
# 시드 0은 다시 돌리지 않는다 -- 아래 경로를 `runs/delta_seeds/`로 심링크해 둔다.
#   d015_s0 <- runs/alpha_y4/{logs,ckpt}/a100_s0
#   d025_s0 <- runs/loss_convergence/{logs,ckpt}/d025_s0
#   d030_s0 <- runs/loss_convergence/{logs,ckpt}/d030_s0
#
# ## 읽는 법
#
#     python tools/report_seed_jitter.py --log_root=runs/delta_seeds \
#         --cells=d015,d025,d030 --seeds=0,1,2,3,4
#
# **안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이다**(설계 문서 §15.6).
# 그래서 그 도구가 `iou_free`·`range_mae`를 같은 표에 찍는다. 시드 jitter가 `range_mae`
# 자체보다 훨씬 작아야 "같은 모델을 일관되게 준다"가 성립한다.
#
# 실행: RUNS_SPEC="0.15|1 0.15|2" CUDA_VISIBLE_DEVICES=0 bash configs/sweep_delta_seeds.sh
#   RUNS_SPEC은 `DELTA_M|SEED`를 공백으로 이어 붙인 것이다.
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-runs/delta_seeds}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
read -r -a SPECS <<< "${RUNS_SPEC:?RUNS_SPEC가 필요하다}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${SPECS[@]}"; do
    IFS='|' read -r delta seed <<< "${entry}"
    name="d$(printf '%03d' "$(python3 -c "print(int(round(${delta}*100)))")")"
    run="${name}_s${seed}"
    if [ -n "$(ls -A "${OUT_ROOT}/logs/${run}" 2>/dev/null)" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (delta=${delta} seed=${seed}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M="${delta}" LAMBDA_B=0.5 SIGMA_ALPHA=1.0 SIGMA_M=None BAND_KAPPA=1.0 LABEL_EPS=0.0 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
    HEIGHT_BINS=4 HEIGHT_MIN_M=-0.25 HEIGHT_MAX_M=1.75 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    LR=1e-4 \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done
