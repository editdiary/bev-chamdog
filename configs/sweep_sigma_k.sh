#!/usr/bin/env bash
# 파라미터를 `(sigma, k)`로 다시 세운 계열 (2026-08-27).
#
# ## 왜 다시 세우나
#
# `soft_target`의 gaussian은 **[-delta, delta]로 절단된 정규 사전분포에서의 정확한
# 사후확률**이다(`soft_boundary.soft_target` docstring). 그러면 물리적으로 의미 있는 양은
# `sigma`(라벨 경계를 몇 cm 모르는가)이고 `delta`는 **어디서 자를지**를 정하는 절단점이다.
#
# 그런데 지금까지는 `delta`를 먼저 정하고 `alpha = sigma/delta`로 `sigma`를 끌어냈다.
# 확정 조건 `delta=0.15, alpha=1.0`은 `k = delta/sigma = 1`, 즉 **1 sigma에서 자른다**.
# 가우시안을 +-1 sigma에서 자르면 꼬리가 날아가서 실효 산포가 지정값의 54 %만 남는다:
#
#     k=1: 8.09cm (54.0%)   k=1.5: 11.14cm (74.3%)   k=2: 13.19cm (88.0%)
#     k=2.5: 14.32cm (95.5%)   k=3: 14.80cm (98.7%)      (전부 sigma=0.15 지정)
#
# 즉 "경계가 15 cm 불확실하다"고 지정했는데 target이 실제로 말하는 것은 **"8 cm 불확실하고
# 15 cm 밖은 절대 확신"**이었다. 사용자가 delta=0.15를 고른 사전 지식은 `delta`가 아니라
# **`sigma`**에 들어갔어야 하는 값이다.
#
#     sigma = 라벨 경계 불확실성 [m]      <- 사전 지식이 들어가는 유일한 자리
#     delta = k*sigma,  k = 2~3           <- 절단이 sigma를 먹지 않을 만큼만
#     alpha = 1/k                          <- 유도값, 자유 파라미터가 아니다
#
# `SIGMA_M`으로 `sigma`를 미터로 직접 준다(`SIGMA_ALPHA`와 동시 지정 금지).
#
# ## 이름 규약
#
# `s{sigma*1000}k{k}` -- `report_seed_jitter.py`가 `{cell}_s{seed}`를 요구하므로 맞춘다.
#   s100k2 = sigma 0.10, delta 0.20    s125k2 = 0.125 / 0.25    s150k2 = 0.15 / 0.30
#   s175k2 = 0.175 / 0.35              s200k2 = 0.20  / 0.40    s150k3 = 0.15 / 0.45
#
# 시드 0이 이미 있는 셋(s100k2/s125k2/s150k2)은 `runs/delta_alpha`의 alpha=0.5 런과
# **수치적으로 같은 config**이므로 심링크로 재사용한다.
#
# ## 읽는 법
#
#   품질:   python tools/report_alpha_convergence.py --log_root=runs/sigma_k
#   경계:   python tools/report_boundary_calibration.py --runs=...   <- config 간 비교는 이것만
#   재현성: python tools/report_seed_jitter.py --log_root=runs/sigma_k --cells=... --seeds=0,1,2,3,4
#
# **안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이 된다**(설계 문서 §15.6).
# `sigma`를 키우면 확신이 단조로 낮아지므로 `iou_free`/`f1@10cm`/`fatal_rate`가 같이
# 안 무너졌는지를 반드시 같은 표에서 본다. delta=0.45를 alpha=1.0에서 쟀을 때 `f1@10cm`이
# 6 sigma 무너졌던 것이 그 벽이고, `s150k3`이 같은 delta를 다른 alpha에서 다시 보는 것이다.
#
# 실행: RUNS_SPEC="s150k3|0.15|0.45|0" CUDA_VISIBLE_DEVICES=0 bash configs/sweep_sigma_k.sh
#   RUNS_SPEC은 `name|SIGMA_M|DELTA_M|SEED`를 공백으로 이어 붙인 것이다.
set -uo pipefail          # **`-e`를 뺀다** -- 런 하나가 OOM으로 죽어도 큐 전체가 멈추면 안 된다.

OUT_ROOT="${OUT_ROOT:-runs/sigma_k}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
read -r -a SPECS <<< "${RUNS_SPEC:?RUNS_SPEC가 필요하다}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for entry in "${SPECS[@]}"; do
    IFS='|' read -r name sigma delta seed <<< "${entry}"
    run="${name}_s${seed}"
    if [ -n "$(ls -A "${OUT_ROOT}/logs/${run}" 2>/dev/null)" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (sigma=${sigma} delta=${delta} k=$(python3 -c "print(round(${delta}/${sigma},2))") seed=${seed}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M="${delta}" LAMBDA_B=0.5 SIGMA_M="${sigma}" SIGMA_ALPHA=None \
    BAND_KAPPA=1.0 LABEL_EPS=0.0 \
    LAMBDA_R=0.3 DELTA_R_M=0.20 DELTA_R_OVER_M=None HUBER_BETA_M=0.10 \
    HEIGHT_BINS=4 HEIGHT_MIN_M=-0.25 HEIGHT_MAX_M=1.75 \
    FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
    LR=1e-4 \
    TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
    VAL_SEQUENCES="raws1,rawos3" \
    LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
    bash configs/train_robot_bev_finetune.sh
    if [ $? -ne 0 ]; then
        # 실패한 런의 디렉터리를 지운다 -- 안 그러면 "이미 있음"으로 영원히 건너뛴다.
        echo "!!! ${run} 실패 -- 디렉터리를 지우고 다음으로 넘어간다 ($(date +%H:%M:%S)) !!!"
        rm -rf "${OUT_ROOT}/logs/${run}" "${OUT_ROOT}/ckpt/${run}"
        continue
    fi
    echo "=== ${run} 완료 ($(date +%H:%M:%S)) ==="
done
echo "=== 큐 전체 종료 ($(date +%H:%M:%S)) ==="
