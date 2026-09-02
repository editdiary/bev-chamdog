#!/usr/bin/env bash
# `L_range`의 두 손잡이 `delta_R`(허용 반폭)와 `beta`(Huber 전환점)를 확정 config 위에서 훑는다.
#
# ## 무엇을 묻는가
#
# 확정값은 `delta_R = 0.20 m`, `beta = 0.10 m`다(`projects/common/range_loss.py`의 기본값).
# 둘 다 라벨에서 추정할 수 없는 하이퍼파라미터이고, 근거는 각각
#   delta_R = 0.20  <- 주 지표 `f1@10cm`의 허용 오차(2셀)의 2배
#   beta    = 0.10  <- dead zone 통과 후 잔차 규모(`range_mae` 0.235 − delta_R 0.20)
# 이다. 즉 **한 번도 스윕으로 확인된 값이 아니다.** `runs/robot_bev_cv/loss_sweep`의 3차
# 스윕은 `delta_R = 0`(dead zone ablation)만 봤고 `beta`는 건드리지 않았다.
#
# 둘을 같이 좁히면(`delta_R` 0.20 → 0.15, `beta` 0.10 → 0.15) 두 변화가 겹친다:
#   - `delta_R`을 좁히면 무벌점 구간이 4셀 → 3셀이 되어 **더 많은 광선이 gradient를 받는다.**
#   - `beta`를 키우면 실효 오차 `e^eff <= beta` 구간이 넓어져 **작은 잔차가 이차로 처리된다**
#     (= 작은 오차에 대한 gradient가 `e/beta`로 줄어든다).
# 방향이 반대라 상쇄될 수도 있다. 그래서 한 축씩 따로 보는 대조군도 같은 root에 둔다.
#
# 판정은 `iou_free`/`f1@10cm`/`fatal_rate`(품질)와 시드 간 산포(재현성)를 **같은 표에서**
# 본다 -- 안정성 지표를 단독으로 읽으면 아무것도 안 배우는 것이 1등이 된다(설계 문서 §15.6).
# `loss_range` 자체는 config 간 비교에 쓸 수 없다(`delta_R`이 다르면 정의가 다르다).
#
# ## 대조군
#
# 확정 config(`delta_R=0.20 beta=0.10`)는 **같은 root에서 같은 시드로 새로 돌린다**
# (`dr200b100`). `runs/sigma_k`의 `s100k3_s0`이 수치적으로 같은 config지만 다른 시점의
# 환경에서 나왔으므로, ±0.004를 판정하는 비교에 torch/cuDNN 버전이 교란으로 들어갈 수 있다.
# 심링크로 재사용하고 싶으면 `LINK_BASE=1`을 준다.
#
# ## 이름 규약
#
# `dr{delta_R*1000}b{beta*1000}` -- `report_seed_jitter.py`가 `{cell}_s{seed}`를 요구한다.
#   dr200b100 = delta_R 0.20 / beta 0.10   (확정값 = 대조군)
#   dr150b150 = 0.15 / 0.15                (사용자 요청)
#   dr150b100 = 0.15 / 0.10                (delta_R만 좁힘)
#   dr200b150 = 0.20 / 0.15                (beta만 키움)
#
# ## 읽는 법
#
#   품질:   python tools/report_alpha_convergence.py --log_root=runs/range_dr_beta
#   경계:   python tools/report_boundary_calibration.py --runs=...
#   재현성: python tools/report_seed_jitter.py --log_root=runs/range_dr_beta \
#             --cells=dr200b100,dr150b150 --seeds=0,1,2,3,4   <- 시드를 여러 개 돌렸을 때만
#
# **런 하나씩 비교하면 시드 간 산포와 config 효과를 구별할 수 없다.** 확정 config에서
# `iou_free`의 시드 간 표준편차는 0.002~0.003 규모다(설계 문서 §21.3). 그보다 작은 차이는
# 이 스윕으로 판정할 수 없고, 시드를 늘려야 한다.
#
# 실행: RUNS_SPEC="dr200b100|0.20|0.10|0 dr150b150|0.15|0.15|0" \
#         CUDA_VISIBLE_DEVICES=0 bash configs/sweep_range_dr_beta.sh
#   RUNS_SPEC은 `name|DELTA_R_M|HUBER_BETA_M|SEED`를 공백으로 이어 붙인 것이다.
set -uo pipefail          # **`-e`를 뺀다** -- 런 하나가 OOM으로 죽어도 큐 전체가 멈추면 안 된다.

OUT_ROOT="${OUT_ROOT:-runs/range_dr_beta}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
LINK_BASE="${LINK_BASE:-0}"     # 1이면 확정 config 대조군을 runs/sigma_k에서 심링크로 가져온다
BASE_ROOT="${BASE_ROOT:-runs/sigma_k}"
BASE_CELL="${BASE_CELL:-s100k3}"
read -r -a SPECS <<< "${RUNS_SPEC:?RUNS_SPEC가 필요하다}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

if [ "${LINK_BASE}" = "1" ]; then
    for seed in 0 1 2 3 4; do
        src_log="${BASE_ROOT}/logs/${BASE_CELL}_s${seed}"
        [ -d "${src_log}" ] || continue
        for kind in logs ckpt; do
            src="$(realpath "${BASE_ROOT}/${kind}/${BASE_CELL}_s${seed}" 2>/dev/null)" || continue
            dst="${OUT_ROOT}/${kind}/${BASE_CELL}_s${seed}"
            [ -e "${dst}" ] || ln -s "${src}" "${dst}"
        done
    done
fi

for entry in "${SPECS[@]}"; do
    IFS='|' read -r name delta_r beta seed <<< "${entry}"
    run="${name}_s${seed}"
    if [ -n "$(ls -A "${OUT_ROOT}/logs/${run}" 2>/dev/null)" ]; then
        echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
        continue
    fi
    echo "=== ${run} (delta_R=${delta_r} beta=${beta} seed=${seed}) 시작 ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    EXP_NAME="${name}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
    LOSS=soft_boundary SOFT_TARGET=gaussian \
    DELTA_M=0.30 LAMBDA_B=0.5 SIGMA_M=0.10 SIGMA_ALPHA=None \
    BAND_KAPPA=1.0 LABEL_EPS=0.0 \
    LAMBDA_R=0.3 DELTA_R_M="${delta_r}" DELTA_R_OVER_M=None HUBER_BETA_M="${beta}" \
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
