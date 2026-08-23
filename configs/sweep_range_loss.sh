#!/usr/bin/env bash
# 3차 스윕 -- **방위각 자유거리 보조항 `L_range`** (docs/soft_boundary_loss_design.md §13).
#
# 베이스는 프론티어 중간점(gaussian, delta=0.15, alpha=0.5 -> f1@10cm 0.5491 / 되올림 +16.5%)이다.
# alpha=0.5를 고른 이유: §10.3에서 alpha가 성능<->수렴의 단일 교환 축이었고, 그 중간점에서
# 재면 프론티어가 **옮겨졌는지**(곡선 이동)와 **곡선 위를 움직였는지**가 구별된다.
#
# lambda_R은 추측하지 않았다 -- `tools/measure_range_gradient.py`로 gradient 비를 실측해
# 정한 값이다(§13.3). lambda_R=1에서의 G_R/G_B가 학습 진행에 따라 1.55(init) -> 0.42(ep10)
# -> 0.27(ep20) -> 0.15(ep40)로 떨어지므로 **한 값으로 전 구간에서 0.1을 유지할 수 없다.**
# 그래서 0.1과 0.3으로 괄호를 친다: 0.3이 ep10~20에서 0.1 근처, 0.1은 전 구간에서 확실히
# 보조 규모다.
#
# 무엇을 묻는가 (§13.4):
#   r00        dead zone이 필요한가                  -- delta_R=0 대조
#   r10 / r30  A: 순수 보조항으로 지표가 오르나       -- lambda_B는 0.5 그대로
#   lb25/lb12  B: 경계 감독의 몫을 옮기면 프론티어가 옮겨지나
#
# B가 핵심이다. lambda_B=0.5를 그대로 두면 관용 없는 항(셀당 계수 5.17/|V|)이 관용 있는
# 보조항보다 10배 세게 남아, 원인 (3)의 암기 압력이 줄지 않는다. lambda_B=0.25 단독 결과는
# §10.4에 이미 있으므로(f1 0.5445 / 되올림 +14.8%) lb25의 대조군이 공짜로 갖춰져 있다.
#
# **판정은 f1 한 점이 아니다**(§12.2의 검증 예측): train/val KL 격차(현재 28~42배)가 줄고
# val KL 곡선이 내려가서 **머무는지**. 그리고 `arc_bias`가 양수 방향으로 커지지 않는지 --
# dead zone이 |e|에 대칭이라 자유공간 과대예측(= fatal 방향)이 무벌점으로 늘 수 있다(§4.5).
#
# **주의: 이 루프가 도는 동안 이 스크립트나 train_robot_bev_finetune.sh를 편집하지 말 것.**
# bash가 파일을 게으르게 읽어 실행 중 수정하면 죽는다.
#
# 실행: bash configs/sweep_range_loss.sh
# 집계: python tools/report_convergence.py --log_root=runs/robot_bev_cv/loss_sweep/logs
set -euo pipefail

SEEDS="${SEEDS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
OUT_ROOT="${OUT_ROOT:-runs/robot_bev_cv/loss_sweep}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# name|LAMBDA_B|LAMBDA_R|DELTA_R_M
CONFIGS=(
    "sb_r30|0.5|0.3|0.20"
    "sb_r10|0.5|0.1|0.20"
    "sb_r30_dr00|0.5|0.3|0.0"
    "sb_r30_lb25|0.25|0.3|0.20"
    "sb_r30_lb12|0.125|0.3|0.20"
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r name lambda_b lambda_r delta_r <<< "${entry}"
        echo "=== seed ${seed} / ${name} 시작 ($(date +%H:%M:%S)) ==="
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        EXP_NAME="${name}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS=soft_boundary SOFT_TARGET=gaussian \
        DELTA_M=0.15 SIGMA_ALPHA=0.5 LAMBDA_B="${lambda_b}" \
        LAMBDA_R="${lambda_r}" DELTA_R_M="${delta_r}" HUBER_BETA_M=0.10 \
        FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
        TRAIN_SEQUENCES="raws2,raws3,rawos1,rawos2,rawos4" \
        VAL_SEQUENCES="raws1,rawos3" \
        LOG_DIR="${OUT_ROOT}/logs" CKPT_DIR="${OUT_ROOT}/ckpt" \
        bash configs/train_robot_bev_finetune.sh

        # 체크포인트는 지운다 -- 집계는 로그만으로 되고, 승자는 확정 후 한 번 더 돌려 만든다.
        for dir in "${OUT_ROOT}/ckpt/${name}_"*"_s${seed}_"*; do
            [ -d "${dir}" ] && rm -rf "${dir}"
        done
        echo "=== seed ${seed} / ${name} 완료 ($(date +%H:%M:%S)) ==="
    done
done
echo "RANGE_SWEEP_DONE"
