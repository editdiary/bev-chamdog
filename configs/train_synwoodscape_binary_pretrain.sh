#!/bin/bash
# SynWoodScape -> Simple-BEV **binary + soft_boundary** 사전학습.
# 캠페인 `02_projection_and_prior` 축 B의 source 학습이다 (원장 §6.3).
#
# **왜 3-class(`train_synwoodscape_threeclass_pretrain.sh`)를 안 쓰나.** 이 실험이 재려는
# 것은 "합성 자율주행 도메인의 prior가 얼마나 해로운가"이고, 그러려면 prior를 **최대한 많이**
# 물려받아야 한다. source를 확정 설정과 같은 정식화(binary)·같은 loss·같은 `Y=4`로 맞추면
# 출력 head까지 형상이 맞아 **네트워크 전체(668 텐서, skipped 0)**가 전이된다. 3-class로
# 두면 head 2개 텐서가 빠져 prior의 마지막 한 겹을 일부러 잘라내고 재는 셈이 된다.
#
# **격자가 다른 것은 의도한 것이다.** source는 240x240(전 8 / 후 4 / 횡 ±6 m),
# target은 120x120(전 4 / 후 2 / 횡 ±3 m)이지만 **셀이 둘 다 5 cm**다. BEV decoder가 전부
# convolution이라 전이되는 것은 "필터 하나가 몇 미터를 보는가"이고 그것이 보존된다.
# 범위가 2배인 이유는 SynWoodScape의 ego가 풀사이즈 승용차라 좁은 격자에서는 절반이 제 차체에
# 덮여 상수 GT가 되기 때문이다(`projects/bev_gt/grid.py`).
#
# 입력 해상도도 다르다(source 512x384, target 512x288) -- 원본 종횡비가 달라서이고,
# encoder가 완전 합성곱이라 전이에는 영향이 없다.
#
# 실행: SEED=0 bash configs/train_synwoodscape_binary_pretrain.sh
set -e
cd "$(dirname "$0")/.."

# conda 환경 강제. 근거는 `configs/train_robot_bev_finetune.sh` 머리말과 동일하다
# (2026-09-01에 `base`에서 20런을 통째로 돌린 사고).
REQUIRED_CONDA_ENV="${REQUIRED_CONDA_ENV:-bev-chamdog}"
if [ "${ALLOW_ANY_ENV:-0}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "${REQUIRED_CONDA_ENV}" ]; then
    echo "!!! conda 환경이 '${CONDA_DEFAULT_ENV:-없음}'이다 -- '${REQUIRED_CONDA_ENV}'가 필요하다." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-swscape_binary_pretrain}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_s${SEED}}"

# **확정 설정(원장 §3)과 같은 값이다.** 여기를 바꾸면 "데이터만 바꿔서 비교한다"가 깨진다.
NUM_EPOCHS="${NUM_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-7}"

OUT_ROOT="${OUT_ROOT:-runs/paper_final/02_projection_and_prior/source_pretrain}"

python tools/train_synwoodscape.py \
    --exp_name="${EXP_NAME}" \
    --run_name="${RUN_NAME}" \
    --seed="${SEED}" \
    --formulation=binary \
    --loss=soft_boundary \
    --delta_m=0.30 \
    --sigma_m=0.10 \
    --lambda_b=0.5 \
    --soft_target=gaussian \
    --lambda_r=0.3 \
    --delta_r_m=0.15 \
    --huber_beta_m=0.15 \
    --band_kappa=1.0 \
    --label_eps=0.0 \
    --num_epochs="${NUM_EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --num_workers=8 \
    --val_fraction=0.2 \
    --split_seed=0 \
    --encoder_type=res101 \
    --use_fisheye=True \
    --augment=True \
    --val_freq_epochs=1 \
    --save_freq_epochs="${NUM_EPOCHS}" \
    --log_dir="${OUT_ROOT}/logs" \
    --ckpt_dir="${OUT_ROOT}/ckpt"
