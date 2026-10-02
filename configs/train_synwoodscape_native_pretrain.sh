#!/bin/bash
# SynWoodScape **원본 기하**(30 m, 200x200, 0.15 m/cell) -> Simple-BEV binary 사전학습.
# 캠페인 `02b_native_source_prior`의 source 학습이다.
#
# ## 무엇이 `train_synwoodscape_binary_pretrain.sh`와 다른가 -- **격자 하나**
#
# 기존 사전학습은 SynWoodScape를 **타깃 과제에 맞춰** 잘라 썼다: 전방8/후방4/횡±6 m를
# 셀 0.05 m로(240x240). 셀 크기를 타깃과 같게 유지한 것은 의도였다 -- BEV decoder가 전부
# convolution이라 전이되는 것이 "필터 하나가 몇 미터를 보는가"이고, 셀이 같으면 그게 보존된다.
#
# **이 스크립트는 그 배려를 일부러 뺀다.** 소스가 실제로 주는 30 m 전체를 자율주행 관행
# 해상도(0.15 m/cell, 200x200)로 학습한다 -- 남이 공개한 주행 BEV 모델을 그대로 가져다
# 쓰는 상황이다. 그래서 재는 것은 **"도메인 내용 불일치"가 아니라 "내용 + 공간 스케일 불일치"**다.
#
# 기존 팔(`02_projection_and_prior/source_prior`)과 나란히 놓으면 스케일 몫이 분해된다.
#
# ⚠ **200이어야 한다.** decoder가 /8로 줄였다가 skip으로 복원하므로 격자가 8로 나눠떨어져야
# 한다. 100x100(0.30 m/cell)은 forward에서 26 대 25로 깨진다(실측).
#
# ## ⚠ 라벨에 수동 보정이 없다
#
# 보정본은 ROI 8/4/±6에만 존재한다. 여기 쓰는 `dataset/synwoodscape_native_roi_15_15_15_h08`은
# 시맨틱 라벨의 **자동 크롭**이다(`tools/build_synwoodscape_native_labels.py`). 그래서 기존
# 팔과의 차이에는 **기하와 라벨 품질이 섞여 있다** -- scratch 대비로 읽는 것이 주 비교다.
#
# ## 나머지는 전부 같다
#
# binary 정식화·soft_boundary·같은 `Y=4`라 출력 head까지 형상이 맞아 네트워크 전체가 전이된다.
# `DELTA_M=0.30`/`SIGMA_M=0.10`도 그대로 둔다 -- 셀이 3배 커져도 대역 안에 셀 4개가 남고
# target 엔트로피가 0.2972 nats로 0.05 m일 때(0.2965)와 사실상 같다(실측).
# **보조항 기본값은 `arc_huber`, `λ_R=0.3`이다**(기존 런 재현). 캠페인을 다른 형태로
# 옮길 때는 `RANGE_LOSS_MODE`/`LAMBDA_R`로 넘긴다 -- 대조군·기존 팔과 **같아야** 한다.
#
# 실행: SEED=0 bash configs/train_synwoodscape_native_pretrain.sh
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
EXP_NAME="${EXP_NAME:-swscape_native_pretrain}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_s${SEED}}"

# **확정 설정(원장 §3)과 같은 값이다.** 여기를 바꾸면 "데이터만 바꿔서 비교한다"가 깨진다.
NUM_EPOCHS="${NUM_EPOCHS:-100}"   # [2026-10-02] tools/paper_final_epochs.py의 DEFAULT_PRETRAIN_EPOCHS
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-7}"

# 보조항 `L_range`. **[2026-10-02] 기본값을 0(보조항 없음)으로 바꿨다** -- 다음 캠페인은
# soft-boundary BCE까지만 쓴다(사용자 결정). 0이면 trainer가 항을 만들지도 않는다.
# 옛 캠페인 재현: `arc_huber` λ_R=0.3(`runs/paper_final/`), `cumulative_l1` λ_R=0.15
# (`runs/paper_final_cumulative/`) -- 큐 러너의 `--range_loss_mode`가 넘긴다
# (`tools/paper_final_aux_loss.py`).
# 이름은 `configs/train_robot_bev_finetune.sh`와 **글자 그대로 같게** 둔다 -- 사전학습과
# 미세조정에 서로 다른 변수명을 쓰면 한쪽만 바뀐 것을 알아채지 못한다.
LAMBDA_R="${LAMBDA_R:-0.0}"
RANGE_LOSS_MODE="${RANGE_LOSS_MODE:-arc_huber}"
DELTA_R_M="${DELTA_R_M:-0.15}"
HUBER_BETA_M="${HUBER_BETA_M:-0.15}"

OUT_ROOT="${OUT_ROOT:-runs/paper_final/02b_native_source_prior/source_pretrain_native}"

python tools/train_synwoodscape.py \
    --exp_name="${EXP_NAME}" \
    --run_name="${RUN_NAME}" \
    --seed="${SEED}" \
    --source_profile=native \
    --formulation=binary \
    --loss=soft_boundary \
    --delta_m=0.30 \
    --sigma_m=0.10 \
    --lambda_b=0.5 \
    --soft_target=gaussian \
    --lambda_r="${LAMBDA_R}" \
    --range_loss_mode="${RANGE_LOSS_MODE}" \
    --delta_r_m="${DELTA_R_M}" \
    --huber_beta_m="${HUBER_BETA_M}" \
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
