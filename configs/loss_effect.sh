#!/usr/bin/env bash
# **loss 재설계의 영향력을 확정 config에서 측정한다.** 산출물은 `runs/loss_effect/`.
#
# 계획 정본: `docs/temp/loss_stability_evaluation_plan_for_claude_2026-09-01.md` (2026-09-01, 사용자 제공).
#
# ## 무엇이 이전 실험과 다른가
#
# `runs/ablation`(2026-08-23)이 같은 사다리를 돌렸지만 **그 런들은 지금 코드와 다른 모델을
# 학습한다.** 그 뒤로 확정 config가 세 번 바뀌었다.
#
#   - lifting 높이 축 `Y = 1 -> 4`  (2026-08-27)   <- 기하 자체가 다르다
#   - soft target `(δ, α) = (0.15, 0.5) -> (δ, σ) = (0.30, 0.10)`  (2026-08-28)
#   - `L_range`의 `(δ_R, β) = (0.20, 0.10) -> (0.15, 0.15)`  (2026-09-01)
#
# 그래서 이 스윕은 옛 표를 갱신하는 것이 아니라 **처음부터 다시 재는 것**이다. 옛 숫자와
# 한 표에 세우지 않는다.
#
# ## 사다리 -- 인접한 두 칸은 손잡이 하나만 다르다
#
#   A_ce      weighted_ce                       대조군. 역빈도 가중 CE
#   B_perset  soft_boundary, λ_B = 0            A -> B: per-set 평균 (½L_F + ½L_N).
#                                               경계 대역 ±δ의 셀은 감독에서 빠진다
#   C_hard    + λ_B = 0.5, σ -> 0               B -> C_hard: 경계 대역에 **hard** 감독
#   C_soft    같은 것, σ = 0.10                  C_hard -> C_soft: target만 hard -> soft
#   D_range   + λ_R = 0.3, arc_huber          C_soft -> D: **보조항** L_range (종점 arc)
#   E_cumulative + λ_R = 0.15, cumulative_l1  C_soft -> E: **같은 계단, 다른 식** (누적 profile)
#
# **`D`와 `E`는 이어지는 계단이 아니라 서로 대체 팔이다.** 둘 다 `C_soft`에서 한 칸이고,
# 논문에 싣는 것은 **`E`다**(`loss_effect_results.md` §16).
#
# **주 비교는 `A_ce` 대 `D_range`다**(= 기존 BCE 대 튜닝한 loss). 나머지는 원인을 가르는
# 용도이고 단독으로 보고하지 않는다.
#
# ## `C_hard`가 왜 필요한가 (2026-09-02 추가)
#
# `A_ce -> B_perset` 한 계단에서 **두 가지가 동시에 바뀐다**: 집계 방식(역빈도 가중 ->
# per-set 평균)과 경계 대역 감독 제거. 그래서 그 계단에서 관찰된 변화를 어느 쪽에도
# 귀속시킬 수 없었다. `C_hard`는 `C_soft`와 **모든 것이 같고 대역 target만 hard 0/1**이라
# 두 축을 완전히 분리한다.
#
#   A_ce  ->  C_hard   집계 방식만 다르다 (둘 다 hard target, 모든 셀 감독)
#   C_hard -> C_soft   대역 target만 다르다 (hard -> soft)
#
# **코드 변경이 필요 없다.** gaussian target은 `sigma -> 0`에서 계단 함수로 수렴하고,
# 5 cm 격자에서 대역 안 셀 중심의 |d|가 최소 0.05 m이므로 `SIGMA_M=0.001`이면 target이
# **정확히 0/1**이 된다(target 엔트로피 0.000e+00, 실측 확인). 즉 `L_B`가 hard CE가 된다.
#
# ## 시드 5개, 바깥 루프
#
# 시드를 바깥에 둔다 -- 중간에 멈춰도 그 시점까지 **네 칸이 같은 n**을 갖는다. 한 칸만
# 시드가 많으면 그 칸의 σ만 신뢰구간이 좁아져 재현성 비교가 기울어진다.
#
# `iou_free`의 시드 간 σ는 확정 config에서 0.002~0.003이다. n=5의 평균 표준오차는 그
# 0.45배이므로 **0.002보다 작은 평균 차이는 이 스윕으로 판정할 수 없다.**
#
# ## 체크포인트
#
# `SAVE_FREQ_EPOCHS=NUM_EPOCHS`라 주기 저장은 **마지막 epoch 하나**만 남는다. 분석에
# 필요한 것은 `model_best`(선택 규칙 = val `iou_free` 최고점)와 마지막 epoch 둘뿐이다 --
# 전자는 "이 loss로 학습하면 어떤 모델을 얻나", 후자는 **사전 선언된 고정 epoch**이라
# 선택 편향이 없는 값이다. epoch별 곡선은 전부 TensorBoard에 있으므로 체크포인트를 더
# 남길 이유가 없다(런당 471 MB x 2 = 약 0.9 GB, 20런에 약 19 GB).
#
# `batch_size`는 셸이 8로 고정한다 -- 노출된 손잡이가 아니라 여기서 줄 수 없다.
#
# ## 20런 x 약 14분 = 약 4.7시간(GPU 1개) (RTX PRO 6000, 100 epoch = 8.2 s/epoch, train 192 / val 75프레임)
# **이미 있는 런은 건너뛴다.** 분석은 `configs/loss_effect_analysis.sh`.
#
# 실행:   CUDA_VISIBLE_DEVICES=0 bash configs/loss_effect.sh
# 집계:   python tools/report_ablation.py --log_root=runs/loss_effect/logs
#         python tools/summarize_repeats.py --log_root=runs/loss_effect/logs --fixed_epoch=40
#         python tools/report_seed_jitter.py --log_root=runs/loss_effect \
#                 --cells=A_ce,B_perset,C_hard,C_soft,D_range,E_cumulative --seeds=0,1,2,3,4
#         python tools/report_threshold_sweep.py --log_root=runs/loss_effect \
#                 --cells=A_ce,B_perset,C_hard,C_soft,D_range,E_cumulative --seeds=0,1,2,3,4
#         python tools/report_decision_disagreement.py --log_root=runs/loss_effect \
#                 --cells=A_ce,B_perset,C_hard,C_soft,D_range,E_cumulative --seeds=0,1,2,3,4
set -uo pipefail          # **`-e`를 뺀다** -- 런 하나가 OOM으로 죽어도 큐 전체가 멈추면 안 된다.
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-runs/loss_effect}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"   # [2026-10-02] 기본 100 (tools/paper_final_epochs.py)
SEEDS="${SEEDS:-0 1 2 3 4}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
# 일부 칸만 돌릴 때(공백 구분 칸 이름). 비우면 사다리 전체다. 칸의 설정은 아래 표에서만 온다 --
# 이 변수는 **고르기만** 한다(예: `ONLY_CELLS="A_ce C_soft"`, 2026-10-02 epoch 길이 사전 실험).
ONLY_CELLS="${ONLY_CELLS:-}"
# **사전 실험 전용 높이 손잡이**(2026-10-02, 높이 범위 사전 실험 `configs/height_range_probe.sh`).
# 이름이 `HEIGHT_*`가 아닌 이유: 호출 셸에 남은 옛 `HEIGHT_BINS`가 사다리를 조용히 오염시키지 못하게
# 하는 계약(`test_inherited_environment_cannot_pollute_the_ladder`)을 지키려는 것이다. 비우면 확정값.
PROBE_HEIGHT_BINS="${PROBE_HEIGHT_BINS:-4}"
PROBE_HEIGHT_MIN_M="${PROBE_HEIGHT_MIN_M:--0.25}"
PROBE_HEIGHT_MAX_M="${PROBE_HEIGHT_MAX_M:-1.75}"

# name|LOSS|LAMBDA_B|LAMBDA_R|SIGMA_M|RANGE_LOSS_MODE
# `SIGMA_M`은 `C_hard`를 위해 칸마다 준다 -- 0.001이 hard target 극한이다.
# `weighted_ce`와 `λ_B=0`인 칸에서는 쓰이지 않지만 자리를 비우지 않는다(파싱이 단순해진다).
# `RANGE_LOSS_MODE`는 `λ_R=0`인 칸에서 쓰이지 않지만 같은 이유로 자리를 채운다.
#
# **[2026-09-21] `E_cumulative`를 사다리에 붙였다.** `C_soft -> E`는 `C_soft -> D`와 **같은
# 계단**이다(보조항을 켠다) -- 다른 것은 그 보조항의 식뿐이다. 그래서 `D`와 `E`는 서로
# **대체 팔**이고, 둘을 이어 붙인 5단 사다리로 읽으면 안 된다.
#
#   C_soft --(+arc_huber      λ_R=0.3 )--> D_range
#          --(+cumulative_l1  λ_R=0.15)--> E_cumulative
#
# `λ_R`이 다른 이유는 **두 식의 gradient 기여를 맞췄기** 때문이다 -- 숫자가 작은 것이 약한
# 것이 아니다(명세 §9.1). 0.15는 거기서 나온 0.1526을 **측정 정밀도에 맞춰 반올림한 값**이다
# (batch 간 σ가 0.58/0.78이라 유효숫자 4자리는 측정이 뒷받침하지 않는다).
#
# **[2026-10-02 사용자 결정] 광선 보조항을 뺀다 -- 사다리는 `C_soft`(soft-boundary BCE)까지
# 넷이다.** 옛 두 팔은 재현용으로 아래에 주석으로 남긴다(`runs/loss_effect/`에 결과가 있다).
#
# **주의: 보조항이 빠지면 `C_soft`는 `01_overall` 고정 split 대조군과 같은 설정이 된다**
# (같은 split·epoch·시드·δ/σ/λ_B). 캠페인에서 둘을 따로 돌릴지 재사용할지는 캠페인 설계 때
# 정한다 -- 재사용하려면 env 전부가 같은지 먼저 대조한다(`docs/paper_campaign_protocol.md`).
CELLS=(
    "A_ce|weighted_ce|0.0|0.0|0.10|arc_huber"
    "B_perset|soft_boundary|0.0|0.0|0.10|arc_huber"
    "C_hard|soft_boundary|0.5|0.0|0.001|arc_huber"
    "C_soft|soft_boundary|0.5|0.0|0.10|arc_huber"
    # "D_range|soft_boundary|0.5|0.3|0.10|arc_huber"            # 2026-09-02, 보조항(뺐다)
    # "E_cumulative|soft_boundary|0.5|0.15|0.10|cumulative_l1"  # 2026-09-21, 보조항(뺐다)
)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/ckpt"

for seed in ${SEEDS}; do
    for entry in "${CELLS[@]}"; do
        IFS='|' read -r name loss lambda_b lambda_r sigma_m range_mode <<< "${entry}"
        if [ -n "${ONLY_CELLS}" ] && [[ " ${ONLY_CELLS} " != *" ${name} "* ]]; then
            continue
        fi
        run="${name}_s${seed}"
        if [ -n "$(ls -A "${OUT_ROOT}/logs/${run}" 2>/dev/null)" ]; then
            echo "=== ${run} 이미 있음 -- 건너뛴다 ==="
            continue
        fi
        echo "=== ${run} 시작 ($(date +%H:%M:%S)) ==="
        # **하이퍼파라미터를 전부 명시한다.** 기본값에 기대면 나중에 기본값이 바뀔 때
        # 이 스윕이 무엇을 돌린 것인지 스크립트만 보고는 알 수 없게 된다 -- 실제로
        # `δ_R`·`β`의 기본값이 2026-09-01에 바뀌었다.
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        EXP_NAME="${name}" RUN_NAME="${run}" SEED="${seed}" NUM_EPOCHS="${NUM_EPOCHS}" \
        LOSS="${loss}" SOFT_TARGET=gaussian \
        DELTA_M=0.30 SIGMA_M="${sigma_m}" SIGMA_ALPHA=None LAMBDA_B="${lambda_b}" \
        BAND_KAPPA=1.0 LABEL_EPS=0.0 \
        RANGE_LOSS_MODE="${range_mode}" \
        LAMBDA_R="${lambda_r}" DELTA_R_M=0.15 DELTA_R_OVER_M=None HUBER_BETA_M=0.15 \
        HEIGHT_BINS="${PROBE_HEIGHT_BINS}" HEIGHT_MIN_M="${PROBE_HEIGHT_MIN_M}" \
        HEIGHT_MAX_M="${PROBE_HEIGHT_MAX_M}" \
        FORMULATION=binary ENCODER_TYPE=res101 AUGMENT=True INIT_CHECKPOINT=none \
        LR=1e-4 WEIGHT_DECAY=1e-7 MAX_CLASS_WEIGHT=20 \
        LABEL_SMOOTHING=0.0 FLIP_AUGMENT=False FREEZE_ENCODER=False \
        PROJECTION=ds_native PINHOLE_HFOV_DEG=None \
        PIXEL_CONVENTION=pixel_center PIXEL_OFFSET=0.0 \
        FRAME_SPLIT_FRACTION=0.0 FRAME_BLOCK_LEN=0 \
        SAVE_FREQ_EPOCHS="${NUM_EPOCHS}" \
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
done
echo "LOSS_EFFECT_DONE ($(date +%H:%M:%S))"
