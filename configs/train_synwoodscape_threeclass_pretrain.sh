#!/bin/bash
# SynWoodScape -> Simple-BEV 3-class 단일 head pretraining 설정.
#
# Simple-BEV에는 config 파일 체계가 없고(원본 `train_nuscenes.py`도 Fire 키워드 인자 +
# 셸 스크립트 관례), 여기서도 같은 방식을 따른다. 값을 바꿔가며 튜닝할 때는 이 파일을
# 복사해서(`configs/train_synwoodscape_<실험명>.sh`) 실험별로 남기는 걸 권장한다.
# 각 값의 의미와 튜닝 기준은 `docs/training_guide.md` 참고.
#
# 실행: bash configs/train_synwoodscape_threeclass_pretrain.sh
#
# 한 값만 바꿔보는 스윕은 파일을 복사하는 대신 환경변수로 덮어쓴다 -- 나머지 인자가
# 한 곳에만 있어야 두 런이 정말 같은 조건인지 확인하기 쉽다:
#   EXP_NAME=threeclass_pretrain_wd1e-4 WEIGHT_DECAY=1e-4 bash configs/train_synwoodscape_threeclass_pretrain.sh
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXP_NAME="${EXP_NAME:-threeclass_pretrain_photo_aug}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-7}"

python tools/train_synwoodscape.py \
    --exp_name="${EXP_NAME}" \
    --num_epochs=60 \
    --batch_size=16 \
    --lr=3e-4 \
    --weight_decay="${WEIGHT_DECAY}" \
    --num_workers=8 \
    --val_fraction=0.2 \
    --split_seed=0 \
    --encoder_type=res101 \
    --use_fisheye=True \
    --augment=True \
    --val_freq_epochs=1 \
    --save_freq_epochs=5 \
    --log_dir=runs/synwoodscape_threeclass/logs \
    --ckpt_dir=runs/synwoodscape_threeclass/ckpt
