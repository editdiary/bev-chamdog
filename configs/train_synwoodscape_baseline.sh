#!/bin/bash
# SynWoodScape -> Simple-BEV baseline 학습 설정.
#
# Simple-BEV에는 config 파일 체계가 없고(원본 `train_nuscenes.py`도 Fire 키워드 인자 +
# 셸 스크립트 관례), 여기서도 같은 방식을 따른다. 값을 바꿔가며 튜닝할 때는 이 파일을
# 복사해서(`configs/train_synwoodscape_<실험명>.sh`) 실험별로 남기는 걸 권장한다.
# 각 값의 의미는 `docs/training_pipeline_walkthrough.md` 참고
# (`docs/archive/training_guide.md`는 2-head 시절 문서다).
#
# 실행: bash configs/train_synwoodscape_baseline.sh
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python tools/train_synwoodscape.py \
    --exp_name=baseline \
    --num_epochs=3 \
    --batch_size=8 \
    --lr=3e-4 \
    --weight_decay=1e-7 \
    --num_workers=8 \
    --val_fraction=0.1 \
    --split_seed=0 \
    --encoder_type=res101 \
    --use_fisheye=True \
    --val_freq_epochs=1 \
    --save_freq_epochs=5 \
    --log_dir=runs/temp_train/logs \
    --ckpt_dir=runs/temp_train/ckpt
