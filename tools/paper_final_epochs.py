"""캠페인 러너가 공유하는 학습 길이. **[2026-10-02 사용자 결정] 기본 100 epoch.**

근거: 사전 실험 `runs/99_epoch_exp`(원장 §5 "2026-10-02 (3)"). 40 epoch은 모델이 튜닝되기 전에
관습적으로 정한 값이었다. 100에서 soft-BCE는 진단 CE가 평평한 채 `iou_free`가 epoch 87까지
오르고, 경계 지표 BF@0.10은 두 손실 모두 epoch 60 근처까지 오른다. 대가는 시간이다(2.5배).

**사전학습(SynWoodScape) 길이는 따로 둔다.** 사전 실험은 로봇 데이터 학습만 봤다 -- 사전학습을
100으로 할 근거는 아직 없다. 값은 미세조정과 같게 두되 손잡이(`--pretrain_epochs`)를 분리해,
바꾸기로 하면 이 상수 하나만 고친다. **고정 epoch 판정(`FIXED_EPOCH`)은 학습 길이와 같다** --
분석 셸·결과 묶음·패키지 생성기의 기본값도 100이다.

**OneCycleLR이 학습 길이에 묶여 있다**(`tools/train_robot_bev.py`). 길이를 바꾸면 같은 epoch에서도
LR이 다르므로, 길이가 다른 런끼리 같은 epoch 값을 비교하지 않는다.
"""
DEFAULT_NUM_EPOCHS = 100
DEFAULT_PRETRAIN_EPOCHS = 100
