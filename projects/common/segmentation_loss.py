"""BEV 분할 loss의 공통 부품 -- 마스킹된 가중 CE와 역빈도 가중치.

3-class(`three_class_metrics`)와 binary(`binary_metrics`)가 **같은 함수**를 쓴다. 두 정식화의
숫자를 나란히 읽는 것이 (D) 판정의 전부이므로, 한쪽만 조용히 다른 정규화나 다른 캡 규칙을
쓰게 되면 비교 자체가 무너진다. 클래스가 몇 개인지만 다르고 나머지는 전부 같아야 한다.
"""
import torch
import torch.nn.functional as F

# 역빈도 가중치의 기본 상한. 3-class 로봇 split에서 `occupied`를 67.9 -> 20으로 자른다.
#
# **2026-08-18 실측 결론: 20은 해롭고, 그렇다고 상한을 고르는 문제도 아니다.**
# 근본 원인은 CE가 **면적** loss인데 `occupied`가 두께 1셀 **표면**이라는 것이다
# (`docs/finetune_overfitting_diagnosis.md` §12-§13). binary 정식화에서는 클래스가
# free/not-free 둘뿐이라 순수 역빈도가 3.94에 그쳐 이 캡이 아예 걸리지 않는다 --
# 그래도 같은 기본값을 쓰는 이유는 `--max_class_weight` 플래그가 두 경로에서 같은 뜻이어야
# 하기 때문이다.
DEFAULT_MAX_CLASS_WEIGHT = 20.0


def masked_weighted_ce(logits, class_index, valid, class_weights, part_by_class,
                       label_smoothing=0.0) -> tuple:
    """`valid` 셀에 한정한 가중 cross-entropy. `(총 loss, 항별 dict)`.

    `loss_*`는 그 클래스 셀의 **평균**이고 `share_*`는 그 클래스가 **총 loss에 실제로 기여하는
    몫**이다. 둘을 같이 내놓는 이유: 평균만 보면 병리가 안 보인다. 2026-08-18 실측에서 val
    `loss_occupied`가 145였는데, 그것만으로는 occupied가 셀의 1.1 %인데 **총 loss의 67 %**를
    차지한다는 사실이 드러나지 않았다 -- 셀 비율을 손으로 곱해 봐야 알 수 있었다.
    `share_*`는 정의상 합이 1이므로 어느 클래스가 학습을 지배하는지 바로 읽힌다.

    `part_by_class`는 `{클래스 인덱스: 이름}`이고 이것이 곧 로그에 찍히는 항의 집합이다.

    `label_smoothing`은 **과신을 직접 겨냥한 손잡이**다. 실측(§16.2)에서 val loss가 오르는
    이유가 "더 많이 틀려서"가 아니라 "같은 만큼 틀리되 확신이 커져서"였다 -- 정답 확률의
    기하평균이 train 0.995 대 val 0.620이다. smoothing은 한 셀이 낼 수 있는 loss에 상한을
    씌워 그 발산을 막는다. 기본 0.0이면 동작이 전과 완전히 같다.
    """
    valid_f = valid.float()
    per_cell = F.cross_entropy(
        logits,
        class_index.squeeze(1),
        weight=class_weights.to(logits.device),
        reduction="none",
        label_smoothing=label_smoothing,
    ).unsqueeze(1)
    weighted_sum = (per_cell * valid_f).sum()
    total = weighted_sum / (valid_f.sum() + 1e-6)

    parts = {}
    valid_b = valid.bool()
    for class_id, name in part_by_class.items():
        selector = ((class_index == class_id) & valid_b).float()
        class_sum = (per_cell * selector).sum()
        parts[f"loss_{name}"] = class_sum / (selector.sum() + 1e-6)
        parts[f"share_{name}"] = class_sum / (weighted_sum + 1e-6)
    return total, parts


def inverse_frequency_weights(counts, max_class_weight=None) -> torch.Tensor:
    """셀 카운트 -> 역빈도 가중치. 최빈 클래스를 1.0으로 정규화한 뒤 상한으로 자른다.

    `cap=1`은 "가중치를 아예 쓰지 않는다"가 된다 -- 최소값이 1로 정규화돼 있으므로 전부 1로
    눌린다. 스윕의 한쪽 끝이다.
    """
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.min().clamp(min=1e-12)
    cap = DEFAULT_MAX_CLASS_WEIGHT if max_class_weight is None else float(max_class_weight)
    return weights.clamp(max=cap).float()
