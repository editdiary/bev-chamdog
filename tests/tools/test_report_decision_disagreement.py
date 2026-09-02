"""`tools/report_decision_disagreement.py`의 집계 항등식을 고정한다.

이 도구의 핵심은 시드 **쌍**을 실제로 돌지 않고 `k(S−k)/C(S,2)`로 세는 것이다. 그 항등식이
깨지면 숫자는 여전히 0~1 사이의 그럴듯한 값으로 나오는데 뜻이 달라진다 -- 그러면 "결정
재현성이 3배 좋아졌다" 같은 문장이 조용히 틀린다. 그래서 **쌍을 직접 도는 참조 구현**과
대조한다.
"""
import itertools

import numpy as np

from tools.report_decision_disagreement import pairwise_disagreement


def _reference(binary, weights=None):
    """정의 그대로 -- 시드 쌍을 전부 돌아 평균한다. 느리지만 의심의 여지가 없다."""
    n_seeds = binary.shape[0]
    pairs = list(itertools.combinations(range(n_seeds), 2))
    per_cell = np.mean([(binary[s] != binary[t]).astype(np.float64) for s, t in pairs], axis=0)
    if weights is None:
        return float(per_cell.mean())
    return float((per_cell * weights).sum() / weights.sum())


def test_closed_form_matches_pairwise_enumeration():
    rng = np.random.default_rng(0)
    for n_seeds in (2, 3, 5, 7):
        binary = (rng.random((n_seeds, 500)) > 0.5).astype(np.int8)
        assert np.isclose(pairwise_disagreement(binary), _reference(binary))


def test_weights_select_a_subset():
    """가중치는 부분집합 평균이어야 한다 -- 경계 근방만 따로 읽는 것이 이 도구의 요점이다."""
    rng = np.random.default_rng(1)
    binary = (rng.random((5, 400)) > 0.5).astype(np.int8)
    mask = np.zeros(400)
    mask[:120] = 1.0
    assert np.isclose(pairwise_disagreement(binary, mask),
                      _reference(binary[:, :120]))


def test_unanimous_seeds_disagree_nowhere():
    """전부 같은 답이면 0이다. **아무것도 안 배우는 모델이 1등**이 되는 자리이고,
    그래서 도구가 품질을 같은 표에 찍는다."""
    binary = np.ones((4, 50), dtype=np.int8)
    assert pairwise_disagreement(binary) == 0.0
    assert pairwise_disagreement(np.zeros((4, 50), dtype=np.int8)) == 0.0


def test_even_split_is_the_maximum():
    """`S`개가 반반으로 갈리면 어긋나는 쌍이 최대다. `S=4`면 `2*2/6 = 2/3`."""
    binary = np.array([[1], [1], [0], [0]], dtype=np.int8)
    assert np.isclose(pairwise_disagreement(binary), 2 / 3)


def test_single_seed_is_undefined_not_zero():
    """시드가 하나면 '완벽히 재현된다'가 아니라 **잴 수 없다**. 0을 주면 표에서 1등이 된다."""
    assert np.isnan(pairwise_disagreement(np.ones((1, 10), dtype=np.int8)))
