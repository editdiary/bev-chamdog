import numpy as np
import torch

from projects.common.baselines import all_free_map, as_batch, constant_free_map


def test_constant_free_map_takes_the_per_cell_majority():
    masks = [
        np.array([[True, True], [False, False]]),
        np.array([[True, False], [False, False]]),
        np.array([[True, False], [True, False]]),
    ]

    result = constant_free_map(masks)

    # 셀별 free 비율: 1.0, 0.33, 0.33, 0.0 -> 0.5 초과만 True
    assert result.tolist() == [[True, False], [False, False]]
    assert result.dtype == bool


def test_constant_free_map_uses_strict_majority_threshold():
    """정확히 50% 경계에서의 동작을 검증 — 다수결은 > 0.5이지 >= 0.5가 아님."""
    masks = [
        np.array([[True, False]]),
        np.array([[False, True]]),
    ]

    result = constant_free_map(masks)

    # 각 셀이 정확히 0.5 비율: (0.5) strict majority는 None, 따라서 False로 취급
    assert result.tolist() == [[False, False]]


def test_all_free_map_is_free_everywhere():
    result = all_free_map((2, 3))
    assert result.shape == (2, 3)
    assert result.all()
    assert result.dtype == bool


def test_as_batch_broadcasts_a_single_map_over_the_batch():
    batched = as_batch(np.array([[True, False]]), batch_size=4, device="cpu")

    assert batched.shape == (4, 1, 1, 2)
    assert batched.dtype == torch.bool
    assert batched[0, 0].tolist() == [[True, False]]
