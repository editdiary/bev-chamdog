import numpy as np

from projects.common.metrics import class_consistency_rate


def test_class_consistency_rate_all_match():
    predicted = np.array([1, 2, 3])
    reference = np.array([1, 2, 3])

    assert class_consistency_rate(predicted, reference) == 1.0


def test_class_consistency_rate_partial_match():
    predicted = np.array([1, 2, 3, 4])
    reference = np.array([1, 0, 3, 0])

    assert class_consistency_rate(predicted, reference) == 0.5


def test_class_consistency_rate_respects_valid_mask():
    predicted = np.array([1, 99, 3])
    reference = np.array([1, 2, 3])
    valid_mask = np.array([True, False, True])

    assert class_consistency_rate(predicted, reference, valid_mask) == 1.0
