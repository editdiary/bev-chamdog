import numpy as np


def class_consistency_rate(predicted: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray = None) -> float:
    """Fraction of entries where `predicted` equals `reference`, restricted to `valid_mask` if given."""
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    matches = predicted == reference
    if valid_mask is not None:
        matches = matches[np.asarray(valid_mask, dtype=bool)]
    return float(matches.mean()) if matches.size else float("nan")
