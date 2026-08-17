import pytest
import torch

from projects.common.free_space import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    decompose,
    decompose_from_class_index,
    partition_defect_count,
    to_class_index,
)


def _grid(values):
    """(H, W) 리스트 -> (1, 1, H, W) float 텐서."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def test_decompose_splits_valid_cells_into_exactly_three_parts():
    occ = _grid([[1, 1], [0, 0]])
    vis = _grid([[1, 0], [1, 0]])
    valid = _grid([[1, 1], [1, 1]])

    parts = decompose(occ, vis, valid)

    assert parts["free"].squeeze().tolist() == [[True, False], [False, False]]
    assert parts["occupied"].squeeze().tolist() == [[False, False], [True, False]]
    # unknown은 vis=0인 셀 전부 -- occ 값과 무관하다
    assert parts["unknown"].squeeze().tolist() == [[False, True], [False, True]]
    assert partition_defect_count(parts, valid.bool()) == 0


def test_invalid_cells_belong_to_no_part():
    """valid=0은 unknown이 아니다 -- 배포 때 존재하지 않는 수집 아티팩트라 gradient도 지표도 받지 않는다."""
    occ = _grid([[1, 1]])
    vis = _grid([[0, 0]])
    valid = _grid([[1, 0]])

    parts = decompose(occ, vis, valid)

    assert parts["unknown"].squeeze().tolist() == [True, False]
    assert partition_defect_count(parts, valid.bool()) == 0


def test_class_index_roundtrip_preserves_the_partition():
    occ = _grid([[1, 0], [1, 0]])
    vis = _grid([[1, 1], [0, 0]])
    valid = _grid([[1, 1], [1, 1]])
    parts = decompose(occ, vis, valid)

    index = to_class_index(parts)
    assert index.squeeze().tolist() == [[FREE, OCCUPIED], [UNKNOWN, UNKNOWN]]

    restored = decompose_from_class_index(index, valid.bool())
    for key in ("free", "occupied", "unknown"):
        assert torch.equal(restored[key], parts[key])


def test_decompose_thresholds_fractional_probabilities_strictly_above_half():
    """`_as_bool`은 `> 0.5`다 -- 0.5는 정확히 False로 떨어져야 한다(모델의 sigmoid 출력이 실수라서).

    각 셀:
        0: occ=0.50(F), vis=0.90(T) -> occupied  (0.5는 경계 그 자체, occ가 아니다)
        1: occ=0.51(T), vis=0.90(T) -> free      (경계 바로 위)
        2: occ=0.90(T), vis=0.49(F) -> unknown   (vis가 경계 바로 아래)
        3: occ=0.49(F), vis=0.51(T) -> occupied  (occ가 경계 바로 아래)
    비교 방향이 `>=`나 `<`로 바뀌면 셀 0/2/3 중 하나 이상이 뒤집혀 아래 assert가 깨진다.
    """
    occ = _grid([[0.50, 0.51, 0.90, 0.49]])
    vis = _grid([[0.90, 0.90, 0.49, 0.51]])
    valid = _grid([[1.0, 1.0, 1.0, 1.0]])

    parts = decompose(occ, vis, valid)

    assert parts["free"].squeeze().tolist() == [False, True, False, False]
    assert parts["occupied"].squeeze().tolist() == [True, False, False, True]
    assert parts["unknown"].squeeze().tolist() == [False, False, True, False]
    assert partition_defect_count(parts, valid.bool()) == 0


def test_partition_defect_count_catches_overlap():
    valid = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    broken = {
        "free": torch.tensor([[[[True, False]]]]),
        "occupied": torch.tensor([[[[True, False]]]]),   # free와 겹친다
        "unknown": torch.tensor([[[[False, True]]]]),
    }

    assert partition_defect_count(broken, valid) == 1
