"""`projects/common/npsafe.bool_not`이 원본을 건드리지 않는 것을 고정한다.

이 테스트가 지키는 것은 **우리 코드의 관용구**이지 numpy의 버그가 아니다. 이 환경
(numpy 1.26.4 + Python 3.14.6)에서는 `x & ~y`가 `y`를 제자리에서 뒤집는 일이 실제로
일어나며(`npsafe` 모듈 docstring에 재현이 있다), 그때 예외도 경고도 나지 않는다.
그래서 뒤에서 다시 쓸 배열에는 `~` 대신 `bool_not`을 쓰기로 했고, 여기서 그 성질을 못박는다.

`elision_is_broken()`은 **환경 진단**이라 True/False 어느 쪽도 실패로 보지 않는다 --
numpy를 올리면 False가 되어야 하고, 그때도 `bool_not`은 그대로 안전하다.
"""
import numpy as np

from projects.common.npsafe import bool_not, elision_is_broken

# numpy의 임시 소거는 256 KB보다 큰 배열에서만 걸린다. 그보다 작게 잡으면 이 테스트가
# 아무것도 안 지키면서 통과한다.
BIG = 1_080_000


def test_bool_not_matches_tilde():
    rng = np.random.default_rng(0)
    a = rng.random(1000) < 0.4
    assert np.array_equal(bool_not(a), ~a.copy())


def test_bool_not_leaves_the_input_alone_in_a_compound_expression():
    """`&`의 오른쪽에 넣어도 원본이 살아 있어야 한다 -- 이것이 실제로 깨졌던 자리다."""
    rng = np.random.default_rng(1)
    keep = rng.random(BIG) < 0.97
    probe = (rng.random(BIG) < 0.12) & keep
    before = probe.copy()
    result = keep & bool_not(probe)
    assert np.array_equal(probe, before), "bool_not이 입력을 덮어썼다"
    assert int(result.sum()) == int((keep & ~before.copy()).sum())


def test_bool_not_survives_being_the_only_reference():
    """임시로 만들어 바로 넘겨도 결과가 맞아야 한다."""
    rng = np.random.default_rng(2)
    keep = rng.random(BIG) < 0.9
    total = int((keep & bool_not(rng.random(BIG) < 0.5)).sum())
    assert 0 < total < BIG


def test_environment_probe_runs_and_returns_a_bool():
    """진단용. **결과로 실패시키지 않는다** -- 환경 상태를 기록하는 것이 목적이다."""
    assert isinstance(elision_is_broken(BIG), bool)
