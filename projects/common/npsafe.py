"""numpy 임시 배열 소거(temporary elision)가 **살아 있는 배열을 덮어쓰는** 것을 막는다.

## 무엇이 문제인가 (2026-09-01 발견)

이 환경은 **numpy 1.26.4를 Python 3.14.6에서** 돌리고 있다. numpy 1.26은 Python 3.12까지만
지원한다(패키지 분류자에 3.13/3.14가 없다). 그 조합에서 numpy의 "임시 배열 소거" 최적화가
오작동한다 -- 원래는 `a + b + c` 같은 식에서 중간 임시 배열을 재활용해 메모리를 아끼는
기능인데, **어떤 것이 임시인지 판단하려고 호출자의 프레임을 들여다보는 코드**가 3.14에서
틀린 답을 낸다. 그래서 살아 있는 이름 붙은 배열을 임시로 착각하고 **제자리에서 덮어쓴다.**

실측 재현 (bool 배열 108만 개):

    near = (np.abs(d) <= 0.15) & valid      # near.sum() = 131,751
    far  = valid & ~near                    # 여기서 near가 뒤집힌다
    near.sum()                              # 948,249  <- ~near로 바뀌어 있다

**예외도 경고도 없다.** 그리고 조건이 미묘해서(배열이 256 KB보다 커야 하고, 프레임 모양에
따라 걸리기도 안 걸리기도 한다) 같은 식을 다른 자리에서 다시 계산하면 맞는 값이 나온다 --
코드를 아무리 들여다봐도 원인이 안 보인다. 실제로 `tools/report_decision_disagreement.py`의
표 절반이 조용히 틀린 값으로 나왔다.

## 이 모듈의 규칙

**뒤에서 다시 쓸 numpy 배열에는 `~`를 직접 쓰지 않는다.** 대신 `bool_not(a)`를 쓴다.
`out=`을 명시하면 numpy가 소거할 임시가 없으므로 원본을 건드릴 수 없다.

torch 텐서에는 해당되지 않는다(`~tensor`는 torch가 처리한다). 그래서 학습 루프의 지표
대부분은 영향이 없다 -- 위험한 곳은 numpy로 내려간 뒤의 계산이다.

## 근본 해결

**numpy를 Python 3.14를 지원하는 판(>= 2.3)으로 올리는 것이 정본 해결이다.** 이 모듈은
그때까지의 방어이고, 올린 뒤에도 남겨 두면 손해가 없다(`out=`은 원래 안전한 관용구다).
진단은 `python tools/check_numpy_elision.py`.
"""
import numpy as np


def bool_not(a):
    """`~a`와 같되 **`a`를 절대 건드리지 않는다.**

    `out=`을 주면 numpy가 결과를 쓸 버퍼가 이미 정해져 있으므로 임시 소거가 개입할 수 없다.
    입력이 bool이 아니면 bool로 보고 부정한다(`np.logical_not`의 계약과 같다).
    """
    a = np.asarray(a)
    return np.logical_not(a, out=np.empty(a.shape, dtype=bool))


def elision_is_broken(n: int = 1_080_000) -> bool:
    """이 환경에서 임시 소거가 살아 있는 배열을 덮어쓰는지 실측한다.

    `n`이 numpy의 소거 문턱(256 KB)보다 충분히 커야 한다 -- 작은 배열에서는 최적화 자체가
    걸리지 않아 항상 `False`가 나온다.
    """
    rng = np.random.default_rng(0)
    keep = rng.random(n) < 0.97
    probe = (rng.random(n) < 0.12) & keep
    before = int(probe.sum())
    _ = keep & ~probe
    return int(probe.sum()) != before
