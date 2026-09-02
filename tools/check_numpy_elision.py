"""이 환경의 numpy가 **살아 있는 배열을 덮어쓰는지** 진단한다.

배경과 원인은 `projects/common/npsafe.py`의 docstring에 있다. 한 줄로: numpy 1.26은
Python 3.12까지만 지원하는데 이 환경은 3.14이고, 그 조합에서 numpy의 임시 배열 소거
최적화가 **이름 붙은 살아 있는 배열을 임시로 착각해 제자리에서 덮어쓴다.** 예외도 경고도
없고, 조건이 미묘해서 같은 식이 어떤 자리에서는 맞고 어떤 자리에서는 틀린다.

이 도구는 그 상태를 숫자로 찍는다. **결과가 `깨짐`이면 numpy로 내려간 모든 계산을
의심해야 한다** -- 이 저장소에서는 광선 추적(`polar`), 거리 변환 기반 `f1@τ`
(`occupied_metrics`), 그리고 분석 도구들이 그 경로다.

정본 해결은 **Python 3.14를 지원하는 numpy(>= 2.3)로 올리는 것**이다. 그 전까지는
`npsafe.bool_not`으로 막는다.

실행: python tools/check_numpy_elision.py
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.npsafe import bool_not  # noqa: E402

# numpy의 소거 문턱은 256 KB다. 그보다 작은 배열에서는 최적화가 아예 안 걸리므로
# "정상"으로 보이는데, 그건 안전한 것이 아니라 못 재는 것이다.
SIZES = (1_000, 100_000, 300_000, 1_080_000, 5_000_000)


def _trial(n, rng):
    """`keep & ~probe` 뒤에 `probe`가 살아 있는지. `(뒤집혔나, 전, 후)`."""
    keep = rng.random(n) < 0.97
    probe = (rng.random(n) < 0.12) & keep
    before = int(probe.sum())
    _ = keep & ~probe
    after = int(probe.sum())
    return before != after, before, after


def _trial_safe(n, rng):
    keep = rng.random(n) < 0.97
    probe = (rng.random(n) < 0.12) & keep
    before = int(probe.sum())
    _ = keep & bool_not(probe)
    return before != int(probe.sum())


def main(sizes=SIZES, repeats=3):
    sizes = tuple(sizes) if isinstance(sizes, (list, tuple)) else (int(sizes),)
    print(f"numpy {np.__version__} | python {sys.version.split()[0]}"
          f" | numpy 소거 문턱 256 KB (bool 262,144개)\n")
    print(f"  {'배열 크기':>12} {'`x & ~y`':>12} {'`x & bool_not(y)`':>20}")
    broken = False
    for n in sizes:
        rng = np.random.default_rng(0)
        hits = sum(_trial(n, rng)[0] for _ in range(repeats))
        safe_hits = sum(_trial_safe(n, rng) for _ in range(repeats))
        broken = broken or hits > 0
        print(f"  {n:12,} {f'{hits}/{repeats} 깨짐' if hits else 'OK':>12}"
              f" {f'{safe_hits}/{repeats} 깨짐' if safe_hits else 'OK':>20}")

    print()
    if broken:
        print("!! 이 환경의 numpy는 살아 있는 배열을 덮어쓴다.")
        print("   - 뒤에서 다시 쓸 배열에 `~`를 직접 쓰지 말 것 (`npsafe.bool_not` 사용)")
        print("   - 정본 해결: Python 3.14를 지원하는 numpy(>= 2.3)로 올린다")
        print("   - 지금까지 나온 numpy 기반 숫자는 교차 검증 없이 믿지 말 것")
        return 1
    print("소거로 인한 덮어쓰기는 관측되지 않았다.")
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main) or 0)
