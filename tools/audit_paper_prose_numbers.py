"""논문 패키지의 **산문에 박힌 숫자**가 아직 데이터와 맞는지 검사한다.

**왜 필요한가.** `REPORT.md`·`README.md`에 소수 셋넷 자리 숫자가 340개 가까이 박혀 있다.
캠페인을 다시 돌려 CSV가 바뀌면 이 숫자들도 바뀌어야 하는데, 손으로 훑으면 반드시
놓친다. CSV만 갈고 산문을 두면 **논문이 데이터와 어긋난 채로 작성된다.**

판정은 셋이다.

| 분류 | 뜻 | 할 일 |
|---|---|---|
| `stale` | **옛 CSV에는 있는데 새 CSV에는 없다** | 반드시 고친다 |
| `ok` | 새 CSV에서 그 값을 찾았다 | 둔다 |
| `unsourced` | 어느 CSV에서도 못 찾았다 | 사람이 본다 (손계산·외부 출처·우연한 숫자) |

CSV 값은 산문이 쓴 **자릿수로 반올림해서** 비교한다 -- 산문은 `−0.0134`, CSV는
`-0.013354`로 적혀 있기 때문이다.

    python tools/audit_paper_prose_numbers.py \\
        --old=docs/archive/paper_package_v2/01_overall --new=/tmp/stage/01_overall
    python tools/audit_paper_prose_numbers.py --old=docs/archive/paper_package_v2/01_overall
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

#: 소수 셋 자리 이상만 본다. `0.5`나 `2.0` 같은 설정값까지 잡으면 잡음이 된다.
NUMBER_RE = re.compile(r"(?<![\w.])([−+-]?)(\d+\.\d{3,6})(?![\w.])")
PROSE_GLOBS = ("*.md",)


def _json_values(node, out):
    """provenance JSON 안의 모든 수치. baseline 같은 값이 CSV가 아니라 여기 있다."""
    if isinstance(node, dict):
        for v in node.values():
            _json_values(v, out)
    elif isinstance(node, list):
        for v in node:
            _json_values(v, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if math.isfinite(float(node)):
            out.append(float(node))


def _csv_values(package: Path):
    """패키지 `data/`의 모든 수치 값과 `provenance/`의 JSON 수치."""
    values = []
    for path in sorted((package / "provenance").glob("*.json")):
        _json_values(json.loads(path.read_text(encoding="utf-8")), values)
    for path in sorted((package / "data").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                for cell in row:
                    try:
                        v = float(cell)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        values.append(v)
    return values


def _rounded_sets(values):
    """자릿수별로 반올림한 집합. 산문이 쓴 자릿수에 맞춰 조회한다."""
    out = {}
    for d in range(3, 7):
        out[d] = {round(v, d) for v in values}
        # 부호를 뒤집어 쓴 표기(`−0.0134`로 적힌 감소량)도 맞는 것으로 본다.
        out[d] |= {round(-v, d) for v in values}
    return out


def prose_numbers(path: Path):
    """`(줄번호, 원문 표기, 부호 있는 값, 소수 자릿수)`."""
    found = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for sign, body in NUMBER_RE.findall(line):
            value = float(body)
            if sign in ("-", "−"):
                value = -value
            found.append((i, f"{sign}{body}", value, len(body.split(".")[1])))
    return found


def audit(package: Path, old_sets, new_sets):
    report = {"stale": [], "unsourced": [], "ok": 0}
    for path in sorted(p for g in PROSE_GLOBS for p in package.glob(g)):
        for line, text, value, digits in prose_numbers(path):
            d = min(max(digits, 3), 6)
            in_new = new_sets is not None and round(value, d) in new_sets[d]
            in_old = round(value, d) in old_sets[d]
            if in_new or (new_sets is None and in_old):
                report["ok"] += 1
            elif in_old:
                report["stale"].append((path.name, line, text))
            else:
                report["unsourced"].append((path.name, line, text))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="기존 패키지 폴더 (산문도 여기서 읽는다)")
    parser.add_argument("--new", default=None,
                        help="다시 만든 패키지 폴더. 없으면 기존 CSV로 추적률만 낸다")
    parser.add_argument("--max_rows", type=int, default=40)
    args = parser.parse_args()

    old = Path(args.old)
    old_sets = _rounded_sets(_csv_values(old))
    new_sets = _rounded_sets(_csv_values(Path(args.new))) if args.new else None

    report = audit(old, old_sets, new_sets)
    total = report["ok"] + len(report["stale"]) + len(report["unsourced"])
    print(f"=== 산문 숫자 감사: {old} ===")
    print(f"  숫자 {total}개  |  데이터와 일치 {report['ok']}  "
          f"|  **낡음 {len(report['stale'])}**  |  출처 불명 {len(report['unsourced'])}")

    if report["stale"]:
        print("\n!! 낡은 숫자 -- 옛 CSV에는 있는데 새 CSV에는 없다. 반드시 고친다.")
        for name, line, text in report["stale"][:args.max_rows]:
            print(f"   {name}:{line}  {text}")
        if len(report["stale"]) > args.max_rows:
            print(f"   ... 외 {len(report['stale']) - args.max_rows}개")
    if report["unsourced"]:
        print(f"\n   출처 불명 {len(report['unsourced'])}개 (손계산·외부 출처·우연한 숫자일 수 있다)")
        for name, line, text in report["unsourced"][:args.max_rows]:
            print(f"   {name}:{line}  {text}")
        if len(report["unsourced"]) > args.max_rows:
            print(f"   ... 외 {len(report['unsourced']) - args.max_rows}개")
    return 1 if report["stale"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
