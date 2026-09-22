"""논문 패키지의 CSV 두 벌을 칸 단위로 대조한다.

**왜 필요한가.** 캠페인을 다른 목적함수로 다시 돌리면 패키지의 숫자가 바뀐다. 기대는
"논문 문장은 그대로고 자릿수만 바뀐다"인데, 그 기대를 **눈으로 확인하면 놓친다** --
CSV 열두 개에 수천 칸이다. 이 도구는 바뀐 칸만 뽑고, 그중 **판정이 뒤집힌 칸**을
따로 세운다. 원장 §8 Phase 3: "결론이 뒤집히는 칸이 나오면 그것 자체가 보고 대상이다."

    python tools/diff_paper_package_csv.py --old=docs/paper_package/01_overall \\
        --new=/tmp/pkg01 --rel_tol=0.05

`--rel_tol`보다 크게 움직인 수치 칸과, 글자가 바뀐 칸을 모두 보고한다.
"""
import argparse
import csv
import math
from pathlib import Path

#: 값이 아니라 판정인 열. 여기가 바뀌면 논문 문장이 바뀐다.
VERDICT_COLUMNS = ("verdict", "sign_agreement", "better_direction", "decision", "claim")
#: provenance는 커밋 해시·시각이 매번 달라서 따로 취급한다.
NOISY_KEYS = ("git_commit", "generated_at", "updated_at", "date", "executable")


def _num(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if rows else ([], [])


def diff_csv(old_path, new_path, rel_tol, abs_floor):
    """`(구조 문제 목록, 바뀐 칸 목록)`. 칸은 `(행번호, 열이름, 옛값, 새값, 상대변화)`."""
    old_header, old_rows = _read(old_path)
    new_header, new_rows = _read(new_path)
    problems = []
    if old_header != new_header:
        problems.append(f"헤더가 다르다: {old_header} -> {new_header}")
        return problems, []
    if len(old_rows) != len(new_rows):
        problems.append(f"행 수가 다르다: {len(old_rows)} -> {len(new_rows)}")

    changed = []
    for i, (a, b) in enumerate(zip(old_rows, new_rows)):
        for j, name in enumerate(old_header):
            av = a[j] if j < len(a) else ""
            bv = b[j] if j < len(b) else ""
            if av == bv:
                continue
            na, nb = _num(av), _num(bv)
            if na is None or nb is None:
                changed.append((i + 2, name, av, bv, None))
                continue
            scale = max(abs(na), abs(nb), abs_floor)
            rel = abs(nb - na) / scale
            if rel > rel_tol:
                changed.append((i + 2, name, av, bv, rel))
    return problems, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="기존 패키지 폴더")
    parser.add_argument("--new", required=True, help="다시 만든 패키지 폴더")
    parser.add_argument("--rel_tol", type=float, default=0.02,
                        help="이보다 작은 상대 변화는 보고하지 않는다 (기본 2%%)")
    parser.add_argument("--abs_floor", type=float, default=1e-4,
                        help="0 근처에서 상대 변화가 폭발하지 않게 하는 바닥")
    parser.add_argument("--max_rows", type=int, default=20, help="파일당 출력할 칸 수")
    args = parser.parse_args()

    old_data, new_data = Path(args.old) / "data", Path(args.new) / "data"
    old_files = {p.name for p in old_data.glob("*.csv")}
    new_files = {p.name for p in new_data.glob("*.csv")}

    print(f"=== 패키지 CSV 대조 ===\n  기존 {old_data}\n  신규 {new_data}"
          f"\n  상대 허용치 {args.rel_tol:.1%}, 바닥 {args.abs_floor}\n")
    for name in sorted(old_files - new_files):
        print(f"!! 신규에 없다: {name}")
    for name in sorted(new_files - old_files):
        print(f"++ 신규에만 있다: {name}")

    total_changed = verdict_changed = 0
    for name in sorted(old_files & new_files):
        problems, changed = diff_csv(old_data / name, new_data / name,
                                     args.rel_tol, args.abs_floor)
        verdicts = [c for c in changed if c[1] in VERDICT_COLUMNS]
        total_changed += len(changed)
        verdict_changed += len(verdicts)
        if not problems and not changed:
            print(f"  {name:<44} 변화 없음")
            continue
        print(f"\n--- {name}")
        for p in problems:
            print(f"  !! {p}")
        if verdicts:
            print(f"  ** 판정이 바뀐 칸 {len(verdicts)}개 -- 논문 문장을 다시 봐야 한다")
            for row, col, a, b, _ in verdicts:
                print(f"     행{row} {col}: {a!r} -> {b!r}")
        others = [c for c in changed if c[1] not in VERDICT_COLUMNS]
        if others:
            others.sort(key=lambda c: -(c[4] or float("inf")))
            print(f"  수치가 {args.rel_tol:.0%} 넘게 바뀐 칸 {len(others)}개 "
                  f"(큰 것부터 최대 {args.max_rows}개)")
            for row, col, a, b, rel in others[:args.max_rows]:
                shown = "글자" if rel is None else f"{rel:+.1%}"
                print(f"     행{row} {col}: {a} -> {b}   ({shown})")

    print(f"\n=== 합계: 바뀐 칸 {total_changed}개, 그중 판정 {verdict_changed}개 ===")
    if verdict_changed:
        print("!! 판정이 바뀐 칸이 있다. 숫자만 갈아끼우지 말고 원장에 적어야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
