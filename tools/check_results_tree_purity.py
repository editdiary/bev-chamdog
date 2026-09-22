"""산출물이 **한 결과 트리에서만** 나왔는지 검사한다.

**왜 필요한가.** 캠페인을 새 루트(`runs/paper_final_cumulative/`)에 다시 돌리는데,
분석·패키지 도구의 기본값은 전부 옛 루트(`runs/paper_final/`)를 가리킨다. 드라이버가
하나라도 덮어쓰기를 빠뜨리면 **옛 런과 새 런이 섞인 표**가 나오고, 그건 숫자만 보면
알아챌 수 없다. 이 검사기는 그 섞임을 시끄럽게 만든다.

산출물(JSON·CSV·txt)에 적힌 경로 문자열을 훑어서, 허용한 루트가 아닌 결과 트리를
가리키는 것이 있으면 실패한다.

    python tools/check_results_tree_purity.py \\
        --targets=runs/paper_final_cumulative \\
        --expect=runs/paper_final_cumulative
"""
import argparse
import re
from pathlib import Path

#: 결과 트리로 볼 경로 패턴. `runs/<무언가>` 형태를 다 잡는다.
RUN_ROOT_RE = re.compile(r"runs/[A-Za-z0-9_.\-]+")
#: 내용이 아니라 바이너리·대용량이라 훑지 않는 것.
SKIP_SUFFIXES = {".pth", ".npz", ".npy", ".png", ".pdf", ".svg", ".pt"}
SCAN_SUFFIXES = {".json", ".csv", ".txt", ".md", ".log"}


def scan(target: Path, expect: set, ignore: set):
    """`[(파일, 줄번호, 발견한 루트)]`."""
    bad = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            continue
        if path.suffix not in SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for found in RUN_ROOT_RE.findall(line):
                if found in expect or found in ignore:
                    continue
                bad.append((path, i, found))
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True,
                        help="검사할 폴더들 (쉼표로 구분)")
    parser.add_argument("--expect", required=True,
                        help="허용하는 결과 루트들 (쉼표로 구분). 예: runs/paper_final_cumulative")
    parser.add_argument("--ignore", default="",
                        help="일부러 참조하는 다른 트리 (쉼표로 구분). 예: runs/loss_effect")
    parser.add_argument("--max_rows", type=int, default=25)
    args = parser.parse_args()

    expect = {e.strip() for e in args.expect.split(",") if e.strip()}
    ignore = {i.strip() for i in args.ignore.split(",") if i.strip()}
    total = 0
    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        target = Path(name)
        if not target.exists():
            print(f"!! 대상이 없다: {target}")
            total += 1
            continue
        bad = scan(target, expect, ignore)
        print(f"{target}: 허용 밖 결과 트리 참조 {len(bad)}건  "
              f"(허용 {sorted(expect)}{', 무시 ' + str(sorted(ignore)) if ignore else ''})")
        for path, line, found in bad[:args.max_rows]:
            print(f"   {path}:{line}  -> {found}")
        if len(bad) > args.max_rows:
            print(f"   ... 외 {len(bad) - args.max_rows}건")
        total += len(bad)

    if total:
        print("\n!! 옛 결과 트리가 섞였다. 드라이버가 경로를 덮어쓰지 못한 곳이 있다.")
        return 1
    print("\n트리 순수성 통과 -- 산출물이 허용한 루트만 가리킨다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
