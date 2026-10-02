"""끊긴 런(로그는 있는데 마지막 체크포인트가 없다)을 찾아 지운다 -- 캠페인 재시도용.

큐 러너들은 **불완전한 런을 덮어쓰지 않는다**(`blocked_incomplete`) -- 사람이 없을 때 그대로 두면
그 런은 영원히 빠진다. 캠페인 드라이버가 실험마다 러너를 돈 뒤 이 도구로 끊긴 런을 지우고 러너를
**한 번 더** 돌린다. 두 번째에도 끊기면 그 런은 빠진 채로 남고 보고에 적힌다.

판정: `<...>/logs/<run>/`가 있고 `<...>/ckpt/<run>/model-{N:09d}.pth`가 없으면 불완전이다.
N은 그 런의 `config.json`의 `num_epochs`(없으면 `--num_epochs`). 지운 목록은 `--record`에 덧붙인다.

    python tools/clean_incomplete_runs.py --root=runs/99_full_campaign/02_projection_and_prior --dry_run
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from fire import Fire


def find_incomplete(root, num_epochs=100) -> list:
    """`[(log_dir, ckpt_dir, 기대 epoch), ...]`."""
    out = []
    for logs in sorted(Path(root).rglob("logs")):
        if not logs.is_dir() or "analysis" in logs.parts:
            continue
        for run in sorted(d for d in logs.iterdir() if d.is_dir()):
            cfg = run / "config.json"
            n = num_epochs
            if cfg.exists():
                try:
                    n = int(json.loads(cfg.read_text()).get("num_epochs", num_epochs))
                except (ValueError, json.JSONDecodeError):
                    pass
            ckpt = logs.parent / "ckpt" / run.name
            if not (ckpt / f"model-{n:09d}.pth").exists():
                out.append((run, ckpt, n))
    return out


def main(root, num_epochs=100, dry_run=False, record=None):
    found = find_incomplete(root, num_epochs)
    if not found:
        print(f"끊긴 런 없음: {root}")
        return 0
    lines = []
    for log_dir, ckpt_dir, n in found:
        line = f"{'(dry) ' if dry_run else ''}불완전 -> 삭제: {log_dir} (+ {ckpt_dir}), 기대 epoch {n}"
        print(line)
        lines.append(line)
        if not dry_run:
            shutil.rmtree(log_dir, ignore_errors=True)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
    if record and not dry_run:
        with open(record, "a", encoding="utf-8") as fh:
            stamp = datetime.now().strftime("%F %T")
            fh.writelines(f"{stamp}  {line}\n" for line in lines)
    return len(found)


if __name__ == "__main__":
    Fire(main)
    sys.exit(0)
