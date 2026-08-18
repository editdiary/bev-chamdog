"""학습 산출물 정리 -- 체크포인트는 지우고 **로그와 best는 남긴다**.

왜 도구로 두는가: 체크포인트 하나가 약 487 MB이고 한 런이 주기 저장 6개 + best 1개라
**런당 약 3.2 GB**다. ablation을 몇 번만 돌려도 디스크가 수십 GB로 불어난다. 그때마다 `rm`을
손으로 쓰면 `model_best`를 실수로 지우는 사고가 나므로(재학습 말고는 복구 수단이 없다) 선택
규칙을 코드로 고정하고 테스트로 못박는다.

무엇을 남기는가:

- **TensorBoard 로그는 절대 지우지 않는다.** 수십 KB인데 실험의 결론이 전부 여기 있다.
  체크포인트를 지운 뒤에도 곡선은 그대로 읽을 수 있다.
- `split_*.txt`(그 런이 실제로 쓴 샘플 목록)도 남긴다 -- 로그와 같은 폴더에 있고 작다.
- `model_best-*.pth`는 기본적으로 남긴다. 재채점(`tools/rescore_checkpoints.py`)과 시각화가
  이것만 필요로 한다.
- `model-*.pth`(주기 저장)는 지운다. 조사가 끝난 런에서는 다시 볼 일이 없다.

실행 (기본은 dry-run이라 아무것도 지우지 않는다):

    python tools/prune_runs.py --pattern='ft_wd*'            # 무엇이 지워지는지 본다
    python tools/prune_runs.py --pattern='ft_wd*' --apply    # 실제로 지운다
    python tools/prune_runs.py --pattern='ft_cw*' --apply --keep_best=False
"""
import fnmatch
import sys
from pathlib import Path

from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_ROOTS = ("runs/robot_bev/ckpt", "runs/synwoodscape_threeclass/ckpt")

PERIODIC_PREFIX = "model-"
BEST_PREFIX = "model_best-"


def classify(paths, keep_best: bool = True) -> dict:
    """체크포인트 경로들을 `keep`/`remove`로 가른다.

    이름으로만 판단한다: `model_best-*.pth`가 best이고 `model-*.pth`가 주기 저장이다
    (`saverloader.save`의 `model_name` 규약). **`model_best`는 `model`로 시작하므로 접두사
    검사 순서가 중요하다** -- `startswith("model-")`로 먼저 걸러야 best가 주기 저장으로
    오분류되지 않는다.

    `.pth`가 아닌 파일은 어느 쪽에도 넣지 않는다 -- 이 도구가 모르는 파일을 지우지 않게 한다.
    """
    keep, remove = [], []
    for path in sorted(paths):
        name = Path(path).name
        if not name.endswith(".pth"):
            keep.append(path)
        elif name.startswith(BEST_PREFIX):
            (keep if keep_best else remove).append(path)
        elif name.startswith(PERIODIC_PREFIX):
            remove.append(path)
        else:
            keep.append(path)  # 규약을 벗어난 파일은 손대지 않는다
    return {"keep": keep, "remove": remove}


def select_runs(roots, pattern: str) -> list:
    """`pattern`에 맞는 런 디렉터리. `fnmatch`라 셸 glob과 같은 규칙이다."""
    selected = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for run_dir in sorted(root_path.iterdir()):
            if run_dir.is_dir() and fnmatch.fnmatch(run_dir.name, pattern):
                selected.append(run_dir)
    return selected


def _human(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def main(pattern="*", roots=DEFAULT_ROOTS, keep_best=True, apply=False):
    """`apply=False`(기본)이면 무엇이 지워질지만 출력한다."""
    if isinstance(roots, str):
        roots = (roots,)
    runs = select_runs(roots, pattern)
    if not runs:
        print(f"pattern={pattern!r}에 맞는 런이 없다 (roots={list(roots)})")
        return

    freed = 0
    for run_dir in runs:
        groups = classify(list(run_dir.iterdir()), keep_best=keep_best)
        size = sum(Path(p).stat().st_size for p in groups["remove"])
        freed += size
        print(f"\n{run_dir}")
        print(f"  남김 {len(groups['keep']):2d}개: "
              f"{', '.join(Path(p).name for p in groups['keep']) or '-'}")
        print(f"  삭제 {len(groups['remove']):2d}개 ({_human(size)}): "
              f"{', '.join(Path(p).name for p in groups['remove']) or '-'}")
        if apply:
            for path in groups["remove"]:
                Path(path).unlink()

    print(f"\n{'삭제함' if apply else 'dry-run -- 삭제하면'} {_human(freed)} 확보"
          f" ({len(runs)}개 런)")
    if not apply:
        print("실제로 지우려면 `--apply`를 붙인다.")


if __name__ == "__main__":
    Fire(main)
