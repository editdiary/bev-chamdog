"""런의 **모든 epoch 지표를 하나의 CSV로** 뽑는다 -- 분석이 아니라 보존이 목적이다.

## 왜 필요한가

집계 도구들(`report_ablation.py` 등)은 각자 필요한 태그 몇 개만 골라 읽고 표로 찍는다.
그 표에 안 들어간 지표는 **없어진 것이 아니라 TensorBoard event 파일 안에 남아 있는데**,
읽으려면 `EventAccumulator`가 필요하고 그건 tensorboard 버전에 묶여 있다. 나중에 "그때
`kl_free`는 어땠지" 같은 질문이 생기면 그 의존을 다시 세워야 한다.

이 도구는 **태그를 고르지 않는다.** 런 디렉터리에 있는 scalar 전부를 긴 형식(long format)
CSV 한 장으로 떨어뜨린다. 열은 다음과 같다.

    run, cell, seed, tag, epoch, value

`cell`/`seed`는 `{cell}_s{seed}` 이름 규약에서 나눈 것이고, 규약을 안 따르는 런은 `cell`에
런 이름 전체가 들어가고 `seed`가 비어 있다.

같이 나오는 것 둘.

- `configs.json` -- 런마다 `config.json` 전체. 어떤 손잡이로 돌린 런인지가 여기 있다.
- `manifest.json` -- 런마다 epoch 수·태그 목록·체크포인트 파일명·split 파일의 줄 수.
  **"이 CSV가 무엇을 담고 있나"를 CSV를 열지 않고 확인하는 용도**이고, 런이 중간에 죽어
  epoch이 모자란 것을 여기서 바로 본다.

실행:

    python tools/export_run_scalars.py --log_root=runs/loss_effect/logs
    python tools/export_run_scalars.py --log_root=runs/loss_effect/logs \\
        --out_dir=runs/loss_effect/analysis
"""
import csv
import json
import sys
from pathlib import Path

from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def split_run_name(name: str):
    """`{cell}_s{seed}` -> `(cell, seed)`. 규약 밖이면 `(name, None)`이다.

    **뒤에서부터 자른다** -- 셀 이름에 `_`가 들어갈 수 있다(`A_ce`, `B_perset`).
    """
    head, sep, tail = name.rpartition("_s")
    if sep and tail.isdigit():
        return head, int(tail)
    return name, None


def read_run(run_dir: Path) -> dict:
    """한 런의 scalar 전부와 부속 정보. `size_guidance=0`이 **다운샘플링을 끈다** --
    기본값은 태그당 1000점만 남기는데, 그러면 긴 런에서 조용히 값이 빠진다."""
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(acc.Tags()["scalars"])
    series = {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in tags}
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    splits = {p.name: sum(1 for _ in p.open()) for p in sorted(run_dir.glob("split_*.txt"))}
    return {"tags": tags, "series": series, "config": config, "splits": splits}


def main(log_root, out_dir=None, pattern="*"):
    log_root = Path(log_root)
    # 기본 출력은 `logs`의 형제 `analysis`다 -- 로그 디렉터리를 오염시키지 않으면서
    # 런과 같은 root 아래 있어야 통째로 옮겨도 짝이 유지된다.
    out_dir = Path(out_dir) if out_dir else log_root.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(d for d in log_root.iterdir() if d.is_dir() and d.match(pattern))
    if not runs:
        print(f"런이 없다: {log_root}/{pattern}")
        return

    rows, configs, manifest = [], {}, {}
    for run_dir in runs:
        info = read_run(run_dir)
        cell, seed = split_run_name(run_dir.name)
        for tag, points in info["series"].items():
            for epoch, value in points:
                rows.append((run_dir.name, cell, "" if seed is None else seed,
                             tag, epoch, repr(float(value))))
        configs[run_dir.name] = info["config"]
        # **epoch 축과 step 축을 섞으면 안 된다.** `train/loss_step`·`train/lr`은 global step을
        # x로 쓰므로 40 epoch 런에서 x가 960까지 간다. 완주 판정은 `*_epoch` 태그로만 한다.
        epochs = [s for tag, points in info["series"].items() if tag.endswith("_epoch")
                  for s, _ in points]
        step_tags = [t for t in info["tags"] if not t.endswith("_epoch")]
        manifest[run_dir.name] = {
            "cell": cell, "seed": seed,
            "n_tags": len(info["tags"]),
            "max_epoch": max(epochs) if epochs else 0,
            "n_points": sum(len(p) for p in info["series"].values()),
            "tags": info["tags"],
            # step 축 태그는 CSV에 그대로 들어 있지만 `epoch` 열의 뜻이 다르다 -- 명시한다.
            "step_axis_tags": step_tags,
            "splits": info["splits"],
            "num_epochs_requested": info["config"].get("num_epochs"),
            "complete": (bool(epochs) and info["config"].get("num_epochs") is not None
                         and max(epochs) >= int(info["config"]["num_epochs"])),
        }
        print(f"  {run_dir.name:16s} 태그 {len(info['tags']):3d}개, "
              f"{max(epochs) if epochs else 0:3d} epoch,"
              f" {sum(len(p) for p in info['series'].values()):6d}점"
              + ("" if manifest[run_dir.name]["complete"] else "   <- 미완"))

    csv_path = out_dir / "scalars.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("run", "cell", "seed", "tag", "epoch", "value"))
        writer.writerows(rows)
    (out_dir / "configs.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    incomplete = [k for k, v in manifest.items() if not v["complete"]]
    print(f"\n{len(runs)}개 런, {len(rows):,}행 -> {csv_path}")
    print(f"  config -> {out_dir / 'configs.json'}")
    print(f"  manifest -> {out_dir / 'manifest.json'}")
    if incomplete:
        # **미완 런을 표에 넣으면 조용히 틀린다** -- 값이 있으니 찍히기는 한다.
        print(f"  !! 미완 런 {len(incomplete)}개: {', '.join(incomplete)}")


if __name__ == "__main__":
    Fire(main)
