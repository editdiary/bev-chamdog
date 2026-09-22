"""고정 split의 **constant-map 기준선**을 만들어 `analysis/constant_map_baseline.json`에 쓴다.

**왜 도구로 만드나.** 이 값은 예전에 즉석으로 만들어 파일만 남아 있었다. 캠페인을 다른
루트에 다시 돌리자 그 루트에는 파일이 없어 패키지 생성이 멈췄다. 재현 가능해야 한다.

**학습과 무관한 값이다** -- train 시퀀스의 셀별 다수결 free map을 val에 한 번 채점한 것이라
라벨과 split만으로 정해진다. 그래서 재학습해도 같은 값이 나와야 하고, **다르게 나오면 그건
라벨이나 split이 바뀌었다는 뜻이다.**

논문에서 이 값은 "모델이 장면을 실제로 읽었나"의 바닥선이다 -- 자유공간 배치가 늘 비슷하면
상수 지도만으로도 IoU가 꽤 나오므로, 그 위의 margin이 실제 이득이다.

    python tools/make_constant_map_baseline.py --out=runs/.../fixed_split/analysis
"""
import json
import sys
from pathlib import Path

import torch
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.bev_gt.grid import ROBOT_GRID_SPEC as GRID_SPEC  # noqa: E402
from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import iou_free  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT, DEFAULT_DATASET_ROOT, FINETUNE_CAMERA_NAMES,
    build_bev_masks, load_masked_labels, split_samples_by_sequence,
)

FIXED_TRAIN = ("raws2", "raws3", "rawos1", "rawos2", "rawos4")
FIXED_VAL = ("raws1", "rawos3")


def compute(train_sequences, val_sequences, dataset_root, common_root, device):
    """`(기준선, train 프레임 수, val 프레임 수)`. `report_loso.fold_baseline`과 같은 계산이다."""
    root = Path(dataset_root)
    names = list(train_sequences) + list(val_sequences)
    train_samples, val_samples = split_samples_by_sequence(
        [root / n for n in names], list(val_sequences))
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)

    train_free = [decompose(*load_masked_labels(r, s, permanent_blind, invalid))["free"]
                  for r, s in train_samples]
    constant_map = constant_free_map(train_free)

    values, counts = [], []
    for r, s in val_samples:
        triple = load_masked_labels(r, s, permanent_blind, invalid)
        occ, _, valid = triple
        free_gt = torch.from_numpy(decompose(*triple)["free"]).view(1, 1, *occ.shape).to(device)
        valid_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
        value, count = iou_free(as_batch(constant_map, 1, device), free_gt, valid_t)
        values.append(value)
        counts.append(count)
    total = sum(counts)
    baseline = sum(v * c for v, c in zip(values, counts)) / total if total else float("nan")
    return float(baseline), len(train_samples), len(val_samples)


def main(out, train_sequences=",".join(FIXED_TRAIN), val_sequences=",".join(FIXED_VAL),
         dataset_root=DEFAULT_DATASET_ROOT, common_root=DEFAULT_COMMON_ROOT,
         expect=None, tol=1e-9):
    """`out` 폴더에 `constant_map_baseline.json`을 쓴다.

    `expect`를 주면 그 값과 다를 때 **거부한다** -- 학습과 무관한 값이라 달라지면 라벨이나
    split이 바뀐 것이고, 그건 조용히 넘어가면 안 되는 사건이다.
    """
    train = tuple(s for s in str(train_sequences).split(",") if s)
    val = tuple(s for s in str(val_sequences).split(",") if s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline, n_train, n_val = compute(train, val, dataset_root, common_root, device)

    if expect is not None and abs(baseline - float(expect)) > float(tol):
        raise SystemExit(
            f"기준선이 기대값과 다르다: {baseline!r} vs {expect!r}\n"
            f"  이 값은 학습과 무관하므로, 다르면 라벨이나 split이 바뀐 것이다.")

    out_path = Path(out) / "constant_map_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": "fixed_split",
        "train_sequences": list(train),
        "val_sequences": list(val),
        "n_train_frames": n_train,
        "n_val_frames": n_val,
        "constant_map_baseline_iou_free": baseline,
        "method": "report_loso.fold_baseline과 동일: train 시퀀스의 셀별 다수결 free map을 val에 채점",
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"constant-map 기준선 {baseline:.10f}  (train {n_train}프레임, val {n_val}프레임)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    Fire(main)
