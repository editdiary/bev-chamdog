"""val split의 **예측 확률맵 원본**을 런마다 저장한다 -- 모든 예측 기반 분석의 재료다.

## 왜 필요한가

`report_seed_jitter.py`·`report_threshold_sweep.py`·`report_decision_disagreement.py`는
전부 같은 일을 먼저 한다: 체크포인트를 올리고 val을 한 번 forward해서 `p(free)`를 얻는다.
그러고 나서 각자 다른 방식으로 요약한다. 즉 **세 도구의 공통 재료가 `p(free)` 하나**이고,
표에 찍히는 숫자는 전부 그것의 함수다.

그 재료를 저장해 두면 세 가지가 가능해진다.

1. **나중에 다른 질문을 물을 수 있다.** "τ=0.65에서는?" "격자 앞쪽 절반만 보면?"
   "이 프레임에서만?" -- 체크포인트도 GPU도 없이 답한다.
2. **체크포인트를 지워도 분석이 남는다.** 체크포인트가 런당 471 MB인데 확률맵은
   **런당 약 2 MB**다(75프레임 × 120×120 float16). 235배 작다.
3. **재현이 정확해진다.** 같은 체크포인트라도 다른 torch/cuDNN에서 다시 forward하면
   마지막 자리가 달라진다. 저장된 맵을 쓰면 그 흔들림이 아예 없다.

## 무엇을 저장하나

런마다 `predictions/{run}__{which}.npz`:

- `prob_free` -- `(F, H, W)` float16. `softmax(logits)[:, 1]`, 즉 τ를 적용하기 **전** 값이다.
  **float16으로 두는 이유**: 확률의 유효 자릿수가 3자리면 충분하고(τ 격자가 0.01 단위),
  float32면 파일이 두 배다. 극단값에서 float16의 해상도는 1e-4보다 좋다.
- `sample_ids` -- `(F,)` 문자열. `{시퀀스}/{프레임}`이고 **행 순서를 고정한다.**
- `checkpoint` -- 실제로 읽은 체크포인트 파일명.

라벨은 런과 무관하므로 `predictions/labels.npz`에 **한 번만** 둔다:
`free`·`occupied`·`unknown`·`valid` (bool), `d` (float32, GT 경계까지 부호 있는 수직 거리),
`sample_ids`. 예측과 **같은 순서**다.

## `which` -- 어느 체크포인트인가

- `best` -- `model_best-*.pth`. 선택 규칙(val `iou_free` 최고점)이 고른 모델이고, "이 loss로
  학습하면 어떤 모델을 얻나"에 답한다. **max 연산이라 위로 편향돼 있다.**
- `last` -- 마지막 주기 저장(= 마지막 epoch). **사전 선언된 고정 epoch**이라 선택 편향이
  없다. 논문의 주 숫자는 이쪽이어야 한다(`summarize_repeats.py`의 규약과 같다).

둘 다 저장하는 것이 기본이다 -- 합쳐도 런당 4 MB다.

실행:

    python tools/export_val_predictions.py --log_root=runs/loss_effect \\
        --cells=A_ce,B_perset,C_soft,D_range --seeds=0,1,2,3,4
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space import decompose  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402


def _csv(value):
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def find_checkpoint(ckpt_dir: Path, which: str):
    """`best`는 `model_best-*.pth`, `last`는 주기 저장 중 **epoch이 가장 큰 것**이다.

    파일명에 epoch이 0으로 채워져 있어 문자열 정렬이 곧 epoch 정렬이지만, 자릿수가 바뀌면
    깨지므로 숫자로 뽑아 정렬한다.
    """
    prefix = "model_best-" if which == "best" else "model-"
    found = [p for p in ckpt_dir.glob(f"{prefix}*.pth")
             if which == "best" or not p.name.startswith("model_best-")]
    if not found:
        return None
    return max(found, key=lambda p: int(p.stem.split("-")[-1]))


def main(
    log_root="runs/loss_effect",
    cells="A_ce,B_perset,C_hard,C_soft,D_range",
    seeds="0,1,2,3,4",
    which="best,last",
    out_dir=None,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=8,
    num_workers=8,
    overwrite=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells, seeds, which = _csv(cells), [int(s) for s in _csv(seeds)], _csv(which)
    log_root = Path(log_root)
    out_dir = Path(out_dir) if out_dir else log_root / "analysis" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    _, val_samples = split_samples_by_sequence([root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    # **`shuffle=False`가 계약이다** -- 저장된 행 순서가 `sample_ids`와 짝지어야 한다.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    run_dirs = [log_root / "logs" / f"{c}_s{s}" for c in cells for s in seeds]
    ckpt_dirs = [log_root / "ckpt" / f"{c}_s{s}" for c in cells for s in seeds]
    convention, offset = convention_for_run_dirs([d for d in run_dirs if d.exists()])
    height = height_config_for_ckpt_dirs([d for d in ckpt_dirs if d.exists()])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                            pixel_convention=convention, pixel_offset=offset,
                                            height_bins=height["height_bins"],
                                            height_min_m=height["height_min_m"],
                                            height_max_m=height["height_max_m"])
    print(f"val {len(val_samples)}프레임 | 격자 {GRID_SPEC.n_rows}x{GRID_SPEC.n_cols} |"
          f" Y={height['height_bins']} | 표본 규약 {convention} offset {offset}")

    # === 라벨 (런과 무관하므로 한 번만) ==============================================
    labels_path = out_dir / "labels.npz"
    if overwrite or not labels_path.exists():
        # `decompose`는 `free`/`occupied`/`unknown` 셋만 준다 -- `valid`는 라벨 입력이므로
        # 따로 담는다. 셋의 합이 곧 `valid`이지만 명시해 두는 편이 나중에 안 헷갈린다.
        parts = {k: [] for k in ("free", "occupied", "unknown", "valid")}
        d_all, ids = [], []
        for batch in loader:
            seg, vis, valid = (batch[k].to(device)
                               for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g"))
            gt = dict(decompose(seg, vis, valid), valid=valid)
            for key in parts:
                parts[key].append(gt[key].bool().cpu().numpy()[:, 0])
            d_all.append(batch["d_bev_g"].numpy()[:, 0])
            ids.extend(batch["sample_id"])
        np.savez_compressed(labels_path,
                            **{k: np.concatenate(v) for k, v in parts.items()},
                            d=np.concatenate(d_all).astype(np.float32),
                            sample_ids=np.array(ids),
                            grid_cell_m=np.float32(GRID_SPEC.cell_m))
        print(f"  labels.npz 저장 ({labels_path.stat().st_size / 1e6:.1f} MB)")
    else:
        ids = list(np.load(labels_path, allow_pickle=False)["sample_ids"])
        print("  labels.npz 이미 있음 -- 건너뛴다")

    # === 런별 확률맵 =================================================================
    written, missing = [], []
    for cell in cells:
        for seed in seeds:
            run = f"{cell}_s{seed}"
            for kind in which:
                target = out_dir / f"{run}__{kind}.npz"
                if target.exists() and not overwrite:
                    print(f"  {target.name} 이미 있음 -- 건너뛴다")
                    written.append(target.name)
                    continue
                ckpt = find_checkpoint(log_root / "ckpt" / run, kind)
                if ckpt is None:
                    print(f"  !! 체크포인트 없음: {log_root}/ckpt/{run} ({kind})")
                    missing.append(f"{run}:{kind}")
                    continue
                model = ThreeClassSegnet(GRID_SPEC.n_rows, vox_util.Y, GRID_SPEC.n_cols,
                                         vox_util, use_radar=False, use_lidar=False,
                                         do_rgbcompress=True, encoder_type=encoder_type,
                                         rand_flip=False, num_classes=2).to(device)
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state.get("model_state_dict", state), strict=True)
                model.eval()
                chunks = []
                with torch.no_grad():
                    for batch in loader:
                        _, _, logits, _, _ = model(batch["rgb_camXs"].to(device) - 0.5,
                                                   batch["pix_T_cams"].to(device),
                                                   batch["cam0_T_camXs"].to(device), vox_util)
                        chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                del model
                torch.cuda.empty_cache()
                prob = np.concatenate(chunks).astype(np.float16)
                np.savez_compressed(target, prob_free=prob,
                                    sample_ids=np.array(ids),
                                    checkpoint=np.array(ckpt.name))
                written.append(target.name)
                print(f"  {target.name} <- {ckpt.name}  ({prob.shape},"
                      f" {target.stat().st_size / 1e6:.1f} MB)", flush=True)

    manifest = {
        "log_root": str(log_root), "cells": cells, "seeds": seeds, "which": which,
        "val_sequences": val_sequences, "train_sequences": train_sequences,
        "n_frames": len(val_samples), "grid": [GRID_SPEC.n_rows, GRID_SPEC.n_cols],
        "height_bins": height["height_bins"], "pixel_convention": convention,
        "pixel_offset": offset, "files": sorted(written), "missing": missing,
        "note": ("prob_free는 softmax(logits)[:,1]이고 τ 적용 전이다. 행 순서는 labels.npz의"
                 " sample_ids와 같다."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    total = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    print(f"\n{len(written)}개 파일, 합계 {total / 1e6:.1f} MB -> {out_dir}")
    if missing:
        print(f"  !! 빠진 것 {len(missing)}개: {', '.join(missing)}")


if __name__ == "__main__":
    Fire(main)
