"""[서버에서 실행] 보드가 돌려준 `results/`를 논문과 같은 지표 코드로 채점하고 속도와 한 표로 묶는다.

    python tools/jetson/score.py --results=<보드에서 받은 results 폴더> --out_dir=<출력 폴더>

지표 함수는 `tools/verify_val_predictions.py`와 같다(학습 로그와 대조된 경로). 라벨은 캠페인이 저장한
`labels.npz`이고, 행 순서(`sample_ids`)가 맞지 않으면 채점하지 않는다.

출력: `jetson_speed_accuracy.csv`(구성별 한 행) + 화면 표.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space_metrics import (  # noqa: E402
    free_precision, free_recall, iou_free, iou_non_free)
from projects.common.metric_spec import BF_TOLERANCES_M, bf_name, tolerance_key  # noqa: E402
from projects.common.occupied_metrics import (  # noqa: E402
    boundary_f_scores, derive_occupied, summarize_boundary_f)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.datasets.robot_simplebev import GRID_SPEC  # noqa: E402

CONFIG_ORDER = ("pytorch_fp32", "trt_fp16", "trt_int8")


def score(prob, lab, rays, tau=0.5):
    gt = torch.from_numpy(lab["free"]).unsqueeze(1)
    valid = torch.from_numpy(lab["valid"]).unsqueeze(1)
    occ = torch.from_numpy(lab["occupied"]).unsqueeze(1)
    pred = (torch.from_numpy(prob).unsqueeze(1) > tau) & valid
    row = {name: fn(pred, gt, valid)[0] for name, fn in (
        ("iou_free", iou_free), ("iou_non_free", iou_non_free),
        ("precision", free_precision), ("recall", free_recall))}
    occ_pred = derive_occupied(pred, valid, rays) & valid.bool()
    bf = summarize_boundary_f([boundary_f_scores(occ_pred, occ, valid, GRID_SPEC.cell_m)])
    row.update({bf_name(t): bf[tolerance_key(t)]["bf"] for t in BF_TOLERANCES_M})
    return {k: float(v) for k, v in row.items()}


def main(results, out_dir=None,
         labels="runs/99_full_campaign/01_overall/fixed_split/analysis/predictions/labels.npz"):
    results = Path(results)
    out_dir = Path(out_dir) if out_dir else results
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = json.loads((results / "bench.json").read_text())
    with np.load(labels, allow_pickle=False) as blob:
        lab = {k: np.array(blob[k]) for k in blob.files}
    rays = build_ray_index(GRID_SPEC)

    rows = []
    names = [n for n in CONFIG_ORDER if (results / f"{n}_prob_free.npz").exists()]
    for name in names:
        with np.load(results / f"{name}_prob_free.npz", allow_pickle=False) as blob:
            prob, ids = np.array(blob["prob_free"]), list(blob["sample_ids"])
        if ids != list(lab["sample_ids"]):
            raise SystemExit(f"!! {name}: 프레임 순서가 라벨과 다르다")
        speed = bench["speed"].get(name, {})
        acc = bench["accuracy_vs_server"].get(name, {})
        row = {"config": name,
               "end_to_end_ms": speed.get("end_to_end", {}).get("median_of_round_medians"),
               "fps": speed.get("fps"),
               "encoder_ms": speed.get("encoder", {}).get("median_of_round_medians"),
               "bev_stage_ms": speed.get("bev_stage", {}).get("median_of_round_medians"),
               "encoder_share": speed.get("encoder_share_of_end_to_end"),
               "engine_device_memory_mb": speed.get("engine_device_memory_mb"),
               **score(prob, lab, rays),
               "decision_flip_rate_vs_server": acc.get("decision_flip_rate_vs_server"),
               "max_abs_prob_diff_vs_server": acc.get("max_abs_prob_diff_vs_server")}
        rows.append(row)

    base = rows[0] if rows and rows[0]["config"] == "pytorch_fp32" else None
    for row in rows:
        for key in ("iou_free", "precision", "recall", "bf_10cm", "bf_20cm", "bf_30cm"):
            row[f"d_{key}"] = (row[key] - base[key]) if base else None
        row["speedup_vs_pytorch_fp32"] = (base["end_to_end_ms"] / row["end_to_end_ms"]
                                          if base and row["end_to_end_ms"] else None)
    with (out_dir / "jetson_speed_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    env = bench.get("env", {})
    print(f"보드 {env.get('device')} | torch {env.get('torch')} | TensorRT {env.get('tensorrt')}")
    print(f"전력 모드: {str(env.get('nvpmodel', '')).splitlines()[:1]}")
    print("\n구성           전체ms   FPS   가속  인코더ms BEVms   IoU_free  ΔIoU     BF@0.10  ΔBF10   뒤집힘")
    for r in rows:
        print(f"{r['config']:13s} {r['end_to_end_ms']:7.2f} {r['fps']:6.1f} {r['speedup_vs_pytorch_fp32']:5.2f}x"
              f" {r['encoder_ms']:8.2f} {r['bev_stage_ms']:5.2f}   {r['iou_free']:.4f} {r['d_iou_free']:+.4f}"
              f"   {r['bf_10cm']:.4f} {r['d_bf_10cm']:+.4f}  {100 * r['decision_flip_rate_vs_server']:.3f} %")
    print(f"\n-> {out_dir / 'jetson_speed_accuracy.csv'}")


if __name__ == "__main__":
    Fire(main)
