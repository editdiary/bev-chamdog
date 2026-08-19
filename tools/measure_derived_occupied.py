"""GT `occupied`가 free 경계에서 복원되는가 -- (D) binary 정식화의 전제 실측.

**왜 이 측정이 먼저인가.** `docs/finetune_overfitting_diagnosis.md` §13.3의 (D)는 `occupied`를
예측 클래스에서 빼는 안이고, 그 비용으로 "왜 못 가는가(장애물인가 미관측인가)를 구별할 수
없어진다"가 적혀 있다. 그런데 라벨 정의(`free_space.decompose`)에서
`occupied = ~occ & vis`이고 `vis`는 ego 원점 2D raycast이므로, **보이는 장애물 셀은 정의상
어떤 광선의 첫 hit**이다 -- 즉 GT `occupied`는 free 영역의 ego 기준 경계이고 `free`만
있으면 복원된다. 그 비용이 실제로 지불되는지 여부가 여기서 결정된다.

측정하는 것은 **상한**이다: GT free를 입력으로 주므로 free 예측 오차가 섞이지 않는다.
이 값이 나쁘면 (D)는 정보를 잃는다. 좋으면 (D)는 정보 손실 없는 단순화이고, 유도 경로
(`polar.frontier_cells`)를 그대로 학습 루프의 `iou_occupied`/`f1@τ` 계산에 쓸 수 있다.

지표는 학습 루프와 **같은 함수**로 계산한다(`iou_masked`, `tolerance_counts`). 그래야
3-class head의 실측값과 직접 비교된다. 비교 대상(§11.2, scratch+aug best):
`iou_occupied` 0.1041 / `f1@10cm` 0.6182 / `f1@20cm` 0.8227.

실행: python tools/measure_derived_occupied.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import iou_masked  # noqa: E402
from projects.common.occupied_metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    summarize_tolerance_f1,
    tolerance_counts,
    tolerance_key,
)
from projects.common.polar import (  # noqa: E402
    build_ray_index,
    first_free_range,
    frontier_cells,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
)

ALL_SEQUENCES = "raws1,raws2,raws3,rawos1,rawos2,rawos3,rawos4"


def _as_batch(mask: np.ndarray) -> torch.Tensor:
    """(H, W) numpy bool -> (1, 1, H, W) torch bool. 지표 함수의 계약이 텐서다."""
    return torch.from_numpy(mask)[None, None]


def _score(pred_masks, gt_masks, valid_masks, cell_m) -> dict:
    """학습 루프와 같은 함수로 `iou_occupied`와 `f1@τ`를 낸다."""
    ious, counts = [], []
    for pred, gt, valid in zip(pred_masks, gt_masks, valid_masks):
        pred_t, gt_t, valid_t = _as_batch(pred), _as_batch(gt), _as_batch(valid)
        iou, n = iou_masked(pred_t, gt_t, valid_t)
        if n:
            ious.append(iou)
        counts.append(tolerance_counts(pred_t, gt_t, valid_t, cell_m))
    return {"iou": float(np.mean(ious)) if ious else float("nan"),
            "f1": summarize_tolerance_f1(counts)}


def _print_scores(label, scores):
    print(f"  {label:22s} iou_occupied {scores['iou']:.4f}", end="")
    for tolerance in DEFAULT_TOLERANCES_M:
        key = tolerance_key(tolerance)
        stat = scores["f1"][key]
        print(f"  | f1@{key} {stat['f1']:.4f} (P {stat['precision']:.3f}"
              f" R {stat['recall']:.3f})", end="")
    print()


def main(sequences=ALL_SEQUENCES, dataset_root=DEFAULT_DATASET_ROOT, n_thetas=(360, 720)):
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    mask_out = permanent_blind | invalid

    frames = []
    for name in str(sequences).split(","):
        root = Path(dataset_root) / name.strip()
        if not (root / "occupancy_npy").exists():
            print(f"skip {name.strip()} (no occupancy_npy)")
            continue
        for sequence_root, sample_id in list_sequence_samples(root):
            occ, vis, valid = load_masked_labels(
                sequence_root, sample_id, permanent_blind, invalid
            )
            frames.append((decompose(occ, vis, valid), valid))

    cells = sum(int(valid.sum()) for _, valid in frames)
    print(f"frames = {len(frames)}   valid cells = {cells}")
    print("GT 셀 비율 (valid 안):", "  ".join(
        f"{name} {sum(int(parts[name].sum()) for parts, _ in frames) / cells:.4f}"
        for name in ("free", "occupied", "unknown")
    ))

    for n_theta in tuple(n_thetas):
        rays = build_ray_index(GRID_SPEC, n_theta=n_theta)
        derived, kept = [], []
        gt_occupied = [parts["occupied"] for parts, _ in frames]
        valids = [valid for _, valid in frames]
        overlap = dict.fromkeys(("occupied", "unknown", "free", "invalid"), 0)
        for parts, valid in frames:
            free = parts["free"]
            frontier = frontier_cells(*first_free_range(free, rays), rays, free.shape)
            derived.append(frontier)
            # 유도된 표면이 GT에서 무엇으로 떨어지나. `valid=0`은 세 클래스 어디에도
            # 속하지 않으므로(수집 아티팩트) 따로 센다.
            for name in ("occupied", "unknown", "free"):
                overlap[name] += int((frontier & parts[name]).sum())
            overlap["invalid"] += int((frontier & ~valid).sum())
            # 정적 마스크가 설명하는 경계를 뺀 변형 -- 배포 시에도 알고 있는 영역이므로
            # 유도 결과에서 제외할 수 있다(`permanent_blind`는 발밑 원반, `invalid`는 카트).
            kept.append(frontier & ~mask_out)

        n_derived = sum(int(m.sum()) for m in derived)
        n_gt = sum(int(m.sum()) for m in gt_occupied)
        recovered = sum(int((d & g).sum()) for d, g in zip(derived, gt_occupied))
        print(f"\nn_theta = {n_theta}")
        print(f"  유도 표면 셀 {n_derived}  vs  GT occupied 셀 {n_gt}"
              f"   (셀 단위 recall {recovered / n_gt:.4f})")
        print("  유도 표면의 GT 정체: " + "  ".join(
            f"{name} {overlap[name] / n_derived:.4f}" for name in overlap
        ))
        _print_scores("derived", _score(derived, gt_occupied, valids, GRID_SPEC.cell_m))
        _print_scores("derived - static", _score(kept, gt_occupied, valids, GRID_SPEC.cell_m))


if __name__ == "__main__":
    Fire(main)
