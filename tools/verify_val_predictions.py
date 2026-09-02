"""내보낸 확률맵이 **학습 로그와 같은 숫자를 내는지** 확인한다 -- 원시 데이터의 무결성 검사.

## 왜 필요한가

`export_val_predictions.py`가 떨어뜨린 `predictions/*.npz`는 이후 모든 예측 기반 분석의
재료이고, 체크포인트를 지운 뒤에는 **유일한** 재료다. 그런데 이 파일이 조용히 틀릴 수 있는
경로가 셋 있다.

1. **행 순서가 어긋난다** -- `DataLoader(shuffle=True)`거나 라벨과 예측을 다른 순서로 모으면
   프레임이 섞인다. 지표는 여전히 그럴듯한 값이 나온다.
2. **다른 기하로 forward했다** -- `height_bins`나 표본 규약이 런과 다르면 모델은 멀쩡히
   돌지만 다른 모델을 재는 셈이 된다.
3. **체크포인트를 잘못 골랐다** -- `best` 대신 마지막 것을 읽는 식.

셋 다 **학습 루프가 그 epoch에 기록해 둔 val 지표와 대조하면 즉시 드러난다.** 학습 루프는
전혀 다른 코드 경로(`evaluate_split`)로 같은 프레임을 채점했으므로, 두 값이 맞으면
행 순서·기하·체크포인트가 전부 맞은 것이다.

## 허용 오차

`prob_free`를 float16으로 저장하므로 완전히 같을 수는 없다. 실측 차이는 `iou_free`에서
약 2e-5다. 기본 허용 오차 `1e-3`은 그보다 50배 크고, 위 셋 중 무엇이 틀려도 그보다는
훨씬 크게 어긋난다(프레임을 섞으면 0.01~0.1 규모로 벌어진다).

`last`(마지막 epoch) 체크포인트도 같은 방식으로 확인한다 -- 그 epoch의 로그 값과 맞춘다.

실행:

    python tools/verify_val_predictions.py --root=runs/loss_effect
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.free_space_metrics import (  # noqa: E402
    fatal_rate,
    free_miss_rate,
    iou_free,
    range_error,
    summarize_range_error,
)
from projects.common.occupied_metrics import (  # noqa: E402
    derive_occupied,
    summarize_tolerance_f1,
    tolerance_counts,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.datasets.robot_simplebev import GRID_SPEC  # noqa: E402

# **torch로만 계산되는 지표 셋.** numpy 임시 소거 결함(`projects/common/npsafe.py`)의
# 영향을 받지 않는 경로라 기준선으로 쓴다.
TORCH_CHECKS = (
    (iou_free, "val/iou_free_epoch", "iou_free"),
    (fatal_rate, "val/fatal_rate_epoch", "fatal_rate"),
    (free_miss_rate, "val/free_miss_rate_epoch", "free_miss_rate"),
)

# **numpy/scipy로 내려가는 지표들.** 광선 추적(`polar`)과 거리 변환(`occupied_metrics`)을
# 지나므로 위 결함의 사정권이다. 학습 로그(학습 시점 계산)와 여기(수정된 코드로 재계산)가
# 맞으면 학습 시점 값이 오염되지 않았다는 뜻이다 -- 서로 다른 프로세스·다른 시점이라
# 같은 방식으로 동시에 틀릴 이유가 없다.
NUMPY_CHECKS = (
    ("range_mae", "val/range_mae_epoch"),
    ("range_bias", "val/range_bias_epoch"),
    ("missed_obstacle_rate", "val/range_missed_obstacle_rate_epoch"),
    ("f1@10cm", "val/occupied_f1_10cm_epoch"),
    ("f1@20cm", "val/occupied_f1_20cm_epoch"),
)


def epoch_of(checkpoint_name: str) -> int:
    """`model_best-000000021.pth` -> 21."""
    return int(str(checkpoint_name).split("-")[-1].split(".")[0])


def numpy_metrics(pred_free, gt_parts, valid, rays, cell_m) -> dict:
    """학습 루프와 **같은 함수**로 numpy 계열 지표를 다시 낸다.

    프레임을 한 번에 넘긴다 -- `summarize_*`가 표본을 모아 통계를 내므로 배치 크기에
    불변이고(설계상), 그래서 학습 때의 배치 8과 여기 75가 같은 값을 내야 한다.
    """
    occ_pred = derive_occupied(pred_free, valid, rays) & valid.bool()
    rng = summarize_range_error([range_error(pred_free, gt_parts["free"], valid, rays)])
    tol_f1 = summarize_tolerance_f1(
        [tolerance_counts(occ_pred, gt_parts["occupied"], valid, cell_m)])
    return {"range_mae": rng["mae"], "range_bias": rng["bias"],
            "missed_obstacle_rate": rng["missed_obstacle_rate"],
            "f1@10cm": tol_f1["10cm"]["f1"], "f1@20cm": tol_f1["20cm"]["f1"]}


def main(root="runs/loss_effect", tau=0.5, tol=1e-3, numpy_tol=2e-3, pred_dir=None,
         check_numpy=True):
    root = Path(root)
    pred_dir = Path(pred_dir) if pred_dir else root / "analysis" / "predictions"
    labels_path = pred_dir / "labels.npz"
    if not labels_path.exists():
        print(f"!! 라벨이 없다: {labels_path}")
        return 1

    # **`NpzFile`을 들고 다니지 않는다** -- 지연 로딩 상태로 다른 npz를 여러 개 열면
    # 첫 읽기가 다른 데이터를 돌려주는 것을 겪었다(`projects/common/npsafe.py` 참고).
    with np.load(labels_path, allow_pickle=False) as blob:
        lab = {k: np.array(blob[k]) for k in blob.files}
    gt = torch.from_numpy(lab["free"]).unsqueeze(1)
    valid = torch.from_numpy(lab["valid"]).unsqueeze(1)
    gt_parts = {"free": gt, "occupied": torch.from_numpy(lab["occupied"]).unsqueeze(1)}
    rays = build_ray_index(GRID_SPEC)
    label_ids = list(lab["sample_ids"])
    print(f"라벨 {len(label_ids)}프레임, τ={tau}, 허용 오차 {tol}")

    files = sorted(p for p in pred_dir.glob("*.npz") if p.name != "labels.npz")
    if not files:
        print(f"!! 확률맵이 없다: {pred_dir}")
        return 1

    failures, rows = [], []
    for path in files:
        run, _, kind = path.stem.partition("__")
        with np.load(path, allow_pickle=False) as raw:
            blob = {k: np.array(raw[k]) for k in raw.files}
        # **행 순서부터 본다** -- 지표가 맞아도 순서가 다르면 다른 프레임을 재고 있는 것이다.
        if list(blob["sample_ids"]) != label_ids:
            failures.append(f"{path.name}: sample_ids가 labels.npz와 다르다")
            print(f"  {path.name:28s} !! sample_ids 불일치")
            continue
        ckpt = str(blob["checkpoint"])
        epoch = epoch_of(ckpt)
        acc = EventAccumulator(str(root / "logs" / run), size_guidance={"scalars": 0})
        acc.Reload()
        prob = torch.from_numpy(blob["prob_free"].astype(np.float32)).unsqueeze(1)
        pred = (prob > tau) & valid

        line, bad = [], False
        for fn, tag, name in TORCH_CHECKS:
            got = fn(pred, gt, valid)[0]
            series = ({e.step: e.value for e in acc.Scalars(tag)}
                      if tag in acc.Tags()["scalars"] else {})
            want = series.get(epoch, float("nan"))
            delta = abs(got - want)
            if not (delta <= tol):
                bad = True
                failures.append(f"{path.name}: {name} npz {got:.6f} != 로그 {want:.6f}"
                                f" (Δ {delta:.2e})")
            line.append(f"{name} Δ{delta:.1e}")
            rows.append({"file": path.name, "run": run, "which": kind, "epoch": epoch,
                         "metric": name, "from_npz": float(got), "from_log": float(want),
                         "abs_diff": float(delta), "ok": bool(delta <= tol)})
        if check_numpy:
            got_np = numpy_metrics(pred, gt_parts, valid, rays, GRID_SPEC.cell_m)
            for name, tag in NUMPY_CHECKS:
                series = ({e.step: e.value for e in acc.Scalars(tag)}
                          if tag in acc.Tags()["scalars"] else {})
                want = series.get(epoch, float("nan"))
                delta = abs(got_np[name] - want)
                if not (delta <= numpy_tol):
                    bad = True
                    failures.append(f"{path.name}: {name} npz {got_np[name]:.6f}"
                                    f" != 로그 {want:.6f} (Δ {delta:.2e})")
                line.append(f"{name} Δ{delta:.1e}")
                rows.append({"file": path.name, "run": run, "which": kind, "epoch": epoch,
                             "metric": name, "from_npz": float(got_np[name]),
                             "from_log": float(want), "abs_diff": float(delta),
                             "ok": bool(delta <= numpy_tol), "path": "numpy"})
        print(f"  {path.name:28s} ep{epoch:3d}  " + "  ".join(line)
              + ("   !! 불일치" if bad else ""))

    out = pred_dir.parent / "verify_predictions.json"
    out.write_text(json.dumps({"tau": tau, "tol": tol, "numpy_tol": numpy_tol,
                               "check_numpy": check_numpy, "n_files": len(files),
                               "failures": failures, "rows": rows},
                              indent=2, ensure_ascii=False))
    print(f"\n파일 {len(files)}개 검사 -> {out}")
    if failures:
        # **여기서 멈추라는 뜻이다.** 아래 분석 전부가 이 파일들 위에 서 있다.
        print(f"!! 불일치 {len(failures)}건 -- 분석 결과를 믿으면 안 된다")
        for f in failures:
            print(f"   {f}")
        return 1
    print("전부 일치 -- 행 순서·기하·체크포인트 선택이 모두 맞다")
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main) or 0)
