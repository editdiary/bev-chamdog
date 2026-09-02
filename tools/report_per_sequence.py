"""런별·시퀀스별 지표를 한 표로 낸다 -- 계획서 §7(시퀀스 일반화)과 §11-1(시드별 표).

## 입력은 내보내 둔 확률맵이다

`{log_root}/analysis/predictions/`를 읽는다(`tools/export_val_predictions.py` → 검증은
`tools/verify_val_predictions.py`). 모델을 다시 돌리지 않으므로 CPU에서 몇 초면 끝나고,
같은 파일에서 언제 다시 돌려도 같은 숫자가 나온다.

`labels.npz`의 `sample_ids`가 `{시퀀스}/{프레임}` 꼴이라 **시퀀스 분해는 파일 이름에서
나온다** -- 별도 메타데이터가 필요 없다.

## 무엇을 내나

1. **시드별 표** -- 런마다 체크포인트 epoch과 주요 지표. 계획서 §11-1이 요구한 것이고,
   평균만 보면 안 보이는 개별 시드의 흩어짐을 그대로 보여준다.
2. **시퀀스별 표** -- 셀 x 시퀀스로 나눈 평균 ± 시드 간 표준편차, 그리고 worst-sequence.

## 반드시 같이 읽어야 하는 한계

**val 시퀀스가 둘뿐이다**(`raws1`, `rawos3`, 합쳐서 75프레임). 계획서 §7이 요구한
"시퀀스 간 SD"와 "best−worst gap"은 n=2에서 사실상 두 값의 차이일 뿐이라 일반화
지표로 읽으면 안 된다. **그 질문의 정본은 LOSO 7-fold**(진단 문서 §31)이고, 거기서
`σ_fold`가 `σ_seed`의 27배였다 -- 즉 시퀀스 변동이 시드 변동을 압도한다. 이 표는 그
사실을 이 실험 안에서 확인하는 용도이지 새 근거가 아니다.

실행:
    python tools/report_per_sequence.py --log_root=runs/loss_effect \\
        --cells=A_ce,B_perset,C_soft,D_range --seeds=0,1,2,3,4
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire

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

# (표시 이름, 높을수록 좋은가)
METRICS = (("iou_free", True), ("f1@10cm", True), ("fatal_rate", False),
           ("free_miss_rate", False), ("range_mae", False), ("range_bias", None),
           ("missed_obstacle", False))


def _csv(value):
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align=">") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def load_npz(path: Path) -> dict:
    """즉시 통째로 읽고 닫는다 -- 지연 로딩 `NpzFile`을 들고 다니지 않는다
    (이유는 `projects/common/npsafe.py`)."""
    with np.load(path, allow_pickle=False) as blob:
        return {key: np.array(blob[key]) for key in blob.files}


def score(pred_free, gt_free, gt_occ, valid, rays) -> dict:
    """학습 루프와 **같은 함수**로 한 묶음의 프레임을 채점한다."""
    occ_pred = derive_occupied(pred_free, valid, rays) & valid.bool()
    rng = summarize_range_error([range_error(pred_free, gt_free, valid, rays)])
    tol = summarize_tolerance_f1([tolerance_counts(occ_pred, gt_occ, valid, GRID_SPEC.cell_m)])
    return {"iou_free": iou_free(pred_free, gt_free, valid)[0],
            "fatal_rate": fatal_rate(pred_free, gt_free, valid)[0],
            "free_miss_rate": free_miss_rate(pred_free, gt_free, valid)[0],
            "f1@10cm": tol["10cm"]["f1"], "range_mae": rng["mae"],
            "range_bias": rng["bias"], "missed_obstacle": rng["missed_obstacle_rate"]}


def main(log_root="runs/loss_effect", cells="A_ce,B_perset,C_hard,C_soft,D_range",
         seeds="0,1,2,3,4", which="best", tau=0.5, pred_dir=None, json_out=None):
    cells, seeds = _csv(cells), [int(s) for s in _csv(seeds)]
    pred_dir = Path(pred_dir) if pred_dir else Path(log_root) / "analysis" / "predictions"
    lab = load_npz(pred_dir / "labels.npz")
    ids = [str(x) for x in lab["sample_ids"]]
    sequences = sorted({i.split("/")[0] for i in ids})
    seq_index = {s: np.array([j for j, i in enumerate(ids) if i.startswith(s + "/")])
                 for s in sequences}
    rays = build_ray_index(GRID_SPEC)

    def as_t(arr):
        return torch.from_numpy(arr).unsqueeze(1)

    gt_free_all, gt_occ_all = as_t(lab["free"]), as_t(lab["occupied"])
    valid_all = as_t(lab["valid"])

    print(f"val {len(ids)}프레임 | 시퀀스 {len(sequences)}개: "
          + ", ".join(f"{s}({len(seq_index[s])}프레임)" for s in sequences)
          + f" | 체크포인트 {which}, τ={tau}\n")

    rows = {}
    for cell in cells:
        for seed in seeds:
            path = pred_dir / f"{cell}_s{seed}__{which}.npz"
            if not path.exists():
                print(f"!! 없어서 건너뛴다: {path.name}")
                continue
            blob = load_npz(path)
            prob = torch.from_numpy(blob["prob_free"].astype(np.float32)).unsqueeze(1)
            pred_all = (prob > tau) & valid_all
            entry = {"checkpoint": str(blob["checkpoint"]),
                     "all": score(pred_all, gt_free_all, gt_occ_all, valid_all, rays)}
            for name, idx in seq_index.items():
                entry[name] = score(pred_all[idx], gt_free_all[idx], gt_occ_all[idx],
                                    valid_all[idx], rays)
            rows[(cell, seed)] = entry
            print(f"  {cell}_s{seed} ({entry['checkpoint']})", flush=True)

    # === 1. 시드별 표 (계획서 §11-1) ==================================================
    print(f"\n=== 시드별 결과 (전체 val, {which}) ===")
    print("**평균만 보면 안 보이는 흩어짐을 그대로 둔다.**\n")
    head = [("cell", 10), ("seed", 5), ("checkpoint ep", 14)] + \
           [(n, 12) for n, _ in METRICS]
    print("  " + " ".join(_pad(h, w) for h, w in head))
    for cell in cells:
        for seed in seeds:
            if (cell, seed) not in rows:
                continue
            v = rows[(cell, seed)]
            ep = v["checkpoint"].split("-")[-1].split(".")[0].lstrip("0") or "0"
            print("  " + " ".join(_pad(c, w) for c, (_, w) in zip(
                [cell, str(seed), ep] + [f"{v['all'][n]:.4f}" for n, _ in METRICS], head)))

    # === 2. 시퀀스별 표 (계획서 §7) ====================================================
    print(f"\n=== 시퀀스별 (평균 ± 시드 간 표준편차, n={len(seeds)}) ===")
    print("**val 시퀀스가 둘뿐이라 '시퀀스 간 SD'는 두 값의 차이일 뿐이다** -- 일반화")
    print("지표로 읽지 말 것. 그 질문의 정본은 LOSO 7-fold다(σ_fold가 σ_seed의 27배).\n")
    summary = {}
    for metric, higher_better in METRICS:
        print(f"  [{metric}]")
        head2 = [("cell", 10)] + [(s, 18) for s in sequences] + \
                [("전체", 18), ("worst-seq", 18)]
        print("  " + " ".join(_pad(h, w) for h, w in head2))
        for cell in cells:
            live = [rows[(cell, s)] for s in seeds if (cell, s) in rows]
            if not live:
                continue
            cols, per_seq_mean = [], {}
            for name in list(sequences) + ["all"]:
                vals = [v[name][metric] for v in live]
                m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
                per_seq_mean[name] = m
                cols.append(f"{m:.4f}±{sd:.4f}")
            if higher_better is None:
                worst = "-"
            else:
                pick = min if higher_better else max
                key = pick(sequences, key=lambda s: per_seq_mean[s])
                worst = f"{per_seq_mean[key]:.4f} ({key})"
            cols.append(worst)
            summary.setdefault(cell, {})[metric] = per_seq_mean
            print("  " + " ".join(_pad(c, w) for c, (_, w) in zip([cell] + cols, head2)))
        print()

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(
            {"which": which, "tau": tau, "sequences": sequences,
             "frames_per_sequence": {k: int(len(v)) for k, v in seq_index.items()},
             "per_run": {f"{c}_s{s}": rows[(c, s)] for c, s in rows},
             "per_cell_sequence_mean": summary}, indent=2, ensure_ascii=False))
        print(f"  원시 값 -> {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main) or 0)
