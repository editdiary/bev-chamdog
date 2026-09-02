"""같은 셀의 `free`/`non-free` 판정이 **시드마다 뒤집히나** -- 결정 재현성.

## 왜 이 도구가 따로 필요한가

기존 안정성 도구 둘은 **광선**을 본다. `report_seed_jitter.py`는 광선별 자유거리 `R̂`의
시드 간 표준편차를 재고, 거기 딸린 `상태 불일치`는 "한 시드는 장애물을 봤는데 다른 시드는
격자 끝까지 free로 봤다"는 **광선 단위** 사건이다.

그런데 planner가 실제로 읽는 것은 **셀**이다. 자유거리가 2 cm밖에 안 흔들려도 그 2 cm가
셀 경계를 걸치면 그 셀의 상태는 뒤집힌다. 반대로 자유거리가 크게 흔들려도 셀 안에서
움직이면 아무것도 안 뒤집힌다. 즉 셀 단위 뒤집힘은 광선 σ에서 유도되지 않는 별개의 양이다.

## 입력은 **내보내 둔 확률맵**이다 (모델을 다시 돌리지 않는다)

`tools/export_val_predictions.py`가 저장한 `{log_root}/analysis/predictions/`를 읽는다.
그 파일들은 `tools/verify_val_predictions.py`가 학습 로그와 대조해 검증한 것이다.

**처음에는 이 도구가 직접 forward했는데 그 경로에서 마스크가 깨졌다**(2026-09-01):
`valid` 셀 수가 집계 도중 0으로 바뀌어 표의 절반이 `nan`이 되고 일부 칸의 값이 다른 칸의
값으로 섞여 나왔다. 저장된 확률맵으로 같은 식을 다시 계산하면 값이 멀쩡하다. 원인을 더
쫓는 대신 **입력을 검증된 파일 하나로 통일**했다 -- 그러면 이 도구는 순수 CPU 계산이 되고,
같은 파일에서 언제 다시 돌려도 같은 숫자가 나온다.

## 정의

시드 `s`의 이진 예측을 `ŷ_i^(s)(τ) = 1[p_i^(s) >= τ]`라 하면, 셀 `i`의 불일치는 시드
**쌍**에 대한 평균이다.

    D_i(τ) = mean_{s<t} 1[ŷ_i^(s) != ŷ_i^(t)]

시드 `S`개 중 `k`개가 free라고 했다면 어긋나는 쌍이 정확히 `k(S−k)`개이므로

    D_i = k(S−k) / C(S, 2)

이고, 쌍을 실제로 돌 필요가 없다(테스트가 두 계산이 같음을 고정한다). 이것을 `valid` 셀
전체에서 평균한 것이 `D(τ)`다.

**`valid` 셀만 센다** -- 지표가 채점되는 집합과 같아야 이 숫자를 `fatal_rate`·`iou_free`와
나란히 읽을 수 있다.

## 반드시 나눠 봐야 하는 것 둘

**(1) 경계 근방을 따로 본다.** 격자의 87.5 %는 경계에서 멀고 거기서는 모든 시드가 자신
있게 같은 답을 낸다. 섞어서 평균하면 쉬운 셀이 분모를 채워 어떤 개입이든 `D`가 작아 보인다.
그래서 `|d| <= band`(GT 경계까지의 부호 있는 수직 거리, 기본 0.15 m)를 따로 찍는다.

**(2) 동작점을 맞춘 뒤 다시 잰다 -- 이것이 판정 표다.** loss를 바꾸면 확률의 눈금 자체가
옮겨 간다. 그러면 고정 τ=0.5에서 free라고 부르는 셀의 개수부터 달라지고, free 예측이
적어지면 뒤집힐 후보 자체가 줄어 `D`가 기계적으로 작아진다. 설계 문서 §16.2가 `fatal`에서
바로 이 함정에 빠졌고("안전 개선"이 실은 동작점 이동이었다), 같은 함정이 여기에도 있다.

그래서 시드마다 **자기 곡선에서** 목표 `free_miss`를 만족하는 `τ_s`를 역보간해 각자 그
문턱으로 자른 뒤 다시 잰다.

- 고정 τ에서만 좋아지고 맞춘 뒤에는 같다 → **개선의 정체는 보정(calibration) 이동이다**
- 맞춘 뒤에도 낮다 → **결정 재현성이 실제로 좋아졌다**

## 품질과 같이 읽는다

설계 문서 §15.6: 안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이다.
전부 non-free로 찍는 모델은 `D = 0`이다. 그래서 `iou_free`와 `free_miss`를 같은 표에 둔다.

실행:

    python tools/export_val_predictions.py --log_root=runs/loss_effect   # 먼저
    python tools/report_decision_disagreement.py \\
        --log_root=runs/loss_effect --cells=A_ce,D_range --seeds=0,1,2,3,4
"""
import json
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.npsafe import bool_not  # noqa: E402

# 경계 근방의 폭 [m]. **loss의 `δ`와 같을 필요는 없다** -- 여기서 묻는 것은 "감독이 soft인
# 자리인가"가 아니라 "판정이 실제로 어려운 자리인가"이고, 후자는 config가 아니라 기하가
# 정한다. 0.15 m = 3셀이고 주 지표 `f1@10cm`의 허용 오차(2셀)보다 한 셀 넓다.
DEFAULT_BAND_M = 0.15

# 동작점을 맞출 때 τ를 역보간할 격자. 넓게 두는 이유: 칸마다 곡선이 놓인 자리가 다르므로
# 좁으면 어떤 칸이 목표 `free_miss`를 못 덮는다. 확률맵에서 재는 것이라 점을 늘려도 싸다.
DEFAULT_TAUS = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def _csv(value):
    """Fire는 `--cells=a,b`를 **tuple로 파싱한다** -- 스칼라만 가정하면 조용히 깨진다."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align=">") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def pairwise_disagreement(binary, weights=None) -> float:
    """`binary`: `(S, N)` 0/1. 시드 쌍에 대한 평균 불일치율을 셀에 대해 평균한다.

    쌍을 돌지 않고 `k(S−k)/C(S,2)`로 센다 -- `S`개 중 `k`개가 1이면 어긋나는 쌍이 정확히
    `k(S−k)`개다. `S=2`면 이것은 그냥 `1[다름]`이 된다.

    `weights`가 있으면 셀 가중평균이다(부분집합 평균을 낼 때 마스크를 그대로 넘긴다).
    """
    n_seeds = binary.shape[0]
    if n_seeds < 2:
        return float("nan")
    k = binary.sum(axis=0).astype(np.float64)
    per_cell = k * (n_seeds - k) / (n_seeds * (n_seeds - 1) / 2.0)
    if weights is None:
        return float(per_cell.mean()) if per_cell.size else float("nan")
    total = float(weights.sum())
    return float((per_cell * weights).sum() / total) if total > 0 else float("nan")


def iou_free_flat(pred, gt, valid, n_frames) -> float:
    """프레임별 IoU의 평균. **학습 루프와 같은 정의다**(`free_space_metrics.iou_masked`) --
    전체를 한 번에 세는 pooled IoU와는 다른 값이므로 정의를 맞춰야 표가 나란히 읽힌다.

    union이 0인 프레임은 평균에서 뺀다(그 클래스가 없는 장면을 0점으로 세지 않는다).
    """
    p = (pred & valid).reshape(n_frames, -1)
    g = (gt & valid).reshape(n_frames, -1)
    inter = (p & g).sum(axis=1).astype(np.float64)
    union = (p | g).sum(axis=1).astype(np.float64)
    keep = union > 0
    return float((inter[keep] / union[keep]).mean()) if keep.any() else float("nan")


def load_npz(path: Path) -> dict:
    """`.npz`를 **즉시 통째로 읽어 dict로** 돌려주고 파일을 닫는다.

    **`np.load`가 돌려주는 `NpzFile`을 들고 다니면 안 된다** (2026-09-01에 하루를 잡아먹은
    자리다). 그것은 지연 로딩 객체라 `lab["d"]`를 쓸 때마다 zip에서 다시 풀고, 그 사이에
    다른 `.npz`를 여러 개 열어 두면 **첫 읽기가 다른 데이터를 돌려주는 일이 실제로 있었다.**
    증상은 조용하다: 경계 근방 마스크가 131,751개가 아니라 948,249개로 나오고 표의 절반이
    엉뚱한 값이 되는데 예외는 하나도 안 난다. 같은 식을 나중에 다시 계산하면 맞는 값이 나와서
    코드를 아무리 봐도 원인이 안 보인다.

    그래서 여는 즉시 `copy()`로 재료를 떼어 오고 파일을 닫는다. 라벨은 3 MB라 비용이 없다.
    """
    with np.load(path, allow_pickle=False) as blob:
        return {key: np.array(blob[key]) for key in blob.files}


def load_predictions(pred_dir: Path, cells, seeds, which):
    """`{cell}_s{seed}__{which}.npz`를 평평한 float32 배열로 읽는다.

    **행 순서가 라벨과 같은지 확인한다** -- 어긋나면 다른 프레임을 비교하게 되는데 숫자는
    여전히 그럴듯하다. `verify_val_predictions.py`가 이미 보는 것이지만 여기서도 막는다.
    """
    labels_path = pred_dir / "labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"확률맵이 없다: {labels_path}. 먼저 `tools/export_val_predictions.py`를 돌린다.")
    lab = load_npz(labels_path)
    label_ids = list(lab["sample_ids"])
    probs = {}
    for cell in cells:
        for seed in seeds:
            path = pred_dir / f"{cell}_s{seed}__{which}.npz"
            if not path.exists():
                print(f"!! 없어서 건너뛴다: {path.name}")
                continue
            blob = load_npz(path)
            if list(blob["sample_ids"]) != label_ids:
                raise ValueError(f"{path.name}의 프레임 순서가 labels.npz와 다르다")
            probs[(cell, seed)] = blob["prob_free"].reshape(-1).astype(np.float32)
            print(f"  {path.name} <- {blob['checkpoint']}")
    return lab, probs


def main(
    log_root="runs/loss_effect",
    cells="A_ce,B_perset,C_hard,C_soft,D_range",
    seeds="0,1,2,3,4",
    # 어느 체크포인트의 확률맵인가. `best`는 선택 규칙이 고른 것, `last`는 사전 선언된
    # 고정 epoch이라 선택 편향이 없다.
    which="best",
    # 고정 문턱. τ=0.5는 `argmax`와 같다.
    tau=0.5,
    # 동작점을 맞출 목표 `free_miss`. `None`이면 모든 런이 덮는 구간의 중앙을 쓴다.
    target_free_miss=None,
    band_m=DEFAULT_BAND_M,
    taus=DEFAULT_TAUS,
    pred_dir=None,
    json_out=None,
):
    cells, seeds, taus = _csv(cells), [int(s) for s in _csv(seeds)], tuple(taus)
    pred_dir = Path(pred_dir) if pred_dir else Path(log_root) / "analysis" / "predictions"

    lab, probs = load_predictions(pred_dir, cells, seeds, which)
    valid = lab["valid"].reshape(-1)
    gt_free = lab["free"].reshape(-1) & valid
    # **`~`를 쓰지 않는다.** `far = valid & ~near`로 쓰면 numpy 임시 소거가 `near`를
    # 제자리에서 뒤집는다 -- 이 도구에서 실제로 그래서 표의 절반이 틀린 값으로 나왔다
    # (재현과 원인은 `projects/common/npsafe.py`). 거리에서 두 마스크를 각각 만든다.
    abs_d = np.abs(lab["d"].reshape(-1))
    near = (abs_d <= float(band_m)) & valid
    far = (abs_d > float(band_m)) & valid
    n_frames = lab["valid"].shape[0]
    gt_shaped = lab["free"], lab["valid"]
    n_gt_free = max(int(gt_free.sum()), 1)

    print(f"\nval {n_frames}프레임 | valid 셀 {int(valid.sum()):,}개 |"
          f" 경계 근방(|d| <= {band_m:.2f} m) {int(near.sum()):,}개"
          f" ({100 * near.sum() / max(valid.sum(), 1):.1f} %) | 체크포인트 {which}")

    live = [c for c in cells if any((c, s) in probs for s in seeds)]
    if not live:
        print("!! 읽은 확률맵이 없다")
        return 1

    # === 시드마다 free_miss(τ) 곡선을 만들고 목표 동작점의 τ_s를 역보간 =================
    #
    # `free_miss = |~pred & gt_free| / |gt_free|`이고 τ에 대해 단조 증가한다(문턱을 올리면
    # free 예측이 줄어 놓치는 GT free가 는다). 격자 위에서 뒤집히면 `np.interp`가 조용히
    # 틀리므로 검사한다.
    curves = {}
    for key, flat in probs.items():
        curves[key] = np.array([float((bool_not(flat >= t) & gt_free).sum()) / n_gt_free
                                for t in taus])
        if np.any(np.diff(curves[key]) < 0):
            print(f"!! {key[0]}_s{key[1]}: free_miss가 τ에 대해 단조가 아니다")

    lo = max(float(c.min()) for c in curves.values())
    hi = min(float(c.max()) for c in curves.values())
    target = (0.5 * (lo + hi) if lo < hi else None) if target_free_miss is None \
        else float(target_free_miss)
    if target is not None and not lo <= target <= hi:
        print(f"!! 목표 free_miss {target:.4f}가 공통 구간 [{lo:.4f}, {hi:.4f}] 밖이다")
        target = None
    matched_tau = ({k: float(np.interp(target, c, taus)) for k, c in curves.items()}
                   if target is not None else {})

    # === 집계 =========================================================================
    def stack(cell, thresholds):
        return np.stack([(probs[(cell, s)] >= thresholds[(cell, s)]).astype(np.int8)
                         for s in seeds if (cell, s) in probs])

    fixed = dict.fromkeys(probs, float(tau))
    results, per_seed = {}, {}
    for cell in live:
        binary = stack(cell, fixed)
        entry = {"n_seeds": int(binary.shape[0]),
                 "all": pairwise_disagreement(binary, valid.astype(np.float64)),
                 "near": pairwise_disagreement(binary, near.astype(np.float64)),
                 "far": pairwise_disagreement(binary, far.astype(np.float64))}
        ious, misses = [], []
        for s in seeds:
            if (cell, s) not in probs:
                continue
            pred = (probs[(cell, s)] >= tau) & valid
            iou = iou_free_flat(pred.reshape(gt_shaped[0].shape), *gt_shaped, n_frames)
            miss = float((bool_not(pred) & gt_free).sum()) / n_gt_free
            ious.append(iou)
            misses.append(miss)
            per_seed[f"{cell}_s{s}"] = {"iou_free": iou, "free_miss": miss,
                                        "matched_tau": matched_tau.get((cell, s))}
        entry["iou_free"] = float(np.mean(ious))
        entry["iou_free_sd"] = float(np.std(ious, ddof=1)) if len(ious) > 1 else float("nan")
        entry["free_miss"] = float(np.mean(misses))
        if target is not None:
            binary_m = stack(cell, matched_tau)
            entry["all_m"] = pairwise_disagreement(binary_m, valid.astype(np.float64))
            entry["near_m"] = pairwise_disagreement(binary_m, near.astype(np.float64))
            entry["far_m"] = pairwise_disagreement(binary_m, far.astype(np.float64))
            used = [matched_tau[(cell, s)] for s in seeds if (cell, s) in probs]
            entry["tau_mean"] = float(np.mean(used))
            entry["tau_sd"] = float(np.std(used, ddof=1)) if len(used) > 1 else float("nan")
        results[cell] = entry

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(
            {"which": which, "tau": float(tau), "band_m": float(band_m),
             "seeds": seeds, "taus": [float(t) for t in taus],
             "target_free_miss": target, "common_range": [lo, hi],
             "n_valid_cells": int(valid.sum()), "n_near_cells": int(near.sum()),
             "n_frames": int(n_frames), "per_seed": per_seed, "cells": results},
            indent=2, ensure_ascii=False))
        print(f"  요약 -> {json_out}")

    print(f"\n=== free/non-free 판정의 시드 간 불일치 (고정 τ={tau:.2f}, {which}) ===")
    print("`D`는 시드 **쌍**에 대한 평균 불일치율이다. 낮을수록 재현성이 좋다.")
    print("**품질과 같이 읽는다** -- 전부 non-free로 찍는 모델은 D=0이다.\n")
    head = [("cell", 10), ("n", 3), ("iou_free", 16), ("free_miss", 10),
            ("D 전체", 9), ("D 경계근방", 12), ("D 대역밖", 10)]
    print("  " + " ".join(_pad(h, w) for h, w in head))
    for cell in live:
        v = results[cell]
        print("  " + " ".join(_pad(c, w) for c, (_, w) in zip(
            [cell, str(v["n_seeds"]), f"{v['iou_free']:.4f}±{v['iou_free_sd']:.4f}",
             f"{v['free_miss']:.4f}", f"{v['all'] * 100:.3f} %",
             f"{v['near'] * 100:.3f} %", f"{v['far'] * 100:.3f} %"], head)))

    if target is None:
        print("\n  !! 동작점을 맞춘 표를 못 냈다 -- 위 숫자만으로는 개선이 보정 이동인지"
              " 구별할 수 없다.")
        return 0

    print(f"\n=== 동작점을 맞춘 뒤 (시드마다 free_miss = {target:.4f}가 되는 τ_s로 자름) ===")
    print("**이것이 판정 표다.** 고정 τ에서만 좋아지고 여기서 같아지면 그 개선의 정체는")
    print("보정(확률 눈금) 이동이지 결정 재현성이 아니다(설계 문서 §16.2와 같은 함정).\n")
    head = [("cell", 10), ("τ_s 평균", 10), ("σ(τ_s)", 9), ("D 전체", 9),
            ("D 경계근방", 12), ("D 대역밖", 10), ("고정τ 대비 전체", 16)]
    print("  " + " ".join(_pad(h, w) for h, w in head))
    for cell in live:
        v = results[cell]
        ratio = v["all"] / v["all_m"] if v["all_m"] else float("nan")
        print("  " + " ".join(_pad(c, w) for c, (_, w) in zip(
            [cell, f"{v['tau_mean']:.3f}", f"{v['tau_sd']:.4f}",
             f"{v['all_m'] * 100:.3f} %", f"{v['near_m'] * 100:.3f} %",
             f"{v['far_m'] * 100:.3f} %", f"{ratio:.2f}x"], head)))
    print("\n  `σ(τ_s)`는 같은 동작점을 만들기 위해 필요한 문턱이 시드마다 얼마나 다른가다"
          " -- 계획서 §6.1의\n  `V_τ(q)`이고, 작을수록 확률 눈금 자체가 재현된다.")
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main) or 0)
