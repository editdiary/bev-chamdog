"""LOSO 7-fold 집계 -- **§25.6이 정한 읽는 법을 표에 강제한다.**

`summarize_repeats.py --group_by=val_sequences`가 이미 fold별로 묶어 주지만, 그것만으로는
진단 문서 §25.6이 요구한 네 가지가 표에 나오지 않는다. 이 도구가 그 넷을 만든다.

1. **주 열은 `iou_free`가 아니라 "constant-map 기준선 대비 마진"이다.** §21이 실측했듯
   raw 값은 모델보다 **시퀀스에 대해** 말한다 -- `rawos1` 0.836이 `rawos3` 0.790보다 높지만
   기준선 대비로는 +0.171 대 +0.397로 뒤집힌다. 기준선을 안 붙이면 표를 거꾸로 읽는다.
2. **fold 행마다 `(햇빛, 통로폭, 기준선, 프레임 수)`를 라벨링한다.** 그 표가 곧 층별
   분석이라 별도 실험이 필요 없다. `raws3`는 **외삽**이라고 따로 표시한다(§25.1 --
   (햇빛 O, 넓은 통로) 칸의 유일한 시퀀스라 그 fold만 train에 조합이 없다).
3. **fold 하나의 표준오차를 같이 찍는다.** val 36~41프레임 x 프레임별 std 0.082이면
   SE ≈ 0.013이고 이는 `σ_seed`(0.0015)의 **9배**다. 게다가 한 시퀀스의 프레임은 서로 강하게
   상관돼 있어 **실효 표본은 훨씬 작으므로 0.013은 하한**이다. 이 숫자가 안 보이면
   fold끼리 비교하게 된다 -- **하지 말아야 할 일이다.**
4. **`min`을 worst-case로 그대로 쓰지 않는다.** 7개 fold가 모두 같은 참값이어도 각각 SE로
   흔들리므로 `min`의 기대값은 평균보다 **약 1.35 x SE ≈ 0.018 낮다.** 표가 그 편향을 함께
   찍고, 시퀀스 간 참분산을 `σ_참² ≈ σ_관측² − SE²`로 추정해 나란히 둔다.

**그리고 fold 간 차이의 원인이 셋이라는 것을 표 아래에 항상 적는다** -- (i) 실제 장면 난이도,
(ii) 표본 오차, (iii) **시퀀스별 라벨 품질**(§28.5.3 때문에 생겼고 **분리할 수 없다**).
따라서 **fold 차이를 "모델의 일반화 능력"으로만 읽지 않는다.**

`constant-map` 기준선은 로그에 남지 않으므로(학습 스크립트가 콘솔에만 찍는다) 여기서 다시
계산한다. **학습과 같은 함수**(`constant_free_map` / `iou_free`)를 쓰므로 값이 갈리지 않는다.

실행:
    python tools/report_loso.py
    python tools/report_loso.py --log_root=runs/robot_bev_cv/loso/logs --fixed_epoch=40
"""
import math
import json
import statistics
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
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    FINETUNE_CAMERA_NAMES,
    build_bev_masks,
    load_masked_labels,
    split_samples_by_sequence,
)
from tools.summarize_repeats import pick_epochs, read_run  # noqa: E402

ALL_SEQUENCES = ("raws1", "raws2", "raws3", "rawos1", "rawos2", "rawos3", "rawos4")

#: 2x2 요인 구조 (진단 §25.1, 사용자 제공). `raws*` = 햇빛 가림막 없음.
FACTORS = {
    "raws1": ("햇빛", "좁음"), "raws2": ("햇빛", "좁음"), "raws3": ("햇빛", "넓음"),
    "rawos1": ("가림막", "좁음"), "rawos2": ("가림막", "좁음"),
    "rawos3": ("가림막", "넓음"), "rawos4": ("가림막", "넓음"),
}
#: (햇빛, 넓음) 칸의 유일한 시퀀스 -- 이 fold만 train에 그 조합이 없어 **외삽**이다.
EXTRAPOLATION_FOLDS = ("raws3",)

#: 프레임별 `iou_free` 표준편차 실측(진단 §22.1, val 75장). fold SE의 근거다.
FRAME_STD_IOU = 0.0820
#: n=3 시드 분산 실측(진단 §24.1). SE와 대조해 보여 준다.
SIGMA_SEED_IOU = 0.0015
#: E[min] 편향 계수 -- 표준정규 7개의 최솟값 기대값(≈ −1.35σ).
MIN_BIAS_COEF = 1.35

REPORTED = (
    ("val/iou_free_epoch", "iou_free", True),
    ("val/fatal_rate_epoch", "fatal_rate", False),
    ("val/free_miss_rate_epoch", "free_miss_rate", False),
)


def _mean_sd(values) -> tuple:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], float("nan"), 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def _finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def summarize_rows(rows: list[dict]) -> dict:
    """fold별 seed 집계를 다시 fold macro 통계로 요약한다."""
    result = {"n_folds": len(rows)}
    for metric in ("iou_free", "fatal_rate", "free_miss_rate"):
        values = [row[metric][0] for row in rows]
        within = [row[metric][1] for row in rows if math.isfinite(row[metric][1])]
        mean, sd, n = _mean_sd(values)
        result[metric] = {
            "mean": _finite_or_none(mean),
            "sd_across_folds": _finite_or_none(sd),
            "n_folds": n,
            "values": values,
            "mean_within_fold_seed_sd": _finite_or_none(
                statistics.fmean(within) if within else None
            ),
        }
    for metric in ("baseline", "margin"):
        values = [row[metric] for row in rows]
        mean, sd, n = _mean_sd(values)
        result[metric] = {"mean": _finite_or_none(mean),
                          "sd_across_folds": _finite_or_none(sd),
                          "n_folds": n, "values": values}
    return result


def build_report_payload(rows: list[dict], fixed_epoch: int, missing: list[str]) -> dict:
    """사람용 표와 같은 내용을 손실 없이 JSON으로 표현한다."""
    folds = []
    for row in rows:
        sun, width = FACTORS[row["fold"]]
        metrics = {}
        for metric in ("iou_free", "fatal_rate", "free_miss_rate"):
            mean, sd, n = row[metric]
            metrics[metric] = {
                "mean": _finite_or_none(mean),
                "sd": _finite_or_none(sd),
                "n": n,
                "values": row.get("seed_values", {}).get(metric, []),
            }
        folds.append({
            "fold": row["fold"],
            "factors": {"lighting": sun, "width": width},
            "extrapolation": row["fold"] in EXTRAPOLATION_FOLDS,
            "n_val_frames": row["n_val"],
            "n_seed": row["n_seed"],
            "seeds": row.get("seeds", []),
            "constant_map_baseline": row["baseline"],
            "margin_over_constant_map": row["margin"],
            "metrics": metrics,
        })
    return {
        "fixed_epoch": int(fixed_epoch),
        "fold_order": [row["fold"] for row in rows],
        "folds": folds,
        "macro": summarize_rows(rows),
        "warnings": {
            "missing_folds": list(missing),
            "folds_are_correlated": True,
            "do_not_report_fold_sd_over_sqrt_n_as_se": True,
        },
        "interpretation": [
            "주 비교는 raw iou_free보다 constant-map 대비 margin이다.",
            "fold 차이는 장면 난이도, 표본 오차, 라벨 품질을 분리하지 못한다.",
            "LOSO fold는 train 데이터를 공유하므로 독립 표본이 아니다.",
        ],
    }


def fold_baseline(held, dataset_root, common_root, device) -> tuple:
    """그 fold의 constant-map 기준선과 val 프레임 수.

    **학습 스크립트와 같은 계산이다** -- train 6시퀀스의 셀별 다수결 free map을 만들어
    held-out 시퀀스에 한 번 채점한다. 로그에 안 남는 값이라 여기서 다시 만든다.
    """
    root = Path(dataset_root)
    roots = [root / name for name in ALL_SEQUENCES]
    train_samples, val_samples = split_samples_by_sequence(roots, [held])
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)

    train_free = [
        decompose(*load_masked_labels(r, s, permanent_blind, invalid))["free"]
        for r, s in train_samples
    ]
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
    return float(baseline), len(val_samples)


def collect(log_root, fixed_epoch) -> dict:
    """`{fold: [런...]}`. fold는 `config.json`의 `val_sequences`가 정한다."""
    root = Path(log_root)
    if not root.exists():
        raise FileNotFoundError(f"로그 폴더가 없다: {root}")
    folds = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run = read_run(run_dir)
        held = str(run["config"].get("val_sequences", "")).strip()
        if not held:
            print(f"  [건너뜀] {run_dir.name}: config.json에 val_sequences가 없다")
            continue
        pick = pick_epochs(run["series"], fixed_epoch)
        if pick["fixed"] is None:
            print(f"  [건너뜀] {run_dir.name}: epoch {fixed_epoch}이 없다"
                  f" (마지막 {pick['last']}) -- 미완료 런")
            continue
        run["pick"] = pick
        folds.setdefault(held, []).append(run)
    return folds


def main(log_root="runs/robot_bev_cv/loso/logs", fixed_epoch=40,
         dataset_root=DEFAULT_DATASET_ROOT, common_root=DEFAULT_COMMON_ROOT,
         json_out=None, legacy_sampling_diagnostics=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = collect(log_root, fixed_epoch)
    if not folds:
        raise SystemExit("집계할 런이 없다.")

    print(f"\n=== LOSO 7-fold (고정 epoch {fixed_epoch}, fold당 시드 평균) ===")
    print("**고정 epoch이다** -- val이 시퀀스 1개일 때 그 시퀀스로 체크포인트를 고르면")
    print("그 fold에 맞춘 선택 편향이 생긴다. 따라서 모든 fold를 같은 epoch에서 읽는다.\n")

    rows = []
    for held in ALL_SEQUENCES:
        runs = folds.get(held)
        if not runs:
            continue
        baseline, n_val = fold_baseline(held, dataset_root, common_root, device)
        row = {"fold": held, "baseline": baseline, "n_val": n_val, "n_seed": len(runs),
               "seed_values": {}, "seeds": [r["config"].get("seed") for r in runs]}
        for tag, label, _ in REPORTED:
            values = [r["series"].get(tag, {}).get(r["pick"]["fixed"]) for r in runs]
            row[label] = _mean_sd(values)
            row["seed_values"][label] = values
        row["margin"] = row["iou_free"][0] - baseline
        row["se"] = (FRAME_STD_IOU / math.sqrt(n_val)
                     if legacy_sampling_diagnostics and n_val else float("nan"))
        rows.append(row)

    missing = [s for s in ALL_SEQUENCES if s not in folds]
    header = (f"{'fold':9}{'요인':12}{'프레임':>6}{'시드':>4}"
              f"{'기준선':>9}{'iou_free':>18}{'마진↑':>9}"
              + (f"{'fold SE':>9}" if legacy_sampling_diagnostics else "")
              + f"{'fatal_rate':>17}")
    print(header)
    print("-" * len(header))
    for r in rows:
        sun, width = FACTORS[r["fold"]]
        mark = " *외삽" if r["fold"] in EXTRAPOLATION_FOLDS else ""
        iou_m, iou_sd, _ = r["iou_free"]
        fat_m, fat_sd, _ = r["fatal_rate"]
        print(f"{r['fold']:9}{sun + '/' + width:12}{r['n_val']:>6}{r['n_seed']:>4}"
              f"{r['baseline']:>9.3f}{iou_m:>11.4f}±{iou_sd:.4f}{r['margin']:>+9.3f}"
              + (f"{r['se']:>9.3f}" if legacy_sampling_diagnostics else "")
              + f"{fat_m:>10.4f}±{fat_sd:.4f}{mark}")

    # --- 요약 통계 -----------------------------------------------------------
    ious = [r["iou_free"][0] for r in rows]
    margins = [r["margin"] for r in rows]
    fatals = [r["fatal_rate"][0] for r in rows]
    obs_sd = statistics.stdev(ious) if len(ious) > 1 else float("nan")
    worst = min(rows, key=lambda r: r["iou_free"][0])
    worst_margin = min(rows, key=lambda r: r["margin"])
    worst_fatal = max(rows, key=lambda r: r["fatal_rate"][0])

    print(f"\n--- 요약 (fold {len(rows)}개) ---")
    print(f"  `iou_free`   평균 {statistics.fmean(ious):.4f} | fold 간 관측 std {obs_sd:.4f}")
    print(f"  기준선 마진   평균 {statistics.fmean(margins):+.4f} | fold 간 std "
          f"{statistics.stdev(margins) if len(margins) > 1 else float('nan'):.4f}")
    print(f"  `fatal_rate` 평균 {statistics.fmean(fatals):.4f} | fold 간 std "
          f"{statistics.stdev(fatals) if len(fatals) > 1 else float('nan'):.4f}")
    if legacy_sampling_diagnostics:
        mean_se = statistics.fmean(r["se"] for r in rows)
        true_var = obs_sd ** 2 - mean_se ** 2
        true_sd = math.sqrt(true_var) if true_var > 0 else float("nan")
        print(f"\n  fold 하나의 표준오차(SE)  평균 {mean_se:.4f}"
              f"   <- σ_seed({SIGMA_SEED_IOU:.4f})의 {mean_se / SIGMA_SEED_IOU:.0f}배")
        print(f"  시퀀스 간 **참**분산 추정   σ_참 ≈ √(std² − SE²) = {true_sd:.4f}")
        print("    (프레임이 서로 강하게 상관돼 있어 **SE는 하한**이므로 σ_참은 상한 추정이다)")

    print(f"\n  worst fold (`iou_free`)  {worst['fold']} {worst['iou_free'][0]:.4f}")
    print(f"  worst fold (마진)         {worst_margin['fold']} {worst_margin['margin']:+.4f}"
          "   <- **이쪽이 주 열이다**")
    print(f"  worst fold (`fatal_rate`) {worst_fatal['fold']} {worst_fatal['fatal_rate'][0]:.4f}")
    if legacy_sampling_diagnostics:
        print(f"\n  ⚠ `min`은 아래로 편향된 통계다. 7개 fold가 모두 같은 참값이어도 `min`의")
        print(f"    기대값이 평균보다 약 {MIN_BIAS_COEF * mean_se:.4f} 낮다"
              f" ({MIN_BIAS_COEF} x SE). **worst를 그대로 인용하지 않는다.**")

    if missing:
        print(f"\n  ⚠ **미완료 fold: {', '.join(missing)}** -- 위 요약은 부분 집계다.")

    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_report_payload(rows, fixed_epoch, missing)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        print(f"\n구조화 결과 -> {path}")

    print("""
--- 이 표를 읽는 법 (진단 §25.6, **실행 전에 정해 두었다**) ---
  1. **fold끼리 단순 비교하지 않는다.** 장면과 라벨 분포가 서로 다르다.
  2. **주 열은 `iou_free`가 아니라 기준선 대비 마진이다** -- raw 값은 모델보다
     시퀀스 난이도에 대해 말한다(§21: rawos1 +0.171 대 rawos3 +0.397).
  3. **`*외삽` 표시된 fold는 train에 그 요인 조합이 없다**(§25.1).
  4. **fold 간 차이의 원인이 셋이고 분리할 수 없다** -- (i) 실제 장면 난이도,
     (ii) 표본 오차, (iii) **시퀀스별 라벨 품질**(§28.5.3).
     **fold 차이를 "모델의 일반화 능력"으로만 읽지 않는다.**
  5. **LOSO 평균의 표준오차는 σ_fold/√7이 아니다** -- 7개 fold가 train을 6/7씩 공유해
     결과가 서로 상관돼 있다(LOO-CV의 알려진 성질). 평균은 점추정으로 보고하고
     fold 7개의 분포를 전부 싣는다.""")


if __name__ == "__main__":
    Fire(main)
