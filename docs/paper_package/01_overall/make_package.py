#!/usr/bin/env python
"""`01_overall` 논문 패키지의 데이터 파일을 결과 bundle에서 다시 만든다.

**이 스크립트가 `data/`와 `provenance/`의 정본이다.** CSV를 손으로 고치지 말고 여기를
고친 뒤 다시 돌린다. 입력은 서버의 `runs/paper_final/01_overall/`이므로 노트북에서는
돌아가지 않는다 -- 노트북에서는 이미 생성된 CSV를 읽기만 하면 된다.

실행:
    python make_package.py
    python make_package.py --root=/다른/경로/01_overall
"""
import csv
import json
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from projects.common.metric_spec import (  # noqa: E402
    BF_DETAIL_METRICS, PAPER_METRICS, RING_METRICS, ring_names,
)

DEFAULT_ROOT = Path("/data/home/dhlee/Desktop/bev-chamdog/runs/paper_final/01_overall")

#: 고정 split이 보고하는 지표 순서 -- **정본은 `projects/common/metric_spec.py`다**(2026-10-02:
#: IoU_free, IoU_non-free, Precision, Recall, BF@0.1/0.2/0.3, 전부 프레임 macro).
FIXED_METRICS = tuple(name for _, name, _ in PAPER_METRICS)
#: 본문 표에는 안 싣지만 CSV로 남기는 것: BF의 precision/recall, 링별 free 지표 넷.
SUPPLEMENTARY_METRICS = tuple(name for _, name, _ in BF_DETAIL_METRICS + RING_METRICS)
#: LOSO도 같은 일곱 개다(`tools/report_loso.py`가 `PAPER_METRICS`를 집계한다).
LOSO_METRICS = FIXED_METRICS

#: epoch 곡선으로 남길 TensorBoard 태그.
CURVE_TAGS = {tag: name for tag, name, _ in PAPER_METRICS}


def _shown(path: Path):
    """패키지 밖(diff용 임시 폴더)에 쓸 때도 죽지 않게."""
    try:
        return path.relative_to(HERE)
    except ValueError:
        return path


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {_shown(path)}  ({len(rows)} rows)")


def _folds(loso: dict) -> list:
    folds = loso["folds"]
    return folds if isinstance(folds, list) else list(folds.values())


def fixed_tables(fixed: dict, baseline: float, out: Path) -> None:
    """고정 split: 시드별 원자료, 요약, 논문 Table 1."""
    per_seed = []
    for run in fixed["per_run"]:
        row = {"run": run["run"], "seed": run["seed"], "epoch": run["fixed_epoch"]}
        row.update({m: run["metrics"][m] for m in FIXED_METRICS})
        per_seed.append(row)
    _write_csv(out / "fixed_split_per_seed.csv",
               ["run", "seed", "epoch", *FIXED_METRICS], per_seed)

    agg = fixed["metrics_at_fixed_epoch"]
    summary = [{"metric": m, "mean": agg[m]["mean"], "sd": agg[m]["sd"], "n": agg[m]["n"]}
               for m in FIXED_METRICS]
    _write_csv(out / "fixed_split_summary.csv", ["metric", "mean", "sd", "n"], summary)

    # 보조 지표(BF의 P/R, 링별) -- 긴 형식. 논문 본문에는 안 쓰지만 사용자가 "혹시 몰라서"
    # 남겨 달라고 했다(2026-10-02).
    supp = [{"run": run["run"], "seed": run["seed"], "epoch": run["fixed_epoch"],
             "metric": m, "value": run["metrics"].get(m)}
            for run in fixed["per_run"] for m in SUPPLEMENTARY_METRICS]
    _write_csv(out / "fixed_split_supplementary_per_seed.csv",
               ["run", "seed", "epoch", "metric", "value"], supp)

    # 링별 요약 -- 링 하나가 한 행. 3-4 m 링은 전방만 남은 부분 고리다(격자 전방 4 / 후방 2 m).
    rings = []
    for ring in ring_names():
        row = {"ring": ring, "n_seeds": agg[f"ring_{ring}_iou_free"]["n"]}
        for m in ("iou_free", "iou_non_free", "precision", "recall"):
            block = agg[f"ring_{ring}_{m}"]
            row[m], row[f"{m}_sd"] = block["mean"], block["sd"]
        rings.append(row)
    _write_csv(out / "fixed_split_ring_summary.csv", list(rings[0]), rings)

    # 논문 Table 1 -- baseline 행을 모델 행 위에 둔다 (§1-6: baseline 없는 iou_free 금지).
    table1 = [
        {"method": "Constant-map baseline", "n_seeds": "",
         "iou_free": round(baseline, 4), "iou_free_sd": "", "margin_vs_constant_map": ""},
        {"method": "Ours (Simple-BEV, ResNet-101, soft-boundary)", "n_seeds": 5,
         "iou_free": round(agg["iou_free"]["mean"], 4),
         "iou_free_sd": round(agg["iou_free"]["sd"], 4),
         "margin_vs_constant_map": round(agg["iou_free"]["mean"] - baseline, 4)},
    ]
    for m in FIXED_METRICS[1:]:
        table1[0][m] = ""
        table1[0][f"{m}_sd"] = ""
        table1[1][m] = round(agg[m]["mean"], 4)
        table1[1][f"{m}_sd"] = round(agg[m]["sd"], 4)
    fields = (["method", "n_seeds", "iou_free", "iou_free_sd", "margin_vs_constant_map"]
              + [c for m in FIXED_METRICS[1:] for c in (m, f"{m}_sd")])
    _write_csv(out / "table1_fixed_split.csv", fields, table1)


def loso_tables(loso: dict, out: Path) -> None:
    """LOSO: fold×seed 원자료, fold 요약, macro, 논문 Table 2, 그림 입력."""
    folds = _folds(loso)

    per_fold_seed = []
    for fold in folds:
        for i, seed in enumerate(fold["seeds"]):
            row = {"fold": fold["fold"], "seed": seed, "epoch": loso["experiment"]["fixed_epoch"]}
            row.update({m: fold["metrics"][m]["values"][i] for m in LOSO_METRICS})
            per_fold_seed.append(row)
    _write_csv(out / "loso_per_fold_seed.csv",
               ["fold", "seed", "epoch", *LOSO_METRICS], per_fold_seed)

    fold_rows = []
    for fold in folds:
        row = {
            "fold": fold["fold"],
            "lighting": fold["factors"].get("lighting"),
            "corridor_width": fold["factors"].get("width"),
            "extrapolation": fold["extrapolation"],
            "n_val_frames": fold["n_val_frames"],
            "n_seeds": fold["n_seed"],
            "constant_map_baseline": fold["constant_map_baseline"],
            "margin_over_constant_map": fold["margin_over_constant_map"],
        }
        for m in LOSO_METRICS:
            row[f"{m}_mean"] = fold["metrics"][m]["mean"]
            row[f"{m}_sd"] = fold["metrics"][m]["sd"]
        fold_rows.append(row)
    fold_fields = (["fold", "lighting", "corridor_width", "extrapolation", "n_val_frames",
                    "n_seeds", "constant_map_baseline", "margin_over_constant_map"]
                   + [f"{m}_{s}" for m in LOSO_METRICS for s in ("mean", "sd")])
    _write_csv(out / "loso_fold_summary.csv", fold_fields, fold_rows)

    macro = loso["macro"]
    macro_rows = []
    for key in (*LOSO_METRICS, "baseline", "margin"):
        block = macro[key]
        macro_rows.append({
            "metric": key,
            "macro_mean": block["mean"],
            "sd_across_folds": block["sd_across_folds"],
            "n_folds": block["n_folds"],
            "mean_within_fold_seed_sd": block.get("mean_within_fold_seed_sd", ""),
        })
    _write_csv(out / "loso_macro.csv",
               ["metric", "macro_mean", "sd_across_folds", "n_folds",
                "mean_within_fold_seed_sd"], macro_rows)

    # 논문 Table 2 -- fold 행 + macro 행. 반올림은 여기서 한 번만 한다.
    table2 = []
    for row in fold_rows:
        table2.append({
            "fold": row["fold"],
            "lighting": row["lighting"],
            "corridor_width": row["corridor_width"],
            "extrapolation": "yes" if row["extrapolation"] else "",
            "n_val_frames": row["n_val_frames"],
            "constant_map_baseline": round(row["constant_map_baseline"], 3),
            "iou_free": round(row["iou_free_mean"], 3),
            "iou_free_sd": round(row["iou_free_sd"], 4),
            "margin": round(row["margin_over_constant_map"], 3),
            **{m: round(row[f"{m}_mean"], 3) for m in LOSO_METRICS[1:]},
        })
    table2.append({
        "fold": "macro (7 folds)", "lighting": "", "corridor_width": "",
        "extrapolation": "", "n_val_frames": sum(r["n_val_frames"] for r in fold_rows),
        "constant_map_baseline": round(macro["baseline"]["mean"], 3),
        "iou_free": round(macro["iou_free"]["mean"], 3),
        "iou_free_sd": round(macro["iou_free"]["sd_across_folds"], 4),
        "margin": round(macro["margin"]["mean"], 3),
        **{m: round(macro[m]["mean"], 3) for m in LOSO_METRICS[1:]},
    })
    _write_csv(out / "table2_loso.csv", list(table2[0]), table2)

    # 그림 입력 -- Figure 1이 읽는 유일한 파일.
    fig = [{
        "fold": r["fold"],
        "corridor_width": r["corridor_width"],
        "lighting": r["lighting"],
        "extrapolation": r["extrapolation"],
        "constant_map_baseline": r["constant_map_baseline"],
        "model_iou_free": r["iou_free_mean"],
        "model_iou_free_sd": r["iou_free_sd"],
        "margin": r["margin_over_constant_map"],
    } for r in fold_rows]
    _write_csv(out / "figure_margin_inversion.csv", list(fig[0]), fig)


def val_curves(root: Path, out: Path) -> None:
    """두 프로토콜의 epoch별 validation 곡선. 수렴 그림·부록용."""
    rows = []
    for protocol in ("fixed_split", "loso"):
        path = root / protocol / "analysis" / "scalars.csv"
        if not path.exists():
            print(f"  (건너뜀: {path} 없음)")
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                metric = CURVE_TAGS.get(row["tag"])
                if metric is None:
                    continue
                rows.append({"protocol": protocol, "run": row["run"], "seed": row["seed"],
                             "epoch": int(row["epoch"]), "metric": metric,
                             "value": float(row["value"])})
    rows.sort(key=lambda r: (r["protocol"], r["run"], r["metric"], r["epoch"]))
    _write_csv(out / "val_curves.csv",
               ["protocol", "run", "seed", "epoch", "metric", "value"], rows)


def provenance(root: Path, fixed: dict, loso: dict, combined: dict,
               baseline_doc: dict, out: Path) -> None:
    """환경·동결 설정·무결성·런 목록. 논문 재현 절에 그대로 옮겨 쓸 수 있게."""
    common = dict(fixed["configuration"]["common"])
    for key in ("ckpt_dir", "log_dir", "train_sequences", "val_sequences"):
        common.pop(key, None)
    payload = {
        "experiment": "paper_final / 01_overall",
        "date": "2026-09-18",
        "git_commit": fixed["git_commit"],
        "environment": fixed["environment"],
        "protocols": {
            "fixed_split": {
                "train_sequences": baseline_doc["train_sequences"],
                "val_sequences": baseline_doc["val_sequences"],
                "n_train_frames": baseline_doc["n_train_frames"],
                "n_val_frames": baseline_doc["n_val_frames"],
                "seeds": [r["seed"] for r in fixed["per_run"]],
                "n_runs": fixed["experiment"]["n_runs"],
                "constant_map_baseline_iou_free": baseline_doc[
                    "constant_map_baseline_iou_free"],
            },
            "loso": {
                "folds": [f["fold"] for f in _folds(loso)],
                "seeds_per_fold": loso["experiment"]["seeds_per_fold"],
                "n_runs": loso["experiment"]["n_runs"],
                "held_out_is_validation": True,
                "train_sequences_per_fold": 6,
            },
        },
        "frozen_configuration": common,
        "primary_checkpoint": "fixed epoch 40 (validation-best is diagnostic only)",
    }
    (out / "environment_and_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {_shown(out / 'environment_and_config.json')}")

    (out / "integrity.json").write_text(
        json.dumps(combined["integrity"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {_shown(out / 'integrity.json')}")

    manifest = []
    for protocol, payload_ in (("fixed_split", fixed), ("loso", loso)):
        for name, entry in payload_["manifest"].items():
            manifest.append({
                "protocol": protocol, "run": name, "seed": entry.get("seed"),
                "cell": entry.get("cell"), "max_epoch": entry.get("max_epoch"),
                "n_scalar_tags": entry.get("n_tags"),
            })
    manifest.sort(key=lambda r: (r["protocol"], r["run"]))
    _write_csv(out / "run_manifest.csv",
               ["protocol", "run", "seed", "cell", "max_epoch", "n_scalar_tags"], manifest)


def main(root: str = str(DEFAULT_ROOT), skip_curves: bool = False,
         out_dir: str = str(HERE)):
    """`out_dir`는 재학습본을 **덮어쓰기 전에 diff**하려고 열어 둔 구멍이다."""
    root, out_dir = Path(root), Path(out_dir)
    fixed = json.loads((root / "fixed_split/analysis/RESULTS.json").read_text())
    loso = json.loads((root / "loso/analysis/RESULTS.json").read_text())
    combined = json.loads((root / "analysis/RESULTS.json").read_text())
    baseline_doc = json.loads(
        (root / "fixed_split/analysis/constant_map_baseline.json").read_text())

    if not combined["integrity"]["passed"]:
        raise SystemExit("무결성이 통과하지 않은 bundle이다. 패키지를 만들지 않는다.")

    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance").mkdir(parents=True, exist_ok=True)
    print("data/")
    fixed_tables(fixed, baseline_doc["constant_map_baseline_iou_free"], out_dir / "data")
    loso_tables(loso, out_dir / "data")
    if not skip_curves:
        val_curves(root, out_dir / "data")
    print("provenance/")
    provenance(root, fixed, loso, combined, baseline_doc, out_dir / "provenance")
    print("\n완료.")


if __name__ == "__main__":
    fire.Fire(main)
