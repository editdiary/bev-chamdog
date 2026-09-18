"""Overall Performance의 고정 split·LOSO 결과를 자립형 파일로 묶는다."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


METRICS = {
    "val/iou_free_epoch": "iou_free",
    "val/fatal_rate_epoch": "fatal_rate",
    "val/free_miss_rate_epoch": "free_miss_rate",
    "val/occupied_f1_10cm_epoch": "occupied_f1_10cm",
    "val/occupied_f1_20cm_epoch": "occupied_f1_20cm",
    "val/occupied_f1_40cm_epoch": "occupied_f1_40cm",
    "val/range_mae_epoch": "range_mae",
    "val/range_bias_epoch": "range_bias",
    "val/range_missed_obstacle_rate_epoch": "range_missed_obstacle_rate",
}


def _load_json(path: Path, missing: list[str], default):
    if not path.exists():
        missing.append(str(path))
        return default
    return json.loads(path.read_text())


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _summary(values):
    clean = [float(value) for value in values if _finite(value)]
    if not clean:
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else None,
        "n": len(clean),
        "values": clean,
    }


def _environment():
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_env": Path(sys.prefix).name,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }


def _git_commit(root: Path):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_scalars(path: Path, missing: list[str]):
    if not path.exists():
        missing.append(str(path))
        return []
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            try:
                row["epoch"] = int(row["epoch"])
                row["value"] = float(row["value"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
        return rows


def _config_partition(configs):
    if not configs:
        return {}, {}
    keys = set.intersection(*(set(config) for config in configs.values()))
    common = {
        key: next(iter(configs.values()))[key]
        for key in sorted(keys)
        if all(config[key] == next(iter(configs.values()))[key]
               for config in configs.values())
    }
    different = {
        run: {key: value for key, value in config.items()
              if key not in common}
        for run, config in sorted(configs.items())
    }
    return common, different


def _integrity(paths, missing):
    failures, diffs, n_files = [], [], 0
    sources = []
    for path in paths:
        payload = _load_json(path, missing, {})
        if not payload:
            continue
        sources.append(str(path))
        n_files += int(payload.get("n_files", 0))
        failures.extend(payload.get("failures", []))
        for row in payload.get("rows", []):
            if _finite(row.get("abs_diff")):
                diffs.append(float(row["abs_diff"]))
    return {
        "passed": not failures and not any(str(path) in missing for path in paths),
        "n_prediction_files": n_files,
        "max_abs_diff": max(diffs) if diffs else None,
        "failures": failures,
        "sources": sources,
    }


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _write_csv(path: Path, rows):
    fields = ("protocol", "scope", "run", "fold", "seed", "epoch", "metric", "value")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _fixed_bundle(root: Path, fixed_epoch: int, env, commit):
    analysis = root / "fixed_split" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    missing = []
    configs = _load_json(analysis / "configs.json", missing, {})
    manifest = _load_json(analysis / "manifest.json", missing, {})
    rows = _read_scalars(analysis / "scalars.csv", missing)
    common, different = _config_partition(configs)

    lookup = defaultdict(dict)
    seeds = {}
    for row in rows:
        if row["tag"] in METRICS:
            lookup[row["run"]][(row["tag"], row["epoch"])] = row["value"]
            if str(row.get("seed", "")).strip():
                seeds[row["run"]] = int(row["seed"])

    per_run, aggregate, aggregate_best, csv_rows = [], {}, {}, []
    for run in sorted(configs):
        fixed_values = {
            name: lookup[run].get((tag, fixed_epoch)) for tag, name in METRICS.items()
        }
        iou_points = [(epoch, value) for (tag, epoch), value in lookup[run].items()
                      if tag == "val/iou_free_epoch"]
        best_epoch = max(iou_points, key=lambda item: item[1])[0] if iou_points else None
        best_values = {
            name: lookup[run].get((tag, best_epoch)) if best_epoch is not None else None
            for tag, name in METRICS.items()
        }
        seed = configs[run].get("seed", seeds.get(run))
        per_run.append({"run": run, "seed": seed, "fixed_epoch": fixed_epoch,
                        "metrics": fixed_values, "best_iou_epoch": best_epoch,
                        "metrics_at_best_iou": best_values})
        for metric, value in fixed_values.items():
            if _finite(value):
                csv_rows.append({"protocol": "fixed_split", "scope": "run",
                                 "run": run, "seed": seed, "epoch": fixed_epoch,
                                 "metric": metric, "value": value})

    for metric in METRICS.values():
        aggregate[metric] = _summary(row["metrics"].get(metric) for row in per_run)
        aggregate_best[metric] = _summary(
            row["metrics_at_best_iou"].get(metric) for row in per_run
        )
        for scope, key in (("fixed_mean", "mean"), ("fixed_sd", "sd")):
            if _finite(aggregate[metric][key]):
                csv_rows.append({"protocol": "fixed_split", "scope": scope,
                                 "epoch": fixed_epoch, "metric": metric,
                                 "value": aggregate[metric][key]})

    incomplete = sorted(run for run, item in manifest.items() if not item.get("complete"))
    integrity = _integrity([analysis / "verify_predictions.json"], missing)
    integrity["incomplete_runs"] = incomplete
    structural = []
    if len(configs) != 5:
        structural.append(f"expected 5 runs, found {len(configs)}")
    if aggregate["iou_free"]["n"] != 5:
        structural.append(
            f"iou_free: expected 5 fixed-epoch values, found {aggregate['iou_free']['n']}"
        )
    integrity["structural_failures"] = structural
    integrity["passed"] = integrity["passed"] and not incomplete and not structural
    payload = {
        "schema_version": 1,
        "experiment": {"protocol": "fixed_split", "n_runs": len(configs),
                       "fixed_epoch": fixed_epoch},
        "environment": env, "git_commit": commit,
        "configuration": {"common": common, "per_run_differences": different},
        "manifest": manifest, "per_run": per_run,
        "metrics_at_fixed_epoch": aggregate,
        "diagnostic_metrics_at_each_runs_best_iou": aggregate_best,
        "integrity": integrity, "missing_files": sorted(set(missing)),
        "_source": {"scalars": str(analysis / "scalars.csv"),
                    "configs": str(analysis / "configs.json"),
                    "manifest": str(analysis / "manifest.json")},
    }
    _write_json(analysis / "RESULTS.json", payload)
    _write_csv(analysis / "RESULTS.csv", csv_rows)
    (analysis / "README.md").write_text(
        "# Fixed-split overall results\n\n"
        f"Five-seed summary at fixed epoch {fixed_epoch}. RESULTS.json is the canonical bundle; "
        "RESULTS.csv is the flat table. Best-IoU values are diagnostics only.\n"
    )
    return payload, csv_rows


def _loso_bundle(root: Path, fixed_epoch: int, env, commit):
    analysis = root / "loso" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    missing = []
    configs = _load_json(analysis / "configs.json", missing, {})
    manifest = _load_json(analysis / "manifest.json", missing, {})
    report = _load_json(analysis / "report_loso.json", missing, {})
    common, different = _config_partition(configs)
    fold_names = report.get(
        "fold_order", [item.get("fold") for item in report.get("folds", [])]
    )
    verification_paths = [
        analysis / f"verify_predictions_{fold}.json" for fold in fold_names if fold
    ]
    integrity = _integrity(verification_paths, missing)
    incomplete = sorted(run for run, item in manifest.items() if not item.get("complete"))
    integrity["incomplete_runs"] = incomplete
    structural = []
    if len(configs) != 35:
        structural.append(f"expected 35 runs, found {len(configs)}")
    if len(report.get("folds", [])) != 7:
        structural.append(f"expected 7 folds, found {len(report.get('folds', []))}")
    for fold in report.get("folds", []):
        if int(fold.get("n_seed", 0)) != 5:
            structural.append(
                f"{fold.get('fold')}: expected 5 seeds, found {fold.get('n_seed', 0)}"
            )
    integrity["structural_failures"] = structural
    integrity["passed"] = integrity["passed"] and not incomplete and not structural

    csv_rows = []
    for fold in report.get("folds", []):
        fold_name = fold["fold"]
        seeds = fold.get("seeds", [])
        for metric, summary in fold.get("metrics", {}).items():
            for index, value in enumerate(summary.get("values", [])):
                csv_rows.append({"protocol": "loso", "scope": "fold_seed",
                                 "fold": fold_name,
                                 "seed": seeds[index] if index < len(seeds) else index,
                                 "epoch": fixed_epoch, "metric": metric, "value": value})
            for scope, key in (("fold_mean", "mean"), ("fold_sd", "sd")):
                if _finite(summary.get(key)):
                    csv_rows.append({"protocol": "loso", "scope": scope,
                                     "fold": fold_name, "epoch": fixed_epoch,
                                     "metric": metric, "value": summary[key]})
        for metric, value in (("constant_map_baseline", fold.get("constant_map_baseline")),
                              ("margin_over_constant_map", fold.get("margin_over_constant_map"))):
            if _finite(value):
                csv_rows.append({"protocol": "loso", "scope": "fold_mean",
                                 "fold": fold_name, "epoch": fixed_epoch,
                                 "metric": metric, "value": value})
    for metric, summary in report.get("macro", {}).items():
        if isinstance(summary, dict) and _finite(summary.get("mean")):
            csv_rows.append({"protocol": "loso", "scope": "macro",
                             "epoch": fixed_epoch, "metric": metric,
                             "value": summary["mean"]})

    payload = {
        "schema_version": 1,
        "experiment": {"protocol": "loso", "n_runs": len(configs),
                       "n_folds": len(report.get("folds", [])),
                       "seeds_per_fold": 5, "fixed_epoch": fixed_epoch},
        "environment": env, "git_commit": commit,
        "configuration": {"common": common, "per_run_differences": different},
        "manifest": manifest, "folds": report.get("folds", []),
        "macro": report.get("macro", {}), "warnings": report.get("warnings", {}),
        "interpretation": report.get("interpretation", []),
        "integrity": integrity, "missing_files": sorted(set(missing)),
        "_source": {"report": str(analysis / "report_loso.json"),
                    "configs": str(analysis / "configs.json"),
                    "manifest": str(analysis / "manifest.json")},
    }
    _write_json(analysis / "RESULTS.json", payload)
    _write_csv(analysis / "RESULTS.csv", csv_rows)
    (analysis / "README.md").write_text(
        "# LOSO overall results\n\n"
        f"Seven-fold, five-seed summary at fixed epoch {fixed_epoch}. Fold means are kept "
        "separate; the folds share training data and are not independent samples.\n"
    )
    return payload, csv_rows


def build_bundles(root, fixed_epoch=40):
    root = Path(root).resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    env = _environment()
    commit = _git_commit(Path(__file__).resolve().parents[1])
    fixed, fixed_csv = _fixed_bundle(root, int(fixed_epoch), env, commit)
    loso, loso_csv = _loso_bundle(root, int(fixed_epoch), env, commit)
    missing = sorted(set(fixed["missing_files"] + loso["missing_files"]))
    combined = {
        "schema_version": 1,
        "experiment": "paper_final_overall",
        "protocols": {
            "fixed_split": {"result": "../fixed_split/analysis/RESULTS.json",
                            "n_runs": fixed["experiment"]["n_runs"]},
            "loso": {"result": "../loso/analysis/RESULTS.json",
                     "n_runs": loso["experiment"]["n_runs"],
                     "n_folds": loso["experiment"]["n_folds"]},
        },
        "integrity": {
            "passed": fixed["integrity"]["passed"] and loso["integrity"]["passed"],
            "fixed_split": fixed["integrity"], "loso": loso["integrity"],
        },
        "missing_files": missing,
        "note": "Fixed split and LOSO answer different questions and are never pooled.",
    }
    _write_json(output / "RESULTS.json", combined)
    _write_csv(output / "RESULTS.csv", fixed_csv + loso_csv)
    (output / "README.md").write_text(
        "# Paper final overall results\n\n"
        "This index links the fixed-split and LOSO bundles. The protocols are not averaged "
        "together. See each protocol's RESULTS.json for canonical values and provenance.\n"
    )
    return {"fixed_split": fixed, "loso": loso, "combined": combined}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/paper_final/01_overall")
    parser.add_argument("--fixed_epoch", type=int, default=40)
    args = parser.parse_args(argv)
    bundles = build_bundles(args.root, args.fixed_epoch)
    if not bundles["combined"]["integrity"]["passed"]:
        raise SystemExit("결과 번들은 생성했지만 무결성 검사를 통과하지 못했다.")
    print(f"결과 번들 생성 완료: {Path(args.root).resolve() / 'analysis'}")


if __name__ == "__main__":
    main()
