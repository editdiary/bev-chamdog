import csv
import json

import pytest

from projects.common.metric_spec import PAPER_METRICS
from tools.build_overall_results_bundle import build_bundles


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_builds_separate_fixed_loso_and_combined_publication_results(tmp_path):
    fixed = tmp_path / "fixed_split" / "analysis"
    loso = tmp_path / "loso" / "analysis"
    fixed.mkdir(parents=True)
    loso.mkdir(parents=True)

    with (fixed / "scalars.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "cell", "seed", "tag", "epoch", "value"))
        for seed, value in enumerate((0.80, 0.81, 0.82, 0.83, 0.84)):
            writer.writerow((f"final_s{seed}", "final", seed,
                             "val/iou_free_epoch", 40, value))
            # 논문 지표가 하나라도 5개가 아니면 무결성 실패다 -- 나머지도 채운다.
            for tag, _, _ in PAPER_METRICS[1:]:
                writer.writerow((f"final_s{seed}", "final", seed, tag, 40, 0.9))
    fixed_configs = {
        f"final_s{seed}": {"seed": seed, "num_epochs": 40, "height_bins": 4,
                            "val_sequences": "raws1,rawos3"}
        for seed in range(5)
    }
    _write_json(fixed / "configs.json", fixed_configs)
    _write_json(fixed / "manifest.json", {
        name: {"complete": True, "max_epoch": 40} for name in fixed_configs
    })
    _write_json(fixed / "verify_predictions.json", {
        "n_files": 10, "failures": [], "rows": [{"abs_diff": 0.0002}]
    })

    loso_configs = {
        f"loso_fold{fold}_s{seed}": {"seed": seed, "num_epochs": 40,
                                      "height_bins": 4, "val_sequences": f"fold{fold}"}
        for fold in range(7) for seed in range(5)
    }
    _write_json(loso / "configs.json", loso_configs)
    _write_json(loso / "manifest.json", {
        name: {"complete": True, "max_epoch": 40} for name in loso_configs
    })
    folds = [{
        "fold": f"fold{fold}", "n_seed": 5, "constant_map_baseline": 0.4,
        "margin_over_constant_map": 0.4,
        "metrics": {"iou_free": {"mean": 0.8, "sd": 0.01, "n": 5,
                                    "values": [0.8] * 5}},
    } for fold in range(7)]
    _write_json(loso / "report_loso.json", {
        "fixed_epoch": 40, "folds": folds,
        "macro": {"n_folds": 7, "iou_free": {"mean": 0.8,
                                                  "sd_across_folds": 0.02}},
    })
    for fold in range(7):
        _write_json(loso / f"verify_predictions_fold{fold}.json", {
            "n_files": 10, "failures": [], "rows": [{"abs_diff": 0.0003}]
        })

    bundles = build_bundles(tmp_path, fixed_epoch=40)

    fixed_result = bundles["fixed_split"]
    loso_result = bundles["loso"]
    combined = bundles["combined"]
    assert fixed_result["experiment"]["n_runs"] == 5
    assert fixed_result["metrics_at_fixed_epoch"]["iou_free"]["n"] == 5
    assert fixed_result["metrics_at_fixed_epoch"]["iou_free"]["mean"] == pytest.approx(0.82)
    assert loso_result["experiment"]["n_runs"] == 35
    assert loso_result["macro"]["n_folds"] == 7
    assert combined["integrity"]["passed"] is True
    assert "joint_mean" not in combined
    for path in (
        fixed / "RESULTS.json", fixed / "RESULTS.csv", fixed / "README.md",
        loso / "RESULTS.json", loso / "RESULTS.csv", loso / "README.md",
        tmp_path / "analysis" / "RESULTS.json",
        tmp_path / "analysis" / "RESULTS.csv",
        tmp_path / "analysis" / "README.md",
    ):
        assert path.exists(), path


def test_a_paper_metric_missing_from_any_run_fails_integrity(tmp_path):
    """옛 지표 정의로 기록된 런이 섞이면 그 열만 조용히 빈다. `iou_free`만 세면 놓치므로
    `PAPER_METRICS` 전부를 센다(2026-10-02)."""
    fixed = tmp_path / "fixed_split" / "analysis"
    fixed.mkdir(parents=True)
    with (fixed / "scalars.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "cell", "seed", "tag", "epoch", "value"))
        for seed in range(5):
            for tag, _, _ in PAPER_METRICS:
                if seed == 4 and tag == "val/bf_30cm_epoch":
                    continue                  # 한 런에서 한 지표만 빠졌다
                writer.writerow((f"final_s{seed}", "final", seed, tag, 40, 0.8))
    configs = {f"final_s{s}": {"seed": s} for s in range(5)}
    _write_json(fixed / "configs.json", configs)
    _write_json(fixed / "manifest.json", {n: {"complete": True} for n in configs})
    _write_json(fixed / "verify_predictions.json", {"n_files": 10, "failures": [],
                                                    "rows": [{"abs_diff": 0.0}]})

    bundles = build_bundles(tmp_path, fixed_epoch=40)

    failures = bundles["fixed_split"]["integrity"]["structural_failures"]
    assert any(f.startswith("bf_30cm:") for f in failures), failures
    assert bundles["fixed_split"]["integrity"]["passed"] is False
