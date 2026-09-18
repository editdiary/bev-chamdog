import json
import math

import pytest

import tools.report_loso as report_loso
from tools.report_loso import build_report_payload, summarize_rows


def test_loso_summary_averages_fold_means_instead_of_flattening_seed_rows():
    rows = [
        {
            "fold": "raws1",
            "iou_free": (0.8, 0.01, 5),
            "fatal_rate": (0.1, 0.01, 5),
            "free_miss_rate": (0.2, 0.01, 5),
            "baseline": 0.6,
            "margin": 0.2,
            "se": 0.01,
            "n_val": 38,
            "n_seed": 5,
        },
        {
            "fold": "rawos3",
            "iou_free": (0.7, 0.02, 5),
            "fatal_rate": (0.2, 0.02, 5),
            "free_miss_rate": (0.3, 0.02, 5),
            "baseline": 0.4,
            "margin": 0.3,
            "se": 0.02,
            "n_val": 37,
            "n_seed": 5,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["iou_free"]["mean"] == pytest.approx(0.75)
    assert summary["margin"]["mean"] == pytest.approx(0.25)
    assert summary["n_folds"] == 2
    assert summary["iou_free"]["mean_within_fold_seed_sd"] == pytest.approx(0.015)


def test_report_payload_marks_extrapolation_and_keeps_fold_seed_values():
    rows = [{
        "fold": "raws3",
        "iou_free": (0.81, 0.003, 5),
        "fatal_rate": (0.12, 0.004, 5),
        "free_miss_rate": (0.08, 0.002, 5),
        "seed_values": {"iou_free": [0.806, 0.808, 0.81, 0.812, 0.814]},
        "seeds": [0, 1, 2, 3, 4],
        "baseline": 0.39,
        "margin": 0.42,
        "se": 0.014,
        "n_val": 36,
        "n_seed": 5,
    }]

    payload = build_report_payload(rows, fixed_epoch=40, missing=[])

    fold = payload["folds"][0]
    assert payload["fixed_epoch"] == 40
    assert fold["fold"] == "raws3"
    assert fold["extrapolation"] is True
    assert fold["factors"] == {"lighting": "햇빛", "width": "넓음"}
    assert fold["metrics"]["iou_free"]["values"] == [0.806, 0.808, 0.81, 0.812, 0.814]


def test_report_payload_is_strict_json_even_with_one_seed():
    rows = [{
        "fold": "raws1",
        "iou_free": (0.8, math.nan, 1),
        "fatal_rate": (0.1, math.nan, 1),
        "free_miss_rate": (0.2, math.nan, 1),
        "seed_values": {"iou_free": [0.8]},
        "seeds": [0],
        "baseline": 0.6,
        "margin": 0.2,
        "se": 0.01,
        "n_val": 38,
        "n_seed": 1,
    }]

    payload = build_report_payload(rows, fixed_epoch=1, missing=[])

    json.dumps(payload, allow_nan=False)
    assert payload["folds"][0]["metrics"]["iou_free"]["sd"] is None


def test_current_report_does_not_print_legacy_y1_sampling_diagnostics(
    tmp_path, monkeypatch, capsys
):
    run = {
        "config": {"seed": 0},
        "series": {
            "val/iou_free_epoch": {40: 0.8},
            "val/fatal_rate_epoch": {40: 0.1},
            "val/free_miss_rate_epoch": {40: 0.2},
        },
        "pick": {"fixed": 40},
    }
    monkeypatch.setattr(report_loso, "collect", lambda *_: {"raws1": [run]})
    monkeypatch.setattr(report_loso, "fold_baseline", lambda *_: (0.6, 38))

    report_loso.main(log_root=tmp_path, fixed_epoch=40)

    output = capsys.readouterr().out
    assert "fold 하나의 표준오차" not in output
    assert "σ_seed(0.0015)" not in output
