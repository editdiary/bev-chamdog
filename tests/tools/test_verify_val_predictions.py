import json

from tools.verify_val_predictions import write_verification_report


def test_custom_verification_output_does_not_overwrite_legacy_location(tmp_path):
    pred_dir = tmp_path / "analysis" / "predictions" / "raws1"
    pred_dir.mkdir(parents=True)
    custom = tmp_path / "analysis" / "verify_predictions_raws1.json"
    payload = {"n_files": 10, "failures": [], "rows": []}

    written = write_verification_report(payload, pred_dir, out_path=custom)

    assert written == custom
    assert json.loads(custom.read_text()) == payload
    assert not (pred_dir.parent / "verify_predictions.json").exists()
