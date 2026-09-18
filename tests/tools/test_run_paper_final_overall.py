import json
from types import SimpleNamespace

from tools.run_paper_final_overall import build_run_specs, run_queue, training_environment


def test_all_protocol_builds_five_fixed_and_thirty_five_loso_runs():
    specs = build_run_specs("all", tuple(range(5)), ())

    assert len([spec for spec in specs if spec.protocol == "fixed_split"]) == 5
    assert len([spec for spec in specs if spec.protocol == "loso"]) == 35
    assert len({spec.run_name for spec in specs}) == 40


def test_each_loso_fold_has_six_train_sequences_and_one_held_out_sequence():
    for spec in build_run_specs("loso", tuple(range(5)), ()):
        assert len(spec.train_sequences) == 6
        assert spec.val_sequences == (spec.held_out,)
        assert spec.held_out not in spec.train_sequences


def test_training_environment_spells_out_the_frozen_configuration(tmp_path):
    spec = build_run_specs("fixed", (0,), ())[0]

    env = training_environment(spec, tmp_path, 40, "0")

    expected = {
        "CUDA_VISIBLE_DEVICES": "0",
        "FORMULATION": "binary",
        "LOSS": "soft_boundary",
        "DELTA_M": "0.30",
        "SIGMA_M": "0.10",
        "SIGMA_ALPHA": "None",
        "LAMBDA_B": "0.5",
        "LAMBDA_R": "0.3",
        "DELTA_R_M": "0.15",
        "DELTA_R_OVER_M": "None",
        "HUBER_BETA_M": "0.15",
        "HEIGHT_BINS": "4",
        "HEIGHT_MIN_M": "-0.25",
        "HEIGHT_MAX_M": "1.75",
        "PIXEL_CONVENTION": "pixel_center",
        "PIXEL_OFFSET": "0.0",
        "ENCODER_TYPE": "res101",
        "INIT_CHECKPOINT": "none",
        "NUM_EPOCHS": "40",
        "SAVE_FREQ_EPOCHS": "40",
        "TRAIN_SEQUENCES": "raws2,raws3,rawos1,rawos2,rawos4",
        "VAL_SEQUENCES": "raws1,rawos3",
    }
    assert {key: env[key] for key in expected} == expected
    assert env["LOG_DIR"] == str(tmp_path / "fixed_split" / "logs")
    assert env["CKPT_DIR"] == str(tmp_path / "fixed_split" / "ckpt")


def test_plan_only_writes_manifest_without_starting_training(tmp_path):
    called = []
    specs = build_run_specs("fixed", (0,), ())

    result = run_queue(
        specs,
        tmp_path,
        num_epochs=40,
        gpu="0",
        plan_only=True,
        command_runner=lambda *args, **kwargs: called.append((args, kwargs)),
    )

    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text())
    assert result == 0
    assert called == []
    assert manifest["n_runs"] == 1
    assert manifest["runs"][0]["run_name"] == "final_s0"
    assert manifest["runs"][0]["environment"]["NUM_EPOCHS"] == "40"


def test_complete_run_is_skipped_without_invoking_training(tmp_path):
    specs = build_run_specs("fixed", (0,), ())
    log_dir = tmp_path / "fixed_split" / "logs" / "final_s0"
    ckpt_dir = tmp_path / "fixed_split" / "ckpt" / "final_s0"
    log_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    (log_dir / "config.json").write_text(json.dumps({"num_epochs": 40}))
    (ckpt_dir / "model-000000040.pth").touch()
    called = []

    result = run_queue(
        specs,
        tmp_path,
        num_epochs=40,
        gpu="0",
        command_runner=lambda *args, **kwargs: called.append((args, kwargs))
        or SimpleNamespace(returncode=0),
    )

    status = json.loads((tmp_path / "queue_status.json").read_text())
    assert result == 0
    assert called == []
    assert status["runs"] == [{"run_name": "final_s0", "status": "skipped_complete"}]


def test_incomplete_run_is_preserved_and_blocks_automatic_rerun(tmp_path):
    specs = build_run_specs("fixed", (0,), ())
    log_dir = tmp_path / "fixed_split" / "logs" / "final_s0"
    log_dir.mkdir(parents=True)
    marker = log_dir / "partial.txt"
    marker.write_text("keep me")
    called = []

    result = run_queue(
        specs,
        tmp_path,
        num_epochs=40,
        gpu="0",
        command_runner=lambda *args, **kwargs: called.append((args, kwargs))
        or SimpleNamespace(returncode=0),
    )

    status = json.loads((tmp_path / "queue_status.json").read_text())
    assert result == 1
    assert marker.read_text() == "keep me"
    assert called == []
    assert status["runs"] == [{"run_name": "final_s0", "status": "blocked_incomplete"}]
