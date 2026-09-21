"""cumulative ray probe 셸이 대조군의 동결 설정을 실제 child process에 전달하는지 검증.

이 probe는 `runs/loss_effect`의 `C_soft_s{N}`·`D_range_s{N}`과 **seed끼리 짝지어** 비교된다.
그러려면 셸이 넘기는 설정이 호출 환경에 관계없이 항상 같아야 한다. 호출 셸에 남아 있던
예전 스윕 변수(`FRAME_SPLIT_FRACTION`, `SEED`, `NUM_EPOCHS`, ...)가 하나라도 새어 들어가면
그 런은 대조군과 짝지을 수 없는데, 산출물만 보면 정상으로 보인다.
"""
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# `train_robot_bev_finetune.sh`에 도달해야 하는 값. 호출 환경이 무엇이든 이것이어야 한다.
FROZEN = {
    "FRAME_SPLIT_FRACTION": "0.0",
    "FRAME_BLOCK_LEN": "0",
    "NUM_EPOCHS": "40",
    "LOSS": "soft_boundary",
    "SOFT_TARGET": "gaussian",
    "RANGE_LOSS_MODE": "cumulative_l1",
    "LAMBDA_B": "0.5",
    "SIGMA_M": "0.10",
    "DELTA_M": "0.30",
    "SIGMA_ALPHA": "None",
    "BAND_KAPPA": "1.0",
    "LABEL_EPS": "0.0",
    "HEIGHT_BINS": "4",
    "HEIGHT_MIN_M": "-0.25",
    "HEIGHT_MAX_M": "1.75",
    "FORMULATION": "binary",
    "ENCODER_TYPE": "res101",
    "AUGMENT": "True",
    "INIT_CHECKPOINT": "none",
    "LR": "1e-4",
    "WEIGHT_DECAY": "1e-7",
    "MAX_CLASS_WEIGHT": "20",
    "LABEL_SMOOTHING": "0.0",
    "FLIP_AUGMENT": "False",
    "FREEZE_ENCODER": "False",
    "PROJECTION": "ds_native",
    "PINHOLE_HFOV_DEG": "None",
    "PIXEL_CONVENTION": "pixel_center",
    "PIXEL_OFFSET": "0.0",
    "SAVE_FREQ_EPOCHS": "40",
    "TRAIN_SEQUENCES": "raws2,raws3,rawos1,rawos2,rawos4",
    "VAL_SEQUENCES": "raws1,rawos3",
    "EXP_NAME": "E_cumulative",
}

# child가 찍어야 하는 변수 순서. seed별로 달라지는 둘을 앞에 둔다.
_CAPTURED = ["SEED", "RUN_NAME", "CUDA_VISIBLE_DEVICES", *FROZEN]


def _run_probe(tmp_path, extra_env, expect_success=True):
    """`train_robot_bev_finetune.sh` 자리에 stub을 끼우고 probe를 돌린다.

    stub은 호출마다 한 줄씩 **덧붙이므로**, seed 여러 개를 돌리면 호출 순서가 그대로 남는다.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture = tmp_path / "captured.txt"
    fields = " ".join(f'"${{{name}-}}"' for name in _CAPTURED)
    bash_stub = bin_dir / "bash"
    bash_stub.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$(printf '%s\\t' {fields})\" >> \"${{CAPTURE_PATH}}\"\n"
    )
    bash_stub.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "CAPTURE_PATH": str(capture),
        "LAMBDA_R": "0.1526",
        "OUT_ROOT": str(tmp_path / "runs"),
    })
    env.update(extra_env)
    proc = subprocess.run(
        ["/bin/bash", "configs/probe_cumulative_ray_loss.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if expect_success:
        assert proc.returncode == 0, proc.stderr
    calls = []
    if capture.exists():
        for line in capture.read_text().splitlines():
            if not line.strip():
                continue
            values = line.split("\t")[:len(_CAPTURED)]
            calls.append(dict(zip(_CAPTURED, values)))
    return proc, calls


def test_default_is_still_the_seed_0_probe(tmp_path):
    """`SEEDS`를 주지 않으면 예전과 똑같이 seed 0 한 번이다."""
    _, calls = _run_probe(tmp_path, {})
    assert [c["SEED"] for c in calls] == ["0"]
    assert [c["RUN_NAME"] for c in calls] == ["E_cumulative_s0"]


def test_seeds_run_in_order_with_derived_run_names(tmp_path):
    """`SEEDS`의 각 seed가 `E_cumulative_s{N}`으로 한 번씩 돈다."""
    _, calls = _run_probe(tmp_path, {"SEEDS": "1 2 3 4"})
    assert [c["SEED"] for c in calls] == ["1", "2", "3", "4"]
    assert [c["RUN_NAME"] for c in calls] == [
        "E_cumulative_s1", "E_cumulative_s2", "E_cumulative_s3", "E_cumulative_s4",
    ]


def test_inherited_environment_cannot_pollute_the_frozen_config(tmp_path):
    """호출 셸의 예전 변수는 동결 설정을 하나도 바꾸지 못한다.

    `SEED`·`RUN_NAME`도 마찬가지다 -- 새어 들어오면 seed 4개가 전부 같은 이름으로 겹치거나
    이름과 실제 seed가 어긋난다.
    """
    polluted = {name: f"polluted_{name}" for name in FROZEN}
    polluted.update({
        "SEED": "7",
        "RUN_NAME": "stale_run",
        "SEEDS": "1 2",
        # 옛 probe 스윕이 남기는 값들. 대조군과 split·기하가 달라진다.
        "FRAME_SPLIT_FRACTION": "0.2",
        "FRAME_BLOCK_LEN": "12",
        "HEIGHT_BINS": "1",
        "NUM_EPOCHS": "5",
        "RANGE_LOSS_MODE": "arc_huber",
        "PROJECTION": "pinhole",
        "PINHOLE_HFOV_DEG": "120",
        "VAL_SEQUENCES": "rawos1",
    })
    _, calls = _run_probe(tmp_path, polluted)

    assert [c["SEED"] for c in calls] == ["1", "2"]
    assert [c["RUN_NAME"] for c in calls] == ["E_cumulative_s1", "E_cumulative_s2"]
    for call in calls:
        for name, expected in FROZEN.items():
            assert call[name] == expected, f"{name}이 오염됐다: {call[name]!r}"


def test_gpu_is_pinned_to_device_1(tmp_path):
    """GPU 0의 다른 실험을 건드리지 않도록 물리 인덱스 1로 고정한다."""
    _, calls = _run_probe(tmp_path, {"CUDA_VISIBLE_DEVICES": "0"})
    assert [c["CUDA_VISIBLE_DEVICES"] for c in calls] == ["1"]


def test_lambda_r_must_be_given(tmp_path):
    """`LAMBDA_R` 기본값을 두지 않는다 -- 빠뜨리면 보조항이 꺼진 채로 돈다."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = os.environ.copy()
    env.update({"OUT_ROOT": str(tmp_path / "runs")})
    env.pop("LAMBDA_R", None)
    proc = subprocess.run(
        ["/bin/bash", "configs/probe_cumulative_ray_loss.sh"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "LAMBDA_R" in proc.stderr


def test_existing_run_is_refused_before_any_training_starts(tmp_path):
    """이미 있는 seed가 큐 뒤쪽에 있어도 **한 런도 돌기 전에** 거부한다.

    완료된 `E_cumulative_s0`을 덮어쓰지 않는다는 계약이고, 앞 seed를 돌린 뒤 멈추면
    그 GPU 시간이 버려진다.
    """
    out_root = tmp_path / "runs"
    existing = out_root / "logs" / "E_cumulative_s3"
    existing.mkdir(parents=True)
    (existing / "events.out.tfevents.1").write_text("x")

    proc, calls = _run_probe(
        tmp_path, {"SEEDS": "1 2 3 4"}, expect_success=False,
    )
    assert proc.returncode != 0
    assert "E_cumulative_s3" in proc.stderr
    assert calls == []
