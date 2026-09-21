"""`configs/loss_effect.sh`가 각 칸에 **그 칸의 loss 설정**을 전달하는지 검증.

이 사다리는 "인접한 두 칸은 손잡이 하나만 다르다"가 전제다. 한 칸의 설정이 옆 칸으로 새면
그 계단에서 관찰된 변화를 어디에도 귀속시킬 수 없는데, 산출물만 보면 정상으로 보인다.

특히 `D_range`(`arc_huber`, λ_R=0.3)와 `E_cumulative`(`cumulative_l1`, λ_R=0.15)는 **같은
계단의 대체 팔**이라 `RANGE_LOSS_MODE`가 섞이면 두 팔이 구별되지 않는다.
"""
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# 칸 이름 -> (LOSS, LAMBDA_B, LAMBDA_R, SIGMA_M, RANGE_LOSS_MODE)
LADDER = {
    "A_ce": ("weighted_ce", "0.0", "0.0", "0.10", "arc_huber"),
    "B_perset": ("soft_boundary", "0.0", "0.0", "0.10", "arc_huber"),
    "C_hard": ("soft_boundary", "0.5", "0.0", "0.001", "arc_huber"),
    "C_soft": ("soft_boundary", "0.5", "0.0", "0.10", "arc_huber"),
    "D_range": ("soft_boundary", "0.5", "0.3", "0.10", "arc_huber"),
    "E_cumulative": ("soft_boundary", "0.5", "0.15", "0.10", "cumulative_l1"),
}

# 칸과 무관하게 항상 같아야 하는 값. 여기에 `FRAME_SPLIT_*`가 있는 이유는 그것이 새면
# 런이 **다른 split을 보면서도 같은 이름으로** 쌓이기 때문이다.
FROZEN = {
    "NUM_EPOCHS": "40",
    "SOFT_TARGET": "gaussian",
    "DELTA_M": "0.30",
    "SIGMA_ALPHA": "None",
    "BAND_KAPPA": "1.0",
    "LABEL_EPS": "0.0",
    "DELTA_R_M": "0.15",
    "DELTA_R_OVER_M": "None",
    "HUBER_BETA_M": "0.15",
    "HEIGHT_BINS": "4",
    "HEIGHT_MIN_M": "-0.25",
    "HEIGHT_MAX_M": "1.75",
    "FORMULATION": "binary",
    "ENCODER_TYPE": "res101",
    "AUGMENT": "True",
    "INIT_CHECKPOINT": "none",
    "PROJECTION": "ds_native",
    "PINHOLE_HFOV_DEG": "None",
    "PIXEL_CONVENTION": "pixel_center",
    "PIXEL_OFFSET": "0.0",
    "FRAME_SPLIT_FRACTION": "0.0",
    "FRAME_BLOCK_LEN": "0",
    "TRAIN_SEQUENCES": "raws2,raws3,rawos1,rawos2,rawos4",
    "VAL_SEQUENCES": "raws1,rawos3",
}

_CAPTURED = ["EXP_NAME", "RUN_NAME", "SEED", "LOSS", "LAMBDA_B", "LAMBDA_R", "SIGMA_M",
             "RANGE_LOSS_MODE", *FROZEN]


def _run_ladder(tmp_path, extra_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture = tmp_path / "captured.txt"
    fields = " ".join(f'"${{{name}-}}"' for name in _CAPTURED)
    stub = bin_dir / "bash"
    stub.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$(printf '%s\\t' {fields})\" >> \"${{CAPTURE_PATH}}\"\n"
    )
    stub.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "CAPTURE_PATH": str(capture),
        "OUT_ROOT": str(tmp_path / "runs"),
        "SEEDS": "0",
    })
    env.update(extra_env)
    proc = subprocess.run(["/bin/bash", "configs/loss_effect.sh"], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    calls = []
    for line in capture.read_text().splitlines():
        if line.strip():
            calls.append(dict(zip(_CAPTURED, line.split("\t")[:len(_CAPTURED)])))
    return proc, calls


def test_every_cell_gets_its_own_loss_settings(tmp_path):
    _, calls = _run_ladder(tmp_path, {})
    assert [c["EXP_NAME"] for c in calls] == list(LADDER)
    for call in calls:
        loss, lambda_b, lambda_r, sigma_m, mode = LADDER[call["EXP_NAME"]]
        assert call["RUN_NAME"] == f"{call['EXP_NAME']}_s0"
        assert (call["LOSS"], call["LAMBDA_B"], call["LAMBDA_R"],
                call["SIGMA_M"], call["RANGE_LOSS_MODE"]) == (loss, lambda_b, lambda_r,
                                                              sigma_m, mode)


def test_the_two_range_arms_differ_only_in_the_auxiliary_term(tmp_path):
    """`D_range`와 `E_cumulative`는 `C_soft`에서 보조항 하나만 켠 **대체 팔**이다."""
    _, calls = _run_ladder(tmp_path, {})
    by_name = {c["EXP_NAME"]: c for c in calls}
    shared = [k for k in _CAPTURED if k not in
              ("EXP_NAME", "RUN_NAME", "LAMBDA_R", "RANGE_LOSS_MODE")]
    for name in ("D_range", "E_cumulative"):
        for key in shared:
            assert by_name[name][key] == by_name["C_soft"][key], (
                f"{name}이 C_soft와 {key}에서 다르다 -- 한 계단에 손잡이 하나가 깨졌다")
    assert (by_name["D_range"]["LAMBDA_R"], by_name["D_range"]["RANGE_LOSS_MODE"]) \
        == ("0.3", "arc_huber")
    assert (by_name["E_cumulative"]["LAMBDA_R"], by_name["E_cumulative"]["RANGE_LOSS_MODE"]) \
        == ("0.15", "cumulative_l1")


def test_inherited_environment_cannot_pollute_the_ladder(tmp_path):
    """호출 셸의 예전 스윕 변수가 칸 설정이나 동결 설정을 바꾸지 못한다."""
    polluted = {name: f"polluted_{name}" for name in FROZEN}
    polluted.update({
        "LOSS": "weighted_ce", "LAMBDA_B": "0.0", "LAMBDA_R": "9.9", "SIGMA_M": "0.5",
        "RANGE_LOSS_MODE": "cumulative_l1", "RUN_NAME": "stale", "EXP_NAME": "stale",
        "FRAME_SPLIT_FRACTION": "0.2", "FRAME_BLOCK_LEN": "12", "HEIGHT_BINS": "1",
        "PROJECTION": "pinhole", "PINHOLE_HFOV_DEG": "120", "NUM_EPOCHS": "40",
    })
    _, calls = _run_ladder(tmp_path, polluted)
    assert [c["EXP_NAME"] for c in calls] == list(LADDER)
    for call in calls:
        loss, lambda_b, lambda_r, sigma_m, mode = LADDER[call["EXP_NAME"]]
        assert call["LOSS"] == loss and call["LAMBDA_R"] == lambda_r
        assert call["RANGE_LOSS_MODE"] == mode
        for key, expected in FROZEN.items():
            assert call[key] == expected, f"{key}가 오염됐다: {call[key]!r}"


def test_existing_runs_are_skipped_so_only_new_cells_train(tmp_path):
    """이미 있는 칸은 건너뛴다 -- `E_cumulative`만 추가로 돌릴 수 있어야 한다."""
    out_root = tmp_path / "runs"
    for name in ("A_ce", "B_perset", "C_hard", "C_soft", "D_range"):
        run_dir = out_root / "logs" / f"{name}_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "events.out.tfevents.1").write_text("x")
    _, calls = _run_ladder(tmp_path, {})
    assert [c["EXP_NAME"] for c in calls] == ["E_cumulative"]
