"""`configs/loss_effect.sh`가 각 칸에 **그 칸의 loss 설정**을 전달하는지 검증.

이 사다리는 "인접한 두 칸은 손잡이 하나만 다르다"가 전제다. 한 칸의 설정이 옆 칸으로 새면
그 계단에서 관찰된 변화를 어디에도 귀속시킬 수 없는데, 산출물만 보면 정상으로 보인다.

**[2026-10-02] 광선 보조항을 뺐다** -- 사다리는 `A_ce -> B_perset -> C_hard -> C_soft` 넷이고
어느 칸도 `λ_R > 0`을 받으면 안 된다(옛 `D_range`·`E_cumulative`는 주석으로만 남았다).
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


def test_no_rung_turns_the_ray_term_on(tmp_path):
    """사용자 결정(2026-10-02): 목적함수는 soft-boundary BCE까지다. 칸이 하나라도 `λ_R > 0`을
    받으면 trainer가 보조항을 만든다 -- 주석 처리한 옛 칸이 되살아나는 것을 여기서 막는다."""
    _, calls = _run_ladder(tmp_path, {})
    assert calls, "사다리가 한 칸도 돌지 않았다"
    for call in calls:
        assert float(call["LAMBDA_R"]) == 0.0, call["EXP_NAME"]


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
    """이미 있는 칸은 건너뛴다 -- 빠진 칸만 추가로 돌릴 수 있어야 한다."""
    out_root = tmp_path / "runs"
    for name in ("A_ce", "B_perset", "C_hard"):
        run_dir = out_root / "logs" / f"{name}_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "events.out.tfevents.1").write_text("x")
    _, calls = _run_ladder(tmp_path, {})
    assert [c["EXP_NAME"] for c in calls] == ["C_soft"]


def test_only_cells_selects_rungs_without_changing_their_settings(tmp_path):
    """`ONLY_CELLS`는 칸을 고르기만 하고 설정은 표에서 그대로 온다."""
    _, calls = _run_ladder(tmp_path, {"ONLY_CELLS": "A_ce C_soft", "NUM_EPOCHS": "100"})
    assert [c["EXP_NAME"] for c in calls] == ["A_ce", "C_soft"]
    for call in calls:
        loss, lambda_b, lambda_r, sigma_m, _ = LADDER[call["EXP_NAME"]]
        assert (call["LOSS"], call["LAMBDA_B"], call["LAMBDA_R"], call["SIGMA_M"]) \
            == (loss, lambda_b, lambda_r, sigma_m)
        assert call["NUM_EPOCHS"] == "100"


def test_probe_height_knobs_change_only_the_height(tmp_path):
    """높이 사전 실험 손잡이는 높이 셋만 바꾸고 칸 설정은 그대로 둔다. 기본값은 확정값이다."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, base = _run_ladder(tmp_path / "a", {"ONLY_CELLS": "C_soft"})
    _, probe = _run_ladder(tmp_path / "b", {"ONLY_CELLS": "C_soft", "PROBE_HEIGHT_BINS": "8",
                                            "PROBE_HEIGHT_MIN_M": "-2.0",
                                            "PROBE_HEIGHT_MAX_M": "2.0"})
    assert (base[0]["HEIGHT_BINS"], base[0]["HEIGHT_MIN_M"], base[0]["HEIGHT_MAX_M"]) \
        == ("4", "-0.25", "1.75")
    assert (probe[0]["HEIGHT_BINS"], probe[0]["HEIGHT_MIN_M"], probe[0]["HEIGHT_MAX_M"]) \
        == ("8", "-2.0", "2.0")
    for key in _CAPTURED:
        if key not in ("HEIGHT_BINS", "HEIGHT_MIN_M", "HEIGHT_MAX_M"):
            assert base[0][key] == probe[0][key], key
