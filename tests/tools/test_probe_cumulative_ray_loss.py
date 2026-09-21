"""cumulative ray probe 셸이 대조군의 동결 설정을 실제 child process에 전달하는지 검증."""
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_probe_overrides_inherited_frame_split_settings(tmp_path):
    """호출 셸의 예전 probe 변수가 고정 sequence split을 바꾸면 안 된다."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "captured.txt"
    bash_stub = bin_dir / "bash"
    bash_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"${FRAME_SPLIT_FRACTION-}\" \"${FRAME_BLOCK_LEN-}\" "
        "> \"${CAPTURE_PATH}\"\n"
    )
    bash_stub.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "CAPTURE_PATH": str(capture),
        "LAMBDA_R": "0.1526",
        "OUT_ROOT": str(tmp_path / "runs"),
        "FRAME_SPLIT_FRACTION": "0.2",
        "FRAME_BLOCK_LEN": "12",
    })
    subprocess.run(
        ["/bin/bash", "configs/probe_cumulative_ray_loss.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert capture.read_text().splitlines() == ["0.0", "0"]
