"""watchdog이 실제로 멈춘 런을 죽이는지 고정한다.

이 장치가 조용히 동작하지 않으면 사람이 없는 8시간 동안 큐가 1런에서 서 있게 된다 --
그 사고를 겪었기 때문에 만든 코드다. 그러니 "있다"가 아니라 "죽인다"를 테스트한다.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.paper_final_run_command import (  # noqa: E402
    RETURNCODE_HARD_TIMEOUT,
    RETURNCODE_SILENCE,
    make_command_runner,
    run_training_command,
)


def test_a_normal_command_returns_its_own_code():
    r = run_training_command(["bash", "-c", "echo hi; exit 3"], echo=False)
    assert r.returncode == 3 and r.reason == "exited"


def test_silence_kills_a_hung_command():
    """출력 없이 멈춘 프로세스는 침묵 한도에서 죽는다."""
    start = time.time()
    r = run_training_command(["bash", "-c", "sleep 300"], echo=False,
                             silence_timeout_s=2, hard_timeout_s=300, poll_s=0.2)
    assert r.returncode == RETURNCODE_SILENCE, r
    assert r.reason == "silence"
    assert time.time() - start < 60, "한도를 한참 넘겨서야 죽었다"


def test_output_keeps_a_slow_command_alive():
    """계속 진행 중인 런을 침묵 한도로 잘못 죽이지 않는다."""
    r = run_training_command(
        ["bash", "-c", "for i in 1 2 3 4 5 6; do echo tick; sleep 0.5; done"],
        echo=False, silence_timeout_s=2, hard_timeout_s=300, poll_s=0.2)
    assert r.returncode == 0 and r.reason == "exited", r


def test_the_hard_limit_catches_a_chatty_command_that_never_ends():
    r = run_training_command(["bash", "-c", "while true; do echo tick; sleep 0.2; done"],
                             echo=False, silence_timeout_s=60, hard_timeout_s=2, poll_s=0.2)
    assert r.returncode == RETURNCODE_HARD_TIMEOUT and r.reason == "hard_timeout", r


def test_the_whole_process_group_dies_not_just_the_shell(tmp_path):
    """**손자 프로세스까지 죽는다.** bash만 죽이면 python이 GPU를 붙든 채 남는다."""
    pid_file = tmp_path / "grandchild.pid"
    script = f"bash -c 'echo $$ > {pid_file}; sleep 300' & sleep 300"
    r = run_training_command(["bash", "-c", script], echo=False,
                             silence_timeout_s=2, hard_timeout_s=300, poll_s=0.2)
    assert r.returncode == RETURNCODE_SILENCE
    pid = int(pid_file.read_text().strip())
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return  # 죽었다
        time.sleep(0.5)
    pytest.fail(f"손자 프로세스 {pid}가 살아남았다 -- 프로세스 그룹 종료가 동작하지 않는다")


def test_the_runner_writes_one_log_file_per_run(tmp_path):
    runner = make_command_runner(tmp_path, echo=False)
    runner(["bash", "-c", "echo hello-from-the-run"], env={"RUN_NAME": "final_s0"})
    log = tmp_path / "final_s0.log"
    assert log.exists() and "hello-from-the-run" in log.read_text()
