"""학습 하위 프로세스를 **반드시 끝나게** 실행한다.

**왜 필요한가.** 캠페인 러너 셋은 `subprocess.run(["bash", script])`을 timeout 없이 불렀다.
학습이 DataLoader worker 교착이나 GPU 대기로 멈추면 큐가 그 자리에서 영원히 선다 --
사람이 없는 동안 8시간짜리 큐가 1런에서 멈춰 있어도 알 방법이 없다(실제로 겪은 사고다).

두 가지 한도를 건다.

- **침묵 한도**(`silence_timeout_s`, 기본 20분). 학습은 epoch마다 줄을 찍는다.
  가장 느린 런도 epoch이 30초 미만이므로, 20분 동안 출력이 한 줄도 없으면 그건 진행이
  아니라 정지다. **이쪽이 주 감지기다** -- 몇 분 안에 잡는다.
- **총 한도**(`hard_timeout_s`, 기본 2시간). 출력은 나오는데 끝나지 않는 경우의 안전망.
  가장 긴 런(사전학셥 40 epoch)이 약 19분이라 6배 여유다.

**프로세스 그룹째 죽인다.** `bash script`가 `python`을 낳으므로 bash만 죽이면 학습이
GPU를 붙든 채 고아로 남고, 다음 런이 메모리 부족으로 연쇄 실패한다.
"""
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SILENCE_TIMEOUT_S = 20 * 60
DEFAULT_HARD_TIMEOUT_S = 2 * 60 * 60

#: `returncode`. 셸의 128+n 관례와 겹치지 않는 값을 쓴다.
RETURNCODE_SILENCE = 901
RETURNCODE_HARD_TIMEOUT = 902


@dataclass
class CommandResult:
    returncode: int
    seconds: float
    reason: str = "exited"


def _kill_group(proc):
    """프로세스 그룹 전체에 TERM, 안 죽으면 KILL. 고아 python이 GPU를 붙들지 않게."""
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_training_command(cmd, cwd=None, env=None, tee_path=None,
                         silence_timeout_s=DEFAULT_SILENCE_TIMEOUT_S,
                         hard_timeout_s=DEFAULT_HARD_TIMEOUT_S,
                         poll_s=10.0, echo=True):
    """`cmd`를 실행하고 두 한도 중 하나라도 걸리면 프로세스 그룹째 죽인다."""
    started = time.time()
    last_output = [started]
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace")

    sink = open(tee_path, "a", buffering=1) if tee_path else None

    def pump():
        for line in proc.stdout:
            last_output[0] = time.time()
            if echo:
                print(line, end="", flush=True)
            if sink:
                sink.write(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    reason = "exited"
    try:
        while True:
            try:
                proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if now - last_output[0] > silence_timeout_s:
                reason = "silence"
                break
            if now - started > hard_timeout_s:
                reason = "hard_timeout"
                break
        if reason != "exited":
            quiet = time.time() - last_output[0]
            print(f"\n!!! [watchdog] {reason}: "
                  f"{time.time() - started:.0f}초 경과, 마지막 출력 {quiet:.0f}초 전. "
                  f"프로세스 그룹을 종료한다.", flush=True)
            _kill_group(proc)
    finally:
        reader.join(timeout=15)
        if proc.stdout:
            proc.stdout.close()
        if sink:
            sink.close()

    code = {"silence": RETURNCODE_SILENCE,
            "hard_timeout": RETURNCODE_HARD_TIMEOUT}.get(reason, proc.returncode)
    return CommandResult(returncode=code, seconds=time.time() - started, reason=reason)


def make_command_runner(log_dir, **kw):
    """러너의 `command_runner=`에 그대로 넘길 수 있는 호출체를 만든다.

    런마다 파일 하나에 stdout을 남긴다 -- 텐서보드 이벤트만으로는 무엇을 하다 멈췄는지
    알 수 없다.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def runner(cmd, cwd=None, env=None, **_ignored):
        counter["n"] += 1
        name = (env or {}).get("RUN_NAME") or f"run{counter['n']:03d}"
        return run_training_command(cmd, cwd=cwd, env=env,
                                    tee_path=log_dir / f"{name}.log", **kw)

    return runner
