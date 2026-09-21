"""실험 02 orchestrator의 큐 판정을 고정한다.

**왜 이 테스트가 있나.** 2026-09-21에 `source_prior` 5런이 통째로 실행되지 않았다.
계획 단계에서 "짝지은 사전학습 체크포인트가 없다"로 막혔는데, 실행 루프가 그 판정을
그대로 믿고 건너뛰었기 때문이다 -- 그런데 그 체크포인트는 **같은 큐의 앞 팔이** 만들어
주는 것이라, 계획 시점에 없는 것이 정상이었다. 로그에는 `[skip]`조차 남지 않아
세 시간 뒤에야 알아챘다.
"""
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.run_paper_final_projection_prior import (  # noqa: E402
    build_run_specs,
    run_queue,
    run_state,
)


class _Runner:
    """`bash <script>` 호출을 기록하고, 사전학습이면 체크포인트를 실제로 만든다."""

    def __init__(self, num_epochs=40, fail=()):
        self.calls, self.num_epochs, self.fail = [], num_epochs, set(fail)

    def __call__(self, argv, cwd=None, env=None):
        name = env["RUN_NAME"]
        self.calls.append(name)
        if name in self.fail:
            return type("R", (), {"returncode": 1})()
        root = Path(env["OUT_ROOT"]) if "OUT_ROOT" in env else Path(env["CKPT_DIR"]).parent
        ckpt = root / "ckpt" / name
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / f"model-{self.num_epochs:09d}.pth").write_bytes(b"")
        (root / "logs" / name).mkdir(parents=True, exist_ok=True)
        return type("R", (), {"returncode": 0})()


def _specs(tmp_path, arms=("source_pretrain", "source_prior"), seeds=(0, 1)):
    return build_run_specs(arms, list(seeds), tmp_path, num_epochs=40, gpu="0")


def test_source_prior_runs_after_its_pretrain_in_the_same_queue(tmp_path):
    """**회귀 테스트.** 같은 큐 안에서 앞 팔이 만든 체크포인트를 뒤 팔이 써야 한다."""
    runner = _Runner()
    status = run_queue(_specs(tmp_path), tmp_path, 40, command_runner=runner)

    assert runner.calls == [
        "swscape_binary_pretrain_s0", "swscape_binary_pretrain_s1",
        "source_prior_s0", "source_prior_s1",
    ]
    assert all(status[n]["decision"] == "done" for n in runner.calls)


def test_source_prior_is_skipped_when_its_pretrain_never_runs(tmp_path):
    """사전학습을 요청하지 않았으면 `source_prior`는 돌지 않는다 -- 그리고 기록이 남는다."""
    specs = _specs(tmp_path, arms=("source_prior",))
    runner = _Runner()
    status = run_queue(specs, tmp_path, 40, command_runner=runner)

    assert runner.calls == []
    for name in ("source_prior_s0", "source_prior_s1"):
        assert status[name]["decision"] == "blocked_missing_input"
        assert status[name]["blockers"], "왜 막혔는지 파일에서 읽을 수 있어야 한다"


def test_each_seed_transfers_its_own_paired_checkpoint(tmp_path):
    """시드 1:1 짝짓기가 깨지면 산포가 과소평가된다(원장 §6.3)."""
    for spec in _specs(tmp_path):
        if spec.arm == "source_prior":
            assert spec.environment["INIT_CHECKPOINT"].endswith(
                f"swscape_binary_pretrain_s{spec.seed}/model-000000040.pth")


def test_completed_runs_are_skipped_and_incomplete_ones_are_not_overwritten(tmp_path):
    """`01`과 같은 계약이다 -- 끝난 런은 건너뛰고, 반쯤 된 런은 덮어쓰지 않는다."""
    specs = _specs(tmp_path, arms=("pinhole120",), seeds=(0, 1))
    done = Path(specs[0].environment["CKPT_DIR"]) / "pinhole120_s0"
    done.mkdir(parents=True)
    (done / "model-000000040.pth").write_bytes(b"")
    half = Path(specs[1].environment["LOG_DIR"]) / "pinhole120_s1"
    half.mkdir(parents=True)

    assert run_state(specs[0], 40) == "complete"
    assert run_state(specs[1], 40) == "incomplete"
    runner = _Runner()
    status = run_queue(specs, tmp_path, 40, command_runner=runner)
    assert runner.calls == []
    assert status["pinhole120_s0"]["decision"] == "skip_complete"
    assert status["pinhole120_s1"]["decision"] == "blocked_incomplete"


def test_status_file_is_merged_across_separate_invocations(tmp_path):
    """팔을 나눠 돌려도 앞 호출의 기록이 남아야 한다 -- 왜 안 돌았는지의 유일한 근거다."""
    run_queue(_specs(tmp_path, arms=("pinhole120",)), tmp_path, 40, plan_only=True)
    run_queue(_specs(tmp_path, arms=("pinhole150",)), tmp_path, 40, plan_only=True)
    status = json.loads((tmp_path / "queue_status.json").read_text())
    assert {"pinhole120_s0", "pinhole150_s0"} <= set(status)


def test_a_failed_run_stops_the_queue(tmp_path):
    """실패를 무시하고 계속 돌면 뒤 팔이 잘못된 입력 위에 쌓인다."""
    specs = _specs(tmp_path, seeds=(0, 1))
    runner = _Runner(fail={"swscape_binary_pretrain_s0"})
    status = run_queue(specs, tmp_path, 40, command_runner=runner)
    assert runner.calls == ["swscape_binary_pretrain_s0"]
    assert status["swscape_binary_pretrain_s0"]["decision"] == "failed"


@pytest.mark.parametrize("arm, expected", [
    ("pinhole120", "120.0"),
    ("pinhole150", "150.0"),
])
def test_pinhole_arms_always_carry_their_fov(tmp_path, arm, expected):
    """화각이 빠지면 셸이 멈추지만, 애초에 orchestrator가 빠뜨리지 않아야 한다."""
    for spec in _specs(tmp_path, arms=(arm,)):
        assert spec.environment["PROJECTION"] == "pinhole"
        assert spec.environment["PINHOLE_HFOV_DEG"] == expected
        assert spec.environment["INIT_CHECKPOINT"] == "none"
