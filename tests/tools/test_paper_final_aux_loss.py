"""캠페인 세 러너가 **같은** 보조항으로 런을 만드는지 고정한다.

**왜 이 테스트가 있나.** 2026-09-22까지 `tools/train_synwoodscape.py`는 `range_loss_mode`를
받지 않았고 두 사전학습 셸은 `--lambda_r=0.3`을 하드코딩하고 있었다. 그 상태에서 보조항
형태를 바꾸면 **미세조정만 바뀌고 사전학습은 옛 형태로 남는다** -- 로그를 봐도 티가 나지
않는다. 이 테스트는 그 절반 적용을 막는다.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.paper_final_aux_loss import AUX_LOSS_ENV, DEFAULT_AUX_LOSS, aux_loss_env  # noqa: E402
from tools import run_paper_final_native_prior as native  # noqa: E402
from tools import run_paper_final_overall as overall  # noqa: E402
from tools import run_paper_final_projection_prior as projection  # noqa: E402


def _overall_envs(mode, tmp_path):
    specs = overall.build_run_specs("fixed", [0])
    return {s.run_name: overall.training_environment(s, tmp_path, 40, "0", mode) for s in specs}


def _spec_envs(module, tmp_path, mode, **kw):
    specs = module.build_run_specs(module.ARMS, [0], tmp_path, 40, "0", mode, **kw)
    return {s.run_name: s.environment for s in specs}


def _all_envs(tmp_path, mode):
    envs = {}
    envs.update(_overall_envs(mode, tmp_path / "a"))
    envs.update(_spec_envs(projection, tmp_path / "b", mode))
    envs.update(_spec_envs(native, tmp_path / "c", mode))
    return envs


@pytest.mark.parametrize("mode", sorted(AUX_LOSS_ENV))
def test_every_run_of_the_campaign_carries_the_same_aux_term(mode, tmp_path):
    """**사전학습 런도 포함해서** 모든 런이 같은 보조항 값을 받는다."""
    envs = _all_envs(tmp_path, mode)
    # 시드 하나 · 고정 split일 때 캠페인이 만드는 런 전부. 러너가 하나 빠지면 여기서 걸린다.
    assert sorted(envs) == [
        "final_s0",
        "pinhole120_s0", "pinhole150_s0",
        "source_prior_native_s0", "source_prior_s0",
        "swscape_binary_pretrain_s0", "swscape_native_pretrain_s0",
    ], sorted(envs)
    expected = aux_loss_env(mode)
    for run_name, env in sorted(envs.items()):
        for key, value in expected.items():
            assert env.get(key) == value, (
                f"{run_name}의 {key}가 {env.get(key)!r}다 (기대 {value!r}). "
                f"사전학습과 미세조정이 다른 보조항으로 갈리면 캠페인 비교가 깨진다.")


def test_the_two_pretrain_runs_are_covered_too():
    """사전학습 런 이름이 실제로 목록에 있는지 -- 위 테스트가 빈 집합을 통과하지 않게."""
    assert "source_pretrain" in projection.ARMS
    assert "source_pretrain_native" in native.ARMS


def test_the_default_drops_the_ray_term_entirely(tmp_path):
    """[2026-10-02 사용자 결정] 다음 캠페인은 soft-boundary BCE까지만 쓴다. 기본값으로 러너를
    실행하면 **모든 런**(사전학습 포함)이 λ_R = 0을 받아 trainer가 항을 만들지도 않는다."""
    assert DEFAULT_AUX_LOSS == "none"
    for run_name, env in _all_envs(tmp_path, DEFAULT_AUX_LOSS).items():
        assert float(env["LAMBDA_R"]) == 0.0, run_name


def test_arc_huber_still_reproduces_the_pre_2026_09_22_campaign(tmp_path):
    """옛 캠페인(`runs/paper_final/`) 재현값은 남아 있어야 한다."""
    envs = _all_envs(tmp_path, "arc_huber")
    for env in envs.values():
        assert env["LAMBDA_R"] == "0.3"
        assert env["DELTA_R_M"] == "0.15"
        assert env["HUBER_BETA_M"] == "0.15"


def test_cumulative_matches_the_loss_effect_ladder():
    """`runs/loss_effect`의 `E_cumulative` config와 글자 그대로 같아야 한다."""
    env = aux_loss_env("cumulative_l1")
    assert env["RANGE_LOSS_MODE"] == "cumulative_l1"
    assert env["LAMBDA_R"] == "0.15", "형태가 다르면 gradient 크기도 달라 λ_R을 재보정한 값이다"


def test_an_unknown_mode_is_refused():
    with pytest.raises(SystemExit):
        aux_loss_env("arc_hubber")


@pytest.mark.parametrize("shell", (
    "configs/train_synwoodscape_binary_pretrain.sh",
    "configs/train_synwoodscape_native_pretrain.sh",
))
def test_the_pretrain_shells_forward_the_mode(shell):
    """셸이 값을 하드코딩하지 않고 환경변수를 넘긴다."""
    text = (REPO_ROOT / shell).read_text()
    assert "--range_loss_mode=" in text, f"{shell}이 range_loss_mode를 넘기지 않는다"
    assert '--lambda_r="${LAMBDA_R}"' in text, f"{shell}이 λ_R을 하드코딩하고 있다"
    assert 'LAMBDA_R="${LAMBDA_R:-0.0}"' in text, f"{shell}의 기본값이 보조항을 켠다"


def test_the_synwoodscape_trainer_accepts_the_flag():
    """trainer가 플래그를 실제로 받는다 -- 셸이 넘겨도 trainer가 거부하면 런이 죽는다."""
    out = subprocess.run([sys.executable, "tools/train_synwoodscape.py", "--help"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert "range_loss_mode" in out.stdout + out.stderr
