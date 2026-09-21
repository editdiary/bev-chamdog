"""학습 때 쓴 투영 모델을 런에서 되찾는 계약을 고정한다.

**왜 테스트가 필요한가.** 확률맵을 다시 내보낼 때 투영이 학습 때와 다르면 모델이 본 적 없는
입력을 받는다. 그 결과는 "조금 나쁜 숫자"가 아니라 **무결성 게이트 실패**로 나타나는데,
원인이 투영 불일치인지 실제 재현 실패인지 구분하기 어렵다. 그래서 되찾기가 (a) 맞게 읽고
(b) 섞인 런을 거부하는 것을 고정한다.
"""
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.models.virtual_pinhole import projection_for_run_dirs  # noqa: E402


def _run(tmp_path, name, **config):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_reads_pinhole_projection_and_fov(tmp_path):
    dirs = [_run(tmp_path, f"pinhole120_s{s}", projection="pinhole", pinhole_hfov_deg=120)
            for s in range(3)]
    assert projection_for_run_dirs(dirs, quiet=True) == ("pinhole", 120.0)


def test_defaults_to_the_frozen_setting_for_older_runs(tmp_path):
    """`projection` 키가 없는 옛 런은 확정 설정(DS-native)이다 -- 그때는 그것뿐이었다."""
    dirs = [_run(tmp_path, "final_s0", formulation="binary")]
    assert projection_for_run_dirs(dirs, quiet=True) == ("ds_native", None)


def test_mixed_projections_are_refused(tmp_path):
    """섞인 런을 한 번에 다루면 절반이 틀린 입력을 받는다. 조용히 하나를 고르면 안 된다."""
    dirs = [_run(tmp_path, "a_s0", projection="ds_native"),
            _run(tmp_path, "b_s0", projection="pinhole", pinhole_hfov_deg=120)]
    with pytest.raises(SystemExit, match="투영 모델이 섞인"):
        projection_for_run_dirs(dirs, quiet=True)


def test_different_fovs_are_also_refused(tmp_path):
    """같은 `pinhole`이어도 화각이 다르면 다른 입력이다."""
    dirs = [_run(tmp_path, "a_s0", projection="pinhole", pinhole_hfov_deg=120),
            _run(tmp_path, "b_s0", projection="pinhole", pinhole_hfov_deg=150)]
    with pytest.raises(SystemExit, match="투영 모델이 섞인"):
        projection_for_run_dirs(dirs, quiet=True)


def test_missing_directories_are_skipped(tmp_path):
    """체크포인트가 지워진 런이 섞여 있어도 도구가 죽지 않아야 한다."""
    dirs = [tmp_path / "gone", _run(tmp_path, "ok_s0", projection="pinhole",
                                    pinhole_hfov_deg=150)]
    assert projection_for_run_dirs(dirs, quiet=True) == ("pinhole", 150.0)
