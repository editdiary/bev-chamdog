"""학습 길이 기본값(2026-10-02: 100 epoch)과 사전학습 길이 분리를 고정한다."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import run_paper_final_native_prior as native  # noqa: E402
from tools import run_paper_final_projection_prior as projection  # noqa: E402
from tools.paper_final_epochs import DEFAULT_NUM_EPOCHS, DEFAULT_PRETRAIN_EPOCHS  # noqa: E402


def test_the_campaign_default_is_100_epochs():
    assert DEFAULT_NUM_EPOCHS == 100
    assert DEFAULT_PRETRAIN_EPOCHS == 100


def _by_name(module, tmp_path, **kw):
    specs = module.build_run_specs(module.ARMS, [0], tmp_path, 100, "0", **kw)
    return {s.run_name: s for s in specs}


def test_pretrain_length_is_separate_and_the_finetune_starts_from_that_checkpoint(tmp_path):
    """사전학습 40 / 미세조정 100으로 나누면 사전학습 런만 40을 받고, 미세조정 팔은
    **40 epoch 사전학습 체크포인트**에서 시작해야 한다."""
    for module, pretrain, finetune in (
            (projection, "swscape_binary_pretrain_s0", "source_prior_s0"),
            (native, "swscape_native_pretrain_s0", "source_prior_native_s0")):
        specs = _by_name(module, tmp_path / module.__name__.split(".")[-1], pretrain_epochs=40)
        assert specs[pretrain].environment["NUM_EPOCHS"] == "40"
        assert specs[finetune].environment["NUM_EPOCHS"] == "100"
        assert specs[finetune].environment["INIT_CHECKPOINT"].endswith("model-000000040.pth")


def test_pretrain_length_defaults_to_the_finetune_length(tmp_path):
    specs = _by_name(projection, tmp_path)
    assert specs["swscape_binary_pretrain_s0"].environment["NUM_EPOCHS"] == "100"


def test_run_state_judges_each_run_by_its_own_length(tmp_path):
    """큐 길이(100)와 다른 길이(40)의 사전학습이 완주했으면 complete여야 한다 -- 큐 길이로
    판정하면 영원히 incomplete라 매번 다시 돈다."""
    spec = _by_name(projection, tmp_path, pretrain_epochs=40)["swscape_binary_pretrain_s0"]
    ckpt = Path(spec.environment["OUT_ROOT"]) / "ckpt" / spec.run_name
    ckpt.mkdir(parents=True)
    (ckpt / "model-000000040.pth").write_text("x")
    assert projection.run_state(spec, 100) == "complete"
