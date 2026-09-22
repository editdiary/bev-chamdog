"""결과 트리 순수성 검사기가 **옛 트리 참조를 놓치지 않는지** 고정한다.

이 검사가 조용히 통과하면 옛 런과 새 런이 섞인 표가 논문에 들어간다.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.check_results_tree_purity import scan  # noqa: E402


def _tree(root, files):
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def test_a_pure_tree_passes(tmp_path):
    root = _tree(tmp_path, {"a/m.json": '{"output_root": "runs/new_root/01"}\n'})
    assert scan(root, {"runs/new_root"}, set()) == []


def test_a_stale_reference_is_found(tmp_path):
    root = _tree(tmp_path, {
        "a/m.json": '{"control_arm": "runs/paper_final/01_overall/fixed_split"}\n'})
    bad = scan(root, {"runs/paper_final_cumulative"}, set())
    assert [b[2] for b in bad] == ["runs/paper_final"]


def test_the_new_root_is_not_confused_with_its_prefix(tmp_path):
    """`runs/paper_final`이 `runs/paper_final_cumulative`의 접두사라 정규식이 흔들리기 쉽다."""
    root = _tree(tmp_path, {"a/m.json": '"runs/paper_final_cumulative/01_overall"\n'})
    assert scan(root, {"runs/paper_final_cumulative"}, set()) == []


def test_a_deliberately_referenced_tree_can_be_ignored(tmp_path):
    """03은 일부러 `runs/loss_effect`를 가리킨다 -- 그건 섞임이 아니다."""
    root = _tree(tmp_path, {"a/m.json": '"runs/loss_effect/logs/E_cumulative_s0"\n'})
    assert scan(root, {"runs/new"}, {"runs/loss_effect"}) == []
    assert len(scan(root, {"runs/new"}, set())) == 1


def test_binary_files_are_not_scanned(tmp_path):
    root = tmp_path
    (root / "ckpt").mkdir()
    (root / "ckpt" / "model.pth").write_bytes(b"runs/paper_final\x00\x01binary")
    assert scan(root, {"runs/new"}, set()) == []
