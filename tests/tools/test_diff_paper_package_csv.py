"""패키지 CSV 대조기가 **바뀐 칸을 놓치지 않는지** 고정한다.

이 도구를 믿고 "자릿수만 바뀌었다"고 판단하게 되므로, 놓치는 쪽이 치명적이다.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.diff_paper_package_csv import diff_csv  # noqa: E402


def _csv(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


HEADER = "arm,metric,value,verdict\n"


def test_identical_files_report_nothing(tmp_path):
    body = HEADER + "a,iou_free,0.8119,arm better\n"
    problems, changed = diff_csv(_csv(tmp_path / "o.csv", body),
                                 _csv(tmp_path / "n.csv", body), 0.02, 1e-4)
    assert problems == [] and changed == []


def test_a_small_numeric_move_is_below_the_tolerance(tmp_path):
    old = HEADER + "a,iou_free,0.8119,arm better\n"
    new = HEADER + "a,iou_free,0.8121,arm better\n"
    _, changed = diff_csv(_csv(tmp_path / "o.csv", old),
                          _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert changed == []


def test_a_large_numeric_move_is_reported_with_its_size(tmp_path):
    old = HEADER + "a,iou_free,0.8119,arm better\n"
    new = HEADER + "a,iou_free,0.7500,arm better\n"
    _, changed = diff_csv(_csv(tmp_path / "o.csv", old),
                          _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert len(changed) == 1
    row, col, a, b, rel = changed[0]
    assert (row, col, a, b) == (2, "value", "0.8119", "0.7500")
    assert rel > 0.05


def test_a_flipped_verdict_is_reported_even_though_it_is_text(tmp_path):
    """**판정이 뒤집히는 칸이 이 도구의 존재 이유다.**"""
    old = HEADER + "a,iou_free,0.8119,arm better\n"
    new = HEADER + "a,iou_free,0.8119,control better\n"
    _, changed = diff_csv(_csv(tmp_path / "o.csv", old),
                          _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert [(c[1], c[2], c[3]) for c in changed] == [
        ("verdict", "arm better", "control better")]


def test_near_zero_values_do_not_explode(tmp_path):
    """0 근처에서 상대 변화가 무한대가 되어 모든 칸이 보고되면 도구가 쓸모없어진다."""
    old = HEADER + "a,range_bias,0.000001,-\n"
    new = HEADER + "a,range_bias,0.000002,-\n"
    _, changed = diff_csv(_csv(tmp_path / "o.csv", old),
                          _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert changed == []


def test_a_changed_header_stops_the_comparison(tmp_path):
    old = HEADER + "a,iou_free,0.8,x\n"
    new = "arm,metric,value\na,iou_free,0.8\n"
    problems, changed = diff_csv(_csv(tmp_path / "o.csv", old),
                                 _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert problems and changed == []


def test_a_different_row_count_is_reported(tmp_path):
    old = HEADER + "a,iou_free,0.8,x\nb,iou_free,0.7,x\n"
    new = HEADER + "a,iou_free,0.8,x\n"
    problems, _ = diff_csv(_csv(tmp_path / "o.csv", old),
                           _csv(tmp_path / "n.csv", new), 0.02, 1e-4)
    assert any("행 수" in p for p in problems)
