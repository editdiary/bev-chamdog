"""산문 숫자 감사기가 **낡은 숫자를 놓치지 않는지** 고정한다.

이 도구가 조용히 0을 보고하면 논문이 데이터와 어긋난 채 작성된다.
"""
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.audit_paper_prose_numbers import (  # noqa: E402
    _csv_values, _rounded_sets, audit, prose_numbers,
)


def _package(root, csv_rows, prose, provenance=None):
    (root / "data").mkdir(parents=True, exist_ok=True)
    with (root / "data" / "t.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    (root / "REPORT.md").write_text(prose, encoding="utf-8")
    if provenance is not None:
        (root / "provenance").mkdir(parents=True, exist_ok=True)
        (root / "provenance" / "p.json").write_text(json.dumps(provenance), encoding="utf-8")
    return root


def _sets(root):
    return _rounded_sets(_csv_values(root))


def test_a_number_that_moved_is_reported_as_stale(tmp_path):
    old = _package(tmp_path / "old", [["metric", "value"], ["iou_free", "0.811862"]],
                   "우리 모델은 `iou_free` **0.8119**를 낸다.\n")
    new = _package(tmp_path / "new", [["metric", "value"], ["iou_free", "0.805400"]], "")
    r = audit(old, _sets(old), _sets(new))
    assert [t for _, _, t in r["stale"]] == ["0.8119"], r


def test_a_number_that_survived_is_ok(tmp_path):
    old = _package(tmp_path / "old", [["metric", "value"], ["iou_free", "0.811862"]],
                   "우리 모델은 `iou_free` **0.8119**를 낸다.\n")
    new = _package(tmp_path / "new", [["metric", "value"], ["iou_free", "0.811858"]], "")
    r = audit(old, _sets(old), _sets(new))
    assert r["stale"] == [] and r["ok"] == 1


def test_a_sign_flipped_citation_still_matches(tmp_path):
    """산문은 감소량을 `−0.0134`로 쓰고 CSV는 `-0.013354`로 적는다 -- 둘 다 맞는 것으로 본다."""
    rows = [["metric", "value"], ["diff", "-0.013354"]]
    old = _package(tmp_path / "old", rows, "핀홀이 **−0.0134** 진다.\n")
    r = audit(old, _sets(old), _sets(_package(tmp_path / "new", rows, "")))
    assert r["stale"] == [] and r["unsourced"] == []


def test_a_value_only_in_provenance_is_not_flagged(tmp_path):
    old = _package(tmp_path / "old", [["metric", "value"], ["x", "1.0"]],
                   "constant-map 기준선은 **0.5180**이다.\n",
                   provenance={"constant_map_baseline_iou_free": 0.5180230692525705})
    r = audit(old, _sets(old), _sets(old))
    assert r["unsourced"] == [] and r["stale"] == []


def test_configuration_constants_are_not_picked_up(tmp_path):
    """`λ_B=0.5`, `δ=0.30` 같은 설정값까지 잡으면 보고가 잡음이 된다."""
    old = _package(tmp_path / "old", [["metric", "value"], ["x", "1.0"]],
                   "δ=0.30, σ=0.10, λ_B=0.5, κ=1.0 으로 고정했다.\n")
    assert prose_numbers(old / "REPORT.md") == []


def test_an_unsourced_number_is_separated_from_a_stale_one(tmp_path):
    old = _package(tmp_path / "old", [["metric", "value"], ["iou_free", "0.811862"]],
                   "0.8119 는 데이터, 0.1234 는 어디에도 없다.\n")
    r = audit(old, _sets(old), _sets(_package(tmp_path / "new",
              [["metric", "value"], ["iou_free", "0.811862"]], "")))
    assert r["stale"] == []
    assert [t for _, _, t in r["unsourced"]] == ["0.1234"]
