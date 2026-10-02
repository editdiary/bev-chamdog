import json

from tools.clean_incomplete_runs import find_incomplete, main


def _run(root, name, epochs, complete):
    log = root / "logs" / name
    log.mkdir(parents=True)
    (log / "config.json").write_text(json.dumps({"num_epochs": epochs}))
    ck = root / "ckpt" / name
    ck.mkdir(parents=True)
    if complete:
        (ck / f"model-{epochs:09d}.pth").write_text("x")


def test_finds_only_runs_without_the_final_checkpoint_and_deletes_them(tmp_path):
    _run(tmp_path / "armA", "a_s0", 100, True)
    _run(tmp_path / "armA", "a_s1", 100, False)
    _run(tmp_path / "pre", "p_s0", 40, True)          # 자기 config의 길이로 판정한다
    (tmp_path / "armA" / "analysis" / "logs").mkdir(parents=True)   # 분석 폴더는 무시
    found = find_incomplete(tmp_path)
    assert [f[0].name for f in found] == ["a_s1"]
    record = tmp_path / "deleted.txt"
    assert main(str(tmp_path), record=str(record)) == 1
    assert not (tmp_path / "armA" / "logs" / "a_s1").exists()
    assert (tmp_path / "armA" / "logs" / "a_s0").exists()
    assert "a_s1" in record.read_text()


def test_dry_run_deletes_nothing(tmp_path):
    _run(tmp_path, "x_s0", 100, False)
    assert main(str(tmp_path), dry_run=True) == 1
    assert (tmp_path / "logs" / "x_s0").exists()
