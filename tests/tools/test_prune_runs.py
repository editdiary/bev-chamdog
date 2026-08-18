"""정리 도구의 선택 규칙.

이 도구가 틀리면 **복구 수단이 재학습밖에 없다.** 특히 `model_best`는 재채점·시각화가
의존하는 유일한 산출물이라 실수로 지워지면 그 실험이 사라진다. 그래서 어떤 파일이
`keep`/`remove`로 가는지를 이름 규약 수준에서 못박는다.
"""
import pytest

from tools.prune_runs import BEST_PREFIX, classify, select_runs


def _names(group):
    from pathlib import Path
    return sorted(Path(p).name for p in group)


def test_best_is_kept_and_periodic_is_removed():
    paths = ["r/model-000000010.pth", "r/model-000000060.pth", "r/model_best-000000035.pth"]

    groups = classify(paths)

    assert _names(groups["keep"]) == ["model_best-000000035.pth"]
    assert _names(groups["remove"]) == ["model-000000010.pth", "model-000000060.pth"]


def test_best_is_not_mistaken_for_a_periodic_checkpoint():
    """`model_best-`는 `model`로 시작한다. 접두사 검사 순서가 뒤바뀌면 best가 주기 저장으로
    분류되어 지워진다 -- 이 도구에서 가장 비싼 실수다."""
    groups = classify(["r/model_best-000000048.pth"])

    assert _names(groups["keep"]) == ["model_best-000000048.pth"]
    assert groups["remove"] == []
    assert BEST_PREFIX.startswith("model")


def test_non_checkpoint_files_are_never_touched():
    """TensorBoard 이벤트와 split 목록은 실험의 결론이 담긴 파일이고 크기가 작다.
    이 도구가 모르는 파일도 손대지 않아야 한다."""
    paths = ["r/events.out.tfevents.123.host", "r/split_train_ids.txt", "r/notes.md",
             "r/model-000000010.pth"]

    groups = classify(paths)

    assert _names(groups["keep"]) == ["events.out.tfevents.123.host", "notes.md",
                                      "split_train_ids.txt"]
    assert _names(groups["remove"]) == ["model-000000010.pth"]


def test_unknown_pth_names_are_kept_not_removed():
    """규약을 벗어난 이름(예: 손으로 복사해 둔 체크포인트)은 지우지 않는다 -- 모르는 것을
    지우는 쪽으로 기울면 안 된다."""
    groups = classify(["r/final_for_deployment.pth", "r/ema-000000060.pth"])

    assert groups["remove"] == []
    assert len(groups["keep"]) == 2


def test_keep_best_false_removes_the_best_too():
    """런을 통째로 버릴 때 쓰는 경로. 기본값이 아니어야 한다."""
    paths = ["r/model-000000010.pth", "r/model_best-000000035.pth"]

    assert _names(classify(paths, keep_best=False)["remove"]) == [
        "model-000000010.pth", "model_best-000000035.pth",
    ]
    assert _names(classify(paths)["remove"]) == ["model-000000010.pth"]


def test_run_selection_uses_shell_glob_rules(tmp_path):
    for name in ("ft_wd1e-4_res101", "ft_wd5e-1_res101", "ft_scratch_res101", "not_a_dir.txt"):
        if name.endswith(".txt"):
            (tmp_path / name).write_text("x")
        else:
            (tmp_path / name).mkdir()

    assert [p.name for p in select_runs([tmp_path], "ft_wd*")] == [
        "ft_wd1e-4_res101", "ft_wd5e-1_res101",
    ]
    assert len(select_runs([tmp_path], "*")) == 3           # 파일은 제외된다
    assert select_runs([tmp_path / "missing"], "*") == []    # 없는 경로는 조용히 건너뛴다


def test_dry_run_is_the_default_and_deletes_nothing(tmp_path, capsys):
    from tools.prune_runs import main

    run_dir = tmp_path / "ft_demo_res101"
    run_dir.mkdir()
    (run_dir / "model-000000010.pth").write_bytes(b"x" * 2048)
    (run_dir / "model_best-000000010.pth").write_bytes(b"x" * 2048)

    main(pattern="ft_demo*", roots=(str(tmp_path),))

    assert (run_dir / "model-000000010.pth").exists(), "dry-run이 파일을 지웠다"
    assert "--apply" in capsys.readouterr().out

    main(pattern="ft_demo*", roots=(str(tmp_path),), apply=True)

    assert not (run_dir / "model-000000010.pth").exists()
    assert (run_dir / "model_best-000000010.pth").exists()
