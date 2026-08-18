from pathlib import Path

import numpy as np
import pytest
import torch

from tools.train_robot_bev import (
    _baseline_iou_free,
    compute_label_statistics,
)

DATASET_ROOT = Path("dataset/sj_datasets")
SEQUENCE_ROOT = DATASET_ROOT / "raws1"
requires_dataset = pytest.mark.skipif(
    not (SEQUENCE_ROOT / "occupancy_npy").exists(),
    reason="self-collected dataset not available locally",
)


def _write_sample(root, sample_id, occupancy, visible):
    (root / "occupancy_npy").mkdir(parents=True, exist_ok=True)
    (root / "visibility_npy").mkdir(parents=True, exist_ok=True)
    np.save(root / "occupancy_npy" / f"{sample_id}.npy", occupancy.astype(np.uint8))
    np.save(root / "visibility_npy" / f"{sample_id}.npy", visible.astype(np.uint8))


def test_label_statistics_count_only_masked_cells(tmp_path):
    """라벨 분포는 loss가 실제로 보는 셀에서만 세어야 한다.

    마스크 밖 라벨을 같이 세면 배너에 찍히는 분포가 loss가 보는 분포와 어긋난다 -- 자체
    데이터셋에서는 마스킹 전후로 obstacle 비율이 몇 배 차이 난다.
    """
    occupancy = np.zeros((4, 4), bool)
    occupancy[0] = True                       # 4 drivable, 12 obstacle (마스킹 전)
    visible = np.ones((4, 4), bool)
    _write_sample(tmp_path, "sample_000000", occupancy, visible)

    permanent_blind = np.zeros((4, 4), bool)
    permanent_blind[2:] = True                # 아래 절반은 영구 사각 -> vis=0
    invalid = np.zeros((4, 4), bool)

    stats = compute_label_statistics([(tmp_path, "sample_000000")], permanent_blind, invalid)

    # 남는 셀은 위 두 행(8개): drivable 4, obstacle 4.
    assert stats["trivial_iou"] == pytest.approx(0.5)
    assert stats["supervised_fraction"] == pytest.approx(0.5)
    assert stats["obstacle_fraction"] == pytest.approx(0.5)

    # 마스킹을 무시했다면 16셀 전체를 세어 obstacle 비율이 12/16 = 0.75가 됐을 것이다.
    unmasked = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), invalid
    )
    assert unmasked["obstacle_fraction"] == pytest.approx(0.75)
    assert unmasked["supervised_fraction"] == pytest.approx(1.0)


def test_invalid_mask_also_removes_cells(tmp_path):
    occupancy = np.ones((4, 4), bool)
    visible = np.ones((4, 4), bool)
    _write_sample(tmp_path, "sample_000000", occupancy, visible)

    invalid = np.zeros((4, 4), bool)
    invalid[:, :2] = True
    stats = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), invalid
    )
    assert stats["supervised_fraction"] == pytest.approx(0.5)


def test_label_statistics_handles_a_fully_masked_sample(tmp_path):
    """전부 마스킹된 샘플에서 0으로 나누지 않아야 한다."""
    _write_sample(tmp_path, "sample_000000", np.ones((4, 4), bool), np.zeros((4, 4), bool))
    stats = compute_label_statistics(
        [(tmp_path, "sample_000000")], np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    )
    assert stats["supervised_fraction"] == 0.0
    assert np.isfinite(stats["trivial_iou"])
    assert np.isfinite(stats["obstacle_fraction"])


def test_constant_map_baseline_scores_masked_validation_free_space_on_cpu(tmp_path):
    """NaN baseline은 epoch 로그의 모델-vs-layout 비교를 무력화한다."""
    _write_sample(tmp_path, "sample_000000", np.array([[1, 0], [0, 0]], bool), np.ones((2, 2), bool))
    _write_sample(tmp_path, "sample_000001", np.ones((2, 2), bool), np.ones((2, 2), bool))

    score = _baseline_iou_free(
        [(tmp_path, "sample_000000"), (tmp_path, "sample_000001")],
        permanent_blind=np.zeros((2, 2), bool),
        invalid=np.zeros((2, 2), bool),
        constant_map=np.array([[1, 0], [0, 0]], bool),
        device="cpu",
    )

    assert score == pytest.approx(0.625)


def test_empty_epoch_metrics_matches_the_evaluate_split_contract():
    """val을 건너뛴 epoch의 자리표시자가 `evaluate_split` 반환과 키가 같아야 한다 --
    다르면 로그 포매터나 scalar writer가 KeyError로 죽는다."""
    from projects.common.bev_occupancy_metrics import empty_epoch_metrics, evaluate_split

    empty = empty_epoch_metrics()
    real = evaluate_split(lambda batch: None, [], "cpu", rays=None, ring_masks=[],
                          cell_m=0.05, range_edges_m=(0.0, 1.0))

    assert set(empty) == set(real)


@requires_dataset
def test_real_dataset_statistics_are_in_the_expected_range():
    """실제 라벨에서 마스킹 후 분포가 예상 범위인지 -- 마스크가 어긋나면 여기서 드러난다."""
    from projects.datasets.robot_simplebev import (
        DEFAULT_COMMON_ROOT,
        GRID_SPEC,
        build_bev_masks,
        list_sequence_samples,
    )
    from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES

    permanent_blind, invalid = build_bev_masks(
        DEFAULT_COMMON_ROOT, GRID_SPEC, FINETUNE_CAMERA_NAMES
    )
    stats = compute_label_statistics(
        list_sequence_samples(SEQUENCE_ROOT), permanent_blind, invalid
    )
    # 관측(vis&valid) 셀은 그리드의 일부지만 0은 아니어야 한다.
    assert 0.10 < stats["supervised_fraction"] < 0.30
    # 관측 영역 안에서는 obstacle이 소수 클래스다(통로가 대부분).
    assert 0.01 < stats["obstacle_fraction"] < 0.20
    # 같은 말을 반대쪽에서 한 번 더 고정한다: drivable이 다수여야 한다.
    assert stats["trivial_iou"] > 0.5


def test_periodic_checkpoints_are_kept_for_every_saved_epoch_by_default():
    """`keep_latest=3`이 주기 저장분을 지워 유효 구간이 사라졌던 회귀를 막는다.

    2026-08-18 fine-tuning 진단에서 실제로 겪었다: `save_freq_epochs=10`으로 60 epoch을
    돌렸는데 40/50/60만 남고, val 지표가 최고였던 epoch 10~30이 전부 삭제돼 재채점으로
    복구할 수 없었다(`docs/finetune_overfitting_diagnosis.md` §5).

    기본값이 `num_epochs / save_freq_epochs`(= 60/10 = 6)보다 작아지면 같은 일이 반복된다.
    """
    import inspect

    import tools.train_robot_bev as robot_trainer
    import tools.train_synwoodscape as synwoodscape_trainer

    for trainer in (robot_trainer, synwoodscape_trainer):
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(trainer.main).parameters.items()
        }
        periodic_saves = defaults["num_epochs"] // defaults["save_freq_epochs"]
        assert defaults["keep_checkpoints"] >= periodic_saves, (
            trainer.__name__, defaults["keep_checkpoints"], periodic_saves,
        )


def test_none_as_a_string_means_from_scratch():
    """config가 `INIT_CHECKPOINT`를 필수로 만들었으므로 셸에서 "pretrain 없이"를 표현할 값이
    필요하다. `"none"`을 경로로 취급하면 `load_trunk_weights`가 파일을 못 찾고 죽는다
    (2026-08-18 클래스 가중치 스윕 첫 시도가 이걸로 실패했다)."""
    from tools.train_robot_bev import from_scratch

    for value in (None, "", "none", "None", " NONE ", "no"):
        assert from_scratch(value), value
    for value in ("runs/x/model_best-000000048.pth", "none.pth", "checkpoint_none"):
        assert not from_scratch(value), value
