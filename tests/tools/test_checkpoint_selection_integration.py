import numpy as np
import pytest
import torch

import tools.train_robot_bev as robot_trainer
import tools.train_synwoodscape as synwoodscape_trainer


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _Dataset:
    cameras = ()


class _Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


def _run_batch(*args, **kwargs):
    return (
        torch.tensor(1.0, requires_grad=True),
        {"loss_occ": torch.tensor(0.4), "loss_vis": torch.tensor(0.6)},
        torch.tensor(0.95), torch.tensor(0.80), 1,
        {"false_high": 0.0, "false_low": 0.0}, {}, {},
        {"iou_free": 0.20, "iou_free_count": 1, "fatal_rate": 0.0,
         "fatal_denom": 1, "free_miss_rate": 0.0, "free_miss_denom": 1,
         "partition_defects": 0},
    )


def _install_common_cpu_doubles(monkeypatch, trainer, captured, selected, saved):
    monkeypatch.setattr(trainer, "SummaryWriter", lambda *args, **kwargs: _Writer())
    monkeypatch.setattr(trainer, "DataLoader", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(trainer, "TwoHeadSegnet", _Model)
    monkeypatch.setattr(trainer, "run_batch", _run_batch)
    monkeypatch.setattr(trainer, "append_free_metrics", lambda values, value: values.append(value))
    monkeypatch.setattr(trainer, "summarize_occupancy_diagnostics", lambda values: {})
    monkeypatch.setattr(trainer, "summarize_deployment_metrics", lambda values: {})
    monkeypatch.setattr(trainer, "write_occupancy_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "write_deployment_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer, "format_epoch_log", lambda **kwargs: captured.append(kwargs) or "epoch log"
    )
    monkeypatch.setattr(
        trainer, "select_checkpoint_score",
        lambda **kwargs: selected.append(kwargs) or kwargs["free_metrics"]["iou_free"],
    )
    monkeypatch.setattr(
        trainer.saverloader, "save",
        lambda *args, **kwargs: saved.append(kwargs.get("model_name", "model")),
    )


def _assert_free_score_reaches_checkpoint_path(captured, selected, saved):
    assert len(selected) == 1
    assert selected[0]["d_iou"] == pytest.approx(0.95)
    assert selected[0]["o_iou"] == pytest.approx(0.80)
    assert selected[0]["free_metrics"]["iou_free"] == pytest.approx(0.20)
    assert captured[0]["val_score"] == pytest.approx(0.20)
    assert "model_best" in saved


def test_robot_trainer_uses_free_score_at_checkpoint_selection_boundary(monkeypatch, tmp_path):
    """robot trainer를 옛 IoU 평균으로 되돌리면 selection spy가 호출되지 않아 실패한다."""
    captured, selected, saved = [], [], []
    dataset_root = tmp_path / "data"
    for name in ("train", "val"):
        (dataset_root / name / "occupancy_npy").mkdir(parents=True)
    _install_common_cpu_doubles(monkeypatch, robot_trainer, captured, selected, saved)
    monkeypatch.setattr(robot_trainer, "parse_sequence_names", lambda names: names.split(",") if names else [])
    monkeypatch.setattr(
        robot_trainer, "split_samples_by_sequence",
        lambda roots, val_names: ([(roots[0], "train")], [(roots[1], "val")]),
    )
    monkeypatch.setattr(robot_trainer, "build_bev_masks", lambda *args: (np.zeros((1, 1), bool),) * 2)
    monkeypatch.setattr(
        robot_trainer, "compute_label_statistics",
        lambda *args: {"pos_weight": 1.0, "trivial_iou": 0.5,
                       "supervised_fraction": 1.0, "obstacle_fraction": 0.5},
    )
    monkeypatch.setattr(
        robot_trainer, "load_masked_labels",
        lambda *args: (np.ones((1, 1), bool),) * 3,
    )
    monkeypatch.setattr(robot_trainer, "constant_free_map", lambda masks: masks[0])
    monkeypatch.setattr(robot_trainer, "_baseline_iou_free", lambda *args: 0.5)
    monkeypatch.setattr(robot_trainer, "build_ray_index", lambda *args, **kwargs: object())
    monkeypatch.setattr(robot_trainer, "build_ring_masks", lambda *args: [])
    monkeypatch.setattr(robot_trainer, "RobotBEVDataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(robot_trainer, "build_double_sphere_vox_util", lambda *args, **kwargs: object())
    monkeypatch.setattr(robot_trainer, "_write_free_space_scalars", lambda *args, **kwargs: None)
    monkeypatch.setattr(robot_trainer, "summarize_range_error", lambda values: {})
    monkeypatch.setattr(
        robot_trainer, "_evaluate",
        lambda *args: {"loss": 1.0, "loss_occ": 0.4, "loss_vis": 0.6,
                       "d_iou": 0.95, "o_iou": 0.80, "false_high": 0.0, "false_low": 0.0,
                       "occ": {}, "deploy": {},
                       "free": {"iou_free": 0.20, "fatal_rate": 0.0, "free_miss_rate": 0.0},
                       "range": {}, "rings": {}},
    )

    robot_trainer.main(
        train_sequences="train", val_sequences="val", num_epochs=1, batch_size=2,
        num_workers=0, dataset_root=dataset_root, common_root=tmp_path, log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpts", device="cpu",
    )

    _assert_free_score_reaches_checkpoint_path(captured, selected, saved)


def test_synwoodscape_trainer_uses_free_score_at_checkpoint_selection_boundary(monkeypatch, tmp_path):
    """SynWoodScape trainer를 옛 IoU 평균으로 되돌리면 selection spy가 호출되지 않아 실패한다."""
    captured, selected, saved = [], [], []
    _install_common_cpu_doubles(monkeypatch, synwoodscape_trainer, captured, selected, saved)
    monkeypatch.setattr(synwoodscape_trainer, "discover_all_sample_ids", lambda *args: ["train", "val"])
    monkeypatch.setattr(synwoodscape_trainer, "train_val_split", lambda *args, **kwargs: (["train"], ["val"]))
    monkeypatch.setattr(synwoodscape_trainer, "compute_pos_weight", lambda *args: 1.0)
    monkeypatch.setattr(synwoodscape_trainer, "compute_trivial_baseline_iou", lambda *args: 0.5)
    monkeypatch.setattr(synwoodscape_trainer, "SynWoodScapeSimpleBEVDataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(synwoodscape_trainer, "build_vox_util", lambda *args, **kwargs: object())

    synwoodscape_trainer.main(
        num_epochs=1, batch_size=2, num_workers=0, use_fisheye=False, log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpts", device="cpu",
    )

    _assert_free_score_reaches_checkpoint_path(captured, selected, saved)
