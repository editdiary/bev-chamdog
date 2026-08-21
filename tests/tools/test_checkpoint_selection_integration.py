"""두 학습 스크립트가 실제로 `iou_free`를 체크포인트 선택에 흘리는지 -- 배선 통합 테스트.

`select_checkpoint_score` 단위 테스트만으로는 부족하다는 것이 Task 11 리뷰에서 확인됐다:
공용 함수는 맞게 두고 두 trainer의 호출부만 옛 평균식으로 되돌려도 단위 테스트는 통과했다.
그래서 여기서는 CPU 더블로 `main()`을 끝까지 태워, 선택 지점과 저장 지점까지 값이 도달하는지
본다.
"""
import numpy as np
import pytest
import torch

import tools.train_robot_bev as robot_trainer
import tools.train_synwoodscape as synwoodscape_trainer

_FREE_METRICS = {
    "iou_free": 0.20, "iou_free_count": 1,
    "fatal_rate": 0.0, "fatal_denom": 1,
    "free_miss_rate": 0.0, "free_miss_denom": 1,
    "partition_defects": 0,
}


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _Dataset:
    cameras = ()


class _Model(torch.nn.Module):
    """생성 kwargs를 클래스에 남긴다 -- 정식화가 출력 채널 수까지 바꾸는지 확인하기 위해서다."""

    last_kwargs = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        type(self).last_kwargs = kwargs
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        # 실제 `Segnet`과 같은 이름의 하위 모듈. `--freeze_encoder`가 여기를 얼린다.
        self.encoder = torch.nn.Linear(2, 2)


def _install_common_cpu_doubles(monkeypatch, trainer, captured, selected, saved, calls):
    """인코더·데이터셋·GPU를 전부 더블로 바꾸고, 지표가 흐르는 지점에만 spy를 남긴다."""
    def step(*args, **kwargs):
        calls.append(args)
        return (
            torch.tensor(1.0, requires_grad=True),
            {"loss_unknown": torch.tensor(0.4), "loss_free": torch.tensor(0.2),
             "loss_occupied": torch.tensor(0.3),
             "share_unknown": torch.tensor(0.5), "share_free": torch.tensor(0.2),
             "share_occupied": torch.tensor(0.3)},
            dict(_FREE_METRICS),
        )

    monkeypatch.setattr(trainer, "SummaryWriter", lambda *args, **kwargs: _Writer())
    monkeypatch.setattr(trainer, "DataLoader", lambda *args, **kwargs: [
        {"valid_bev_g": torch.ones(1, 1, 1, 1)}
    ])
    monkeypatch.setattr(trainer, "ThreeClassSegnet", _Model)
    # `class_weights_from_labels`와 `run_batch`는 호출부마다 어느 모듈에서 오는지가 달라
    # (robot trainer는 정식화별 모듈에서 꺼낸다) 각 테스트가 직접 패치한다.
    monkeypatch.setattr(trainer, "build_ring_masks", lambda *args, **kwargs: [])
    monkeypatch.setattr(trainer, "build_ray_index", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        trainer, "evaluate_split",
        lambda *args, **kwargs: {
            "loss": 1.0,
            "loss_parts": {"loss_unknown": 0.4, "loss_free": 0.2, "loss_occupied": 0.3},
            "free": dict(_FREE_METRICS),
            "range": None,
            "range_bins": {},
            "rings": {},
            "tolerance": {},
        },
    )
    monkeypatch.setattr(trainer, "write_epoch_scalars", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer, "format_epoch_log", lambda **kwargs: captured.append(kwargs) or "epoch log"
    )
    monkeypatch.setattr(
        trainer, "select_checkpoint_score",
        lambda free_metrics: selected.append(free_metrics) or free_metrics["iou_free"],
    )
    monkeypatch.setattr(
        trainer.saverloader, "save",
        lambda *args, **kwargs: saved.append(kwargs.get("model_name", "model")),
    )
    return step


def _install_robot_dataset_doubles(monkeypatch):
    """robot trainer의 데이터셋·마스크·lifting을 전부 더블로. 정식화와 무관한 부분이다."""
    monkeypatch.setattr(
        robot_trainer, "parse_sequence_names",
        lambda names: names.split(",") if names else [],
    )
    monkeypatch.setattr(
        robot_trainer, "split_samples_by_sequence",
        lambda roots, val_names: ([(roots[0], "train")], [(roots[1], "val")]),
    )
    monkeypatch.setattr(
        robot_trainer, "build_bev_masks", lambda *args: (np.zeros((1, 1), bool),) * 2
    )
    monkeypatch.setattr(
        robot_trainer, "compute_label_statistics",
        lambda *args: {"trivial_iou": 0.5, "supervised_fraction": 1.0,
                       "obstacle_fraction": 0.5},
    )
    monkeypatch.setattr(
        robot_trainer, "load_masked_labels", lambda *args: (np.ones((1, 1), bool),) * 3
    )
    monkeypatch.setattr(robot_trainer, "constant_free_map", lambda masks: masks[0])
    monkeypatch.setattr(robot_trainer, "_baseline_iou_free", lambda *args: 0.5)
    monkeypatch.setattr(robot_trainer, "RobotBEVDataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(
        robot_trainer, "build_double_sphere_vox_util", lambda *args, **kwargs: object()
    )


def _assert_free_score_reaches_checkpoint_path(captured, selected, saved, calls):
    assert len(calls) == 1, "train step이 정확히 한 번 돌아야 한다"
    assert len(selected) == 1
    assert selected[0]["iou_free"] == pytest.approx(0.20)
    assert captured[0]["val_score"] == pytest.approx(0.20)
    # 세 loss 항이 로그까지 전달돼야 한다 -- 항이 두 칸뿐이던 옛 계약으로 돌아가면 깨진다.
    assert captured[0]["train_loss_parts"] == pytest.approx(
        {"loss_unknown": 0.4, "loss_free": 0.2, "loss_occupied": 0.3,
         "share_unknown": 0.5, "share_free": 0.2, "share_occupied": 0.3}
    )
    assert "model_best" in saved


def test_robot_trainer_uses_free_score_at_checkpoint_selection_boundary(monkeypatch, tmp_path):
    """robot trainer를 옛 IoU 평균으로 되돌리면 selection spy가 호출되지 않아 실패한다."""
    captured, selected, saved, calls = [], [], [], []
    dataset_root = tmp_path / "data"
    for name in ("train", "val"):
        (dataset_root / name / "occupancy_npy").mkdir(parents=True)
    step = _install_common_cpu_doubles(
        monkeypatch, robot_trainer, captured, selected, saved, calls
    )
    monkeypatch.setattr(robot_trainer.three_class_metrics, "run_batch", step)
    monkeypatch.setattr(robot_trainer.three_class_metrics, "class_weights_from_labels",
                        lambda *args, **kwargs: torch.ones(3))
    _install_robot_dataset_doubles(monkeypatch)

    robot_trainer.main(
        train_sequences="train", val_sequences="val", num_epochs=1, batch_size=2,
        num_workers=0, dataset_root=dataset_root, common_root=tmp_path,
        log_dir=tmp_path / "logs", ckpt_dir=tmp_path / "ckpts", device="cpu",
    )

    _assert_free_score_reaches_checkpoint_path(captured, selected, saved, calls)


def test_binary_formulation_switches_head_width_loss_module_and_log_terms(monkeypatch, tmp_path):
    """`--formulation=binary`가 실제로 세 곳을 동시에 바꾸는지 -- 하나라도 빠지면 조용히 깨진다.

    출력 채널만 2로 바꾸고 loss를 3-class 그대로 두면 CE가 클래스 3을 찾다 죽고, 반대로
    loss만 바꾸고 head를 3채널로 두면 안 쓰는 채널이 학습된다. 로그 항 이름이 안 바뀌면
    binary의 두 항이 로그에서 통째로 사라진다(`share nf/f`가 그 확인이다).
    """
    captured, selected, saved, calls = [], [], [], []
    dataset_root = tmp_path / "data"
    for name in ("train", "val"):
        (dataset_root / name / "occupancy_npy").mkdir(parents=True)
    step = _install_common_cpu_doubles(
        monkeypatch, robot_trainer, captured, selected, saved, calls
    )
    monkeypatch.setattr(robot_trainer.binary_metrics, "run_batch", step)
    monkeypatch.setattr(robot_trainer.binary_metrics, "class_weights_from_labels",
                        lambda *args, **kwargs: torch.ones(2))
    # 3-class 경로가 실수로 불리면 즉시 드러나게 둔다.
    monkeypatch.setattr(robot_trainer.three_class_metrics, "run_batch",
                        lambda *args, **kwargs: pytest.fail("binary인데 3-class run_batch가 불렸다"))
    _install_robot_dataset_doubles(monkeypatch)

    robot_trainer.main(
        train_sequences="train", val_sequences="val", num_epochs=1, batch_size=2,
        num_workers=0, dataset_root=dataset_root, common_root=tmp_path,
        log_dir=tmp_path / "logs", ckpt_dir=tmp_path / "ckpts", device="cpu",
        formulation="binary",
    )

    assert _Model.last_kwargs["num_classes"] == 2
    assert captured[0]["loss_part_names"] == ("not_free", "free")
    assert len(calls) == 1


def test_freeze_encoder_takes_the_encoder_out_of_the_optimizer(monkeypatch, tmp_path):
    """`requires_grad=False`만 걸고 optimizer에 그대로 남겨 두면 얼린 것이 로그로 확인되지 않고,
    AdamW가 상태 텐서를 계속 들고 있는다. 파라미터 목록에서도 빠져야 한다."""
    captured, selected, saved, calls = [], [], [], []
    dataset_root = tmp_path / "data"
    for name in ("train", "val"):
        (dataset_root / name / "occupancy_npy").mkdir(parents=True)
    step = _install_common_cpu_doubles(
        monkeypatch, robot_trainer, captured, selected, saved, calls
    )
    monkeypatch.setattr(robot_trainer.three_class_metrics, "run_batch", step)
    monkeypatch.setattr(robot_trainer.three_class_metrics, "class_weights_from_labels",
                        lambda *args, **kwargs: torch.ones(3))
    _install_robot_dataset_doubles(monkeypatch)
    seen = {}
    real_adamw = torch.optim.AdamW

    def spy_adamw(params, **kwargs):
        params = list(params)
        seen["n"] = len(params)
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(robot_trainer.torch.optim, "AdamW", spy_adamw)

    robot_trainer.main(
        train_sequences="train", val_sequences="val", num_epochs=1, batch_size=2,
        num_workers=0, dataset_root=dataset_root, common_root=tmp_path,
        log_dir=tmp_path / "logs", ckpt_dir=tmp_path / "ckpts", device="cpu",
        freeze_encoder=True,
    )

    # `_Model`은 weight 1개 + encoder(weight, bias) 2개다. 얼리면 1개만 남는다.
    assert seen["n"] == 1


def test_an_unknown_formulation_fails_before_training_starts(tmp_path):
    """오타를 조용히 3-class로 떨어뜨리면 잘못된 런을 몇 시간 뒤에 발견하게 된다."""
    with pytest.raises(ValueError, match="formulation"):
        robot_trainer.main(formulation="binry", dataset_root=tmp_path)


def test_synwoodscape_trainer_uses_free_score_at_checkpoint_selection_boundary(
    monkeypatch, tmp_path
):
    """pretrain trainer도 같은 선택 기준을 쓴다 -- 두 경로가 갈리면 숫자를 나란히 못 읽는다."""
    captured, selected, saved, calls = [], [], [], []
    step = _install_common_cpu_doubles(
        monkeypatch, synwoodscape_trainer, captured, selected, saved, calls
    )
    monkeypatch.setattr(synwoodscape_trainer, "run_batch", step)
    monkeypatch.setattr(synwoodscape_trainer, "class_weights_from_labels",
                        lambda *args, **kwargs: torch.ones(3))
    monkeypatch.setattr(
        synwoodscape_trainer, "discover_all_sample_ids", lambda *args: ["train", "val"]
    )
    monkeypatch.setattr(
        synwoodscape_trainer, "train_val_split", lambda *args, **kwargs: (["train"], ["val"])
    )
    monkeypatch.setattr(
        synwoodscape_trainer, "load_label_triples",
        lambda *args: iter([(np.ones((1, 1), bool),) * 3]),
    )
    monkeypatch.setattr(synwoodscape_trainer, "compute_trivial_baseline_iou", lambda *args: 0.5)
    monkeypatch.setattr(synwoodscape_trainer, "constant_free_map", lambda masks: masks[0])
    monkeypatch.setattr(synwoodscape_trainer, "baseline_iou_free", lambda *args: 0.5)
    monkeypatch.setattr(
        synwoodscape_trainer, "SynWoodScapeSimpleBEVDataset",
        lambda *args, **kwargs: _Dataset(),
    )
    monkeypatch.setattr(synwoodscape_trainer, "build_vox_util", lambda *args, **kwargs: object())

    synwoodscape_trainer.main(
        num_epochs=1, batch_size=2, num_workers=0, use_fisheye=False,
        log_dir=tmp_path / "logs", ckpt_dir=tmp_path / "ckpts", device="cpu",
    )

    _assert_free_score_reaches_checkpoint_path(captured, selected, saved, calls)
