import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.models.simplebev_three_class import (  # noqa: F401  (sys.path 부작용이 필요하다)
    DISCARDED_HEADS,
    ThreeClassDecoder,
    head_was_transferred,
    load_trunk_weights,
    unexpected_skips,
)
from nets.segnet import Decoder  # noqa: E402  (위 import가 third_party 경로를 넣어준다)


def test_class_ids_are_distinct():
    assert len({FREE, OCCUPIED, UNKNOWN}) == 3


def test_decoder_emits_three_logit_channels():
    decoder = ThreeClassDecoder(in_channels=8)
    x = torch.randn(2, 8, 16, 16)

    out = decoder(x)

    assert out["three_class"].shape == (2, 3, 16, 16)


def test_unused_instance_heads_are_gone():
    """3-class 학습이 쓰지 않는 head는 만들지도 실행하지도 않는다.

    파라미터가 없어야 임베디드 배포에 안 실리고, forward에서 안 돌아야 연산이 줄어든다.
    `Segnet.forward`가 5-튜플로 언패킹하므로 키 자체는 남아 있어야 한다(값은 None).
    """
    decoder = ThreeClassDecoder(in_channels=8)

    for name in DISCARDED_HEADS:
        assert not hasattr(decoder, name), f"{name}이 아직 남아 있다"
    assert not any(name.startswith(head) for name, _ in decoder.named_parameters()
                   for head in DISCARDED_HEADS)

    out = decoder(torch.randn(1, 8, 16, 16))
    assert set(out) >= {"raw_feat", "feat", "segmentation", "instance_center", "instance_offset"}
    assert out["instance_center"] is None and out["instance_offset"] is None


def test_trunk_matches_the_upstream_decoder_after_removing_the_heads():
    """head를 뺀 forward가 원본 trunk와 **같은 값**을 내는지 대조한다.

    `Decoder.forward`를 복사해 다시 구현했으므로, 원본이 바뀌거나 복사가 틀리면 여기서 잡힌다.
    (이 테스트가 없으면 trunk를 잘못 베껴도 학습은 그냥 돌아간다 -- 값만 조용히 달라진다.)
    """
    torch.manual_seed(0)
    upstream = Decoder(in_channels=8, n_classes=1, predict_future_flow=False).eval()
    three_class = ThreeClassDecoder(in_channels=8).eval()
    # segmentation_head를 뺀 나머지(=trunk)를 그대로 옮긴다.
    three_class.load_state_dict(
        {k: v for k, v in upstream.state_dict().items()
         if k in three_class.state_dict() and three_class.state_dict()[k].shape == v.shape},
        strict=False,
    )

    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        assert torch.allclose(three_class(x)["raw_feat"], upstream(x)["raw_feat"], atol=1e-6)


def test_head_is_skipped_when_the_checkpoint_head_shape_differs(tmp_path):
    """출력 채널 수가 다른 체크포인트에서는 head가 skip되고 trunk만 온다.

    2-head 체크포인트가 이 경우였다(674 loaded / 6 skipped). 2-head 코드는 제거됐지만
    `load_trunk_weights`의 부분 전이 분기는 남아 있으므로, 1채널 head를 가진 upstream
    `Decoder`로 같은 상황을 만들어 계약을 계속 고정한다.
    """
    source = Decoder(in_channels=8, n_classes=1, predict_future_flow=False)
    path = tmp_path / "single_channel_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    report = load_trunk_weights(target, path, device="cpu")

    assert report["loaded"] > 0
    assert report["skipped"]
    assert not head_was_transferred(report["skipped"])
    # skip되는 것은 형상이 다른 출력 head와, 이 모델이 갖고 있지 않은 DISCARDED_HEADS뿐이다.
    # trunk가 하나라도 빠지면 unexpected_skips가 비지 않는다.
    assert unexpected_skips(report["skipped"]) == []


def test_trunk_transfer_actually_changes_the_weights(tmp_path):
    """형상이 맞는 trunk 값이 실제로 옮겨졌는지 확인한다."""
    source = Decoder(in_channels=8, n_classes=1, predict_future_flow=False)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if not name.startswith("segmentation_head"):
                parameter.fill_(0.5)
    path = tmp_path / "single_channel_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    load_trunk_weights(target, path, device="cpu")

    shared_weight = target.state_dict()["first_conv.weight"]
    assert torch.allclose(shared_weight, torch.full_like(shared_weight, 0.5))


def test_three_class_checkpoint_transfers_its_head_too(tmp_path):
    """3-class -> 3-class 전이에서는 출력 head까지 와야 하고 skipped가 비어야 한다.

    Phase 4의 요점이 바로 이것이다 -- 2-head pretrain에서는 head가 랜덤 초기화로 남았고
    (`docs/free_space_metric_migration.md` §8.3), 3-class pretrain은 그 교란을 없앤다.
    `skipped == 0`이 "head가 전이됐다"의 유일한 기계적 확인이라 테스트로 고정한다.
    """
    source = ThreeClassDecoder(in_channels=8)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    path = tmp_path / "three_class.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    report = load_trunk_weights(target, path, device="cpu")

    assert report["skipped"] == []
    head_keys = [k for k in target.state_dict() if "segmentation_head" in k]
    assert head_keys, "segmentation_head 파라미터가 없으면 이 테스트는 아무것도 검증하지 않는다"
    for key in head_keys:
        assert torch.allclose(
            target.state_dict()[key], torch.full_like(target.state_dict()[key], 0.25)
        ), f"{key}가 전이되지 않았다"
