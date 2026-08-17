import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.models.simplebev_three_class import ThreeClassDecoder, load_trunk_weights
from projects.models.simplebev_two_head import TwoHeadDecoder


def test_class_ids_are_distinct():
    assert len({FREE, OCCUPIED, UNKNOWN}) == 3


def test_decoder_emits_three_logit_channels():
    decoder = ThreeClassDecoder(in_channels=8)
    x = torch.randn(2, 8, 16, 16)

    out = decoder(x)

    assert out["three_class"].shape == (2, 3, 16, 16)


def test_trunk_weights_transfer_from_a_two_head_checkpoint(tmp_path):
    """pretrain 자산은 trunk다. 최종 1x1 conv만 새로 배운다."""
    source = TwoHeadDecoder(in_channels=8)
    path = tmp_path / "two_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    report = load_trunk_weights(target, path, device="cpu")

    assert report["loaded"] > 0
    assert report["skipped"]
    assert all("segmentation_head" in name for name in report["skipped"])


def test_trunk_transfer_actually_changes_the_weights(tmp_path):
    """형상이 맞는 trunk 값이 실제로 옮겨졌는지 확인한다."""
    source = TwoHeadDecoder(in_channels=8)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if not name.startswith("segmentation_head"):
                parameter.fill_(0.5)
    path = tmp_path / "two_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    load_trunk_weights(target, path, device="cpu")

    shared_weight = target.state_dict()["first_conv.weight"]
    assert torch.allclose(shared_weight, torch.full_like(shared_weight, 0.5))
