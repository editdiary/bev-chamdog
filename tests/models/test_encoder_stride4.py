"""stride-4 encoder가 **표본 격자만** 바꾸는지 고정한다 (진단 문서 §29).

이 실험의 판정은 "표본 간격을 절반으로 줄이면 원거리 정밀도가 따라 움직이나"이므로,
간격 말고 다른 것이 같이 바뀌면 그 판정이 오염된다. 그래서 여기서 고정하는 것은 셋이다.
(1) 해상도가 정확히 2배다, (2) 출력 채널이 `latent_dim` 그대로다, (3) stride-8 가지가
upstream과 같은 계산이다.
"""
import torch

from projects.models.encoder_stride4 import (  # noqa: F401  (sys.path 부작용이 필요하다)
    STRIDE4_ENCODER_TYPE,
    Encoder_res101_stride4,
)
from nets.segnet import Encoder_res101  # noqa: E402  (위 import가 third_party 경로를 넣어준다)


def test_feature_map_resolution_doubles_and_channels_do_not():
    """64x36 -> 128x72. 채널이 같이 바뀌면 BEV lifting 출력 형상이 달라져 격자가 흔들린다."""
    x = torch.randn(1, 3, 288, 512)

    with torch.no_grad():
        stride8 = Encoder_res101(128).eval()(x)
        stride4 = Encoder_res101_stride4(128).eval()(x)

    assert stride8.shape == (1, 128, 36, 64)
    assert stride4.shape == (1, 128, 72, 128)


def test_stride8_branch_computes_the_same_thing_as_upstream():
    """새 encoder의 stride-8 단계는 upstream `Encoder_res101`의 그 단계와 **같아야 한다**.

    §29.2가 `layer1` 단독을 쓰지 않기로 한 이유가 이것이다 -- 새 특징이 현재 특징의
    **상위집합**이어야 실패했을 때 "밀도가 안 통했다"와 "의미가 약해졌다"를 가를 수 있다.
    upstream이 바뀌거나 우리가 가지를 잘못 이으면 여기서 잡힌다.
    """
    torch.manual_seed(0)
    upstream = Encoder_res101(128).eval()
    ours = Encoder_res101_stride4(128).eval()
    # upstream의 `backbone`(conv1~layer2)을 우리 쪽 `stem`/`layer1`/`layer2`로 옮긴다.
    ours.stem.load_state_dict({
        k.split(".", 1)[1]: v for k, v in upstream.backbone.state_dict().items()
        if k.split(".", 1)[0] in {"0", "1"}
    }, strict=False)
    ours.layer1.load_state_dict(
        {k[2:]: v for k, v in upstream.backbone.state_dict().items() if k.startswith("4.")})
    ours.layer2.load_state_dict(
        {k[2:]: v for k, v in upstream.backbone.state_dict().items() if k.startswith("5.")})
    ours.layer3.load_state_dict(upstream.layer3.state_dict())
    ours.upsampling_layer.load_state_dict(upstream.upsampling_layer.state_dict())

    x = torch.randn(1, 3, 288, 512)
    with torch.no_grad():
        x1 = upstream.backbone(x)
        expected = upstream.upsampling_layer(upstream.layer3(x1), x1)

        h = ours.layer1(ours.stem(x))
        h2 = ours.layer2(h)
        actual = ours.upsampling_layer(ours.layer3(h2), h2)

    torch.testing.assert_close(actual, expected)


def test_the_encoder_type_string_is_not_one_upstream_knows():
    """upstream `Segnet.__init__`의 assert를 통과하는 문자열이면 래퍼가 조용히 무시된다.

    즉 이 문자열이 `["res101","res50","effb0","effb4"]`에 없어야, 래퍼가 소비하는 것을
    잊었을 때 학습이 **stride-8로 조용히 도는 대신 즉시 죽는다.**
    """
    assert STRIDE4_ENCODER_TYPE not in ["res101", "res50", "effb0", "effb4"]
