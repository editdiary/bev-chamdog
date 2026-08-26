"""stride-4 특징맵을 내는 ResNet-101 encoder.

**왜 있나.** lifting(`unproject_image_to_mem`)이 표본하는 2D 특징맵이 stride 8이라
입력 512x288에서 64x36이고, 4 m 거리에서 BEV 셀 하나(5 cm)당 표본 간격이 20~24 cm다
(진단 문서 §27). §28.10의 프로브가 **표본 간격 가설을 기각하지 않았으므로** 간격을
절반으로 줄여 원거리 정밀도가 따라 움직이는지 본다. 사전 선언은 진단 문서 **§29**이고,
**예측한 무늬(§29.5)와 채택 규칙(§29.6)은 런 하나도 없는 상태에서 이미 적혀 있다.**

**`layer1`만 쓰지 않는 이유가 판정에 직결된다**(§29.2). `layer1`은 위치가 선명한 대신
수용영역이 좁고 채널이 적다(256 대 512). 그것만 표본하면 실패했을 때 "밀도가 안 통했다"와
"의미가 약해졌다"를 가를 수 없다. 그래서 **기존 stride-8 특징을 그대로 만든 뒤 그것을 올려
`layer1`과 합친다** -- 새 특징이 현재 특징의 상위집합이라 그 교란이 사라진다.

    현재:  stride8 = Fuse(layer2, Up(layer3))                   -> 64x36
    여기:  stride4 = Fuse(layer1, Up(Fuse(layer2, Up(layer3))))  -> 128x72

**출력 채널은 `C`(=`latent_dim`) 그대로다.** 그래서 lifting 출력 `(B, C, Z, Y, X)`도,
BEV decoder도, 격자도 전부 불변이다 -- 바뀌는 것은 표본되는 2D 격자 하나뿐이다.
upstream `Segnet.forward`가 `sy = Hf/H, sx = Wf/W`로 내부 파라미터를 다시 스케일하므로
**stride 변화는 기하에 자동으로 반영된다**(§29.3에서 확인).

`third_party/`는 고치지 않는다(`AGENTS.md`). upstream `Encoder_res101`은 `conv1`~`layer2`를
`nn.Sequential`로 한 덩어리로 묶어 `layer1`을 노출하지 않으므로 여기서 다시 쪼갠다.
"""
import sys
from pathlib import Path

import torch.nn as nn
import torchvision

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

from nets.segnet import UpsamplingConcat  # noqa: E402

#: `--encoder_type`으로 이 encoder를 고르는 문자열. upstream `Segnet.__init__`의
#: `assert encoder_type in [...]`가 모르는 값이므로 **우리 래퍼가 소비하고 upstream에는
#: 전달하지 않는다**(`ThreeClassSegnet` 참고).
STRIDE4_ENCODER_TYPE = "res101_s4"


class Encoder_res101_stride4(nn.Module):
    """upstream `Encoder_res101`과 같은 계약(이미지 -> `C`채널 특징맵), 해상도만 2배.

    이름을 upstream 스타일(`Encoder_res101`)에 맞춘 것은 나란히 읽히라는 뜻이다.
    """

    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)

        # upstream의 `backbone`(conv1~layer2 한 덩어리)을 stride 단계별로 쪼갠다.
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # stride 4,   256ch
        self.layer2 = resnet.layer2  # stride 8,   512ch
        self.layer3 = resnet.layer3  # stride 16, 1024ch

        # stride-8 가지는 upstream과 **완전히 같다**(1536 = 512 + 1024를 concat). 이름까지
        # 맞춰 두면 두 encoder의 이 단계가 같은 것임이 state_dict 키에서도 보인다.
        self.upsampling_layer = UpsamplingConcat(1536, 512)
        # 새로 붙는 단계. 768 = layer1의 256 + 위 단계의 512.
        self.upsampling_layer_s4 = UpsamplingConcat(768, 512)

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer(x3, x2)     # stride 8  -- upstream이 내보내던 특징
        x = self.upsampling_layer_s4(x, x1)   # stride 4  -- 그 상위집합
        return self.depth_layer(x)
