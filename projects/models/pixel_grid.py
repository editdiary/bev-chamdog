"""특징맵 표본 좌표의 **규약** -- `grid_sample` 정규화와 상수 오프셋.

근거: `docs/finetune_overfitting_diagnosis.md` §18.3.

upstream Simple-BEV(`utils/vox.py:337`)와 우리 두 래퍼(`double_sphere_vox`,
`fisheye_vox`)는 표본 좌표를 `utils.basic.normalize_grid2d`로 정규화한 뒤
`F.grid_sample(align_corners=False)`로 표본한다. **두 함수의 규약이 다르다.**

- `normalize_grid2d`: 픽셀 **인덱스** 규약. `g = 2x/(W-1) - 1`. 즉 `x=0`과 `x=W-1`이
  각각 `-1`, `+1`로 간다 -- 이것은 `align_corners=True`의 규약이다.
- `grid_sample(align_corners=False)`: 픽셀 **가장자리** 규약. `g=-1`이 텐서의 왼쪽
  가장자리(픽셀 0의 중심보다 0.5px 왼쪽), `g=+1`이 오른쪽 가장자리다.

두 규약을 합성하면 실제 표본 위치는

    x' = x * W/(W-1) - 0.5

가 된다. **상수 편이가 아니라 배율 오차(W/(W-1))와 편이가 섞여 있어서**, 오프셋
하나로는 없앨 수 없다. W=64에서 실측한 값(§18.3의 표):

    목표 x |  8     16     32     48     63
    실제   |  7.63  15.75  32.01  48.26  63.5 (-> zero-padding과 섞여 값이 절반)

중심에서 0이고 가장자리로 갈수록 커지는 계통 오차다.

## 이 모듈이 나누는 두 가지

1. **정규화 규약**(`convention`) -- 위의 배율 오차. `pixel_center`가 옳고, 그러면
   `x' = x`가 **정확히** 성립한다(`tests/models/test_pixel_grid.py`가 고정한다).
   `legacy_index`는 기존 동작을 비트 단위로 보존하는 **대조군**이다.

2. **상수 오프셋**(`pixel_offset`, 특징픽셀 단위) -- 규약을 고쳐도 남는 부분.
   native -> 입력 리사이즈 규약과 stride 8 특징맵의 수용영역 중심이 어디인지는
   유도로 확정되지 않는다. 두 극단을 계산해 보면

   - 리사이즈가 half-pixel center(PIL/cv2 기본)이고 stride 8 특징 `j`의 중심이
     입력 픽셀 `8j+3.5`라면  ->  **-0.475** 특징픽셀
   - 같은 리사이즈에 중심이 `8j`(torchvision ResNet의 stride-2 conv는 좌상단을
     집는다)라면  ->  **-0.0375** 특징픽셀

   여기에 `Encoder_res101`의 `upsampling_layer`(stride16 -> stride8 보간)가 자기
   규약을 더한다. 그래서 **유도로 정하지 않고 스윕으로 실측한다**
   (`configs/sweep_pixel_offset.sh`).

   **오프셋은 이 모듈이 아니라 `unproject_image_to_mem`이 적용한다** -- native -> 특징
   좌표 변환 직후, 좌우 반전(`mirror_x`)과 유효 영역 판정보다 **앞**이어야 하기 때문이다.
   반전 뒤에 더하면 x축에서 부호가 뒤집히고(`(W-1) - (x+o)` != `(W-1) - x + o`),
   유효 영역도 실제 표본 위치가 아닌 좌표로 판정된다.

## 기본값은 `pixel_center`다 (2026-08-25 전환, 사용자 승인)

**성능이 근거가 아니다.** 15런 스윕(오프셋 4칸 × 시드 3 + 대조군 시드 3)에서 품질·안전
지표가 **전부 노이즈**로 나왔고 잔여 오프셋에 최적점도 없었다(진단 문서 §18.3.5).
그래도 옮긴 이유는 둘이다.

1. **표본 좌표가 어긋난 것은 성능과 무관하게 틀린 것이다.** 버그 수정이 성능 향상을
   증명해야 채택되는 것은 순서가 거꾸로다.
2. **지금이 비교 가능성을 끊기에 가장 싼 시점이다.** loss 연구가 설계 문서 §16으로
   닫혀 어차피 새 실험 줄기가 시작되고, 미루면 어긋난 기하로 돌린 런만 더 쌓인다.

**그래서 이 시점 이후의 런은 `runs/ablation`·설계 문서 §15·§16의 숫자와 기하가 다르다.**
그 숫자들과 비교할 때는 반드시 그 사실을 적는다.

`legacy_index`는 **지우지 않는다** -- 옛 런을 재현·재채점할 때 필요하고,
`--loss=weighted_ce`를 대조군으로 남겨 둔 것과 같은 이유다.

**`pixel_offset`의 기본값은 0이다.** 스윕이 평평했으므로 0이 아닌 값을 넣을 근거가 없고,
넣으면 노이즈를 적합하는 것이다.

## 옛 체크포인트를 재채점할 때 (2026-08-25에 추가)

기본값을 옮긴 순간 **재채점 도구가 조용히 틀리기 시작한다.** `runs/ablation`의 12런은
`legacy_index`로 학습됐는데 도구가 기본값(`pixel_center`)으로 vox util을 만들면 학습과
평가의 표본 기하가 달라진다. 규약을 인자로 노출하는 것만으로는 부족하다 -- 사람이 매번
옳은 값을 타이핑해야 하고, 틀려도 에러가 나지 않는다.

그래서 규약을 **런의 `config.json`에서 되찾는다**(`convention_for_run_dirs`,
`convention_for_checkpoints`). 규칙은 하나다.

    `pixel_convention` 키가 없으면 그 런은 `legacy_index` / offset 0이다.

이 규칙이 성립하는 근거: 이 인자는 2026-08-25에 생겼고 그 전 런의 `config.json`에는
키 자체가 없다(`runs/ablation` 12런 전부 확인). 그리고 여러 런을 한 표에 넣을 때
**규약이 섞여 있으면 즉시 멈춘다** -- 그 표는 서로 다른 기하의 숫자를 한 열에 세운 것이므로
경고로 넘길 것이 아니다.
"""
import json
import sys
from pathlib import Path

import torch

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.basic  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)

LEGACY_INDEX = "legacy_index"
PIXEL_CENTER = "pixel_center"
CONVENTIONS = (LEGACY_INDEX, PIXEL_CENTER)

DEFAULT_CONVENTION = PIXEL_CENTER


def normalize_pixel_grid2d(y, x, H, W, convention=DEFAULT_CONVENTION):
    """픽셀 좌표 `(y, x)`를 `grid_sample(align_corners=False)`용 `[-1, 1]` 좌표로 보낸다.

    반환은 `utils.basic.normalize_grid2d`와 같은 `(grid_y, grid_x)` 순서다.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention은 {CONVENTIONS} 중 하나여야 한다: {convention!r}")

    if convention == LEGACY_INDEX:
        # 기존 동작 그대로. 비트 단위 보존을 위해 upstream 함수를 그대로 부른다.
        return utils.basic.normalize_grid2d(y, x, H, W)

    # 픽셀 **중심** 규약: `x = 0`이 픽셀 0의 중심을 가리키고, `align_corners=False`의
    # 역변환 `x' = ((g+1)*W - 1)/2`에 넣으면 `x' = x`가 정확히 나온다.
    grid_x = (2.0 * x + 1.0) / float(W) - 1.0
    grid_y = (2.0 * y + 1.0) / float(H) - 1.0

    # upstream과 같은 clamp -- 화각 밖으로 크게 튄 투영이 grid_sample에 inf로 들어가는 것을
    # 막는다. 유효 영역 판정은 호출자(`unproject_image_to_mem`)가 clamp 이전 좌표로 한다.
    grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
    return grid_y, grid_x


def convention_from_config(config):
    """`config.json` 하나에서 `(convention, offset)`을 되찾는다.

    **키가 없으면 `legacy_index` / 0.0이다** -- 모듈 docstring의 "옛 체크포인트를
    재채점할 때"를 보라.
    """
    config = config or {}
    convention = str(config.get("pixel_convention", LEGACY_INDEX))
    if convention not in CONVENTIONS:
        raise ValueError(f"config.json의 pixel_convention이 이상하다: {convention!r}")
    return convention, float(config.get("pixel_offset", 0.0) or 0.0)


def convention_for_run_dirs(log_dirs, *, quiet=False):
    """여러 런의 로그 폴더에서 학습 때 쓴 `(convention, offset)`을 되찾는다.

    `log_dirs`의 각 항목은 `config.json`을 담은 폴더다. 없는 폴더는 건너뛴다(체크포인트가
    지워진 런이 섞여 있어도 도구가 죽지 않아야 한다). **규약이 섞여 있으면 SystemExit이다.**
    하나도 못 찾으면 기본값(`DEFAULT_CONVENTION`)을 돌려주고 그 사실을 찍는다.
    """
    found = {}
    for log_dir in log_dirs:
        path = Path(log_dir) / "config.json"
        if not path.exists():
            continue
        found.setdefault(convention_from_config(json.loads(path.read_text())), []).append(
            Path(log_dir).name)

    if not found:
        if not quiet:
            print(f"[표본 규약] config.json을 못 찾았다 -> 기본값 {DEFAULT_CONVENTION} / 0.0")
        return DEFAULT_CONVENTION, 0.0

    if len(found) > 1:
        lines = [f"  {conv} / offset {off}: {', '.join(sorted(runs))}"
                 for (conv, off), runs in sorted(found.items())]
        raise SystemExit(
            "학습 때 쓴 표본 규약이 런마다 다르다 -- 한 표에 세우면 서로 다른 기하의 숫자가"
            " 한 열에 섞인다:\n" + "\n".join(lines))

    (convention, offset), runs = next(iter(found.items()))
    if not quiet:
        print(f"[표본 규약] 런 {len(runs)}개의 config.json에서 되찾았다: "
              f"{convention} / offset {offset}")
    return convention, offset


def convention_for_checkpoints(checkpoints, *, quiet=False):
    """체크포인트 경로에서 같은 일을 한다.

    학습이 `<root>/ckpt/<run>/*.pth`와 `<root>/logs/<run>/config.json`을 짝지어 만들므로
    (`tools/train_robot_bev.py`) 경로를 그 규칙으로 되짚는다.
    """
    log_dirs = []
    for checkpoint in checkpoints:
        run_dir = Path(checkpoint).parent
        root = run_dir.parent
        log_dirs.append(root.parent / "logs" / run_dir.name if root.name == "ckpt" else run_dir)
    return convention_for_run_dirs(log_dirs, quiet=quiet)
