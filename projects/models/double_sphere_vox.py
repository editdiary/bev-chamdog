"""Simple-BEV `Vox_util.unproject_image_to_mem`(핀홀 전용)을 Double Sphere 투영으로
바꿔치기하는 래퍼 — 자체 데이터셋 fine-tuning용.

SynWoodScape(radial_poly)용 대응물은 `projects/models/fisheye_vox.py`이고 구조가 같다:
submodule(`third_party/models/simple_bev`)은 건드리지 않고 `Vox_util`을 서브클래싱해
`unproject_image_to_mem` 하나만 오버라이드한다. 원본이 쓰는 `pixB_T_camA`(내·외파라미터를
합친 4x4)는 핀홀 전용이라 무시하고, 외파라미터만 담긴 `camB_T_camA`로 얻은 카메라 좌표계
3D 점에 DS 투영을 직접 적용한다.

`double_sphere_pixel_coords`는 `projects.geometry.double_sphere.DoubleSphereCamera.project`의
torch 판이며, `tests/models/test_double_sphere_vox.py`가 그 numpy 구현과 직접 대조한다.

radial_poly 판과 다른 점 하나: 유효성 판정에 `z_cam > 0`을 쓰면 안 된다. DS는 화각이
180°를 넘을 수 있어(이 리그는 약 175°) 광축 뒤쪽(z<0) 점도 정상적으로 투영되므로,
모델 자신의 정의역 조건 `z > -w2 * |P|`를 써야 화각 주변부가 잘려나가지 않는다.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.basic  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402

from projects.bev_gt.grid import OccupancyGridSpec  # noqa: E402
from projects.datasets.simplebev_vox import vox_bounds, vox_dims  # noqa: E402
from projects.models.pixel_grid import (  # noqa: E402
    CONVENTIONS,
    DEFAULT_CONVENTION,
    normalize_pixel_grid2d,
)

_EPS = 1e-9


def double_sphere_pixel_coords(x_cam, y_cam, z_cam, xi, alpha, fx, fy, cx, cy, domain_cos_limit):
    """카메라 프레임(OpenCV) 3D 점 -> (u, v, valid). 모든 파라미터는 점 텐서와 브로드캐스트 가능해야 한다."""
    d1 = torch.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
    k = xi * d1 + z_cam
    d2 = torch.sqrt(x_cam ** 2 + y_cam ** 2 + k ** 2)
    denom = alpha * d2 + (1.0 - alpha) * k

    valid = (z_cam > -domain_cos_limit * d1) & (denom.abs() > _EPS)
    safe = torch.where(denom.abs() > _EPS, denom, torch.ones_like(denom))
    return fx * x_cam / safe + cx, fy * y_cam / safe + cy, valid


class DoubleSphereVoxUtil(utils.vox.Vox_util):
    """`unproject_image_to_mem`을 Double Sphere 투영으로 교체한 `Vox_util`."""

    # 좌우 반전 증강용. True면 **거울 세계**를 lifting한다 -- 자세한 유도는
    # `set_mirror_x`의 docstring에 있다. 기본은 꺼져 있고, 켜도 캘리브레이션은 건드리지 않는다.
    mirror_x = False

    # 특징맵 표본 좌표의 규약과 상수 오프셋. 근거는 `projects/models/pixel_grid.py`와
    # 진단 문서 §18.3. 기본값은 **기존 동작 보존**이다.
    pixel_convention = DEFAULT_CONVENTION
    pixel_offset = 0.0

    def set_mirror_x(self, mirror: bool) -> None:
        """좌우 반전된 세계를 lifting하도록 전환한다.

        **왜 캘리브레이션이나 extrinsic을 고치지 않는가.** 반전 증강은 "거울 세계를 거울 리그로
        관측한 것"이고, 그 관측 이미지는 원본 이미지의 좌우 반전이다. 그래서 필요한 것은 두
        가지뿐이다:

        1. BEV 질의점을 ref 프레임에서 좌우 반전한다(`x_ref -> -x_ref`). 그러면 투영식이
           **실제 카메라가 실제 점을 보는 픽셀** `u`를 준다.
        2. 그 `u`가 가리키는 내용은 반전된 특징맵에서 `(W-1) - x` 위치에 있다. 그래서 표본
           좌표를 뒤집는다.

        두 단계를 합치면 `V'[i] = F(π(T⁻¹ · D · q_i)) = flip(V)[i]`가 되어, 반전된 입력이
        **정확히** 반전된 BEV를 만든다(`tests/models/test_double_sphere_vox.py`가 실측으로
        고정한다). `cx`를 `W - cx`로 바꾸는 방식도 가능하지만, 상수가 특징맵 해상도와
        encoder stride에 얽혀 있어(upstream이 좌표를 픽셀 **인덱스**로 정규화한다) 값이
        조용히 틀리기 쉽다. 여기서는 그 상수가 `W - 1` 하나로 끝난다.
        """
        self.mirror_x = bool(mirror)

    def set_pixel_grid(self, convention: str, pixel_offset: float = 0.0) -> None:
        """특징맵 표본 좌표의 정규화 규약과 상수 오프셋[특징픽셀]을 정한다.

        근거와 두 규약의 차이는 `projects/models/pixel_grid.py`, 진단 문서 §18.3.
        """
        if convention not in CONVENTIONS:
            raise ValueError(f"convention은 {CONVENTIONS} 중 하나여야 한다: {convention!r}")
        self.pixel_convention = convention
        self.pixel_offset = float(pixel_offset)

    def set_camera_calibrations(self, cameras) -> None:
        """`cameras`: 배치의 카메라 순서(S)와 같은 순서의 `DoubleSphereCamera` 리스트.

        `Segnet.forward`가 (B, S, ...)를 `pack_seqdim`으로 (B*S, ...)로 합치므로(S가 안쪽),
        이 순서는 `Dataset`이 `rgb_camXs`/`cam0_T_camXs`를 쌓은 카메라 순서와 반드시 같아야 한다.
        """
        self.camera_count = len(cameras)
        fields = ("xi", "alpha", "fx", "fy", "cx", "cy", "width", "height")
        self._calib_tensors = tuple(
            torch.tensor([float(getattr(cam, f)) for cam in cameras], dtype=torch.float32)
            for f in fields
        ) + (
            torch.tensor([float(cam.domain_cos_limit) for cam in cameras], dtype=torch.float32),
        )

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X,
                               assert_cube=False, xyz_camA=None):
        assert getattr(self, "camera_count", None), "먼저 set_camera_calibrations()를 호출해야 한다"
        B, C, H, W = list(rgb_camB.shape)
        device = camB_T_camA.device

        if xyz_camA is None:
            xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=device)
            xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        if self.mirror_x:
            # (1) 질의점을 ref 프레임에서 좌우 반전한다. 이후 투영식은 **실제** 카메라가
            # **실제** 점을 보는 픽셀을 낸다(`set_mirror_x` 참고).
            xyz_camA = xyz_camA * torch.tensor(
                [-1.0, 1.0, 1.0], device=xyz_camA.device, dtype=xyz_camA.dtype
            )

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)  # B x N x 3, OpenCV 카메라 프레임
        x_cam, y_cam, z_cam = xyz_camB[:, :, 0], xyz_camB[:, :, 1], xyz_camB[:, :, 2]

        S = self.camera_count
        assert B % S == 0, f"packed batch({B})는 camera 수({S})로 나눠떨어져야 한다"
        n_repeat = B // S
        xi, alpha, fx, fy, cx, cy, native_w, native_h, w2 = (
            t.to(device).repeat(n_repeat).unsqueeze(1) for t in self._calib_tensors
        )  # 각각 (B, 1) -- 점 차원(N)과 브로드캐스트된다

        # native(캘리브레이션 원본) 해상도로 계산한 뒤 지금 rgb_camB의 해상도로 스케일한다.
        # DS 투영은 (fx, cx)/(fy, cy)에 대해 선형이라 사후 스케일과 계수 사전 스케일이 동치다.
        u_native, v_native, valid = double_sphere_pixel_coords(
            x_cam, y_cam, z_cam, xi, alpha, fx, fy, cx, cy, w2
        )
        x = u_native * (float(W) / native_w)
        y = v_native * (float(H) / native_h)
        # 리사이즈 규약과 stride 8 수용영역 중심이 남긴 상수 편이를 여기서 보정한다.
        # **반전과 유효 영역 판정보다 앞이어야 한다** -- 이유는 `pixel_grid` 모듈 docstring.
        if self.pixel_offset:
            x = x + float(self.pixel_offset)
            y = y + float(self.pixel_offset)
        if self.mirror_x:
            # (2) 입력이 좌우 반전된 특징맵이므로 표본 좌표도 뒤집는다. `torch.flip`이 픽셀
            # 인덱스 `j`를 `W-1-j`로 보내므로 상수는 정확히 `W - 1`이다 -- 정규화 규약과도,
            # 해상도·stride와도 무관하다(`x`는 아직 픽셀 인덱스 좌표다).
            x = float(W - 1) - x

        x_valid = (x > -0.5) & (x < float(W - 0.5))
        y_valid = (y > -0.5) & (y < float(H - 0.5))
        valid_mem = (x_valid & y_valid & valid).reshape(B, 1, Z, Y, X).float()

        y_pixB, x_pixB = normalize_pixel_grid2d(y, x, H, W, self.pixel_convention)
        xyz_pixB = torch.stack([x_pixB, y_pixB, torch.zeros_like(x)], dim=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB.unsqueeze(2), xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        return values * valid_mem


def build_double_sphere_vox_util(grid_spec: OccupancyGridSpec, cameras,
                                 height_margin_m: float = 0.25, device="cpu",
                                 mirror_x: bool = False,
                                 pixel_convention: str = DEFAULT_CONVENTION,
                                 pixel_offset: float = 0.0):
    Z, Y, X = vox_dims(grid_spec)
    bounds = vox_bounds(grid_spec, height_margin_m)
    scene_centroid = torch.zeros(1, 3, dtype=torch.float32, device=device)
    vox_util = DoubleSphereVoxUtil(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds,
                                   assert_cube=False)
    vox_util.set_camera_calibrations(cameras)
    vox_util.set_mirror_x(mirror_x)
    vox_util.set_pixel_grid(pixel_convention, pixel_offset)
    return vox_util
