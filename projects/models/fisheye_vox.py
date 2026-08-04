"""Simple-BEV `Vox_util.unproject_image_to_mem`(핀홀 전용)을 실제 어안(`radial_poly`) 투영으로
바꿔치기하는 래퍼 — ROADMAP Phase 3.2.

submodule(`third_party/models/simple_bev`)은 직접 수정하지 않는다. 대신 `Vox_util`을
서브클래싱해서 `unproject_image_to_mem` 하나만 오버라이드한다. `Segnet.forward()`는 이
인스턴스를 그대로 받아 쓰므로(생성자·forward 호출 모두에 같은 `vox_util` 객체를 넘기면 됨),
`nets/segnet.py`는 한 줄도 바꾸지 않는다.

원본 `unproject_image_to_mem`은 외파라미터+내파라미터를 하나로 합친 4x4 행렬
(`pixB_T_camA`)에 원근분할을 적용해 픽셀 좌표를 얻는다 — 핀홀 전용이라 그대로 못 쓴다.
반면 별도 인자로 넘어오는 `camB_T_camA`(외파라미터만, 강체변환)는 어안이든 핀홀이든
그대로 재사용 가능하다. 그래서 이 클래스는 `pixB_T_camA`를 무시하고, `camB_T_camA`로 얻은
카메라 좌표계 3D 점에 WoodScape 공식 `radial_poly` 계수(`projects.geometry.fisheye`가 이미
검증한 것과 같은 `Camera` 객체에서 그대로 읽음)를 적용해 픽셀 좌표를 계산한다.

`radial_poly_pixel_coords`는 WoodScape `RadialPolyCamProjection.project_3d_to_2d` +
`Camera`의 aspect_ratio/principal-point 후처리와 수학적으로 동일한 torch 버전이며,
`tests/models/test_fisheye_vox.py`가 실제 그 numpy 구현과 직접 대조해 검증한다.
"""
import math
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


def radial_poly_pixel_coords(x_cam, y_cam, z_cam, k, cx, cy, aspect_ratio):
    """카메라 프레임(OpenCV: x=오른쪽, y=아래, z=광축) 3D 점 -> 어안 픽셀 좌표 (u, v).

    `k`는 마지막 차원이 4(k1..k4)이고 `x_cam`과 브로드캐스트 가능해야 한다(예: 점별 shape이
    `(B, N)`이면 `k`는 `(B, 1, 4)`). `cx`/`cy`/`aspect_ratio`는 `x_cam`과 바로 브로드캐스트
    가능해야 한다(`(B, 1)` 또는 스칼라).
    """
    chi = torch.sqrt(x_cam ** 2 + y_cam ** 2)
    theta = math.pi / 2.0 - torch.atan2(z_cam, chi)
    theta_powers = torch.stack([theta ** (i + 1) for i in range(k.shape[-1])], dim=-1)
    rho = torch.sum(theta_powers * k, dim=-1)

    scale = torch.where(chi > 1e-9, rho / chi.clamp(min=1e-9), torch.zeros_like(rho))
    lens_x = scale * x_cam
    lens_y = scale * y_cam

    u = lens_x + cx
    v = lens_y * aspect_ratio + cy
    return u, v


class FisheyeVoxUtil(utils.vox.Vox_util):
    """`unproject_image_to_mem`을 실제 어안(radial_poly) 투영으로 교체한 `Vox_util`."""

    def set_camera_calibrations(self, cameras) -> None:
        """`cameras`: 배치의 카메라 순서(S)와 같은 순서의 `projects.geometry.fisheye.Camera` 리스트.

        `Segnet.forward`가 (B, S, ...)를 `pack_seqdim`으로 (B*S, ...)로 합치므로(S가 안쪽,
        b가 바깥쪽), 이 순서는 `Dataset`이 `rgb_camXs`/`cam0_T_camXs`를 쌓은 카메라 순서와
        반드시 같아야 한다.
        """
        self.camera_count = len(cameras)
        self._calib_tensors = (
            torch.tensor([list(cam.lens.coefficients[:4]) for cam in cameras], dtype=torch.float32),
            torch.tensor([float(cam.cx) for cam in cameras], dtype=torch.float32),
            torch.tensor([float(cam.cy) for cam in cameras], dtype=torch.float32),
            torch.tensor([float(cam.aspect_ratio) for cam in cameras], dtype=torch.float32),
            torch.tensor([float(cam.width) for cam in cameras], dtype=torch.float32),
            torch.tensor([float(cam.height) for cam in cameras], dtype=torch.float32),
        )

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False, xyz_camA=None):
        assert getattr(self, "camera_count", None), "먼저 set_camera_calibrations()를 호출해야 한다"
        B, C, H, W = list(rgb_camB.shape)
        device = camB_T_camA.device

        if xyz_camA is None:
            xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=device)
            xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)  # B x N x 3, OpenCV 카메라 프레임
        x_cam, y_cam, z_cam = xyz_camB[:, :, 0], xyz_camB[:, :, 1], xyz_camB[:, :, 2]

        S = self.camera_count
        assert B % S == 0, f"packed batch({B})는 camera 수({S})로 나눠떨어져야 한다"
        n_repeat = B // S
        k, cx, cy, aspect, native_w, native_h = (
            t.to(device).repeat(n_repeat, *([1] * (t.dim() - 1))) for t in self._calib_tensors
        )  # k: (B, 4), 나머지: (B,)

        # native(캘리브레이션 원본 해상도) 좌표로 계산한 뒤, 지금 rgb_camB의 해상도(H, W --
        # 인코더가 낮춘 feature map 해상도일 수도, 원본 이미지 해상도일 수도 있음)로 한 번에
        # 스케일한다. 다항식은 계수에 대해 선형이므로 사후 스케일과 계수 사전 스케일이 동치다.
        u_native, v_native = radial_poly_pixel_coords(
            x_cam, y_cam, z_cam,
            k.unsqueeze(1), cx.unsqueeze(1), cy.unsqueeze(1), aspect.unsqueeze(1),
        )
        x = u_native * (float(W) / native_w).unsqueeze(1)
        y = v_native * (float(H) / native_h).unsqueeze(1)

        x_valid = (x > -0.5) & (x < float(W - 0.5))
        y_valid = (y > -0.5) & (y < float(H - 0.5))
        z_valid = z_cam > 0.0  # radial_poly가 광축 뒤쪽도 유한 픽셀로 접으므로 반드시 필요
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        y_pixB, x_pixB = utils.basic.normalize_grid2d(y, x, H, W)
        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], dim=2)
        rgb_camB_5d = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB_5d, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values


def build_fisheye_vox_util(grid_spec: OccupancyGridSpec, cameras, height_margin_m: float = 0.25, device="cpu"):
    Z, Y, X = vox_dims(grid_spec)
    bounds = vox_bounds(grid_spec, height_margin_m)
    scene_centroid = torch.zeros(1, 3, dtype=torch.float32, device=device)
    vox_util = FisheyeVoxUtil(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds, assert_cube=False)
    vox_util.set_camera_calibrations(cameras)
    return vox_util
