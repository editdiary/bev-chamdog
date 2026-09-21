"""어안을 가상 핀홀로 편 뒤 **통상적인 핀홀 파이프라인**으로 lifting하는 경로.

캠페인 `02_projection_and_prior` 축 A의 대조 팔이다(원장 §6.2). 확정 설정은 DS-native이고,
이 모듈은 "대부분의 BEV 인식 파이프라인처럼 먼저 undistort하면 어떻게 되는가"를 재기 위한
비교 대상을 만든다.

--------------------------------------------------------------------------------
§1. 왜 화각을 정해야 하는가
--------------------------------------------------------------------------------
핀홀 이미지는 평면이라 광축에서 θ 벗어난 광선이 중심에서 `f·tan(θ)`에 찍힌다.
**`tan(90°)`가 무한대**이므로 핀홀은 90° 이상의 광선을 **해상도와 무관하게** 담을 수 없다.
구현 한계가 아니라 원리적 불가능이다. 이 리그의 렌즈는 실측 ±96~101°를 보므로 절반 이상이
어떤 핀홀에도 들어가지 않는다. 그래서 화각을 정해야 하고, 정하면 정보를 버린다.

**동결값은 120°와 150°다**(`tools/measure_projection_tradeoff.py` §2). 512 px 폭 출력에서
120°는 정면 각해상도가 어안과 일치하고(2.58 대 2.55 px/deg) BEV 격자의 93.1 %를 덮는다 --
"베이스라인을 굶겼다"는 반박이 막히는 지점이다. 150°는 커버리지를 96.7 %로 올려 주는 대신
정면 해상도를 0.47배로 떨어뜨린다. 불리한 조건과 유리한 조건 양쪽을 본다.

--------------------------------------------------------------------------------
§2. 비교가 성립하려면 나머지가 전부 같아야 한다
--------------------------------------------------------------------------------
- **출력 해상도**를 DS 팔과 같은 512x288로 둔다. 다르면 연산량이 달라져 비교가 깨진다.
- **광축**을 어안과 같게 둔다. 새 pose를 도입하면 extrinsic 오차가 섞인다.
- **`pixel_center` 규약**을 그대로 쓴다. upstream `Vox_util`은 `legacy_index` 규약의
  배율 오차(`x·W/(W-1) − 0.5`)를 갖고 있어서, 그것을 쓰면 "DS + 옳은 표본 위치" 대
  "핀홀 + 틀린 표본 위치"를 비교하게 된다. 그래서 `build_vox_util`이 아니라
  `DoubleSphereVoxUtil`의 경로를 **투영식만 바꿔** 재사용한다.
- **`permanent_blind` 마스크는 DS 기준 그대로** 둔다. 핀홀이 못 보는 셀도 과제에는 남아
  있어야 `01_overall`과 같은 과제가 되고, 커버리지 손실이 결과에 정직하게 반영된다.
  (커버리지 자체는 별도로 보고한다 -- 원장 §6.2.)
- 화각 밖 화소는 검게 남으므로 **유효로 세지 않는다**(`remap_maps`의 `valid`).
"""
import numpy as np
import torch

from projects.bev_gt.grid import OccupancyGridSpec
from projects.datasets.simplebev_vox import (
    DEFAULT_HEIGHT_BINS,
    DEFAULT_HEIGHT_MAX_M,
    DEFAULT_HEIGHT_MIN_M,
    vox_bounds,
    vox_dims,
)
from projects.models.double_sphere_vox import DoubleSphereVoxUtil
from projects.models.pixel_grid import DEFAULT_CONVENTION

# 실측 이미지 원 반경 [native px] -- `tools/measure_projection_tradeoff.py` §3.
# DS 모델은 반각 121°(전체 242°)까지 유효하다고 주장하지만 실제 렌즈는 ±96~101°이고
# 그 바깥은 검은 화소다. 어안을 펼 때 그 검은 영역을 끌어오면 안 된다.
IMAGE_CIRCLE_PX = {"front": 572.0, "left": 592.0, "right": 588.0}


def pinhole_focal_px(hfov_deg: float, out_w: int) -> float:
    """수평 화각과 출력 폭에서 초점거리 [px]. `f = (W/2) / tan(HFOV/2)`."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"핀홀 화각은 0°와 180° 사이여야 한다(tan이 발산한다): {hfov_deg}")
    return (out_w / 2.0) / float(np.tan(np.deg2rad(hfov_deg) / 2.0))


def pinhole_intrinsics(hfov_deg: float, out_w: int, out_h: int) -> np.ndarray:
    """가상 핀홀의 4x4 intrinsic. 주점은 이미지 중심이다.

    `remap_maps`가 **같은 식**으로 광선을 만들므로 리샘플링과 lifting이 정확히 일치한다.
    절대 반 픽셀 규약은 두 팔에 동일하게 적용되므로 비교에 영향을 주지 않는다.
    """
    f = pinhole_focal_px(hfov_deg, out_w)
    k = np.eye(4, dtype=np.float64)
    k[0, 0] = k[1, 1] = f          # 정사각 화소 -- 가상 카메라이므로 종횡 왜곡을 넣지 않는다
    k[0, 2] = out_w / 2.0
    k[1, 2] = out_h / 2.0
    return k


def remap_maps(camera, hfov_deg: float, out_w: int, out_h: int, circle_px=None,
               src_wh=None):
    """가상 핀홀 화소 -> 원본 어안 화소. `cv2.remap(src, map_x, map_y, INTER_LINEAR)`용.

    `src_wh`를 주면 그 해상도로 **미리 축소된** 어안 이미지 좌표계로 반환한다.

    **왜 축소된 이미지에서 펴는가 (중요).** 두 팔이 **글자 그대로 같은 입력 이미지**를 보게
    하기 위해서다. DS 팔은 1280x720을 PIL로 512x288로 줄여 쓰는데, 핀홀 팔이 1280x720에서
    직접 펴면 원본 해상도를 더 많이 쓰게 되어 "투영 방식" 외의 차이가 하나 더 생긴다.
    게다가 120° 핀홀의 중심 각해상도(2.58 px/deg)는 native(≈6.4 px/deg)보다 낮아서 bilinear
    탭 하나로 끌어오면 **중심부에 에일리어싱**이 생긴다 -- PIL 축소는 안티에일리어싱을 하지만
    `cv2.remap`은 하지 않는다. 512x288에서 펴면 중심이 거의 1:1(2.55 -> 2.58)이라 두 문제가
    동시에 사라진다. 주변부는 어차피 핀홀이 확대하는 영역이라 잃는 실제 정보가 없다.

    반환 `(map_x, map_y, valid)` 전부 `(out_h, out_w)`. `valid=False`인 화소는 렌즈가
    담지 않은 방향이라 검게 둔다.
    """
    f = pinhole_focal_px(hfov_deg, out_w)
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float64),
                       np.arange(out_h, dtype=np.float64))
    # 화소 중심을 쓴다 -- `pinhole_intrinsics`의 주점 규약과 같아야 한다.
    rays = np.stack([(u + 0.5 - out_w / 2.0) / f,
                     (v + 0.5 - out_h / 2.0) / f,
                     np.ones_like(u)], axis=-1).reshape(-1, 3)
    x_src, y_src, ok = camera.project(rays)
    if circle_px is not None:
        ok = ok & (np.hypot(x_src - camera.cx, y_src - camera.cy) <= circle_px)
    ok = ok & (x_src >= 0) & (x_src <= camera.width - 1) & (y_src >= 0) & (y_src <= camera.height - 1)
    if src_wh is not None:
        src_w, src_h = src_wh
        x_src = x_src * (src_w / float(camera.width))
        y_src = y_src * (src_h / float(camera.height))
    map_x = np.where(ok, x_src, 0.0).astype(np.float32).reshape(out_h, out_w)
    map_y = np.where(ok, y_src, 0.0).astype(np.float32).reshape(out_h, out_w)
    return map_x, map_y, ok.reshape(out_h, out_w)


def undistort(image: np.ndarray, map_x, map_y, valid) -> np.ndarray:
    """어안 이미지를 가상 핀홀로 편다. 화각 밖은 0으로 둔다."""
    import cv2
    out = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out * valid[..., None] if out.ndim == 3 else out * valid


class VirtualPinholeVoxUtil(DoubleSphereVoxUtil):
    """투영식만 핀홀로 바꾼 `DoubleSphereVoxUtil`.

    나머지 경로(해상도 스케일, `pixel_offset`, `mirror_x`, 유효 영역 판정, `grid_sample`
    정규화)는 **상속으로 그대로 물려받는다.** 축 A가 투영 방식 하나만 다른 비교가 되려면
    그 경로가 비트 단위로 같아야 한다.
    """

    def set_pinhole(self, hfov_deg: float, out_w: int, out_h: int, n_cameras: int) -> None:
        self.camera_count = int(n_cameras)
        self.pinhole_hfov_deg = float(hfov_deg)
        self._pinhole_f = pinhole_focal_px(hfov_deg, out_w)
        self._pinhole_wh = (int(out_w), int(out_h))

    def _project_to_native(self, x_cam, y_cam, z_cam, B, device):
        """핀홀 투영. `native` 해상도는 편 이미지 자신의 해상도(512x288)다."""
        assert getattr(self, "_pinhole_f", None), "먼저 set_pinhole()을 호출해야 한다"
        out_w, out_h = self._pinhole_wh
        f = self._pinhole_f
        # 카메라 뒤(z<=0)는 핀홀이 담을 수 없다. DS 경로의 `domain_cos_limit`에 대응한다.
        valid = z_cam > 1e-6
        z_safe = torch.where(valid, z_cam, torch.ones_like(z_cam))
        u = f * (x_cam / z_safe) + out_w / 2.0
        v = f * (y_cam / z_safe) + out_h / 2.0
        shape = (1, 1)
        native_w = torch.full(shape, float(out_w), device=device)
        native_h = torch.full(shape, float(out_h), device=device)
        return u, v, valid, native_w, native_h


def build_virtual_pinhole_vox_util(grid_spec: OccupancyGridSpec, n_cameras: int,
                                   hfov_deg: float, out_w: int, out_h: int,
                                   height_margin_m: float = 0.25, device="cpu",
                                   mirror_x: bool = False,
                                   pixel_convention: str = DEFAULT_CONVENTION,
                                   pixel_offset: float = 0.0,
                                   height_bins: int = DEFAULT_HEIGHT_BINS,
                                   height_min_m=DEFAULT_HEIGHT_MIN_M,
                                   height_max_m=DEFAULT_HEIGHT_MAX_M):
    """`build_double_sphere_vox_util`과 같은 인자 규약. 카메라 대신 화각을 받는다."""
    Z, Y, X = vox_dims(grid_spec, height_bins)
    bounds = vox_bounds(grid_spec, height_margin_m, height_min_m, height_max_m, height_bins)
    scene_centroid = torch.zeros(1, 3, dtype=torch.float32, device=device)
    vox_util = VirtualPinholeVoxUtil(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds,
                                     assert_cube=False)
    vox_util.set_pinhole(hfov_deg, out_w, out_h, n_cameras)
    vox_util.set_mirror_x(mirror_x)
    vox_util.set_pixel_grid(pixel_convention, pixel_offset)
    return vox_util
