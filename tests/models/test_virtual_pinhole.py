"""가상 핀홀 경로의 기하 정합성을 고정한다 -- `02_projection_and_prior` 축 A.

**무엇을 고정하나.** 리샘플링(`remap_maps`)과 lifting(`VirtualPinholeVoxUtil`)이 같은
광선 규약을 쓰는가. 둘이 어긋나면 편 이미지에서 특징을 **엉뚱한 화소**에서 뽑게 되는데,
학습은 그래도 돌아가고 성능만 조금 나빠진다 -- 즉 조용히 틀린다. 그러면 축 A의 결론이
"어안이 낫다"가 아니라 "우리 핀홀 구현에 버그가 있었다"가 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.geometry.double_sphere import load_cameras  # noqa: E402
from projects.models.virtual_pinhole import (  # noqa: E402
    IMAGE_CIRCLE_PX,
    pinhole_focal_px,
    pinhole_intrinsics,
    remap_maps,
    VirtualPinholeVoxUtil,
)

_CALIB = _REPO_ROOT / "dataset/sj_datasets/common/calibration/calib.yaml"
OUT_W, OUT_H = 512, 288


@pytest.fixture(scope="module")
def camera():
    if not _CALIB.exists():
        pytest.skip(f"캘리브레이션이 없다: {_CALIB}")
    return load_cameras(_CALIB)["front"]


def test_focal_matches_requested_fov():
    """화각의 정의 그대로: 이미지 가장자리 광선이 정확히 HFOV/2에 있다."""
    for hfov in (90.0, 120.0, 150.0):
        f = pinhole_focal_px(hfov, OUT_W)
        edge_angle = np.degrees(np.arctan((OUT_W / 2.0) / f))
        assert edge_angle == pytest.approx(hfov / 2.0, abs=1e-9)


def test_fov_at_or_above_180_is_rejected():
    """`tan(90°)`가 발산한다 -- 조용히 거대한 초점거리를 내지 말고 멈춰야 한다."""
    for bad in (180.0, 200.0, 0.0, -10.0):
        with pytest.raises(ValueError):
            pinhole_focal_px(bad, OUT_W)


def test_intrinsics_and_remap_share_one_ray_convention(camera):
    """**핵심 계약.** intrinsic이 광선을 보내는 핀홀 화소와, 그 화소가 remap으로 끌어오는
    어안 화소가, 같은 광선을 DS로 직접 투영한 결과와 일치해야 한다."""
    hfov = 120.0
    f = pinhole_focal_px(hfov, OUT_W)
    map_x, map_y, valid = remap_maps(camera, hfov, OUT_W, OUT_H, IMAGE_CIRCLE_PX["front"])

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(400):
        u = float(rng.integers(0, OUT_W))
        v = float(rng.integers(0, OUT_H))
        if not valid[int(v), int(u)]:
            continue
        ray = np.array([[(u + 0.5 - OUT_W / 2.0) / f, (v + 0.5 - OUT_H / 2.0) / f, 1.0]])
        x_src, y_src, ok = camera.project(ray)
        assert ok[0]
        assert x_src[0] == pytest.approx(map_x[int(v), int(u)], abs=1e-3)
        assert y_src[0] == pytest.approx(map_y[int(v), int(u)], abs=1e-3)
        checked += 1
    assert checked > 100, "유효 화소를 충분히 검사하지 못했다"


def test_vox_util_projection_inverts_the_remap(camera):
    """lifting이 쓰는 투영이 intrinsic과 같은가 -- 광선을 넣으면 그 화소가 나와야 한다."""
    hfov = 150.0
    k = pinhole_intrinsics(hfov, OUT_W, OUT_H)
    vox = VirtualPinholeVoxUtil(8, 4, 8, scene_centroid=torch.zeros(1, 3),
                                bounds=(-3.0, 3.0, -0.25, 1.75, -4.0, 2.0), assert_cube=False)
    vox.set_pinhole(hfov, OUT_W, OUT_H, n_cameras=1)

    rng = np.random.default_rng(1)
    pts = rng.normal(size=(64, 3)) * np.array([2.0, 1.0, 0.0]) + np.array([0.0, 0.0, 3.0])
    x, y, z = (torch.tensor(pts[:, i], dtype=torch.float64).unsqueeze(0) for i in range(3))
    u, v, valid, nw, nh = vox._project_to_native(x, y, z, B=1, device="cpu")

    expected_u = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
    expected_v = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
    assert valid.all()
    assert float(nw) == OUT_W and float(nh) == OUT_H
    np.testing.assert_allclose(u.numpy()[0], expected_u, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(v.numpy()[0], expected_v, rtol=1e-9, atol=1e-9)


def test_points_behind_the_camera_are_invalid(camera):
    """핀홀은 z<=0을 담을 수 없다. 조용히 앞쪽 화소로 접히면 안 된다."""
    vox = VirtualPinholeVoxUtil(8, 4, 8, scene_centroid=torch.zeros(1, 3),
                                bounds=(-3.0, 3.0, -0.25, 1.75, -4.0, 2.0), assert_cube=False)
    vox.set_pinhole(120.0, OUT_W, OUT_H, n_cameras=1)
    x = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    y = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    z = torch.tensor([[2.0, -2.0, 0.0]], dtype=torch.float64)
    _, _, valid, _, _ = vox._project_to_native(x, y, z, B=1, device="cpu")
    assert valid.tolist() == [[True, False, False]]


def test_wider_fov_keeps_more_of_the_lens_and_costs_resolution():
    """화각 절충이 실제로 그 방향인가 -- 원장 §6.2 표의 전제."""
    f120, f150 = pinhole_focal_px(120.0, OUT_W), pinhole_focal_px(150.0, OUT_W)
    assert f150 < f120                      # 넓히면 정면 해상도를 판다
    edge120 = np.degrees(np.arctan((OUT_W / 2.0) / f120))
    edge150 = np.degrees(np.arctan((OUT_W / 2.0) / f150))
    assert edge150 > edge120                # 대신 더 넓은 광선을 담는다
