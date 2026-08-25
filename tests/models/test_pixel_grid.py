"""특징맵 표본 좌표 규약을 고정한다 -- 진단 문서 §18.3의 결함과 그 수정.

이 파일이 하는 일은 셋이다.

1. **결함을 숫자로 못박는다.** `legacy_index`(현행)에서 표본 위치가 `x*W/(W-1) - 0.5`가
   되어 §18.3의 표(8 -> 7.63, 63 -> 값이 절반)가 그대로 재현되는지.
2. **수정이 정확한지 고정한다.** `pixel_center`에서 표본 위치가 `x`와 **정확히** 같은지.
3. **오프셋이 적용되는 자리를 고정한다.** 좌우 반전(`mirror_x`)보다 앞이어야 하고,
   뒤에 놓으면 x축에서 부호가 뒤집힌다.

`legacy_index`가 기존 동작을 비트 단위로 보존하는지도 같이 본다 -- 대조군이 대조군이
아니게 되면 스윕 전체가 무의미해진다.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from projects.bev_gt.grid import ROBOT_GRID_SPEC
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES, load_cameras, load_ego_T_cams
from projects.models.double_sphere_vox import build_double_sphere_vox_util
from projects.models.pixel_grid import LEGACY_INDEX, PIXEL_CENTER, normalize_pixel_grid2d

CALIB_PATH = Path("dataset/sj_datasets/common/calibration/calib.yaml")
requires_calib = pytest.mark.skipif(
    not CALIB_PATH.exists(), reason="self-collected calibration not available locally"
)

# 실제 fine-tuning 형상: 1280x720 -> 512x288 입력 -> stride 8 특징맵.
FEAT_H, FEAT_W = 36, 64


def _sample_ramp(x_targets, convention, pixel_offset=0.0, W=FEAT_W, H=FEAT_H):
    """열 인덱스를 값으로 갖는 램프 특징맵을 `x_targets`에서 표본해 **실제 표본 위치**를 읽는다.

    값 = 열 인덱스이므로 반환값이 곧 표본 위치다(zero-padding에 걸리면 그만큼 줄어든다).
    `unproject_image_to_mem`이 쓰는 것과 같은 정규화 + `grid_sample(align_corners=False)` 조합.
    """
    feat = torch.arange(W, dtype=torch.float64).view(1, 1, 1, W).expand(1, 1, H, W).contiguous()

    x = torch.tensor(x_targets, dtype=torch.float64) + float(pixel_offset)
    y = torch.full_like(x, (H - 1) / 2.0)          # 램프가 y에 상수라 어느 행이든 무방
    grid_y, grid_x = normalize_pixel_grid2d(y, x, H, W, convention)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)

    return F.grid_sample(feat, grid, align_corners=False).view(-1)


def test_legacy_normalization_is_bit_identical_to_upstream():
    """대조군은 기존 동작 그대로여야 한다 -- 한 비트라도 다르면 스윕의 기준점이 사라진다."""
    import utils.basic  # simple_bev submodule -- pixel_grid import가 sys.path를 세팅한다

    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(-80.0, 140.0, 5000))
    y = torch.tensor(rng.uniform(-40.0, 80.0, 5000))

    expected_y, expected_x = utils.basic.normalize_grid2d(y, x, FEAT_H, FEAT_W)
    got_y, got_x = normalize_pixel_grid2d(y, x, FEAT_H, FEAT_W, LEGACY_INDEX)

    assert torch.equal(got_x, expected_x)
    assert torch.equal(got_y, expected_y)


def test_legacy_reproduces_the_documented_sampling_defect():
    """§18.3의 표 그대로다 -- 규약 불일치가 `x*W/(W-1) - 0.5`의 계통 오차를 만든다.

    중심에서 0이고 가장자리로 갈수록 커지며, **마지막 열은 zero-padding과 섞여 값이 절반**이
    된다. 이 테스트가 깨진다면 결함이 사라졌거나 다른 결함으로 바뀐 것이다.
    """
    targets = [8.0, 16.0, 32.0, 48.0, 63.0]
    got = _sample_ramp(targets, LEGACY_INDEX).numpy()

    expected = np.array([7.6270, 15.7540, 32.0079, 48.2619, 31.5000])
    np.testing.assert_allclose(got, expected, atol=5e-4)

    # 닫힌 형태와도 대조한다 -- 표의 숫자가 우연이 아니라 규약 합성의 결과임을 고정한다.
    x = np.array(targets)
    analytic = x * FEAT_W / (FEAT_W - 1) - 0.5
    np.testing.assert_allclose(got[:-1], analytic[:-1], atol=1e-9)
    assert analytic[-1] == pytest.approx(63.5)        # 텐서 밖 -> 절반만 남는다


def test_pixel_center_samples_exactly_where_asked():
    """수정의 계약: 표본 위치 = 요청 위치, **정확히**. 마지막 열도 감쇠하지 않는다."""
    targets = [0.0, 8.0, 16.0, 32.0, 48.0, 63.0]
    got = _sample_ramp(targets, PIXEL_CENTER).numpy()

    np.testing.assert_allclose(got, np.array(targets), atol=1e-9)


def test_pixel_center_is_exact_at_subpixel_positions_too():
    """정수 위치만 맞고 사이가 틀리면 배율 오차가 남아 있는 것이다."""
    targets = np.linspace(0.0, float(FEAT_W - 1), 257)
    got = _sample_ramp(targets.tolist(), PIXEL_CENTER).numpy()

    np.testing.assert_allclose(got, targets, atol=1e-9)


@pytest.mark.parametrize("offset", [-1.0, -0.5, 0.5, 1.0])
def test_pixel_offset_shifts_the_sample_by_exactly_that_much(offset):
    """스윕이 재는 양의 정의 -- 오프셋 `o`는 표본 위치를 정확히 `o` 특징픽셀 옮긴다."""
    targets = [8.0, 16.0, 32.0, 48.0]
    got = _sample_ramp(targets, PIXEL_CENTER, pixel_offset=offset).numpy()

    np.testing.assert_allclose(got, np.array(targets) + offset, atol=1e-9)


@requires_calib
@pytest.mark.parametrize("convention", [LEGACY_INDEX, PIXEL_CENTER])
def test_pixel_offset_is_applied_before_the_mirror_flip(convention):
    """오프셋이 있어도 반전 등변성이 유지돼야 한다.

    오프셋을 반전 **뒤에** 더하면 x축에서 부호가 뒤집혀
    `(W-1) - (x+o) != (W-1) - x + o`가 되고, 반전 증강을 켠 순간 좌우가 서로 다른
    기하로 학습된다. `test_double_sphere_vox.py`의 등변성 테스트는 오프셋 0에서만
    돌므로 그 버그를 잡지 못한다.
    """
    torch.manual_seed(0)
    offset = -0.5
    spec = ROBOT_GRID_SPEC
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]
    plain = build_double_sphere_vox_util(
        spec, cameras, pixel_convention=convention, pixel_offset=offset
    )
    mirror = build_double_sphere_vox_util(
        spec, cameras, mirror_x=True, pixel_convention=convention, pixel_offset=offset
    )

    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    feat = torch.randn(len(cameras), 4, FEAT_H, FEAT_W)

    volume = plain.unproject_image_to_mem(feat, camB_T_camA, camB_T_camA, Z, Y, X)
    mirrored = mirror.unproject_image_to_mem(
        torch.flip(feat, dims=[-1]), camB_T_camA, camB_T_camA, Z, Y, X
    )
    expected = torch.flip(volume, dims=[-1])

    assert volume.abs().sum() > 0
    assert (mirrored - expected).abs().max() < 1e-3


@requires_calib
def test_pixel_offset_actually_changes_the_lifted_volume():
    """0.5 특징픽셀(= native 10 px)이 실제로 표본을 옮기는지 -- 조용히 무시되면 스윕이 헛돈다."""
    torch.manual_seed(0)
    spec = ROBOT_GRID_SPEC
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]

    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    feat = torch.randn(len(cameras), 4, FEAT_H, FEAT_W)

    def lift(pixel_offset):
        vox = build_double_sphere_vox_util(
            spec, cameras, pixel_convention=PIXEL_CENTER, pixel_offset=pixel_offset
        )
        return vox.unproject_image_to_mem(feat, camB_T_camA, camB_T_camA, Z, Y, X)

    base = lift(0.0)
    shifted = lift(-0.5)
    # 이웃 특징값 차이가 O(1)인 랜덤 특징맵이므로 반픽셀이면 그 규모의 차이가 나야 한다.
    assert (shifted - base).abs().max() > 0.3


@requires_calib
def test_convention_change_actually_changes_the_lifted_volume():
    """`legacy_index`와 `pixel_center`가 다른 결과를 낸다 -- 스윕의 전제."""
    torch.manual_seed(0)
    spec = ROBOT_GRID_SPEC
    cameras = [load_cameras(CALIB_PATH)[name] for name in FINETUNE_CAMERA_NAMES]

    ego_T_cams = load_ego_T_cams(CALIB_PATH)
    ref_T_cams = np.stack([ref_T_cam_from_ego_T_cam(ego_T_cams[n]) for n in FINETUNE_CAMERA_NAMES])
    camB_T_camA = torch.from_numpy(np.linalg.inv(ref_T_cams)).float()

    Z, Y, X = spec.n_rows, 1, spec.n_cols
    feat = torch.randn(len(cameras), 4, FEAT_H, FEAT_W)

    def lift(convention):
        vox = build_double_sphere_vox_util(spec, cameras, pixel_convention=convention)
        return vox.unproject_image_to_mem(feat, camB_T_camA, camB_T_camA, Z, Y, X)

    assert (lift(PIXEL_CENTER) - lift(LEGACY_INDEX)).abs().max() > 0.3
