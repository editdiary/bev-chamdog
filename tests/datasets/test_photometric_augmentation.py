import pytest
import torch

from projects.datasets.photometric import (
    IDENTITY_PHOTOMETRIC,
    PhotometricParams,
    apply_photometric,
    sample_photometric_params,
)


def _rgb(seed: int = 0, cameras: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(cameras, 3, 8, 12, generator=generator)


def test_identity_params_leave_the_image_unchanged():
    rgb = _rgb()

    assert torch.allclose(apply_photometric(rgb, IDENTITY_PHOTOMETRIC), rgb, atol=1e-6)


def test_brightness_moves_the_mean_in_the_expected_direction():
    rgb = _rgb()

    brighter = apply_photometric(rgb, PhotometricParams(brightness=1.2))
    darker = apply_photometric(rgb, PhotometricParams(brightness=0.8))

    assert brighter.mean() > rgb.mean()
    assert darker.mean() < rgb.mean()


def test_output_stays_inside_the_zero_one_range():
    rgb = _rgb()

    out = apply_photometric(
        rgb,
        PhotometricParams(brightness=1.5, contrast=1.5, saturation=1.5, gamma=0.5, noise_std=0.2),
    )

    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_all_cameras_receive_the_same_transform():
    """카메라마다 다른 색보정을 걸면 실제로는 없는 카메라 간 색차를 학습하게 된다.

    리그 전체가 같은 장면을 같은 조명에서 찍으므로 한 샘플 안에서는 파라미터가 공유돼야 한다.
    """
    rgb = _rgb()
    params = PhotometricParams(brightness=1.1, contrast=0.9, saturation=1.2, gamma=0.9)

    together = apply_photometric(rgb, params)
    per_camera = torch.cat([apply_photometric(rgb[i : i + 1], params) for i in range(rgb.shape[0])])

    assert torch.allclose(together, per_camera, atol=1e-6)


def test_noise_is_the_only_source_of_per_pixel_randomness():
    rgb = _rgb()
    params = PhotometricParams(brightness=1.1, noise_std=0.05)
    generator = torch.Generator().manual_seed(7)

    first = apply_photometric(rgb, params, generator=generator)
    generator = torch.Generator().manual_seed(7)
    second = apply_photometric(rgb, params, generator=generator)

    assert torch.allclose(first, second, atol=1e-6)
    assert not torch.allclose(first, apply_photometric(rgb, PhotometricParams(brightness=1.1)))


def test_sampled_params_stay_within_the_configured_ranges():
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        params = sample_photometric_params(generator=generator)
        assert 0.8 <= params.brightness <= 1.2
        assert 0.8 <= params.contrast <= 1.2
        assert 0.8 <= params.saturation <= 1.2
        assert 0.8 <= params.gamma <= 1.25
        assert 0.0 <= params.noise_std <= 0.02


def test_sampling_is_reproducible_from_a_seeded_generator():
    a = sample_photometric_params(generator=torch.Generator().manual_seed(3))
    b = sample_photometric_params(generator=torch.Generator().manual_seed(3))
    c = sample_photometric_params(generator=torch.Generator().manual_seed(4))

    assert a == b
    assert a != c


def test_sampling_actually_varies_across_draws():
    generator = torch.Generator().manual_seed(0)
    draws = {sample_photometric_params(generator=generator) for _ in range(20)}

    assert len(draws) == 20


def test_empty_camera_stack_is_handled():
    rgb = torch.empty(0, 3, 8, 12)

    assert apply_photometric(rgb, PhotometricParams(brightness=1.2)).shape == rgb.shape
