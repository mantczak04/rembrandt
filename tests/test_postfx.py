"""Tests for sensor-domain post-processing."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random

import numpy as np
import pytest

from rembrandt.postfx import (
    FramePostFx,
    apply_postfx,
    sample_frame_postfx,
)


def _solid_rgb(*, value: int = 128, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_postfx_module_is_bpy_free() -> None:
    import rembrandt.postfx as postfx_module

    source = Path(postfx_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_sample_frame_postfx_respects_ranges() -> None:
    params = sample_frame_postfx(
        frame_index=0,
        gaussian_noise_sigma=(1.0, 2.0),
        blur_radius=(0.5, 1.0),
        jpeg_quality=(60, 80),
        exposure_ev=(-0.5, 0.5),
        seed=7,
    )
    assert 1.0 <= params.gaussian_noise_sigma <= 2.0
    assert 0.5 <= params.blur_radius <= 1.0
    assert 60 <= params.jpeg_quality <= 80
    assert -0.5 <= params.exposure_ev <= 0.5


def test_sample_frame_postfx_deterministic_with_seed() -> None:
    kwargs = {
        "gaussian_noise_sigma": (0.0, 8.0),
        "blur_radius": (0.0, 1.2),
        "jpeg_quality": (55, 95),
        "exposure_ev": (-0.7, 0.7),
        "seed": 42,
    }
    first = sample_frame_postfx(frame_index=3, **kwargs)
    second = sample_frame_postfx(frame_index=3, **kwargs)
    assert first == second
    different = sample_frame_postfx(frame_index=4, **kwargs)
    assert different != first


def test_sample_frame_postfx_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        sample_frame_postfx(
            frame_index=-1,
            gaussian_noise_sigma=(0.0, 1.0),
            blur_radius=(0.0, 1.0),
            jpeg_quality=(55, 95),
            exposure_ev=(-0.5, 0.5),
            seed=0,
        )


def test_apply_postfx_preserves_shape_and_dtype() -> None:
    rgb = _solid_rgb()
    params = FramePostFx(
        gaussian_noise_sigma=2.0,
        blur_radius=0.8,
        jpeg_quality=75,
        exposure_ev=0.3,
    )
    result = apply_postfx(rgb, params, rng=Random(0))
    assert result.shape == rgb.shape
    assert result.dtype == np.uint8


def test_apply_postfx_identity_when_effects_are_neutral() -> None:
    rgb = _solid_rgb(value=200)
    params = FramePostFx(
        gaussian_noise_sigma=0.0,
        blur_radius=0.0,
        jpeg_quality=95,
        exposure_ev=0.0,
    )
    result = apply_postfx(rgb, params, rng=Random(0))
    np.testing.assert_array_equal(result, rgb)


def test_apply_postfx_deterministic_with_rng() -> None:
    rgb = _solid_rgb(value=100)
    params = FramePostFx(
        gaussian_noise_sigma=4.0,
        blur_radius=0.5,
        jpeg_quality=70,
        exposure_ev=0.2,
    )
    first = apply_postfx(rgb, params, rng=Random(99))
    second = apply_postfx(rgb, params, rng=Random(99))
    np.testing.assert_array_equal(first, second)


def test_apply_postfx_rejects_invalid_shape() -> None:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    params = FramePostFx(0.0, 0.0, 95, 0.0)
    with pytest.raises(ValueError, match="shape"):
        apply_postfx(rgba, params)


def test_apply_postfx_changes_image_when_effects_active() -> None:
    rgb = _solid_rgb(value=180, size=64)
    params = FramePostFx(
        gaussian_noise_sigma=5.0,
        blur_radius=1.0,
        jpeg_quality=60,
        exposure_ev=0.5,
    )
    result = apply_postfx(rgb, params, rng=Random(1))
    assert not np.array_equal(result, rgb)


def test_apply_postfx_exposure_scales_brightness() -> None:
    rgb = _solid_rgb(value=100, size=8)
    brighter = apply_postfx(
        rgb,
        FramePostFx(0.0, 0.0, 100, 1.0),
        rng=Random(0),
    )
    assert int(brighter[0, 0, 0]) > 100


def test_frame_postfx_asdict_round_trip() -> None:
    params = sample_frame_postfx(
        frame_index=0,
        gaussian_noise_sigma=(0.0, 8.0),
        blur_radius=(0.0, 1.2),
        jpeg_quality=(55, 95),
        exposure_ev=(-0.7, 0.7),
        seed=1,
    )
    dumped = asdict(params)
    assert set(dumped) == {
        "gaussian_noise_sigma",
        "blur_radius",
        "jpeg_quality",
        "exposure_ev",
    }
