"""Sensor-domain post-processing for rendered frames. NO bpy.

All effects are geometry-preserving (pixel-space only): bounding boxes and
masks derived from the render alpha channel remain valid after post-fx.
Apply only to the final composited RGB image, never before label extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from random import Random
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

PostFxMode = Literal["off", "random"]


@dataclass(frozen=True)
class FramePostFx:
    """Per-frame sampled post-processing parameters."""

    gaussian_noise_sigma: float
    blur_radius: float
    jpeg_quality: int
    exposure_ev: float


def sample_frame_postfx(
    *,
    frame_index: int,
    gaussian_noise_sigma: tuple[float, float],
    blur_radius: tuple[float, float],
    jpeg_quality: tuple[int, int],
    exposure_ev: tuple[float, float],
    seed: int | None,
) -> FramePostFx:
    """Sample independent post-fx parameters for one frame.

    Args:
        frame_index: Zero-based frame index combined with ``seed`` for RNG.
        gaussian_noise_sigma: Inclusive sigma range in 8-bit units.
        blur_radius: Inclusive PIL GaussianBlur radius range.
        jpeg_quality: Inclusive JPEG quality range for encode/decode round trip.
        exposure_ev: Inclusive exposure offset range in EV (``2**ev`` scale).
        seed: Optional seed; combined with ``frame_index`` for a local RNG.

    Returns:
        Sampled parameters for ``apply_postfx``.

    Raises:
        ValueError: If ``frame_index`` is negative.
    """
    if frame_index < 0:
        msg = f"frame_index must be >= 0, got {frame_index}"
        raise ValueError(msg)

    rng = Random(seed + frame_index) if seed is not None else Random()
    noise_lo, noise_hi = gaussian_noise_sigma
    blur_lo, blur_hi = blur_radius
    quality_lo, quality_hi = jpeg_quality
    ev_lo, ev_hi = exposure_ev

    return FramePostFx(
        gaussian_noise_sigma=rng.uniform(noise_lo, noise_hi),
        blur_radius=rng.uniform(blur_lo, blur_hi),
        jpeg_quality=rng.randint(quality_lo, quality_hi),
        exposure_ev=rng.uniform(ev_lo, ev_hi),
    )


def _apply_exposure(rgb: npt.NDArray[np.uint8], ev: float) -> npt.NDArray[np.uint8]:
    if ev == 0.0:
        return rgb
    scale = 2.0**ev
    adjusted = np.clip(rgb.astype(np.float64) * scale, 0.0, 255.0)
    return cast(npt.NDArray[np.uint8], np.rint(adjusted).astype(np.uint8, copy=False))


def _apply_gaussian_noise(
    rgb: npt.NDArray[np.uint8],
    sigma: float,
    rng: Random,
) -> npt.NDArray[np.uint8]:
    if sigma == 0.0:
        return rgb
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    noise = np_rng.normal(0.0, sigma, size=rgb.shape)
    adjusted = np.clip(rgb.astype(np.float64) + noise, 0.0, 255.0)
    return cast(npt.NDArray[np.uint8], np.rint(adjusted).astype(np.uint8, copy=False))


def _apply_blur(rgb: npt.NDArray[np.uint8], radius: float) -> npt.NDArray[np.uint8]:
    if radius <= 0.0:
        return rgb
    image = Image.fromarray(rgb, mode="RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.uint8)


def _apply_jpeg_roundtrip(rgb: npt.NDArray[np.uint8], quality: int) -> npt.NDArray[np.uint8]:
    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def apply_postfx(
    rgb: npt.NDArray[np.uint8],
    params: FramePostFx,
    *,
    rng: Random | None = None,
) -> npt.NDArray[np.uint8]:
    """Apply geometry-preserving sensor effects to a composited RGB frame.

    Args:
        rgb: Input image with shape ``(height, width, 3)`` and dtype ``uint8``.
        params: Sampled effect strengths.
        rng: Optional RNG for noise; defaults to an unseeded generator.

    Returns:
        Processed RGB array with the same shape and dtype as the input.

    Raises:
        ValueError: If ``rgb`` does not have shape ``(h, w, 3)``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = f"rgb must have shape (h, w, 3), got {rgb.shape}"
        raise ValueError(msg)
    if rgb.dtype != np.uint8:
        msg = f"rgb must have dtype uint8, got {rgb.dtype}"
        raise ValueError(msg)

    noise_rng = rng if rng is not None else Random()
    result = _apply_exposure(rgb, params.exposure_ev)
    result = _apply_gaussian_noise(result, params.gaussian_noise_sigma, noise_rng)
    result = _apply_blur(result, params.blur_radius)
    return _apply_jpeg_roundtrip(result, params.jpeg_quality)
