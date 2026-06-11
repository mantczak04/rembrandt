"""Post-render background compositing for dataset frames.

This module is pure Python (no Blender). It indexes local background photos,
deterministically selects one per frame, resizes/crops the background to cover
the render resolution, and alpha-composites the rendered foreground over it.
"""

from __future__ import annotations

from pathlib import Path
from random import Random

import numpy as np
import numpy.typing as npt
from PIL import Image

from rembrandt.errors import BackgroundDirectoryNotFoundError

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def index_backgrounds(directory: str | Path) -> list[Path]:
    """Recursively list usable background images under a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted paths to background image files.

    Raises:
        BackgroundDirectoryNotFoundError: If the directory does not exist.
        ValueError: If no background images are found.
    """
    root = Path(directory)
    if not root.is_dir():
        raise BackgroundDirectoryNotFoundError(str(root))

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _EXTENSIONS
    )
    if not paths:
        msg = f"no background images found in {root}"
        raise ValueError(msg)
    return paths


def choose_background(
    backgrounds: list[Path],
    *,
    frame_index: int,
    seed: int | None,
) -> Path:
    """Pick a background image for a frame using a local RNG.

    Args:
        backgrounds: Non-empty pool of background image paths.
        frame_index: Zero-based frame index in the render run.
        seed: Base seed for reproducible choice; ``None`` uses an unseeded RNG.

    Returns:
        The chosen background path.

    Raises:
        ValueError: If ``backgrounds`` is empty or ``frame_index`` is negative.
    """
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    if not backgrounds:
        raise ValueError("backgrounds must not be empty")

    rng = Random(seed + frame_index) if seed is not None else Random()
    return rng.choice(backgrounds)


def load_cover_resized(
    path: str | Path,
    *,
    width: int,
    height: int,
) -> npt.NDArray[np.uint8]:
    """Load an image, scale to cover ``width`` x ``height``, and center-crop.

    Args:
        path: Filesystem path to the background image.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        RGB array with shape ``(height, width, 3)``.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        src_w, src_h = rgb.size
        scale = max(width / src_w, height / src_h)
        resized_w = max(1, round(src_w * scale))
        resized_h = max(1, round(src_h * scale))
        resized = rgb.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)
        left = (resized_w - width) // 2
        top = (resized_h - height) // 2
        cropped = resized.crop((left, top, left + width, top + height))
        return np.asarray(cropped, dtype=np.uint8)


def composite_over(
    foreground_rgba: npt.NDArray[np.uint8],
    background_rgb: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Alpha-composite an RGBA foreground over an RGB background.

    Args:
        foreground_rgba: Foreground with shape ``(h, w, 4)``.
        background_rgb: Background with shape ``(h, w, 3)``.

    Returns:
        Composited RGB array with shape ``(h, w, 3)``.

    Raises:
        ValueError: If shapes do not match or channel counts are wrong.
    """
    if foreground_rgba.ndim != 3 or foreground_rgba.shape[2] != 4:
        raise ValueError(f"foreground_rgba must have shape (h, w, 4), got {foreground_rgba.shape}")
    if background_rgb.ndim != 3 or background_rgb.shape[2] != 3:
        raise ValueError(f"background_rgb must have shape (h, w, 3), got {background_rgb.shape}")
    if foreground_rgba.shape[:2] != background_rgb.shape[:2]:
        raise ValueError(
            "foreground and background height/width must match: "
            f"{foreground_rgba.shape[:2]} vs {background_rgb.shape[:2]}"
        )

    alpha = foreground_rgba[..., 3:4].astype(np.float64) / 255.0
    fg_rgb = foreground_rgba[..., :3].astype(np.float64)
    bg_rgb = background_rgb.astype(np.float64)
    blended = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    return np.rint(blended).astype(np.uint8)


def composite_over_color(
    foreground_rgba: npt.NDArray[np.uint8],
    color: tuple[float, float, float],
) -> npt.NDArray[np.uint8]:
    """Alpha-composite an RGBA foreground over a flat RGB background color.

    Args:
        foreground_rgba: Foreground with shape ``(h, w, 4)``.
        color: Background RGB in ``[0, 1]`` (linear-ish display values).

    Returns:
        Composited RGB array with shape ``(h, w, 3)``.
    """
    rgb = tuple(int(round(channel * 255.0)) for channel in color)
    height, width = foreground_rgba.shape[:2]
    background = np.full((height, width, 3), rgb, dtype=np.uint8)
    return composite_over(foreground_rgba, background)


def apply_background_to_frame(
    frame_path: str | Path,
    background_path: str | Path,
    *,
    foreground_rgba: npt.NDArray[np.uint8] | None = None,
) -> Path:
    """Composite a background image over a rendered RGBA frame in place.

    Args:
        frame_path: Path to the rendered frame PNG (read and overwritten).
        background_path: Path to the background photo.
        foreground_rgba: Optional pre-loaded RGBA array; when omitted the
            frame is read from ``frame_path``.

    Returns:
        ``frame_path`` as a ``Path`` after writing the composited RGB PNG.
    """
    output = Path(frame_path)
    if foreground_rgba is None:
        with Image.open(output) as frame_image:
            foreground = np.asarray(frame_image.convert("RGBA"), dtype=np.uint8)
    else:
        foreground = foreground_rgba
    height, width = foreground.shape[:2]
    background = load_cover_resized(background_path, width=width, height=height)
    composited = composite_over(foreground, background)
    Image.fromarray(composited, mode="RGB").save(output, format="PNG")
    return output
