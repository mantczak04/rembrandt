"""Alpha-mask to YOLO bounding-box conversion. NO bpy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

BboxPx = tuple[int, int, int, int]


def mask_from_alpha(rgba: npt.NDArray[np.uint8], *, threshold: int = 8) -> npt.NDArray[np.bool_]:
    """Build a boolean object mask from an RGBA frame's alpha channel.

    Args:
        rgba: Array with shape ``(height, width, 4)``.
        threshold: Minimum alpha (0–255) treated as foreground. Values below
            this are ignored to skip faint anti-aliasing fringe pixels.

    Returns:
        Boolean mask with shape ``(height, width)``.

    Raises:
        ValueError: If ``rgba`` does not have shape ``(h, w, 4)``.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba must have shape (h, w, 4), got {rgba.shape}")
    return rgba[..., 3] >= threshold


def bbox_from_mask(mask: npt.NDArray[np.bool_]) -> BboxPx | None:
    """Compute a pixel-space inclusive bounding box from a boolean mask.

    Args:
        mask: Boolean array with shape ``(height, width)``.

    Returns:
        ``(x0, y0, x1, y1)`` in pixel coordinates, or ``None`` when the mask
        is empty.
    """
    if not mask.any():
        return None

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def visible_pixel_count(mask: npt.NDArray[np.bool_]) -> int:
    """Count foreground pixels in a boolean mask.

    Args:
        mask: Boolean array with shape ``(height, width)``.

    Returns:
        Number of ``True`` entries in ``mask``.
    """
    return int(mask.sum())


def yolo_line(
    class_id: int,
    bbox_px: BboxPx,
    *,
    width: int,
    height: int,
) -> str:
    """Format a YOLO detection label line from a pixel bounding box.

    Args:
        class_id: Integer class index for the object.
        bbox_px: Inclusive pixel bounds ``(x0, y0, x1, y1)``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A single YOLO label line: ``"<class_id> <cx> <cy> <w> <h>"`` with
        normalized center and size in ``[0, 1]``.

    Raises:
        ValueError: If ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    x0, y0, x1, y1 = bbox_px
    cx = (x0 + x1 + 1) / 2 / width
    cy = (y0 + y1 + 1) / 2 / height
    box_w = (x1 - x0 + 1) / width
    box_h = (y1 - y0 + 1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"
