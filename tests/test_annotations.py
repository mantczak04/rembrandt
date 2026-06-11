"""Tests for alpha-mask to YOLO label conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.annotations import (
    bbox_from_mask,
    mask_from_alpha,
    visible_pixel_count,
    yolo_line,
)


def test_annotations_module_is_bpy_free() -> None:
    import rembrandt.annotations as annotations_module

    source = Path(annotations_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_mask_from_alpha_threshold_skips_fringe() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[1:3, 1:3, 3] = 255
    rgba[0, 0, 3] = 4

    mask = mask_from_alpha(rgba, threshold=8)

    assert mask[0, 0] == False  # noqa: E712
    assert mask[1, 1]
    assert mask.sum() == 4


def test_bbox_from_mask_single_blob() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:8] = True

    assert bbox_from_mask(mask) == (3, 2, 7, 4)


def test_bbox_from_mask_empty_returns_none() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    assert bbox_from_mask(mask) is None


def test_bbox_from_mask_one_pixel() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 2] = True
    assert bbox_from_mask(mask) == (2, 1, 2, 1)


def test_bbox_from_mask_touching_edges() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    mask[:, -1] = True
    assert bbox_from_mask(mask) == (0, 0, 3, 3)


def test_visible_pixel_count() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert visible_pixel_count(mask) == 2


def test_yolo_line_normalization_round_trip() -> None:
    width, height = 100, 80
    line = yolo_line(0, (10, 20, 49, 59), width=width, height=height)
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, box_w, box_h = (float(value) for value in parts[1:])
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 < box_w <= 1.0
    assert 0.0 < box_h <= 1.0
    assert cx == pytest.approx((10 + 49 + 1) / 2 / width)
    assert cy == pytest.approx((20 + 59 + 1) / 2 / height)
    assert box_w == pytest.approx((49 - 10 + 1) / width)
    assert box_h == pytest.approx((59 - 20 + 1) / height)


def test_yolo_line_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        yolo_line(0, (0, 0, 1, 1), width=0, height=10)
