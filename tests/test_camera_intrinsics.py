"""Tests for camera intrinsics and FOV math."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan

import numpy as np
import pytest

from rembrandt.camera.intrinsics import (
    fov_from_k_matrix,
    get_sensor_size,
    get_view_fac_in_px,
    intrinsics_as_k_matrix,
    limiting_fov_from_camera,
)


@dataclass
class _CameraData:
    lens: float
    shift_x: float
    shift_y: float
    sensor_fit: str
    sensor_width: float
    sensor_height: float


def test_get_sensor_size_horizontal() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "HORIZONTAL", 36.0, 24.0)
    assert get_sensor_size(cam) == 36.0


def test_get_sensor_size_vertical() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "VERTICAL", 36.0, 24.0)
    assert get_sensor_size(cam) == 24.0


def test_get_view_fac_in_px_horizontal() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "HORIZONTAL", 36.0, 24.0)
    assert get_view_fac_in_px(cam, 1.0, 1.0, 640, 480) == 640


def test_get_view_fac_in_px_auto_picks_horizontal() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "AUTO", 36.0, 24.0)
    assert get_view_fac_in_px(cam, 1.0, 1.0, 640, 480) == 640


def test_get_view_fac_in_px_keeps_fractional_vertical_extent() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "VERTICAL", 36.0, 24.0)
    assert get_view_fac_in_px(cam, 3.0, 4.0, 640, 481) == pytest.approx((4.0 / 3.0) * 481)


def test_fov_from_k_matrix_square_symmetric() -> None:
    fx = fy = 500.0
    size = 640
    k = np.array([[fx, 0.0, 319.5], [0.0, fy, 319.5], [0.0, 0.0, 1.0]])
    fov_x, fov_y = fov_from_k_matrix(k, size, size)
    expected = 2 * atan(size / 2 / fx)
    assert fov_x == pytest.approx(expected)
    assert fov_y == pytest.approx(expected)


def test_intrinsics_as_k_matrix_no_shift() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "HORIZONTAL", 36.0, 24.0)
    k = intrinsics_as_k_matrix(
        cam=cam,
        resolution_x_in_px=640,
        resolution_y_in_px=640,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )
    view_fac = 640
    expected_fx = 50.0 / 36.0 * view_fac
    assert k[0, 0] == pytest.approx(expected_fx)
    assert k[1, 1] == pytest.approx(expected_fx)
    assert k[0, 2] == pytest.approx(319.5)
    assert k[1, 2] == pytest.approx(319.5)


def test_limiting_fov_from_camera_uses_smaller_axis() -> None:
    cam = _CameraData(50.0, 0.0, 0.0, "HORIZONTAL", 36.0, 24.0)
    k = intrinsics_as_k_matrix(
        cam=cam,
        resolution_x_in_px=1920,
        resolution_y_in_px=1080,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )
    fov_x, fov_y = fov_from_k_matrix(k, 1920, 1080)
    assert limiting_fov_from_camera(
        cam=cam,
        resolution_x_in_px=1920,
        resolution_y_in_px=1080,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    ) == pytest.approx(min(fov_x, fov_y))
