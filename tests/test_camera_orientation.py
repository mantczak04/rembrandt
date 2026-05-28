"""Tests for camera orientation helpers."""

from __future__ import annotations

import bpy  # noqa: F401
import pytest

from rembrandt.camera.orientation import (
    require_nonzero_direction,
    rotation_euler_from_forward,
)


def test_require_nonzero_direction_returns_input() -> None:
    direction = (1.0, 0.0, 0.0)
    assert require_nonzero_direction(direction) == direction


def test_require_nonzero_direction_raises() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        require_nonzero_direction((0.0, 0.0, 0.0))


def test_require_nonzero_direction_custom_message() -> None:
    with pytest.raises(ValueError, match="custom"):
        require_nonzero_direction((0.0, 0.0, 0.0), error_message="custom error")


def test_rotation_euler_from_forward_along_negative_z() -> None:
    # Camera at (0, 0, 5) looking at origin: forward = (0, 0, -1).
    euler = rotation_euler_from_forward((0.0, 0.0, -1.0))
    assert euler[0] == pytest.approx(0.0, abs=1e-6)
    assert euler[1] == pytest.approx(0.0, abs=1e-6)
    assert euler[2] == pytest.approx(0.0, abs=1e-6)


def test_rotation_euler_from_forward_along_positive_x() -> None:
    euler = rotation_euler_from_forward((1.0, 0.0, 0.0))
    assert euler != (0.0, 0.0, 0.0)


def test_rotation_euler_from_forward_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        rotation_euler_from_forward((0.0, 0.0, 0.0))
