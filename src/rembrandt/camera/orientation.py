"""Camera and light orientation helpers (Blender -Z forward convention)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathutils import Vector


def _vector(coords: tuple[float, float, float]) -> Vector:
    import bpy  # noqa: F401
    from mathutils import Vector

    return Vector(coords)


def require_nonzero_direction(
    direction: tuple[float, float, float],
    *,
    error_message: str | None = None,
) -> tuple[float, float, float]:
    """Return direction if non-zero, else raise ValueError."""
    vec = _vector(direction)
    if vec.length == 0:
        if error_message is not None:
            raise ValueError(error_message)
        raise ValueError("direction cannot be zero length.")
    return direction


def rotation_euler_from_forward(
    forward: tuple[float, float, float],
    *,
    up_axis: str = "Y",
) -> tuple[float, float, float]:
    """Euler rotation aligning local -Z with forward (Blender camera convention)."""
    direction = require_nonzero_direction(forward)
    euler = _vector(direction).to_track_quat("-Z", up_axis).to_euler()
    return (euler.x, euler.y, euler.z)
