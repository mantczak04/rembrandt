"""Shared helpers for preview/render orientation regression tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from bpy.types import Object, Scene


def split_vertices_by_world_z(
    vertices: npt.NDArray[np.float64],
    *,
    high_quantile: float = 0.9,
    low_quantile: float = 0.1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return vertex subsets above/below the given Z quantiles."""
    z_values = vertices[:, 2]
    high_mask = z_values >= np.quantile(z_values, high_quantile)
    low_mask = z_values <= np.quantile(z_values, low_quantile)
    return vertices[high_mask], vertices[low_mask]


def camera_world_up_dot_positive_z(camera: Object) -> float:
    """Dot product of the camera's world-space up vector with +Z."""
    import bpy  # noqa: F401
    from mathutils import Vector

    world_up = camera.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    return float(world_up.dot(Vector((0.0, 0.0, 1.0))))


def mean_camera_view_y(
    scene: Scene,
    camera: Object,
    vertices: npt.NDArray[np.float64],
) -> float:
    """Mean normalized camera-view Y for world-space vertices (0=bottom, 1=top)."""
    import bpy  # noqa: F401
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    if vertices.size == 0:
        msg = "vertices must not be empty"
        raise ValueError(msg)

    y_values = [
        world_to_camera_view(scene, camera, Vector(vertex.tolist())).y for vertex in vertices
    ]
    return float(np.mean(y_values))


def assert_world_z_upright_in_camera_view(
    scene: Scene,
    camera: Object,
    vertices: npt.NDArray[np.float64],
    *,
    high_quantile: float = 0.9,
    low_quantile: float = 0.1,
) -> None:
    """Fail when +Z-extreme mesh points project below -Z-extreme points in the frame."""
    high_vertices, low_vertices = split_vertices_by_world_z(
        vertices,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
    )
    high_y = mean_camera_view_y(scene, camera, high_vertices)
    low_y = mean_camera_view_y(scene, camera, low_vertices)
    if high_y <= low_y:
        msg = (
            "world +Z should appear higher in the rendered frame than world -Z "
            f"(high_z mean view y={high_y:.4f}, low_z mean view y={low_y:.4f})"
        )
        raise AssertionError(msg)
