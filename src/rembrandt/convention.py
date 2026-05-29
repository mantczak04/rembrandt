"""Canonical object orientation and centering for preview and render.

Rembrandt's camera sampler treats **+Z as up** (elevation is measured from the XY
plane). Imported meshes must therefore have their visual up aligned to +Z and sit
with their axis-aligned bounding-box center at the origin — the same frame
``Scene.center_target`` produces after import.

Wavefront ``.obj`` files are typically authored Y-up. Blender's ``obj_import``
operator remaps axes via ``up_axis`` / ``forward_axis``; the constants below match
the default Blender 4.x import that rotates Y-up assets into Blender's Z-up world
(a fixed +90° rotation about +X). The pure function ``orient_and_center`` applies
that same rotation and bbox centering without bpy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Values for bpy.ops.wm.obj_import (Blender 4.x). Keep in sync with orient_and_center.
OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_Z"
OBJ_IMPORT_UP_AXIS = "Y"

# Y-up (OBJ) -> Z-up (Rembrandt). Equivalent to Blender's default obj_import remap.
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def orient_and_center(
    vertices: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate Y-up OBJ vertices to Z-up and center on the axis-aligned bbox.

    Args:
        vertices: Array of shape ``(n, 3)`` with raw OBJ vertex positions.

    Returns:
        A tuple of ``(centered_vertices, bbox)`` where ``bbox`` is a ``(2, 3)``
        array ``[min, max]`` in the centered frame (bbox center at the origin).

    Raises:
        ValueError: If ``vertices`` is empty or not two-dimensional with 3 columns.
    """
    coords = np.asarray(vertices, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        msg = f"vertices must have shape (n, 3), got {coords.shape}"
        raise ValueError(msg)
    if coords.shape[0] == 0:
        raise ValueError("vertices must not be empty")

    oriented = coords @ _Y_UP_TO_Z_UP.T
    bbox_min = oriented.min(axis=0)
    bbox_max = oriented.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    centered = oriented - center
    bbox = np.stack(
        (centered.min(axis=0), centered.max(axis=0)),
    )
    return centered, bbox
