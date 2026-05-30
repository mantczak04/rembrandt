"""Canonical object orientation and centering for preview and render.

Rembrandt's camera sampler treats **+Z as up** (elevation is measured from the XY
plane). Imported meshes must therefore have their visual up aligned to +Z and sit
with their axis-aligned bounding-box center at the origin — the same frame
``Scene.center_target`` produces after import.

Wavefront ``.obj`` files may be authored Y-up or Z-up. Blender's ``obj_import``
operator remaps axes via ``up_axis`` / ``forward_axis``; the declarations below
keep each supported source orientation's pure rotation matrix next to its Blender
import axes so preview and render cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

SourceUpAxis = Literal["Y", "Z"]
BlenderImportAxis = Literal["X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z"]


@dataclass(frozen=True)
class SourceOrientation:
    """Shared pure-Python and Blender import orientation declaration."""

    rotation: npt.NDArray[np.float64]
    forward_axis: BlenderImportAxis
    up_axis: SourceUpAxis


SOURCE_ORIENTATIONS: dict[SourceUpAxis, SourceOrientation] = {
    # Y-up OBJ -> Rembrandt Z-up. Equivalent to Blender's default OBJ import remap.
    "Y": SourceOrientation(
        rotation=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    ),
    # Z-up OBJ -> Rembrandt Z-up. Verified against Blender obj_import with an
    # asymmetric fixture; forward_axis="Y" leaves coordinates in the same frame.
    "Z": SourceOrientation(
        rotation=np.identity(3, dtype=np.float64),
        forward_axis="Y",
        up_axis="Z",
    ),
}

# Default values for bpy.ops.wm.obj_import (Blender 4.x). Z-up is the default
# source convention; pass ``up_axis="Y"`` for legacy Y-up assets.
OBJ_IMPORT_FORWARD_AXIS = SOURCE_ORIENTATIONS["Z"].forward_axis
OBJ_IMPORT_UP_AXIS = SOURCE_ORIENTATIONS["Z"].up_axis


def obj_import_axes(up_axis: SourceUpAxis = "Z") -> tuple[BlenderImportAxis, SourceUpAxis]:
    """Return Blender ``obj_import`` axes for a declared source up-axis.

    Args:
        up_axis: Native up-axis of the source OBJ.

    Returns:
        ``(forward_axis, up_axis)`` values for ``bpy.ops.wm.obj_import``.
    """
    orientation = SOURCE_ORIENTATIONS[up_axis]
    return orientation.forward_axis, orientation.up_axis


def orient_and_center(
    vertices: npt.ArrayLike,
    *,
    up_axis: SourceUpAxis = "Z",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate source OBJ vertices to Z-up and center on the axis-aligned bbox.

    Args:
        vertices: Array of shape ``(n, 3)`` with raw OBJ vertex positions.
        up_axis: Native up-axis of the source OBJ.

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

    oriented = coords @ SOURCE_ORIENTATIONS[up_axis].rotation.T
    bbox_min = oriented.min(axis=0)
    bbox_max = oriented.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    centered = oriented - center
    bbox = np.stack(
        (centered.min(axis=0), centered.max(axis=0)),
    )
    return centered, bbox
