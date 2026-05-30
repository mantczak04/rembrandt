"""OBJ mesh parsing for the bpy-free SPA preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rembrandt.convention import SourceUpAxis, orient_and_center
from rembrandt.errors import ModelFileNotFoundError


@dataclass(frozen=True)
class PreviewMesh:
    """Oriented, centered mesh geometry for a Three.js ``BufferGeometry``.

    Args:
        positions: Flat ``[x0, y0, z0, x1, y1, z1, ...]`` vertex coordinates.
        indices: Flat triangle corner indices into ``positions // 3``.
        bbox: Axis-aligned bounds ``[[min_x, min_y, min_z], [max_x, max_y, max_z]]``
            in the centered frame.
    """

    positions: list[float]
    indices: list[int]
    bbox: list[list[float]]


def load_preview_mesh(path: str | Path, *, up_axis: SourceUpAxis = "Z") -> PreviewMesh:
    """Load an ``.obj`` file, orient it to the canonical frame, and return preview geometry.

    Args:
        path: Filesystem path to a Wavefront ``.obj`` file.
        up_axis: Native up-axis of the source OBJ.

    Returns:
        Serializable mesh data ready for the preview API / Three.js.

    Raises:
        ModelFileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no vertices.
    """
    obj_path = Path(path)
    if not obj_path.is_file():
        raise ModelFileNotFoundError(str(obj_path))

    vertices, _triangles = _parse_obj(obj_path)
    if not vertices:
        msg = f"OBJ file contains no vertices: {obj_path}"
        raise ValueError(msg)

    vertex_array = np.asarray(vertices, dtype=np.float64)
    centered, bbox = orient_and_center(vertex_array, up_axis=up_axis)
    return PreviewMesh(
        positions=centered.reshape(-1).tolist(),
        indices=[index for triangle in _triangles for index in triangle],
        bbox=bbox.tolist(),
    )


def _parse_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                indices = [_parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                triangles.extend(_triangulate(indices))

    return vertices, triangles


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw_index = int(token.split("/")[0])
    if raw_index < 0:
        return vertex_count + raw_index
    return raw_index - 1


def _triangulate(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []

    first = indices[0]
    return [(first, indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]
