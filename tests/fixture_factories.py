"""Deterministic test fixture writers (bpy-free)."""

from __future__ import annotations

from pathlib import Path

# Unit-cube face quads as 0-based vertex indices (bottom, top, +Y sides).
_CUBE_QUADS = (
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (4, 0, 3, 7),
)


def write_two_offset_cubes_obj(path: Path) -> Path:
    """Write two unit cubes: one centered at origin, one offset +3 on X (Z-up).

    Emits separate ``o`` groups so Blender's importer creates two mesh objects.

    Args:
        path: Destination ``.obj`` file path.

    Returns:
        The written path.
    """
    lines: list[str] = []
    offsets = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    for cube_index, (ox, oy, oz) in enumerate(offsets):
        lines.append(f"o cube_{cube_index}")
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    lines.append(f"v {ox + sx} {oy + sy} {oz + sz}")
        base = cube_index * 8 + 1
        for a, b, c, d in _CUBE_QUADS:
            lines.append(f"f {base + a} {base + b} {base + c} {base + d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
