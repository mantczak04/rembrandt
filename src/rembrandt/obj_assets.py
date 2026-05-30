"""Wavefront OBJ/MTL asset helpers (bpy-free)."""

from __future__ import annotations

import re
from pathlib import Path

_MTLIB_RE = re.compile(r"^mtllib\s+(\S+)", re.MULTILINE)
_TEXTURE_MAP_RE = re.compile(r"^\s*map_[A-Za-z]+\s+(\S+)", re.MULTILINE)


def mtllib_names(obj_path: Path) -> list[str]:
    """Return ``mtllib`` filenames declared in an OBJ file."""
    text = obj_path.read_text(encoding="utf-8", errors="replace")
    return _MTLIB_RE.findall(text)


def texture_map_names(mtl_path: Path) -> list[str]:
    """Return texture filenames referenced by ``map_*`` lines in an MTL file."""
    text = mtl_path.read_text(encoding="utf-8", errors="replace")
    return _TEXTURE_MAP_RE.findall(text)


def normalize_mtl_line_endings(mtl_path: Path) -> bool:
    """Normalize an MTL file for Blender's OBJ importer.

    Rewrites CRLF/CR line endings to LF and drops ``map_Ka`` lines, which Blender
    does not support (it warns and ignores them, while ``map_Kd`` carries the
    diffuse texture we need).

    Args:
        mtl_path: Path to a Wavefront MTL file.

    Returns:
        True if the file was rewritten.
    """
    raw = mtl_path.read_bytes()
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    filtered = [line for line in lines if not line.lstrip().startswith("map_Ka ")]
    normalized = "\n".join(filtered)
    if lines and not normalized.endswith("\n"):
        normalized += "\n"
    if filtered == lines and raw.decode("utf-8", errors="replace") == normalized:
        return False
    mtl_path.write_text(normalized, encoding="utf-8")
    return True


def normalize_obj_mtllibs(obj_path: Path) -> None:
    """Normalize line endings for MTL files referenced by an OBJ."""
    asset_dir = obj_path.parent
    for name in mtllib_names(obj_path):
        mtl_path = asset_dir / name
        if mtl_path.is_file():
            normalize_mtl_line_endings(mtl_path)


def resolve_texture_file(asset_dir: Path, filename: str) -> Path | None:
    """Locate a texture file next to an OBJ/MTL asset.

    Tries the path as given, its basename, and a spaces-for-underscores variant
    (some exporters rewrite paths inconsistently).
    """
    candidates = [
        asset_dir / filename,
        asset_dir / Path(filename).name,
        asset_dir / Path(filename).name.replace("_", " "),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None
