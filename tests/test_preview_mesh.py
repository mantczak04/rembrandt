"""Tests for bpy-free OBJ preview mesh loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.convention import bounding_radius_from_bbox
from rembrandt.errors import ModelFileNotFoundError
from rembrandt.preview.mesh import PreviewMesh, load_preview_mesh
from tests.test_paths import sample_object_path, sample_object_up_axis


def test_load_preview_mesh_parses_sample() -> None:
    mesh = load_preview_mesh(sample_object_path(), up_axis=sample_object_up_axis())

    assert isinstance(mesh, PreviewMesh)
    assert len(mesh.positions) > 0
    assert len(mesh.positions) % 3 == 0
    assert len(mesh.indices) > 0
    assert len(mesh.indices) % 3 == 0
    assert len(mesh.bbox) == 2
    assert len(mesh.bbox[0]) == 3


def test_load_preview_mesh_bbox_centered_at_origin() -> None:
    mesh = load_preview_mesh(sample_object_path(), up_axis=sample_object_up_axis())
    bbox_min = np.asarray(mesh.bbox[0], dtype=np.float64)
    bbox_max = np.asarray(mesh.bbox[1], dtype=np.float64)
    np.testing.assert_allclose((bbox_min + bbox_max) / 2.0, 0.0, atol=1e-5)


def test_load_preview_mesh_positions_match_orient_and_center() -> None:
    from rembrandt.convention import orient_and_center

    raw_vertices = _parse_obj_vertices(sample_object_path())
    expected, _bbox = orient_and_center(
        np.asarray(raw_vertices, dtype=np.float64),
        up_axis=sample_object_up_axis(),
    )

    mesh = load_preview_mesh(
        sample_object_path(),
        up_axis=sample_object_up_axis(),
        normalize=False,
    )
    positions = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(positions, expected, atol=1e-5)


def test_load_preview_mesh_normalize_true_yields_unit_radius(tmp_path: Path) -> None:
    obj_path = tmp_path / "scaled_box.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 4 4 4",
                "f 1 2 1",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_preview_mesh(obj_path, normalize=True)
    bbox = np.asarray(mesh.bbox, dtype=np.float64)
    assert bounding_radius_from_bbox(bbox) == pytest.approx(1.0, abs=1e-12)


def test_load_preview_mesh_normalize_false_preserves_extent(tmp_path: Path) -> None:
    obj_path = tmp_path / "scaled_box.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 4 4 4",
                "f 1 2 1",
            ]
        ),
        encoding="utf-8",
    )

    normalized = load_preview_mesh(obj_path, normalize=True)
    raw = load_preview_mesh(obj_path, normalize=False)
    normalized_radius = bounding_radius_from_bbox(np.asarray(normalized.bbox, dtype=np.float64))
    raw_radius = bounding_radius_from_bbox(np.asarray(raw.bbox, dtype=np.float64))
    assert normalized_radius == pytest.approx(1.0, abs=1e-12)
    assert raw_radius > normalized_radius


def test_load_preview_mesh_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.obj"
    with pytest.raises(ModelFileNotFoundError, match="missing.obj"):
        load_preview_mesh(missing)


def test_load_preview_mesh_raises_on_empty_vertices(tmp_path: Path) -> None:
    empty_obj = tmp_path / "empty.obj"
    empty_obj.write_text("# no vertices\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no vertices"):
        load_preview_mesh(empty_obj)


def test_preview_mesh_module_is_bpy_free() -> None:
    import rembrandt.preview.mesh as mesh_module

    source_path = Path(mesh_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def _parse_obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices
