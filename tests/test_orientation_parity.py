"""End-to-end preview vs render-scene orientation parity (bpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.preview.mesh import load_preview_mesh
from rembrandt.scene import Scene
from tests.test_convention import _assert_vertex_sets_allclose
from tests.test_paths import sample_object_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"


@pytest.mark.bpy
def test_preview_mesh_matches_scene_geometry() -> None:
    """Preview API output and bpy scene vertices share the same oriented frame."""
    pytest.importorskip("bpy")

    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    preview = load_preview_mesh(obj_path)
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(obj_path)
    scene.center_target()
    assert scene.target is not None
    scene_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)


@pytest.mark.bpy
def test_preview_mesh_matches_scene_geometry_on_y_up_fixture() -> None:
    """Regression guard using a committed asymmetric Y-up fixture."""
    pytest.importorskip("bpy")

    preview = load_preview_mesh(FIXTURE_OBJ)
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(FIXTURE_OBJ)
    scene.center_target()
    assert scene.target is not None
    scene_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)
