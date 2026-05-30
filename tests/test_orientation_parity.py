"""End-to-end preview vs render-scene orientation parity (bpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.preview.mesh import load_preview_mesh
from rembrandt.scene import Scene
from tests.test_convention import _assert_vertex_sets_allclose
from tests.test_paths import chess_board_object_path, sample_object_path, sample_object_up_axis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"
Z_UP_FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_z_up.obj"


@pytest.mark.bpy
def test_preview_mesh_matches_scene_geometry() -> None:
    """Preview API output and bpy scene vertices share the same oriented frame."""
    pytest.importorskip("bpy")

    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    preview = load_preview_mesh(obj_path, up_axis=sample_object_up_axis())
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(obj_path, up_axis=sample_object_up_axis())
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

    preview = load_preview_mesh(FIXTURE_OBJ, up_axis="Y")
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(FIXTURE_OBJ, up_axis="Y")
    scene.center_target()
    assert scene.target is not None
    scene_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)


@pytest.mark.bpy
def test_preview_mesh_matches_scene_geometry_on_z_up_fixture() -> None:
    """Regression guard using a committed asymmetric Z-up fixture."""
    pytest.importorskip("bpy")

    preview = load_preview_mesh(Z_UP_FIXTURE_OBJ, up_axis="Z")
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(Z_UP_FIXTURE_OBJ, up_axis="Z")
    scene.center_target()
    assert scene.target is not None
    scene_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)


@pytest.mark.bpy
def test_preview_mesh_matches_scene_geometry_on_chess_board_object_z_up() -> None:
    """The reported pawn uses the explicit Z-up path in both preview and render."""
    pytest.importorskip("bpy")

    obj_path = chess_board_object_path()
    if not obj_path.is_file():
        pytest.skip(
            f"reported drift asset not found at {obj_path}; "
            "copy 12951_Stone_Chess_Board_v1_L3.obj into test-obj/ to run this check",
        )

    preview = load_preview_mesh(obj_path, up_axis="Z")
    preview_vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)

    scene = Scene()
    scene.load_object(obj_path, up_axis="Z")
    scene.center_target()
    assert scene.target is not None
    scene_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)
