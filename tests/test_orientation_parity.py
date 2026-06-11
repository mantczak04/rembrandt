"""End-to-end preview vs render-scene orientation parity (bpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.preview.mesh import load_preview_mesh
from rembrandt.scene import Scene
from tests.fixture_factories import write_two_offset_cubes_obj
from tests.test_convention import _assert_vertex_sets_allclose
from tests.test_paths import chess_board_object_path, sample_object_path, sample_object_up_axis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"
Z_UP_FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_z_up.obj"


def _scene_vertices(scene: Scene) -> np.ndarray:
    return np.array(
        [
            [*(obj.matrix_world @ vertex.co)]
            for obj in scene.targets
            for vertex in obj.data.vertices
        ],
        dtype=np.float64,
    )


def _scene_union_bbox(scene: Scene) -> np.ndarray:
    corners = scene._target_world_corners()
    coords = np.array([[corner.x, corner.y, corner.z] for corner in corners], dtype=np.float64)
    return np.stack((coords.min(axis=0), coords.max(axis=0)))


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
    assert scene.targets
    scene_vertices = _scene_vertices(scene)

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
    assert scene.targets
    scene_vertices = _scene_vertices(scene)

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
    assert scene.targets
    scene_vertices = _scene_vertices(scene)

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
    assert scene.targets
    scene_vertices = _scene_vertices(scene)

    _assert_vertex_sets_allclose(preview_vertices, scene_vertices)


@pytest.mark.bpy
def test_multi_mesh_union_bbox_centered_at_origin_and_matches_preview(tmp_path: Path) -> None:
    """OBJ geometry with spatially separated parts matches preview union bounds."""
    pytest.importorskip("bpy")

    obj_path = write_two_offset_cubes_obj(tmp_path / "two_offset_cubes.obj")
    preview = load_preview_mesh(obj_path, up_axis="Z")
    preview_bbox = np.asarray(preview.bbox, dtype=np.float64)

    scene = Scene()
    scene.load_object(obj_path, up_axis="Z")
    assert len(scene.targets) == 2
    scene.center_target()

    scene_bbox = _scene_union_bbox(scene)
    np.testing.assert_allclose(scene_bbox, preview_bbox, atol=1e-5)
    np.testing.assert_allclose(scene_bbox.mean(axis=0), 0.0, atol=1e-5)


@pytest.mark.bpy
def test_multiple_target_objects_center_on_union_bbox() -> None:
    """Separate mesh objects are translated together to the union bbox center."""
    bpy = pytest.importorskip("bpy")

    scene = Scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube_a = bpy.context.object
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0))
    cube_b = bpy.context.object
    scene.targets = [cube_a, cube_b]

    scene.center_target()

    scene_bbox = _scene_union_bbox(scene)
    np.testing.assert_allclose(scene_bbox.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(scene_bbox[0], [-2.0, -0.5, -0.5], atol=1e-5)
    np.testing.assert_allclose(scene_bbox[1], [2.0, 0.5, 0.5], atol=1e-5)
