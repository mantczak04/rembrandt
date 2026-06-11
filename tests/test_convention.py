"""Tests for canonical object orientation and centering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.convention import (
    OBJ_IMPORT_FORWARD_AXIS,
    OBJ_IMPORT_UP_AXIS,
    SourceUpAxis,
    obj_import_axes,
    orient_and_center,
)
from tests.test_paths import PROJECT_ROOT, chess_board_object_path, sample_object_path

Y_UP_FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"
Z_UP_FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_z_up.obj"


def _parse_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(vertices, dtype=np.float64)


def _assert_vertex_sets_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-5,
) -> None:
    sorted_actual = _sort_vertices(actual)
    sorted_expected = _sort_vertices(expected)
    np.testing.assert_allclose(sorted_actual, sorted_expected, atol=atol)


def _sort_vertices(vertices: np.ndarray) -> np.ndarray:
    return vertices[np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))]


def test_obj_import_axis_constants() -> None:
    assert OBJ_IMPORT_FORWARD_AXIS == "Y"
    assert OBJ_IMPORT_UP_AXIS == "Z"
    assert obj_import_axes() == ("Y", "Z")
    assert obj_import_axes("Y") == ("NEGATIVE_Z", "Y")
    assert obj_import_axes("Z") == ("Y", "Z")


def test_orient_and_center_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        orient_and_center(np.empty((0, 3)))


def test_orient_and_center_raises_on_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        orient_and_center(np.zeros((4, 2)))


def test_orient_and_center_maps_y_up_to_z_up() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    centered, bbox = orient_and_center(vertices, up_axis="Y")
    assert bbox[1, 2] - bbox[0, 2] == pytest.approx(2.0)
    assert bbox[0, 1] == pytest.approx(bbox[1, 1])
    np.testing.assert_allclose(centered[:, 2], [-1.0, 1.0], atol=1e-12)


def test_orient_and_center_preserves_z_up() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    centered, bbox = orient_and_center(vertices)
    assert bbox[1, 2] - bbox[0, 2] == pytest.approx(2.0)
    assert bbox[0, 1] == pytest.approx(bbox[1, 1])
    np.testing.assert_allclose(centered[:, 2], [-1.0, 1.0], atol=1e-12)


def test_orient_and_center_places_bbox_at_origin() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float64,
    )
    centered, bbox = orient_and_center(vertices)
    np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(bbox[0], centered.min(axis=0))
    np.testing.assert_allclose(bbox[1], centered.max(axis=0))
    np.testing.assert_allclose((bbox[0] + bbox[1]) / 2.0, 0.0, atol=1e-12)


@pytest.mark.bpy
def test_orient_and_center_matches_bpy_import() -> None:
    _assert_orient_and_center_matches_bpy_import(sample_object_path(), up_axis="Y")


@pytest.mark.bpy
def test_orient_and_center_matches_bpy_import_on_y_up_fixture() -> None:
    _assert_orient_and_center_matches_bpy_import(Y_UP_FIXTURE_OBJ, up_axis="Y")


@pytest.mark.bpy
def test_orient_and_center_matches_bpy_import_on_z_up_fixture() -> None:
    _assert_orient_and_center_matches_bpy_import(Z_UP_FIXTURE_OBJ, up_axis="Z")


@pytest.mark.bpy
def test_orient_and_center_matches_bpy_import_on_chess_board_object() -> None:
    """Parity on the asset named in the original orientation bug report."""
    obj_path = chess_board_object_path()
    if not obj_path.is_file():
        pytest.skip(
            f"reported drift asset not found at {obj_path}; "
            "copy 12951_Stone_Chess_Board_v1_L3.obj into test-obj/ to run this check",
        )

    _assert_orient_and_center_matches_bpy_import(obj_path, up_axis="Z")


@pytest.mark.bpy
def test_chess_board_object_z_up_axis_lands_on_world_z() -> None:
    """The reported pawn is Z-up native and should stand on world Z."""
    pytest.importorskip("bpy")

    obj_path = chess_board_object_path()
    if not obj_path.is_file():
        pytest.skip(
            f"reported drift asset not found at {obj_path}; "
            "copy 12951_Stone_Chess_Board_v1_L3.obj into test-obj/ to run this check",
        )

    raw_vertices = _parse_obj_vertices(obj_path)
    centered, _bbox = orient_and_center(raw_vertices, up_axis="Z")
    extents = centered.max(axis=0) - centered.min(axis=0)
    assert np.argmax(extents) == 2

    scene = _load_scene_object(obj_path, up_axis="Z")
    assert scene.targets
    scene_vertices = np.array(
        [
            [*(obj.matrix_world @ vertex.co)]
            for obj in scene.targets
            for vertex in obj.data.vertices
        ],
        dtype=np.float64,
    )
    scene_extents = scene_vertices.max(axis=0) - scene_vertices.min(axis=0)
    assert np.argmax(scene_extents) == 2


def _load_scene_object(obj_path: Path, *, up_axis: SourceUpAxis):
    from rembrandt.scene import Scene

    scene = Scene()
    scene.load_object(obj_path, up_axis=up_axis)
    scene.center_target()
    return scene


def _assert_orient_and_center_matches_bpy_import(
    obj_path: Path,
    *,
    up_axis: SourceUpAxis,
) -> None:
    bpy = pytest.importorskip("bpy")
    from mathutils import Vector

    raw_vertices = _parse_obj_vertices(obj_path)
    pure_vertices, _bbox = orient_and_center(raw_vertices, up_axis=up_axis)

    scene = _load_scene_object(obj_path, up_axis=up_axis)
    assert scene.targets
    scene_vertices = np.array(
        [
            [*(obj.matrix_world @ vertex.co)]
            for obj in scene.targets
            for vertex in obj.data.vertices
        ],
        dtype=np.float64,
    )
    _assert_vertex_sets_allclose(pure_vertices, scene_vertices)

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    forward_axis, import_up_axis = obj_import_axes(up_axis)
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis=forward_axis,
        up_axis=import_up_axis,
    )
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"][0]
    corners = [imported.matrix_world @ Vector(corner) for corner in imported.bound_box]
    imported.location -= sum(corners, Vector()) / 8
    bpy.context.view_layer.update()
    direct_vertices = np.array(
        [[*(imported.matrix_world @ vertex.co)] for vertex in imported.data.vertices],
        dtype=np.float64,
    )
    _assert_vertex_sets_allclose(pure_vertices, direct_vertices)
