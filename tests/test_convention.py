"""Tests for canonical object orientation and centering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rembrandt.convention import (
    OBJ_IMPORT_FORWARD_AXIS,
    OBJ_IMPORT_UP_AXIS,
    orient_and_center,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHESS_BOARD_OBJ = PROJECT_ROOT / "test-obj" / "12951_Stone_Chess_Board_v1_L3.obj"


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
    assert OBJ_IMPORT_FORWARD_AXIS == "NEGATIVE_Z"
    assert OBJ_IMPORT_UP_AXIS == "Y"


def test_orient_and_center_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        orient_and_center(np.empty((0, 3)))


def test_orient_and_center_raises_on_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        orient_and_center(np.zeros((4, 2)))


def test_orient_and_center_maps_y_up_to_z_up() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
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
    bpy = pytest.importorskip("bpy")
    from mathutils import Vector

    from rembrandt.scene import Scene

    raw_vertices = _parse_obj_vertices(CHESS_BOARD_OBJ)

    scene = Scene()
    scene.load_object(CHESS_BOARD_OBJ)
    scene.center_target()
    assert scene.target is not None
    bpy_vertices = np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )

    pure_vertices, _bbox = orient_and_center(raw_vertices)
    _assert_vertex_sets_allclose(pure_vertices, bpy_vertices)

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.obj_import(
        filepath=str(CHESS_BOARD_OBJ),
        forward_axis=OBJ_IMPORT_FORWARD_AXIS,
        up_axis=OBJ_IMPORT_UP_AXIS,
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
