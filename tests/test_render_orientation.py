"""Rendered-frame orientation checks (bpy): geometry parity cannot cover camera roll."""

from __future__ import annotations

from pathlib import Path

import bpy  # noqa: F401
import numpy as np
import pytest

from rembrandt.camera_poses import sample_camera_poses
from rembrandt.convention import SourceUpAxis
from rembrandt.scene import Scene
from tests.orientation_checks import (
    assert_world_z_is_dominant_axis,
    assert_world_z_upright_in_camera_view,
)
from tests.test_paths import chess_board_object_path, sample_object_path, sample_object_up_axis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"
Z_UP_FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_z_up.obj"


def _scene_vertices(scene: Scene) -> np.ndarray:
    assert scene.target is not None
    return np.array(
        [[*(scene.target.matrix_world @ vertex.co)] for vertex in scene.target.data.vertices],
        dtype=np.float64,
    )


def _assert_upright_for_poses(
    obj_path: Path,
    *,
    up_axis: SourceUpAxis,
    n_poses: int,
    seed: int,
) -> None:
    poses = sample_camera_poses(
        n=n_poses,
        seed=seed,
        azimuth_range=(180.0, 360.0),
        elevation_range=(-90.0, 90.0),
        distance_range=(7.0, 13.0),
    )

    for pose in poses:
        scene = Scene()
        scene.load_object(obj_path, up_axis=up_axis)
        scene.center_target()
        vertices = _scene_vertices(scene)
        assert_world_z_is_dominant_axis(vertices)
        scene.add_camera(focal_length=50.0)
        scene.move_camera(
            location=pose.location,
            look_at=pose.look_at,
            fit_target=False,
        )
        assert scene.camera is not None
        assert_world_z_upright_in_camera_view(
            bpy.context.scene,
            scene.camera,
            vertices,
        )


@pytest.mark.bpy
@pytest.mark.parametrize(
    ("obj_path", "up_axis"),
    [
        (sample_object_path(), sample_object_up_axis()),
        (FIXTURE_OBJ, "Y"),
        (Z_UP_FIXTURE_OBJ, "Z"),
    ],
)
def test_rendered_view_keeps_world_z_upright(
    obj_path: Path,
    up_axis: SourceUpAxis,
) -> None:
    """High world-Z mesh points must project above low-Z points in the camera view."""
    pytest.importorskip("bpy")

    if not obj_path.is_file():
        pytest.skip(f"object not found: {obj_path}")

    _assert_upright_for_poses(obj_path, up_axis=up_axis, n_poses=6, seed=0)


@pytest.mark.bpy
def test_rendered_view_keeps_world_z_upright_on_chess_board_object() -> None:
    """End-to-end render orientation on the asset from the original bug report."""
    pytest.importorskip("bpy")

    obj_path = chess_board_object_path()
    if not obj_path.is_file():
        pytest.skip(
            f"reported drift asset not found at {obj_path}; "
            "copy 12951_Stone_Chess_Board_v1_L3.obj into test-obj/ to run this check",
        )

    _assert_upright_for_poses(obj_path, up_axis="Z", n_poses=8, seed=42)
