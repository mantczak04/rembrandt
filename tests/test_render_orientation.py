"""Rendered-frame orientation checks (bpy): geometry parity cannot cover camera roll."""

from __future__ import annotations

from pathlib import Path

import bpy  # noqa: F401
import numpy as np
import pytest
from PIL import Image

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
    assert scene.targets
    return np.array(
        [
            [*(obj.matrix_world @ vertex.co)]
            for obj in scene.targets
            for vertex in obj.data.vertices
        ],
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


@pytest.mark.bpy
def test_jitter_translates_without_scaling(tmp_path: Path) -> None:
    """Look-at jitter shifts the object in-frame without changing apparent size."""
    pytest.importorskip("bpy")

    import bpy

    from rembrandt.annotations import bbox_from_mask, mask_from_alpha
    from rembrandt.framing import (
        fill_to_fit_margin,
        fitted_camera_distance,
        jitter_look_at,
        limiting_fov_for_focal_length,
    )

    resolution = 640
    fill = 0.4
    pose_location = (0.0, -6.0, 0.0)
    pose_look_at = (0.0, 0.0, 0.0)
    focal_length = 50.0
    fit_margin = fill_to_fit_margin(fill)

    def render_target(
        *,
        look_at: tuple[float, float, float],
        frame_path: Path,
    ) -> tuple[int, int, int, int]:
        scene = Scene()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
        scene.targets = [bpy.context.object]
        scene.center_target()
        render_settings = bpy.context.scene.render
        render_settings.resolution_x = resolution
        render_settings.resolution_y = resolution
        scene.add_camera(focal_length=focal_length)
        scene.move_camera(
            location=pose_location,
            look_at=look_at,
            fit_margin=fit_margin,
            fit_about=pose_look_at,
        )
        scene.render(
            frame_path,
            resolution=(resolution, resolution),
            samples=1,
            transparent_film=True,
        )
        with Image.open(frame_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        bbox = bbox_from_mask(mask_from_alpha(rgba))
        assert bbox is not None
        return bbox

    no_jitter_path = tmp_path / "no_jitter.png"
    jitter_path = tmp_path / "jitter.png"

    bbox_no_jitter = render_target(look_at=pose_look_at, frame_path=no_jitter_path)

    scene_for_radius = Scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
    scene_for_radius.targets = [bpy.context.object]
    scene_for_radius.center_target()
    target_radius = scene_for_radius.target_radius_about(pose_look_at)

    limiting_fov = limiting_fov_for_focal_length(focal_length, (resolution, resolution))
    fitted_dist = fitted_camera_distance(
        camera_location=pose_location,
        look_at=pose_look_at,
        target_radius=target_radius,
        focal_length=focal_length,
        resolution=(resolution, resolution),
        fit_margin=fit_margin,
    )
    jittered_look_at = jitter_look_at(
        look_at=pose_look_at,
        camera_location=pose_location,
        fitted_distance=fitted_dist,
        limiting_fov_rad=limiting_fov,
        center_jitter=0.5,
        jitter_uv=(0.9, 0.0),
    )
    bbox_jitter = render_target(look_at=jittered_look_at, frame_path=jitter_path)

    height_no_jitter = bbox_no_jitter[3] - bbox_no_jitter[1] + 1
    height_jitter = bbox_jitter[3] - bbox_jitter[1] + 1
    assert abs(height_no_jitter - height_jitter) <= 2

    center_x_no_jitter = (bbox_no_jitter[0] + bbox_no_jitter[2]) / 2
    center_x_jitter = (bbox_jitter[0] + bbox_jitter[2]) / 2
    assert abs(center_x_jitter - center_x_no_jitter) >= 0.2 * resolution
