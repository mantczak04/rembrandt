"""Tests for camera orientation helpers."""

from __future__ import annotations

import bpy  # noqa: F401
import pytest

from rembrandt.camera.orientation import (
    require_nonzero_direction,
    rotation_euler_from_forward,
)


def test_require_nonzero_direction_returns_input() -> None:
    direction = (1.0, 0.0, 0.0)
    assert require_nonzero_direction(direction) == direction


def test_require_nonzero_direction_raises() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        require_nonzero_direction((0.0, 0.0, 0.0))


def test_require_nonzero_direction_custom_message() -> None:
    with pytest.raises(ValueError, match="custom"):
        require_nonzero_direction((0.0, 0.0, 0.0), error_message="custom error")


def test_rotation_euler_from_forward_along_negative_z() -> None:
    # Camera at (0, 0, 5) looking at origin: forward = (0, 0, -1).
    euler = rotation_euler_from_forward((0.0, 0.0, -1.0))
    assert euler[0] == pytest.approx(0.0, abs=1e-6)
    assert euler[1] == pytest.approx(0.0, abs=1e-6)
    assert euler[2] == pytest.approx(0.0, abs=1e-6)


def test_rotation_euler_from_forward_along_positive_x() -> None:
    euler = rotation_euler_from_forward((1.0, 0.0, 0.0))
    assert euler != (0.0, 0.0, 0.0)


def test_rotation_euler_from_forward_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        rotation_euler_from_forward((0.0, 0.0, 0.0))


@pytest.mark.bpy
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Blender to_track_quat(-Z, Y) does not lock world +Z as camera up the way "
        "the Three.js preview does (camera.up = (0, 0, 1)); see Branch B in "
        ".ai/specs/ui-vs-backend-camera-angle-investigation-30-05.md"
    ),
)
def test_camera_world_up_aligns_with_positive_z_when_above_horizon() -> None:
    """Match Three.js preview: cameras above the look-at point should use world +Z as up."""
    pytest.importorskip("bpy")

    from rembrandt.camera_poses import sample_camera_poses
    from rembrandt.scene import Scene
    from tests.orientation_checks import camera_world_up_dot_positive_z
    from tests.test_paths import sample_object_path

    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    poses = sample_camera_poses(
        n=12,
        seed=42,
        azimuth_range=(180.0, 360.0),
        elevation_range=(-90.0, 90.0),
        distance_range=(7.0, 13.0),
    )

    checked = 0
    for pose in poses:
        if pose.location[2] <= pose.look_at[2]:
            continue
        checked += 1
        scene = Scene()
        scene.load_object(obj_path)
        scene.center_target()
        scene.add_camera(focal_length=50.0)
        scene.move_camera(
            location=pose.location,
            look_at=pose.look_at,
            fit_target=False,
        )
        assert scene.camera is not None
        dot_positive_z = camera_world_up_dot_positive_z(scene.camera)
        assert dot_positive_z > 0.85, (
            f"camera above horizon should use world +Z as up for pose {pose.location}, "
            f"got dot={dot_positive_z:.4f}"
        )

    assert checked >= 4, "expected several above-horizon poses in the sample band"
