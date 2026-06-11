"""Tests for Scene camera fitting integration."""

from __future__ import annotations

import bpy

from rembrandt.scene import Scene


def test_scene_refits_camera_after_render_resolution_changes() -> None:
    scene = Scene()
    render = bpy.context.scene.render
    render.resolution_x = 640
    render.resolution_y = 640

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    scene.targets = [bpy.context.object]

    camera = scene.add_camera(
        location=(0.0, 0.0, 2.0),
        look_at=(0.0, 0.0, 0.0),
        focal_length=50.0,
        fit_target=True,
        fit_margin=1.0,
    )
    square_distance = camera.location.length

    render.resolution_x = 480
    render.resolution_y = 1920
    scene._refit_camera_for_current_render_settings()

    assert camera.location.length > square_distance
