"""Tests for Scene light management."""

from __future__ import annotations

from pathlib import Path

import pytest

from rembrandt.light_poses import DEFAULT_LIGHT_ENERGY


@pytest.mark.bpy
def test_clear_lights_removes_objects_and_data(tmp_path: Path) -> None:
    bpy = pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    scene = Scene()
    lights_before = len(bpy.data.lights)
    scene.add_light(light_type="POINT", location=(1.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.0))
    scene.add_light(light_type="SUN", location=(2.0, 1.0, 3.0), look_at=(0.0, 0.0, 0.0))
    scene.add_light(light_type="AREA", location=(0.0, 2.0, 4.0), look_at=(0.0, 0.0, 0.0))
    assert len(bpy.data.lights) == lights_before + 3

    scene.clear_lights()

    assert scene.lights == []
    assert not any(obj.type == "LIGHT" for obj in bpy.data.objects)
    assert len(bpy.data.lights) == lights_before


@pytest.mark.bpy
def test_clear_lights_twice_is_noop() -> None:
    pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    scene = Scene()
    scene.add_light()
    scene.clear_lights()
    scene.clear_lights()
    assert scene.lights == []


@pytest.mark.bpy
def test_add_light_after_clear_uses_default_energy() -> None:
    pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    scene = Scene()
    scene.add_light(light_type="POINT")
    scene.clear_lights()

    for light_type, expected in DEFAULT_LIGHT_ENERGY.items():
        obj = scene.add_light(
            light_type=light_type,
            location=(1.0, 2.0, 3.0),
            look_at=(0.0, 0.0, 0.0),
        )
        assert obj.data.energy == expected
        scene.clear_lights()


@pytest.mark.bpy
def test_clear_lights_does_not_affect_clear_behavior() -> None:
    bpy = pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    scene = Scene()
    scene.add_light()
    scene.add_camera()
    scene.clear()

    assert scene.lights == []
    assert scene.camera is None
    assert len(bpy.data.objects) == 0


@pytest.mark.bpy
def test_add_light_after_clear_renders(tmp_path: Path) -> None:
    bpy = pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    scene = Scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    scene.target = bpy.context.object
    scene.add_light()
    scene.clear_lights()
    scene.add_light(location=(3.0, 3.0, 5.0), look_at=(0.0, 0.0, 0.0))
    scene.add_camera(location=(4.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.0))

    frame_path = tmp_path / "after_clear.png"
    scene.render(frame_path, resolution=(64, 64), samples=1)

    assert frame_path.stat().st_size > 0
