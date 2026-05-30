"""Tests for the config-driven render CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rembrandt.config import RembrandtConfig, dump_config, load_config
from rembrandt.render import render, render_from_config, resolve_object_path
from tests.test_paths import PROJECT_ROOT, sample_object_path


def test_resolve_object_path_relative_to_project_root() -> None:
    obj_path = sample_object_path()
    relative = obj_path.relative_to(PROJECT_ROOT)
    config_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    resolved = resolve_object_path(config_path, str(relative))

    assert resolved == obj_path.resolve()


def test_resolve_object_path_relative_to_config_directory(tmp_path: Path) -> None:
    obj = tmp_path / "model.obj"
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("object:\n  path: model.obj\n", encoding="utf-8")

    assert resolve_object_path(config_path, "model.obj") == obj.resolve()


def test_render_from_config_wires_scene(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 3, "seed": 1},
        lights=[
            {
                "light_type": "SUN",
                "location": (1.0, 2.0, 3.0),
                "look_at": (0.0, 0.0, 0.0),
                "energy": 2.0,
            }
        ],
        render={"focal_length": 35.0, "resolution": (128, 128), "samples": 4},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    scene = MagicMock()
    scene.render.side_effect = lambda path, **kwargs: Path(path)

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="test-run",
    )

    assert output_dir == tmp_path / "frames" / "test-run"
    scene.load_object.assert_called_once_with(sample_object_path().resolve())
    scene.center_target.assert_called_once()
    assert scene.add_light.call_count == 1
    scene.add_camera.assert_called_once_with(focal_length=35.0)
    assert scene.move_camera.call_count == 3
    assert scene.render.call_count == 3
    scene.render.assert_any_call(
        output_dir / "frame_0000.png",
        resolution=(128, 128),
        engine="EEVEE",
        samples=4,
    )


def test_render_loads_config_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 1, "seed": 0},
        output={"dir": str(tmp_path / "out")},
    )
    dump_config(cfg, config_path)

    captured: dict[str, Any] = {}

    def fake_render_from_config(
        loaded: RembrandtConfig,
        *,
        config_path: Path,
        scene_factory: object = None,
        stamp: str | None = None,
    ) -> Path:
        captured["cfg"] = loaded
        captured["config_path"] = config_path
        return tmp_path / "out" / "stamp"

    monkeypatch.setattr("rembrandt.render.render_from_config", fake_render_from_config)
    result = render(config_path)

    assert result == tmp_path / "out" / "stamp"
    assert captured["config_path"] == config_path
    assert captured["cfg"].camera.n == 1


@pytest.mark.bpy
def test_render_smoke_writes_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bpy")
    monkeypatch.chdir(PROJECT_ROOT)

    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    config_path = tmp_path / "smoke.yaml"
    output_root = tmp_path / "rendered"
    object_path = (
        str(obj_path.relative_to(PROJECT_ROOT))
        if obj_path.is_relative_to(PROJECT_ROOT)
        else str(obj_path)
    )
    cfg = RembrandtConfig(
        object={"path": object_path},
        camera={"n": 2, "seed": 0},
        render={"resolution": (64, 64), "samples": 1},
        output={"dir": str(output_root)},
    )
    dump_config(cfg, config_path)

    output_dir = render(config_path)
    frames = sorted(output_dir.glob("frame_*.png"))

    assert output_dir.is_dir()
    assert len(frames) == 2
    assert all(frame.stat().st_size > 0 for frame in frames)


def test_render_module_only_imports_bpy_through_scene() -> None:
    import rembrandt.render as render_module

    source = Path(render_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source
