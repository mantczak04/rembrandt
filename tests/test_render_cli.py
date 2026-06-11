"""Tests for the config-driven render CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rembrandt.backgrounds import choose_background
from rembrandt.config import RembrandtConfig, dump_config, load_config
from rembrandt.errors import BackgroundDirectoryNotFoundError
from rembrandt.render import (
    render,
    render_from_config,
    resolve_background_dir,
    resolve_object_path,
    resolve_output_dir,
)
from tests.test_paths import PROJECT_ROOT, sample_object_path, sample_object_up_axis


def _mock_render_scene() -> MagicMock:
    scene = MagicMock()
    scene.target_radius_about.return_value = 1.0
    return scene


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


def test_resolve_output_dir_absolute(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("output:\n  dir: /tmp/out\n", encoding="utf-8")
    assert resolve_output_dir(config_path, "/tmp/out") == Path("/tmp/out").resolve()


def test_resolve_output_dir_relative_to_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("output:\n  dir: frames\n", encoding="utf-8")
    assert resolve_output_dir(config_path, "frames") == (tmp_path / "frames").resolve()


def test_resolve_output_dir_prefers_existing_directory_at_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "cfg.yaml"
    config_path.write_text("output:\n  dir: shared\n", encoding="utf-8")

    cwd_dir = tmp_path / "shared"
    cwd_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir(config_path, "shared") == cwd_dir.resolve()


def test_render_from_config_wires_scene(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
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

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=128),
        Path(path),
    )[1]

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="test-run",
    )

    assert output_dir == tmp_path / "frames" / "test-run"
    scene.load_object.assert_called_once_with(
        sample_object_path().resolve(),
        up_axis=sample_object_up_axis(),
    )
    scene.center_target.assert_called_once()
    scene.clear_lights.assert_not_called()
    assert scene.add_light.call_count == 1
    scene.add_camera.assert_called_once_with(focal_length=35.0)
    assert scene.move_camera.call_count == 3
    assert scene.render.call_count == 3
    scene.render.assert_any_call(
        output_dir / "frame_0000.png",
        resolution=(128, 128),
        engine="EEVEE",
        samples=4,
        transparent_film=True,
    )

    assert (output_dir / "frame_0000.txt").is_file()

    run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["resolved_object_path"] == str(sample_object_path().resolve())
    assert len(run_metadata["frames"]) == 3
    assert run_metadata["frames"][0]["camera_pose"]["location"]
    assert "light_rig" not in run_metadata["frames"][0]
    assert "framing" in run_metadata["frames"][0]
    first_move = scene.move_camera.call_args_list[0]
    assert "fit_margin" in first_move.kwargs


def test_render_from_config_framing_determinism(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 3, "seed": 1},
        framing={"center_jitter": 0.35, "fill_range": (0.2, 0.6), "seed": 99},
        render={"resolution": (64, 64), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    def collect_move_calls() -> list[dict[str, object]]:
        scene = _mock_render_scene()
        scene.render.side_effect = lambda path, **kwargs: (
            _write_rgba_frame(Path(path), size=64),
            Path(path),
        )[1]
        render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
            stamp="framing-det",
        )
        return [call.kwargs for call in scene.move_camera.call_args_list]

    first = collect_move_calls()
    second = collect_move_calls()
    assert first == second
    assert any(call["fit_margin"] != 1.2 for call in first)


def test_render_from_config_random_lights_call_order(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 3, "seed": 1},
        lights=[
            {
                "light_type": "SUN",
                "location": (99.0, 99.0, 99.0),
                "look_at": (0.0, 0.0, 0.0),
                "energy": 999.0,
            }
        ],
        light_randomization={
            "mode": "random",
            "count_range": (2, 2),
            "light_types": ["POINT"],
            "seed": 7,
        },
        render={"resolution": (64, 64), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=64),
        Path(path),
    )[1]

    render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="random-lights",
    )

    assert scene.clear_lights.call_count == 3
    assert scene.add_light.call_count == 6
    for light_call in scene.add_light.call_args_list:
        assert light_call.kwargs.get("energy") != 999.0

    relevant = {"clear_lights", "add_light", "move_camera", "render"}
    sequence = [name for name, *_ in scene.mock_calls if name in relevant]
    assert sequence == [
        "clear_lights",
        "add_light",
        "add_light",
        "move_camera",
        "render",
        "clear_lights",
        "add_light",
        "add_light",
        "move_camera",
        "render",
        "clear_lights",
        "add_light",
        "add_light",
        "move_camera",
        "render",
    ]


def test_render_from_config_random_lights_determinism(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 2, "seed": 0},
        light_randomization={"mode": "random", "count_range": (1, 1), "seed": 42},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    def collect_add_light_calls() -> list[dict[str, object]]:
        scene = _mock_render_scene()
        scene.render.side_effect = lambda path, **kwargs: (
            _write_rgba_frame(Path(path), size=64),
            Path(path),
        )[1]
        render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
            stamp="det",
        )
        return [call.kwargs for call in scene.add_light.call_args_list]

    first = collect_add_light_calls()
    second = collect_add_light_calls()
    assert first == second

    cfg.light_randomization = cfg.light_randomization.model_copy(update={"seed": 43})
    dump_config(cfg, config_path)
    different = collect_add_light_calls()
    assert different != first


def test_resolve_background_dir_relative_to_config_directory(tmp_path: Path) -> None:
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("background:\n  image_dir: backgrounds\n", encoding="utf-8")

    assert resolve_background_dir(config_path, "backgrounds") == bg_dir.resolve()


def _write_rgba_frame(path: Path, *, size: int = 32) -> None:
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    frame.paste((255, 0, 0, 255), (8, 8, 24, 24))
    frame.save(path)


def test_render_from_config_with_backgrounds(tmp_path: Path) -> None:
    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    colors = ((0, 255, 0), (0, 0, 255), (255, 255, 0))
    for index, color in enumerate(colors):
        Image.new("RGB", (64, 64), color).save(bg_dir / f"bg_{index}.png")

    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 2, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
        background={"mode": "image", "image_dir": str(bg_dir), "seed": 7},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()

    def render_side_effect(path: Path, **kwargs: object) -> Path:
        assert kwargs.get("transparent_film") is True
        _write_rgba_frame(Path(path))
        return Path(path)

    scene.render.side_effect = render_side_effect

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="bg-run",
    )

    for frame_path in sorted(output_dir.glob("frame_*.png")):
        with Image.open(frame_path) as frame:
            assert frame.mode == "RGB"
            assert frame.getpixel((0, 0)) != (0, 0, 0, 0)


def test_render_from_config_background_determinism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(bg_dir / "red.png")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(bg_dir / "green.png")

    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 3, "seed": 0},
        output={"dir": str(tmp_path / "frames")},
        background={"mode": "image", "image_dir": str(bg_dir), "seed": 42},
    )
    dump_config(cfg, config_path)

    chosen: list[Path] = []
    original_choose = choose_background

    def capture_choose(*args: object, **kwargs: object) -> Path:
        path = original_choose(*args, **kwargs)
        chosen.append(path)
        return path

    monkeypatch.setattr("rembrandt.render.choose_background", capture_choose)

    scene = _mock_render_scene()

    def render_side_effect(path: Path, **kwargs: object) -> Path:
        _write_rgba_frame(Path(path))
        return Path(path)

    scene.render.side_effect = render_side_effect

    render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="det",
    )
    first_run = list(chosen)
    chosen.clear()

    render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="det2",
    )

    assert chosen == first_run


def test_render_from_config_fail_fast_missing_background_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 1, "seed": 0},
        output={"dir": str(tmp_path / "frames")},
        background={"mode": "image", "image_dir": str(tmp_path / "missing")},
    )
    dump_config(cfg, config_path)

    scene = MagicMock()
    with pytest.raises(BackgroundDirectoryNotFoundError):
        render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
        )
    scene.render.assert_not_called()


def test_render_from_config_fail_fast_empty_background_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 1, "seed": 0},
        output={"dir": str(tmp_path / "frames")},
        background={"mode": "image", "image_dir": str(empty)},
    )
    dump_config(cfg, config_path)

    scene = MagicMock()
    with pytest.raises(ValueError, match="no background images found"):
        render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
        )
    scene.render.assert_not_called()


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
        frames_only: bool = False,
    ) -> Path:
        captured["cfg"] = loaded
        captured["config_path"] = config_path
        captured["frames_only"] = frames_only
        return tmp_path / "out" / "stamp"

    def fake_write_yolo_dataset(*args: object, **kwargs: object) -> Path:
        return tmp_path / "out" / "stamp" / "dataset" / "data.yaml"

    monkeypatch.setattr("rembrandt.render.render_from_config", fake_render_from_config)
    monkeypatch.setattr("rembrandt.render.write_yolo_dataset", fake_write_yolo_dataset)
    run_dir, data_yaml = render(config_path)

    assert run_dir == tmp_path / "out" / "stamp"
    assert data_yaml == tmp_path / "out" / "stamp" / "dataset" / "data.yaml"
    assert captured["config_path"] == config_path
    assert captured["cfg"].camera.n == 1
    assert captured["frames_only"] is False


@pytest.mark.bpy
def test_render_smoke_writes_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        object={"path": object_path, "up_axis": sample_object_up_axis()},
        camera={"n": 2, "seed": 0},
        render={"resolution": (64, 64), "samples": 1},
        output={"dir": str(output_root)},
    )
    dump_config(cfg, config_path)

    run_dir, data_yaml = render(config_path)

    assert run_dir.is_dir()
    assert data_yaml is not None
    assert data_yaml.is_file()
    train_images = list((run_dir / "dataset" / "images" / "train").glob("*.png"))
    val_images = list((run_dir / "dataset" / "images" / "val").glob("*.png"))
    assert len(train_images) + len(val_images) == 2
    assert (run_dir / "dataset" / "labels" / "train").is_dir()
    assert (run_dir / "dataset" / "train_yolo.py").is_file()
    assert (run_dir / "dataset" / "README.md").is_file()


def test_render_from_config_postfx_records_sampled_values(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 2, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
        postfx={"mode": "random", "seed": 5},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=32),
        Path(path),
    )[1]

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="postfx-run",
    )

    run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert "postfx" in run_metadata["frames"][0]
    postfx = run_metadata["frames"][0]["postfx"]
    assert set(postfx) == {
        "gaussian_noise_sigma",
        "blur_radius",
        "jpeg_quality",
        "exposure_ev",
    }


def test_render_from_config_postfx_off_is_no_op(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 1, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
        postfx={"mode": "off"},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=32),
        Path(path),
    )[1]

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="postfx-off",
    )

    run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert "postfx" not in run_metadata["frames"][0]


def test_render_from_config_postfx_determinism(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 2, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
        postfx={"mode": "random", "seed": 99},
    )
    dump_config(cfg, config_path)

    def collect_postfx() -> list[dict[str, object]]:
        scene = _mock_render_scene()
        scene.render.side_effect = lambda path, **kwargs: (
            _write_rgba_frame(Path(path), size=32),
            Path(path),
        )[1]
        output_dir = render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
            stamp="postfx-det",
        )
        run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        return [frame["postfx"] for frame in run_metadata["frames"]]

    assert collect_postfx() == collect_postfx()


def test_render_from_config_frames_only_skips_labels(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 2, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=32),
        Path(path),
    )[1]

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="frames-only",
        frames_only=True,
    )

    assert list(output_dir.glob("frame_*.txt")) == []
    scene.render.assert_any_call(
        output_dir / "frame_0000.png",
        resolution=(32, 32),
        engine="EEVEE",
        samples=1,
        transparent_film=False,
    )


def test_render_from_config_labels_disabled_uses_opaque_film(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 1, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
        labels={"enabled": False},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: Path(path)

    output_dir = render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        stamp="no-labels",
    )

    assert list(output_dir.glob("frame_*.txt")) == []
    scene.render.assert_called_once_with(
        output_dir / "frame_0000.png",
        resolution=(32, 32),
        engine="EEVEE",
        samples=1,
        transparent_film=False,
    )


@pytest.mark.bpy
def test_render_transparent_film_writes_rgba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    monkeypatch.chdir(PROJECT_ROOT)
    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    scene = Scene()
    scene.load_object(obj_path, up_axis=sample_object_up_axis())
    scene.center_target()
    scene.add_camera()
    scene.move_camera(location=(4.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.0))

    frame_path = tmp_path / "transparent.png"
    scene.render(frame_path, resolution=(64, 64), samples=1, transparent_film=True)

    with Image.open(frame_path) as image:
        assert image.mode == "RGBA"
        pixels = image.load()
        assert pixels[0, 0][3] == 0
        center_alpha = pixels[32, 32][3]
        assert center_alpha > 0


@pytest.mark.bpy
def test_render_default_writes_rgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bpy")
    from rembrandt.scene import Scene

    monkeypatch.chdir(PROJECT_ROOT)
    obj_path = sample_object_path()
    if not obj_path.is_file():
        pytest.skip(f"sample object not found: {obj_path}")

    scene = Scene()
    scene.load_object(obj_path, up_axis=sample_object_up_axis())
    scene.center_target()
    scene.add_camera()
    scene.move_camera(location=(4.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.0))

    frame_path = tmp_path / "opaque.png"
    scene.render(frame_path, resolution=(64, 64), samples=1, transparent_film=False)

    with Image.open(frame_path) as image:
        assert image.mode == "RGB"


def test_render_module_only_imports_bpy_through_scene() -> None:
    import rembrandt.render as render_module

    source = Path(render_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source
