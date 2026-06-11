"""Tests for parallel render coordination and frame-level determinism."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rembrandt.backgrounds import choose_background
from rembrandt.config import RembrandtConfig, dump_config, load_config
from rembrandt.framing import sample_frame_framing
from rembrandt.light_poses import sample_light_rig
from rembrandt.postfx import sample_frame_postfx
from rembrandt.render import (
    merge_run_metadata,
    parse_frame_range,
    render,
    render_from_config,
    worker_frame_indices,
)
from tests.test_paths import sample_object_path, sample_object_up_axis


def test_parse_frame_range() -> None:
    assert parse_frame_range("0:4") == (0, 4)
    assert parse_frame_range("2:10") == (2, 10)


@pytest.mark.parametrize(
    ("frame_range", "message"),
    [
        ("bad", "start:end"),
        ("1:1", "start must be < end"),
        ("-1:2", ">= 0"),
    ],
)
def test_parse_frame_range_rejects_invalid(frame_range: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_frame_range(frame_range)


def test_worker_frame_indices_round_robin() -> None:
    assert worker_frame_indices(n_frames=10, worker_index=0, num_workers=4) == [0, 4, 8]
    assert worker_frame_indices(n_frames=10, worker_index=1, num_workers=4) == [1, 5, 9]
    assert worker_frame_indices(n_frames=10, worker_index=2, num_workers=4) == [2, 6]
    assert worker_frame_indices(n_frames=10, worker_index=3, num_workers=4) == [3, 7]


def test_worker_frame_indices_respects_frame_range() -> None:
    indices = worker_frame_indices(
        n_frames=12,
        worker_index=0,
        num_workers=3,
        frame_range=(2, 8),
    )
    assert indices == [3, 6]


def test_merge_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.frames.worker_0001.json").write_text(
        json.dumps({"frames": [{"frame": 1, "camera_pose": {"location": [1, 0, 0]}}]}),
        encoding="utf-8",
    )
    (run_dir / "run.frames.worker_0000.json").write_text(
        json.dumps({"frames": [{"frame": 0, "camera_pose": {"location": [0, 0, 0]}}]}),
        encoding="utf-8",
    )

    cfg = RembrandtConfig(
        object={"path": "model.obj"},
        camera={"n": 2, "seed": 0},
    )
    merge_run_metadata(
        run_dir,
        cfg=cfg,
        resolved_object_path=tmp_path / "model.obj",
    )

    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert [frame["frame"] for frame in run_metadata["frames"]] == [0, 1]
    assert not list(run_dir.glob("run.frames.worker_*.json"))


def _mock_render_scene() -> MagicMock:
    scene = MagicMock()
    scene.target_radius_about.return_value = 1.0
    return scene


def _write_rgba_frame(path: Path, *, size: int = 32) -> None:
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    frame.paste((255, 0, 0, 255), (8, 8, 24, 24))
    frame.save(path)


def test_render_from_config_frame_indices_subset(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 5, "seed": 1},
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
        stamp="subset",
        frame_indices=[1, 3],
    )

    assert sorted(path.name for path in output_dir.glob("frame_*.png")) == [
        "frame_0001.png",
        "frame_0003.png",
    ]
    assert scene.render.call_count == 2


def test_render_worker_mode_writes_partial_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    run_dir = tmp_path / "frames" / "parallel-run"
    run_dir.mkdir(parents=True)
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 4, "seed": 1},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    scene = _mock_render_scene()
    scene.render.side_effect = lambda path, **kwargs: (
        _write_rgba_frame(Path(path), size=32),
        Path(path),
    )[1]

    render_from_config(
        load_config(config_path),
        config_path=config_path,
        scene_factory=lambda: scene,
        output_dir=run_dir,
        frame_indices=[0, 2],
        write_run_metadata=False,
        worker_partial_metadata_path=run_dir / "run.frames.worker_0000.json",
    )

    partial = json.loads((run_dir / "run.frames.worker_0000.json").read_text(encoding="utf-8"))
    assert [frame["frame"] for frame in partial["frames"]] == [0, 2]
    assert not (run_dir / "run.json").exists()


def test_render_coordinator_spawns_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path())},
        camera={"n": 3, "seed": 0},
        output={"dir": str(tmp_path / "out")},
        labels={"enabled": False},
    )
    dump_config(cfg, config_path)

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    render(config_path, workers=2, frames_only=True)

    assert len(commands) == 2
    for index, command in enumerate(commands):
        assert command[1:3] == ["-m", "rembrandt.render"]
        assert "--worker-index" in command
        assert command[command.index("--worker-index") + 1] == str(index)
        assert "--workers-total" in command
        assert command[command.index("--workers-total") + 1] == "2"
        assert "--frames-only" in command


def test_parallel_render_matches_sequential_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    cfg = RembrandtConfig(
        object={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
        camera={"n": 4, "seed": 1},
        light_randomization={"mode": "random", "count_range": (1, 1), "seed": 7},
        framing={"center_jitter": 0.2, "fill_range": (0.2, 0.5), "seed": 3},
        postfx={"mode": "random", "seed": 11},
        render={"resolution": (32, 32), "samples": 1},
        output={"dir": str(tmp_path / "frames")},
    )
    dump_config(cfg, config_path)

    def run_sequential() -> list[dict[str, Any]]:
        scene = _mock_render_scene()
        scene.render.side_effect = lambda path, **kwargs: (
            _write_rgba_frame(Path(path), size=32),
            Path(path),
        )[1]
        output_dir = render_from_config(
            load_config(config_path),
            config_path=config_path,
            scene_factory=lambda: scene,
            stamp="sequential",
        )
        run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        return run_metadata["frames"]

    def run_parallel_workers() -> list[dict[str, Any]]:
        run_dir = tmp_path / "frames" / "parallel"
        run_dir.mkdir(parents=True, exist_ok=True)
        loaded = load_config(config_path)
        for worker_index in range(2):
            scene = _mock_render_scene()
            scene.render.side_effect = lambda path, **kwargs: (
                _write_rgba_frame(Path(path), size=32),
                Path(path),
            )[1]
            indices = worker_frame_indices(
                n_frames=loaded.camera.n,
                worker_index=worker_index,
                num_workers=2,
            )
            render_from_config(
                loaded,
                config_path=config_path,
                scene_factory=lambda scene=scene: scene,
                output_dir=run_dir,
                frame_indices=indices,
                write_run_metadata=False,
                worker_partial_metadata_path=run_dir / f"run.frames.worker_{worker_index:04d}.json",
            )
        merge_run_metadata(
            run_dir,
            cfg=loaded,
            resolved_object_path=sample_object_path().resolve(),
        )
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        return run_metadata["frames"]

    sequential_frames = run_sequential()
    parallel_frames = run_parallel_workers()
    assert parallel_frames == sequential_frames


def test_frame_sampling_depends_only_on_seed_and_index(tmp_path: Path) -> None:
    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(bg_dir / "a.png")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(bg_dir / "b.png")
    backgrounds = sorted(bg_dir.glob("*.png"))

    def sample_for_index(frame_index: int) -> dict[str, object]:
        rig = sample_light_rig(frame_index=frame_index, seed=42, count_range=(2, 2))
        background = choose_background(backgrounds, frame_index=frame_index, seed=7)
        framing = sample_frame_framing(
            frame_index=frame_index,
            camera_location=(4.0, 0.0, 2.0),
            look_at=(0.0, 0.0, 0.0),
            target_radius=1.0,
            focal_length=50.0,
            resolution=(640, 480),
            center_jitter=0.35,
            fill_range=(0.2, 0.6),
            seed=99,
        )
        postfx = sample_frame_postfx(
            frame_index=frame_index,
            gaussian_noise_sigma=(0.0, 8.0),
            blur_radius=(0.0, 1.2),
            jpeg_quality=(55, 95),
            exposure_ev=(-0.7, 0.7),
            seed=5,
        )
        return {
            "rig": rig,
            "background": background.name,
            "framing": framing,
            "postfx": postfx,
        }

    direct = {index: sample_for_index(index) for index in range(6)}
    shuffled = list(range(6))
    shuffled.sort(reverse=True)
    for index in shuffled:
        assert sample_for_index(index) == direct[index]
