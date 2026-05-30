"""Tests for the shared YAML render config schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rembrandt.config import (
    CameraConfig,
    LightConfig,
    OutputConfig,
    RembrandtConfig,
    RenderConfig,
    dump_config,
    load_config,
)
from tests.test_paths import SAMPLE_OBJECT_PATH


def _minimal_config(*, n: int = 10, seed: int | None = 42) -> RembrandtConfig:
    return RembrandtConfig(
        object={"path": SAMPLE_OBJECT_PATH},
        camera={"n": n, "seed": seed},
    )


def test_load_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(missing)


def test_load_config_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_config(empty)


def test_config_round_trip_equality(tmp_path: Path) -> None:
    cfg = _minimal_config()
    out = tmp_path / "roundtrip.yaml"

    dump_config(cfg, out)
    loaded = load_config(out)

    assert loaded == cfg


def test_config_round_trip_full_fields(tmp_path: Path) -> None:
    cfg = RembrandtConfig(
        object={"path": SAMPLE_OBJECT_PATH},
        camera={
            "n": 25,
            "azimuth_range": (20.0, 120.0),
            "elevation_range": (-15.0, 25.0),
            "distance_range": (4.0, 6.0),
            "strategy": "fibonacci",
            "seed": 7,
            "look_at": (1.0, 2.0, 3.0),
        },
        lights=[
            LightConfig(light_type="AREA", location=(0.0, 0.0, 4.0), size=2.0),
        ],
        render=RenderConfig(
            focal_length=35.0,
            resolution=(512, 384),
            engine="CYCLES",
            samples=64,
        ),
        output=OutputConfig(dir="datasets/run1", train_val_split=0.75),
    )
    out = tmp_path / "full.yaml"

    dump_config(cfg, out)
    loaded = load_config(out)

    assert loaded == cfg


def test_config_defaults_applied() -> None:
    cfg = _minimal_config()

    assert cfg.object.up_axis == "Z"
    assert cfg.camera.azimuth_range == (0.0, 360.0)
    assert cfg.camera.elevation_range == (-10.0, 30.0)
    assert cfg.camera.distance_range == (3.0, 5.0)
    assert cfg.camera.strategy == "random"
    assert cfg.camera.look_at == (0.0, 0.0, 0.0)
    assert len(cfg.lights) == 2
    assert cfg.lights[0].light_type == "SUN"
    assert cfg.lights[1].light_type == "POINT"
    assert cfg.render.focal_length == 50.0
    assert cfg.render.resolution == (640, 640)
    assert cfg.render.engine == "EEVEE"
    assert cfg.render.samples == 32
    assert cfg.output.dir == "output"
    assert cfg.output.train_val_split == 0.8


@pytest.mark.parametrize(
    ("camera_kwargs", "message"),
    [
        ({"n": 0}, "n"),
        ({"n": 1, "azimuth_range": (350.0, 10.0)}, "azimuth_range"),
        ({"n": 1, "elevation_range": (-91.0, 10.0)}, "elevation_range"),
        ({"n": 1, "elevation_range": (10.0, 91.0)}, "elevation_range"),
        ({"n": 1, "elevation_range": (10.0, -10.0)}, "elevation_range"),
        ({"n": 1, "distance_range": (0.0, 1.0)}, "distance_range"),
        ({"n": 1, "distance_range": (1.0, 0.0)}, "distance_range"),
        ({"n": 1, "strategy": "grid"}, "strategy"),
    ],
)
def test_camera_config_rejects_invalid_ranges(
    camera_kwargs: dict[str, object],
    message: str,
) -> None:
    base: dict[str, object] = {"n": 10}
    base.update(camera_kwargs)
    with pytest.raises(ValidationError, match=message):
        CameraConfig.model_validate(base)


def test_output_train_val_split_out_of_range() -> None:
    with pytest.raises(ValidationError, match="train_val_split"):
        OutputConfig(train_val_split=1.0)
    with pytest.raises(ValidationError, match="train_val_split"):
        OutputConfig(train_val_split=0.0)


def test_load_config_from_yaml_list_ranges(tmp_path: Path) -> None:
    """YAML lists should coerce to tuples like the sampler expects."""
    yaml_text = f"""
object:
  path: {SAMPLE_OBJECT_PATH}
camera:
  n: 5
  azimuth_range: [10, 90]
  elevation_range: [-5, 15]
  distance_range: [2.5, 4.5]
"""
    path = tmp_path / "lists.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(path)

    assert cfg.camera.azimuth_range == (10.0, 90.0)
    assert cfg.camera.elevation_range == (-5.0, 15.0)
    assert cfg.camera.distance_range == (2.5, 4.5)


def test_load_config_accepts_object_up_axis(tmp_path: Path) -> None:
    yaml_text = f"""
object:
  path: {SAMPLE_OBJECT_PATH}
  up_axis: Z
camera:
  n: 5
"""
    path = tmp_path / "z_up.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(path)

    assert cfg.object.up_axis == "Z"


def test_object_config_rejects_invalid_up_axis() -> None:
    with pytest.raises(ValidationError, match="up_axis"):
        RembrandtConfig.model_validate(
            {
                "object": {"path": SAMPLE_OBJECT_PATH, "up_axis": "X"},
                "camera": {"n": 1},
            }
        )
