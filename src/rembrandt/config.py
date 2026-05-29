"""Pydantic schema and YAML I/O for the shared render config contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from rembrandt.camera_poses import SamplingStrategy, validate_camera_pose_inputs

LightType = Literal["POINT", "SUN", "AREA"]
RenderEngine = Literal["EEVEE", "CYCLES"]


class ObjectConfig(BaseModel):
    """Input object for rendering."""

    path: str


class CameraConfig(BaseModel):
    """Camera pose sampling parameters (mirrors ``sample_camera_poses``)."""

    n: int
    azimuth_range: tuple[float, float] = (0.0, 360.0)
    elevation_range: tuple[float, float] = (-10.0, 30.0)
    distance_range: tuple[float, float] = (3.0, 5.0)
    strategy: SamplingStrategy = "random"
    seed: int | None = None
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_sampling_params(self) -> Self:
        validate_camera_pose_inputs(
            n=self.n,
            azimuth_range=self.azimuth_range,
            elevation_range=self.elevation_range,
            distance_range=self.distance_range,
            strategy=self.strategy,
        )
        return self


class LightConfig(BaseModel):
    """A single light matching ``Scene.add_light`` parameters."""

    light_type: LightType = "POINT"
    location: tuple[float, float, float] = (5.0, 5.0, 5.0)
    look_at: tuple[float, float, float] = (1.0, 1.0, 1.0)
    energy: float | None = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: float = 1.0


class RenderConfig(BaseModel):
    """Render engine settings."""

    focal_length: float = 50.0
    resolution: tuple[int, int] = (640, 640)
    engine: RenderEngine = "EEVEE"
    samples: int = 32


class OutputConfig(BaseModel):
    """Dataset output layout (``train_val_split`` reserved for a later phase)."""

    dir: str = "output"
    train_val_split: float = Field(default=0.8, gt=0.0, lt=1.0)


class RembrandtConfig(BaseModel):
    """Top-level YAML config shared by the SPA preview and ``rembrandt render``."""

    object: ObjectConfig
    camera: CameraConfig
    lights: list[LightConfig] = Field(
        default_factory=lambda: [
            LightConfig(
                light_type="SUN",
                location=(2.0, -3.0, 5.0),
                look_at=(0.0, 0.0, 0.0),
                energy=3.0,
            ),
            LightConfig(light_type="POINT", location=(-2.0, 2.0, 3.0)),
        ]
    )
    render: RenderConfig = Field(default_factory=RenderConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> RembrandtConfig:
    """Load and validate a render config from a YAML file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        The validated config model.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    try:
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in {config_path}: {exc}"
        raise ValueError(msg) from exc

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    return RembrandtConfig.model_validate(data)


def dump_config(cfg: RembrandtConfig, path: str | Path) -> None:
    """Write a render config to a YAML file.

    Args:
        cfg: Config to serialize.
        path: Destination file path (parent directories are created).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = cfg.model_dump(mode="json")
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
