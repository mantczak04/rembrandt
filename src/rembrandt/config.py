"""Pydantic schema and YAML I/O for the shared render config contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from rembrandt.camera_poses import SamplingStrategy, validate_camera_pose_inputs
from rembrandt.convention import SourceUpAxis
from rembrandt.light_poses import LightType, validate_light_rig_inputs

RenderEngine = Literal["EEVEE", "CYCLES"]


def _default_light_types() -> list[LightType]:
    return ["POINT", "SUN", "AREA"]


class ObjectConfig(BaseModel):
    """Input object for rendering."""

    path: str
    up_axis: SourceUpAxis = "Z"
    class_name: str = "object"
    class_id: int = Field(default=0, ge=0)


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
    """Dataset output layout."""

    dir: str = "output"
    train_val_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    split_seed: int | None = None


class LightRandomizationConfig(BaseModel):
    """Per-frame randomized light rigs. Off (``static``) by default.

    In ``random`` mode the static ``lights:`` list is ignored (not merged).
    ``seed`` is independent of ``camera.seed`` and ``background.seed``.
    ``energy_scale_range`` multiplies per-type defaults in
    ``light_poses.DEFAULT_LIGHT_ENERGY`` (POINT/AREA in Watts, SUN unitless).
    """

    mode: Literal["static", "random"] = "static"
    count_range: tuple[int, int] = (1, 3)
    light_types: list[LightType] = Field(default_factory=_default_light_types)
    azimuth_range: tuple[float, float] = (0.0, 360.0)
    elevation_range: tuple[float, float] = (10.0, 80.0)
    distance_range: tuple[float, float] = (4.0, 8.0)
    energy_scale_range: tuple[float, float] = (0.5, 2.0)
    color_jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    area_size_range: tuple[float, float] = (1.0, 3.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        validate_light_rig_inputs(
            count_range=self.count_range,
            light_types=self.light_types,
            azimuth_range=self.azimuth_range,
            elevation_range=self.elevation_range,
            distance_range=self.distance_range,
            energy_scale_range=self.energy_scale_range,
            color_jitter=self.color_jitter,
            area_size_range=self.area_size_range,
        )
        return self


class BackgroundConfig(BaseModel):
    """Randomized background compositing (post-render). Off by default.

    ``image_dir`` may be absolute, relative to the config file, or relative to
    the working directory (resolved at render time). ``seed`` is independent of
    ``camera.seed``; ``None`` means non-reproducible background choice.
    """

    mode: Literal["none", "image"] = "none"
    image_dir: str | None = None
    seed: int | None = None
    color: tuple[float, float, float] = (0.05, 0.05, 0.05)

    @model_validator(mode="after")
    def _check_image_dir(self) -> Self:
        if self.mode == "image" and not self.image_dir:
            raise ValueError("background.image_dir is required when mode is 'image'")
        return self


class LabelsConfig(BaseModel):
    """YOLO label generation from rendered alpha masks."""

    enabled: bool = True
    min_visible_pixels: int = Field(default=25, ge=0)


class PostFxConfig(BaseModel):
    """Sensor-domain post-processing applied after background compositing.

    Effects simulate camera/sensor artifacts that YOLO train-time augmentation
    does not model. All are geometry-preserving; labels are derived from the
    render alpha mask before post-fx runs. ``seed`` is independent of other
    config seeds.
    """

    mode: Literal["off", "random"] = "off"
    gaussian_noise_sigma: tuple[float, float] = (0.0, 8.0)
    blur_radius: tuple[float, float] = (0.0, 1.2)
    jpeg_quality: tuple[int, int] = (55, 95)
    exposure_ev: tuple[float, float] = (-0.7, 0.7)
    seed: int | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        noise_lo, noise_hi = self.gaussian_noise_sigma
        if noise_lo < 0 or noise_hi < 0 or noise_lo > noise_hi:
            raise ValueError("postfx.gaussian_noise_sigma must be non-negative with lo <= hi")

        blur_lo, blur_hi = self.blur_radius
        if blur_lo < 0 or blur_hi < 0 or blur_lo > blur_hi:
            raise ValueError("postfx.blur_radius must be non-negative with lo <= hi")

        quality_lo, quality_hi = self.jpeg_quality
        if not (1 <= quality_lo <= 100 and 1 <= quality_hi <= 100):
            raise ValueError("postfx.jpeg_quality values must be in [1, 100]")
        if quality_lo > quality_hi:
            raise ValueError("postfx.jpeg_quality lower bound must not exceed upper bound")

        ev_lo, ev_hi = self.exposure_ev
        if ev_lo > ev_hi:
            raise ValueError("postfx.exposure_ev lower bound must not exceed upper bound")

        return self


class FramingConfig(BaseModel):
    """Per-frame framing diversity (scale and in-image translation).

    When active, ``camera.distance_range`` is a lower bound on camera distance;
    the sampled ``fill_range`` determines how large the object appears via
    ``fit_margin = 1 / fill``. Center jitter offsets the look-at point in the
    camera image plane at render time (not shown in the SPA preview).
    """

    center_jitter: float = Field(default=0.35, ge=0.0)
    fill_range: tuple[float, float] = (0.15, 0.75)
    seed: int | None = None

    @model_validator(mode="after")
    def _check_fill_range(self) -> Self:
        lo, hi = self.fill_range
        if lo <= 0 or hi <= 0:
            raise ValueError("framing.fill_range values must be positive")
        if lo > hi:
            raise ValueError("framing.fill_range lower bound must not exceed upper bound")
        return self


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
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    light_randomization: LightRandomizationConfig = Field(default_factory=LightRandomizationConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    framing: FramingConfig = Field(default_factory=FramingConfig)
    postfx: PostFxConfig = Field(default_factory=PostFxConfig)


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
