"""Per-frame light rig sampling for dataset rendering.

The samplers in this module are pure Python math helpers. They do not import
Blender, so they can be tested quickly without the bpy runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, radians, sin
from random import Random
from typing import Literal, TypeAlias

from rembrandt.camera_poses import Point3D, position_from_spherical, validate_spherical_ranges

LightType: TypeAlias = Literal["POINT", "SUN", "AREA"]

DEFAULT_LIGHT_ENERGY: dict[LightType, float] = {
    "POINT": 1000.0,
    "SUN": 5.0,
    "AREA": 100.0,
}

_VALID_LIGHT_TYPES = frozenset({"POINT", "SUN", "AREA"})


@dataclass(frozen=True)
class SampledLight:
    """One light in a per-frame rig, mirroring ``Scene.add_light`` params.

    Args:
        light_type: Blender light type.
        location: World-space position. For SUN lights only the direction from
            ``location`` toward ``look_at`` affects shading.
        look_at: World-space point the light aims at.
        energy: Light intensity in type-appropriate units.
        color: RGB in [0, 1].
        size: AREA light side length in meters (ignored for other types).
    """

    light_type: LightType
    location: Point3D
    look_at: Point3D
    energy: float
    color: Point3D
    size: float


def validate_light_rig_inputs(
    *,
    count_range: tuple[int, int],
    light_types: Sequence[LightType],
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
    energy_scale_range: tuple[float, float],
    color_jitter: float,
    area_size_range: tuple[float, float],
) -> None:
    """Validate light rig sampling inputs.

    Args:
        count_range: Inclusive integer range for lights per frame.
        light_types: Allowed light types to sample from.
        azimuth_range: Inclusive degree range around +Z.
        elevation_range: Inclusive degree range above the XY plane.
        distance_range: Inclusive world-unit range for distance sampling.
        energy_scale_range: Multiplier range applied to per-type default energy.
        color_jitter: Maximum per-channel deviation from white before normalization.
        area_size_range: Inclusive world-unit range for AREA light size.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if count_range[0] < 1 or count_range[0] > count_range[1]:
        raise ValueError(f"count_range min must be >= 1 and <= max, got {count_range}")
    if not light_types:
        raise ValueError("light_types must be non-empty")
    invalid = [t for t in light_types if t not in _VALID_LIGHT_TYPES]
    if invalid:
        raise ValueError(f"light_types contains invalid entries: {invalid}")
    validate_spherical_ranges(
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
    )
    if energy_scale_range[0] <= 0 or energy_scale_range[1] <= 0:
        raise ValueError(f"energy_scale_range values must be > 0, got {energy_scale_range}")
    if energy_scale_range[0] > energy_scale_range[1]:
        raise ValueError(f"energy_scale_range min must be <= max, got {energy_scale_range}")
    if area_size_range[0] <= 0 or area_size_range[1] <= 0:
        raise ValueError(f"area_size_range values must be > 0, got {area_size_range}")
    if area_size_range[0] > area_size_range[1]:
        raise ValueError(f"area_size_range min must be <= max, got {area_size_range}")
    if color_jitter < 0.0 or color_jitter > 1.0:
        raise ValueError(f"color_jitter must be within [0.0, 1.0], got {color_jitter}")


def sample_light_rig(
    *,
    frame_index: int,
    count_range: tuple[int, int] = (1, 3),
    light_types: Sequence[LightType] = ("POINT", "SUN", "AREA"),
    azimuth_range: tuple[float, float] = (0.0, 360.0),
    elevation_range: tuple[float, float] = (10.0, 80.0),
    distance_range: tuple[float, float] = (4.0, 8.0),
    energy_scale_range: tuple[float, float] = (0.5, 2.0),
    color_jitter: float = 0.0,
    area_size_range: tuple[float, float] = (1.0, 3.0),
    look_at: Point3D = (0.0, 0.0, 0.0),
    seed: int | None = None,
) -> list[SampledLight]:
    """Sample a randomized light rig for one frame.

    Lights are placed on a spherical band around ``look_at`` using the same
    +Z-up convention as camera pose sampling. For SUN lights, only the
    direction from ``location`` toward ``look_at`` affects shading; the band
    position still determines that direction.

    Args:
        frame_index: Zero-based frame index combined with ``seed`` for RNG.
        count_range: Inclusive integer range for the number of lights.
        light_types: Allowed light types to sample from.
        azimuth_range: Inclusive degree range around +Z.
        elevation_range: Inclusive degree range above the XY plane.
        distance_range: Inclusive world-unit range for distance sampling.
        energy_scale_range: Multiplier range applied to per-type default energy.
        color_jitter: Maximum per-channel deviation from white before normalization.
        area_size_range: Inclusive world-unit range for AREA light size (sampled
            for every light so the RNG stream does not depend on drawn types).
        look_at: World-space point all sampled lights aim at.
        seed: Optional seed; combined with ``frame_index`` for a local RNG.

    Returns:
        A list of sampled lights for this frame.

    Raises:
        ValueError: If ``frame_index`` or any range is invalid.
    """
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    validate_light_rig_inputs(
        count_range=count_range,
        light_types=light_types,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
        energy_scale_range=energy_scale_range,
        color_jitter=color_jitter,
        area_size_range=area_size_range,
    )

    rng = Random(seed + frame_index) if seed is not None else Random()
    sin_el_min = sin(radians(elevation_range[0]))
    sin_el_max = sin(radians(elevation_range[1]))
    az_min = radians(azimuth_range[0])
    az_max = radians(azimuth_range[1])

    count = rng.randint(count_range[0], count_range[1])
    lights: list[SampledLight] = []
    for _ in range(count):
        light_type = rng.choice(light_types)
        sin_el = rng.uniform(sin_el_min, sin_el_max)
        elevation = asin(sin_el)
        azimuth = rng.uniform(az_min, az_max)
        distance = rng.uniform(distance_range[0], distance_range[1])
        location = position_from_spherical(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            look_at=look_at,
        )
        energy = DEFAULT_LIGHT_ENERGY[light_type] * rng.uniform(
            energy_scale_range[0],
            energy_scale_range[1],
        )
        color = (
            rng.uniform(1.0 - color_jitter, 1.0),
            rng.uniform(1.0 - color_jitter, 1.0),
            rng.uniform(1.0 - color_jitter, 1.0),
        )
        if color_jitter > 0.0:
            max_channel = max(color)
            color = (color[0] / max_channel, color[1] / max_channel, color[2] / max_channel)
        size = rng.uniform(area_size_range[0], area_size_range[1])
        lights.append(
            SampledLight(
                light_type=light_type,
                location=location,
                look_at=look_at,
                energy=energy,
                color=color,
                size=size,
            )
        )
    return lights
