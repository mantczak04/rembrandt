"""Camera pose sampling for dataset rendering.

The samplers in this module are pure Python math helpers. They do not import
Blender, so they can be tested quickly without the bpy runtime.

Scene.add_camera and Scene.move_camera default to fit_target=True. If a sampled
camera distance is too close to frame the loaded object, the scene moves the
camera back to the fit distance. This keeps the object visible, but it means
small sampled distances may be overridden. Pass fit_target=False to either
method if exact sampled distances are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, pi, radians, sin, sqrt
from random import Random
from typing import Literal, TypeAlias

SamplingStrategy: TypeAlias = Literal["random", "fibonacci"]
Point3D: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class CameraPose:
    """A camera position and its look-at target in world coordinates.

    Args:
        location: Camera position as an (x, y, z) tuple.
        look_at: World-space point the camera aims at.
    """

    location: Point3D
    look_at: Point3D


def sample_camera_poses(
    *,
    n: int,
    azimuth_range: tuple[float, float] = (0.0, 360.0),
    elevation_range: tuple[float, float] = (-10.0, 30.0),
    distance_range: tuple[float, float] = (3.0, 5.0),
    strategy: SamplingStrategy = "random",
    look_at: Point3D = (0.0, 0.0, 0.0),
    seed: int | None = None,
) -> list[CameraPose]:
    """Generate camera poses on a spherical band around a target point.

    Args:
        n: Number of poses to generate.
        azimuth_range: Inclusive degree range around +Z, measured
            counterclockwise from +X when viewed from above.
        elevation_range: Inclusive degree range above the XY plane.
        distance_range: Inclusive world-unit range for independent uniform
            distance sampling.
        strategy: Sampling strategy. "random" uses independent uniform samples.
            "fibonacci" gives more even angular coverage.
        look_at: World-space point all sampled cameras aim at.
        seed: Optional seed for a local random number generator.

    Returns:
        A list of sampled camera poses.

    Raises:
        ValueError: If any range, count, or strategy is invalid.
    """
    validate_camera_pose_inputs(
        n=n,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
        strategy=strategy,
    )

    rng = Random(seed)
    sin_el_min = sin(radians(elevation_range[0]))
    sin_el_max = sin(radians(elevation_range[1]))
    az_min = radians(azimuth_range[0])
    az_max = radians(azimuth_range[1])

    if strategy == "random":
        return _sample_random(
            n=n,
            sin_el_range=(sin_el_min, sin_el_max),
            azimuth_range=(az_min, az_max),
            distance_range=distance_range,
            look_at=look_at,
            rng=rng,
        )

    return _sample_fibonacci(
        n=n,
        sin_el_range=(sin_el_min, sin_el_max),
        azimuth_range=(az_min, az_max),
        distance_range=distance_range,
        look_at=look_at,
        rng=rng,
    )


def validate_camera_pose_inputs(
    *,
    n: int,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
    strategy: str,
) -> None:
    """Validate camera pose sampling inputs.

    Args:
        n: Number of poses to generate.
        azimuth_range: Inclusive degree range around +Z.
        elevation_range: Inclusive degree range above the XY plane.
        distance_range: Inclusive world-unit range for distance sampling.
        strategy: Sampling strategy name.

    Raises:
        ValueError: If any range, count, or strategy is invalid.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    validate_spherical_ranges(
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
    )
    if strategy not in {"random", "fibonacci"}:
        raise ValueError(f"strategy must be one of 'random' or 'fibonacci', got {strategy!r}")


def validate_spherical_ranges(
    *,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
) -> None:
    """Validate azimuth, elevation, and distance sampling ranges.

    Args:
        azimuth_range: Inclusive degree range around +Z.
        elevation_range: Inclusive degree range above the XY plane.
        distance_range: Inclusive world-unit range for distance sampling.

    Raises:
        ValueError: If any range is invalid.
    """
    if azimuth_range[0] > azimuth_range[1]:
        raise ValueError(f"azimuth_range min must be <= max, got {azimuth_range}")
    if elevation_range[0] < -90 or elevation_range[1] > 90:
        raise ValueError(f"elevation_range must be within [-90, 90], got {elevation_range}")
    if elevation_range[0] > elevation_range[1]:
        raise ValueError(f"elevation_range min must be <= max, got {elevation_range}")
    if distance_range[0] <= 0 or distance_range[1] <= 0:
        raise ValueError(f"distance_range values must be > 0, got {distance_range}")
    if distance_range[0] > distance_range[1]:
        raise ValueError(f"distance_range min must be <= max, got {distance_range}")


def _validate_inputs(
    *,
    n: int,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
    strategy: str,
) -> None:
    """Deprecated private wrapper for older in-repo callers."""
    validate_camera_pose_inputs(
        n=n,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
        strategy=strategy,
    )


def _sample_random(
    *,
    n: int,
    sin_el_range: tuple[float, float],
    azimuth_range: tuple[float, float],
    distance_range: tuple[float, float],
    look_at: Point3D,
    rng: Random,
) -> list[CameraPose]:
    poses: list[CameraPose] = []
    for _ in range(n):
        sin_el = rng.uniform(sin_el_range[0], sin_el_range[1])
        elevation = asin(sin_el)
        azimuth = rng.uniform(azimuth_range[0], azimuth_range[1])
        distance = rng.uniform(distance_range[0], distance_range[1])
        poses.append(_pose_from_spherical(azimuth, elevation, distance, look_at))
    return poses


def _sample_fibonacci(
    *,
    n: int,
    sin_el_range: tuple[float, float],
    azimuth_range: tuple[float, float],
    distance_range: tuple[float, float],
    look_at: Point3D,
    rng: Random,
) -> list[CameraPose]:
    poses: list[CameraPose] = []
    golden_angle = pi * (3.0 - sqrt(5.0))
    az_span = azimuth_range[1] - azimuth_range[0]

    for i in range(n):
        t = (i + 0.5) / n
        sin_el = sin_el_range[0] + t * (sin_el_range[1] - sin_el_range[0])
        elevation = asin(sin_el)
        raw_azimuth = (i * golden_angle) % (2 * pi)
        azimuth = azimuth_range[0] + (raw_azimuth / (2 * pi)) * az_span
        distance = rng.uniform(distance_range[0], distance_range[1])
        poses.append(_pose_from_spherical(azimuth, elevation, distance, look_at))

    return poses


def position_from_spherical(
    *,
    azimuth: float,
    elevation: float,
    distance: float,
    look_at: Point3D,
) -> Point3D:
    """World-space point at spherical coordinates around ``look_at`` (radians).

    Args:
        azimuth: Azimuth in radians, counterclockwise from +X when viewed from above.
        elevation: Elevation in radians above the XY plane.
        distance: Distance from ``look_at`` in world units.
        look_at: World-space center of the spherical band.

    Returns:
        The sampled world-space position.
    """
    horizontal_distance = distance * cos(elevation)
    return (
        look_at[0] + horizontal_distance * cos(azimuth),
        look_at[1] + horizontal_distance * sin(azimuth),
        look_at[2] + distance * sin(elevation),
    )


def _pose_from_spherical(
    azimuth: float,
    elevation: float,
    distance: float,
    look_at: Point3D,
) -> CameraPose:
    location = position_from_spherical(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        look_at=look_at,
    )
    return CameraPose(location=location, look_at=look_at)
