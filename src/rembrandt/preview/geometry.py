"""Camera band, sampled poses, and ground-plane geometry for the SPA preview."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import cos, radians, sin
from typing import Literal

import numpy as np
import numpy.typing as npt

from rembrandt.camera_poses import CameraPose, SamplingStrategy, sample_camera_poses

_AZIMUTH_STEPS = 80
_ELEVATION_STEPS = 40
_EDGE_AZIMUTH_STEPS = 120
_EDGE_ELEVATION_STEPS = 80

BBoxLike = Sequence[Sequence[float]]


@dataclass(frozen=True)
class BandShell:
    """A spherical band surface at a fixed distance."""

    distance: float
    positions: list[float]
    azimuth_count: int
    elevation_count: int


@dataclass(frozen=True)
class BandEdgeLine:
    """A polyline along a constant azimuth or elevation on the band."""

    kind: Literal["azimuth", "elevation"]
    value_deg: float
    positions: list[float]


@dataclass(frozen=True)
class PreviewBand:
    """Angular band surface and outline edges."""

    surface: BandShell
    edges: list[BandEdgeLine]


@dataclass(frozen=True)
class PreviewCameras:
    """Sampled camera locations and look-at rays for the preview."""

    locations: list[list[float]]
    look_at: list[float]
    rays: list[list[list[float]]]


@dataclass(frozen=True)
class PreviewGroundPlane:
    """Horizontal quad under the object at the bbox base."""

    positions: list[float]
    indices: list[int]


@dataclass(frozen=True)
class PreviewPoseGeometry:
    """Band, camera points, and ground plane for the SPA 3D view."""

    band: PreviewBand
    cameras: PreviewCameras
    ground_plane: PreviewGroundPlane
    display_radius: float


def build_preview_pose_geometry(
    *,
    bbox: BBoxLike,
    n: int,
    azimuth_range: tuple[float, float] = (0.0, 360.0),
    elevation_range: tuple[float, float] = (-10.0, 30.0),
    distance_range: tuple[float, float] = (3.0, 5.0),
    strategy: SamplingStrategy = "random",
    seed: int | None = None,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PreviewPoseGeometry:
    """Build preview band, camera points, and ground plane from camera parameters.

    Args:
        bbox: Oriented mesh bounds ``[[min], [max]]`` from ``PreviewMesh``.
        n: Number of camera poses to sample.
        azimuth_range: Azimuth limits in degrees.
        elevation_range: Elevation limits in degrees.
        distance_range: Distance limits for pose sampling and inner band shell.
        strategy: ``sample_camera_poses`` strategy name.
        seed: Optional RNG seed for pose sampling.
        look_at: World-space look-at target shared by poses and band geometry.

    Returns:
        Serializable geometry for the preview API / Three.js.
    """
    display_radius = band_display_radius(bbox, distance_range)
    band = PreviewBand(
        surface=sphere_band_shell(
            distance=display_radius,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            look_at=look_at,
        ),
        edges=sphere_band_edges(
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            distance=display_radius,
            look_at=look_at,
        ),
    )
    poses = sample_camera_poses(
        n=n,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
        strategy=strategy,
        seed=seed,
        look_at=look_at,
    )
    return PreviewPoseGeometry(
        band=band,
        cameras=_camera_points_from_poses(poses),
        ground_plane=build_ground_plane(bbox=bbox, radius=display_radius),
        display_radius=display_radius,
    )


def band_display_radius(
    bbox: BBoxLike,
    distance_range: tuple[float, float],
) -> float:
    """Cosmetic band / ground-plane radius for legible wrapping around the object."""
    bbox_arr = _bbox_array(bbox)
    mins = bbox_arr[0]
    maxs = bbox_arr[1]
    corner_radius = max(
        float(np.linalg.norm(np.array(corner, dtype=np.float64)))
        for corner in product(
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2]),
        )
    )
    return max(corner_radius, distance_range[1])


def spherical_to_cartesian(
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Convert spherical coordinates to world-space Cartesian."""
    azimuth = radians(azimuth_deg)
    elevation = radians(elevation_deg)
    horizontal = distance * cos(elevation)
    offset = (
        horizontal * cos(azimuth),
        horizontal * sin(azimuth),
        distance * sin(elevation),
    )
    return (
        look_at[0] + offset[0],
        look_at[1] + offset[1],
        look_at[2] + offset[2],
    )


def sphere_band_shell(
    *,
    distance: float,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BandShell:
    """Sample a spherical band surface grid as a flat position buffer."""
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], _AZIMUTH_STEPS)
    elevations = np.linspace(elevation_range[0], elevation_range[1], _ELEVATION_STEPS)
    positions: list[float] = []
    for elevation_deg in elevations:
        for azimuth_deg in azimuths:
            point = spherical_to_cartesian(distance, azimuth_deg, elevation_deg, look_at)
            positions.extend(point)
    return BandShell(
        distance=distance,
        positions=positions,
        azimuth_count=_AZIMUTH_STEPS,
        elevation_count=_ELEVATION_STEPS,
    )


def sphere_band_edges(
    *,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance: float,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[BandEdgeLine]:
    """Build azimuth and elevation outline polylines on the band."""
    elevations = np.linspace(elevation_range[0], elevation_range[1], _EDGE_ELEVATION_STEPS)
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], _EDGE_AZIMUTH_STEPS)
    edges: list[BandEdgeLine] = []

    for azimuth_deg in azimuth_range:
        positions: list[float] = []
        for elevation_deg in elevations:
            positions.extend(
                spherical_to_cartesian(distance, azimuth_deg, float(elevation_deg), look_at),
            )
        edges.append(
            BandEdgeLine(kind="azimuth", value_deg=azimuth_deg, positions=positions),
        )

    for elevation_deg in elevation_range:
        positions = []
        for azimuth_deg in azimuths:
            positions.extend(
                spherical_to_cartesian(distance, float(azimuth_deg), elevation_deg, look_at),
            )
        edges.append(
            BandEdgeLine(kind="elevation", value_deg=elevation_deg, positions=positions),
        )

    return edges


def build_ground_plane(
    *,
    bbox: BBoxLike,
    radius: float,
) -> PreviewGroundPlane:
    """Build a horizontal quad at the bbox base, sized to the band display radius."""
    z_base = float(_bbox_array(bbox)[0][2])
    corners = (
        (-radius, -radius, z_base),
        (radius, -radius, z_base),
        (radius, radius, z_base),
        (-radius, radius, z_base),
    )
    positions = [coord for corner in corners for coord in corner]
    return PreviewGroundPlane(positions=positions, indices=[0, 1, 2, 0, 2, 3])


def _bbox_array(bbox: BBoxLike) -> npt.NDArray[np.float64]:
    bbox_arr = np.asarray(bbox, dtype=np.float64)
    if bbox_arr.shape != (2, 3):
        msg = f"bbox must have shape (2, 3), got {bbox_arr.shape}"
        raise ValueError(msg)
    if not np.isfinite(bbox_arr).all():
        raise ValueError("bbox must contain only finite values")
    return bbox_arr


def _camera_points_from_poses(poses: list[CameraPose]) -> PreviewCameras:
    if not poses:
        return PreviewCameras(locations=[], look_at=[0.0, 0.0, 0.0], rays=[])

    look_at = list(poses[0].look_at)
    locations = [list(pose.location) for pose in poses]
    rays = [[list(pose.location), look_at] for pose in poses]
    return PreviewCameras(locations=locations, look_at=look_at, rays=rays)
