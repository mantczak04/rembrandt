"""Per-frame framing jitter and scale diversity. NO bpy."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, tan
from random import Random
from typing import TypeAlias

from rembrandt.camera.fit import fit_distance
from rembrandt.camera.intrinsics import SensorFit, limiting_fov_from_camera

Point3D: TypeAlias = tuple[float, float, float]


class _DefaultCameraData:
    """Blender default camera intrinsics for pure-FOV math."""

    def __init__(self, lens: float) -> None:
        self.lens = lens
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.sensor_fit: SensorFit = "AUTO"
        self.sensor_width = 36.0
        self.sensor_height = 24.0


@dataclass(frozen=True)
class FrameFraming:
    """Per-frame framing parameters consumed by the render loop."""

    look_at: Point3D
    fit_margin: float
    fill: float
    jitter_uv: tuple[float, float]


def _vec_sub(a: Point3D, b: Point3D) -> Point3D:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: Point3D, b: Point3D) -> Point3D:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(vector: Point3D, scalar: float) -> Point3D:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _vec_len(vector: Point3D) -> float:
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _vec_normalize(vector: Point3D) -> Point3D:
    length = _vec_len(vector)
    if length == 0:
        msg = "cannot normalize zero vector"
        raise ValueError(msg)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _vec_cross(a: Point3D, b: Point3D) -> Point3D:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def sample_fill(fill_range: tuple[float, float], rng: Random) -> float:
    """Sample a target object fill fraction from ``fill_range``.

    Args:
        fill_range: Inclusive ``(lo, hi)`` fraction of image height the object
            should occupy after camera fitting.
        rng: Random number generator.

    Returns:
        Sampled fill value in ``fill_range``.
    """
    lo, hi = fill_range
    return rng.uniform(lo, hi)


def fill_to_fit_margin(fill: float) -> float:
    """Convert a target fill fraction to a camera ``fit_margin``.

    Args:
        fill: Target fraction of the limiting image axis occupied by the
            object (must be positive).

    Returns:
        ``1 / fill`` — passed to ``Scene.move_camera(fit_margin=...)``.
    """
    if fill <= 0:
        msg = f"fill must be positive, got {fill}"
        raise ValueError(msg)
    return 1.0 / fill


def camera_image_plane_basis(
    view_direction: Point3D,
    *,
    world_up: Point3D = (0.0, 0.0, 1.0),
) -> tuple[Point3D, Point3D]:
    """Build unit right/up vectors spanning the camera image plane.

    Args:
        view_direction: Vector from the camera toward the look-at point
            (need not be unit length).
        world_up: World up reference for building the basis (+Z by default).

    Returns:
        ``(right, up)`` unit vectors perpendicular to the view direction.
    """
    forward = _vec_normalize(view_direction)
    right = _vec_cross(forward, world_up)
    if _vec_len(right) < 1e-8:
        right = _vec_cross(forward, (0.0, 1.0, 0.0))
    right = _vec_normalize(right)
    up = _vec_normalize(_vec_cross(right, forward))
    return right, up


def limiting_fov_for_focal_length(
    focal_length: float,
    resolution: tuple[int, int],
) -> float:
    """Return the limiting FOV in radians for default Blender camera intrinsics."""
    width, height = resolution
    cam = _DefaultCameraData(focal_length)
    return limiting_fov_from_camera(
        cam=cam,
        resolution_x_in_px=width,
        resolution_y_in_px=height,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )


def fitted_camera_distance(
    *,
    camera_location: Point3D,
    look_at: Point3D,
    target_radius: float,
    focal_length: float,
    resolution: tuple[int, int],
    fit_margin: float,
) -> float:
    """Minimum camera distance after fit-target clamping along the view ray.

    When framing is enabled, ``distance_range`` becomes a lower bound: the
    camera may be pushed farther back so the object fills the sampled
    ``fit_margin``.

    Args:
        camera_location: Sampled camera position before fitting.
        look_at: Look-at point (before center jitter).
        target_radius: Bounding-sphere radius of the target about ``look_at``.
        focal_length: Camera focal length in millimeters.
        resolution: ``(width, height)`` in pixels.
        fit_margin: Framing margin from ``fill_to_fit_margin``.

    Returns:
        Fitted camera distance along the view ray.

    Raises:
        ValueError: If ``camera_location`` equals ``look_at``.
    """
    requested = _vec_sub(look_at, camera_location)
    sampled_distance = _vec_len(requested)
    if sampled_distance == 0:
        msg = "camera location and look_at cannot be the same point"
        raise ValueError(msg)

    limiting_fov = limiting_fov_for_focal_length(focal_length, resolution)
    min_distance = fit_distance(
        target_radius=target_radius,
        fov_rad=limiting_fov,
        margin=fit_margin,
    )
    return max(sampled_distance, min_distance)


def jitter_look_at(
    *,
    look_at: Point3D,
    camera_location: Point3D,
    fitted_distance: float,
    limiting_fov_rad: float,
    center_jitter: float,
    jitter_uv: tuple[float, float],
) -> Point3D:
    """Offset ``look_at`` in the camera image plane for in-frame translation.

    Args:
        look_at: Base look-at point from the camera pose sampler.
        camera_location: Camera position (unchanged by jitter).
        fitted_distance: Camera distance after target fitting.
        limiting_fov_rad: Smaller of horizontal/vertical FOV in radians.
        center_jitter: Fraction of half-frame the bbox center may wander.
        jitter_uv: Uniform samples in ``[-1, 1]`` for each image-plane axis.

    Returns:
        Jittered look-at point in world coordinates.
    """
    if center_jitter == 0.0:
        return look_at

    view = _vec_sub(look_at, camera_location)
    right, up = camera_image_plane_basis(view)
    u, v = jitter_uv
    scale = fitted_distance * tan(limiting_fov_rad / 2.0) * center_jitter
    offset = _vec_add(_vec_scale(right, u * scale), _vec_scale(up, v * scale))
    return _vec_add(look_at, offset)


def sample_frame_framing(
    *,
    frame_index: int,
    camera_location: Point3D,
    look_at: Point3D,
    target_radius: float,
    focal_length: float,
    resolution: tuple[int, int],
    center_jitter: float,
    fill_range: tuple[float, float],
    seed: int | None,
) -> FrameFraming:
    """Sample per-frame fill and look-at jitter for framing diversity.

    Args:
        frame_index: Zero-based frame index combined with ``seed`` for RNG.
        camera_location: Sampled camera position.
        look_at: Base look-at from the pose sampler.
        target_radius: Bounding-sphere radius about ``look_at``.
        focal_length: Camera focal length in millimeters.
        resolution: ``(width, height)`` in pixels.
        center_jitter: Image-plane jitter fraction (0 disables translation).
        fill_range: Inclusive fill fraction range for scale diversity.
        seed: Optional seed; combined with ``frame_index`` for a local RNG.

    Returns:
        Framing parameters for ``Scene.move_camera``.

    Raises:
        ValueError: If ``frame_index`` is negative.
    """
    if frame_index < 0:
        msg = f"frame_index must be >= 0, got {frame_index}"
        raise ValueError(msg)

    rng = Random(seed + frame_index) if seed is not None else Random()
    fill = sample_fill(fill_range, rng)
    fit_margin = fill_to_fit_margin(fill)
    jitter_uv = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))

    fitted_distance = fitted_camera_distance(
        camera_location=camera_location,
        look_at=look_at,
        target_radius=target_radius,
        focal_length=focal_length,
        resolution=resolution,
        fit_margin=fit_margin,
    )
    limiting_fov = limiting_fov_for_focal_length(focal_length, resolution)
    jittered_look_at = jitter_look_at(
        look_at=look_at,
        camera_location=camera_location,
        fitted_distance=fitted_distance,
        limiting_fov_rad=limiting_fov,
        center_jitter=center_jitter,
        jitter_uv=jitter_uv,
    )

    return FrameFraming(
        look_at=jittered_look_at,
        fit_margin=fit_margin,
        fill=fill,
        jitter_uv=jitter_uv,
    )
