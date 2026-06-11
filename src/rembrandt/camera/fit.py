"""Pure camera framing math."""

from __future__ import annotations

from math import sin, sqrt


def fit_distance(*, target_radius: float, fov_rad: float, margin: float = 1.2) -> float:
    """Minimum camera distance so a sphere of target_radius fits in the FOV cone."""
    if margin <= 0:
        raise ValueError("fit_margin must be greater than 0.")
    return (target_radius * margin) / sin(fov_rad / 2)


def fit_camera_location(
    *,
    requested_location: tuple[float, float, float],
    fit_about: tuple[float, float, float],
    target_radius: float,
    fov_rad: float,
    fit_margin: float,
) -> tuple[float, float, float]:
    """Push the camera back along the anchor ray until the target fits in frame.

    Distance (apparent object size) is determined relative to ``fit_about``. The
    camera aims at ``look_at`` separately after this step.

    Args:
        requested_location: Sampled camera position before fitting.
        fit_about: Anchor for bounding-sphere radius and push-back direction.
        target_radius: Bounding-sphere radius about ``fit_about``.
        fov_rad: Limiting field of view in radians.
        fit_margin: Extra framing margin around the target.

    Returns:
        Fitted camera location in world coordinates.

    Raises:
        ValueError: If ``requested_location`` equals ``fit_about``.
    """
    dx = fit_about[0] - requested_location[0]
    dy = fit_about[1] - requested_location[1]
    dz = fit_about[2] - requested_location[2]
    direction_len = sqrt(dx * dx + dy * dy + dz * dz)
    if direction_len == 0:
        msg = "camera location and fit_about cannot be the same point"
        raise ValueError(msg)

    min_distance = fit_distance(
        target_radius=target_radius,
        fov_rad=fov_rad,
        margin=fit_margin,
    )
    distance = max(direction_len, min_distance)
    scale = distance / direction_len
    return (
        fit_about[0] - dx * scale,
        fit_about[1] - dy * scale,
        fit_about[2] - dz * scale,
    )
