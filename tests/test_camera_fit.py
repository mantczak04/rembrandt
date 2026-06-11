"""Pure tests for camera push-back fit geometry."""

from __future__ import annotations

from math import isclose, sqrt

from rembrandt.camera.fit import fit_camera_location
from rembrandt.framing import (
    fill_to_fit_margin,
    fitted_camera_distance,
    limiting_fov_for_focal_length,
)


def test_fit_camera_location_is_determined_by_anchor_only() -> None:
    """Fitted position depends on fit_about and radius, not a separate aim point."""
    requested_location = (5.0, 0.0, 0.0)
    anchor = (0.0, 0.0, 0.0)
    target_radius = 1.0
    fov_rad = limiting_fov_for_focal_length(50.0, (640, 640))
    fit_margin = fill_to_fit_margin(0.5)

    first = fit_camera_location(
        requested_location=requested_location,
        fit_about=anchor,
        target_radius=target_radius,
        fov_rad=fov_rad,
        fit_margin=fit_margin,
    )
    second = fit_camera_location(
        requested_location=requested_location,
        fit_about=anchor,
        target_radius=target_radius,
        fov_rad=fov_rad,
        fit_margin=fit_margin,
    )

    assert first == second
    distance = sqrt(
        (first[0] - anchor[0]) ** 2 + (first[1] - anchor[1]) ** 2 + (first[2] - anchor[2]) ** 2
    )
    assert isclose(
        distance,
        fitted_camera_distance(
            camera_location=requested_location,
            look_at=anchor,
            target_radius=target_radius,
            focal_length=50.0,
            resolution=(640, 640),
            fit_margin=fit_margin,
        ),
        rel_tol=1e-9,
    )


def test_fitted_camera_distance_matches_fit_camera_location() -> None:
    """Framing distance math matches the scene push-back helper for any jitter."""
    camera_location = (5.0, 0.0, 0.0)
    look_at = (0.0, 0.0, 0.0)
    target_radius = 1.0
    focal_length = 50.0
    resolution = (640, 640)
    fit_margin = fill_to_fit_margin(0.5)
    fov_rad = limiting_fov_for_focal_length(focal_length, resolution)

    framing_distance = fitted_camera_distance(
        camera_location=camera_location,
        look_at=look_at,
        target_radius=target_radius,
        focal_length=focal_length,
        resolution=resolution,
        fit_margin=fit_margin,
    )
    fitted_location = fit_camera_location(
        requested_location=camera_location,
        fit_about=look_at,
        target_radius=target_radius,
        fov_rad=fov_rad,
        fit_margin=fit_margin,
    )
    scene_distance = sqrt(
        (fitted_location[0] - look_at[0]) ** 2
        + (fitted_location[1] - look_at[1]) ** 2
        + (fitted_location[2] - look_at[2]) ** 2
    )

    assert isclose(framing_distance, scene_distance, rel_tol=1e-9)
