"""Tests for per-frame framing jitter and scale diversity."""

from __future__ import annotations

from math import isclose
from pathlib import Path

import pytest

from rembrandt.framing import (
    camera_image_plane_basis,
    fill_to_fit_margin,
    fitted_camera_distance,
    jitter_look_at,
    limiting_fov_for_focal_length,
    sample_fill,
    sample_frame_framing,
)


def test_framing_module_is_bpy_free() -> None:
    import rembrandt.framing as framing_module

    source = Path(framing_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_fill_to_fit_margin_inverts_fill() -> None:
    assert isclose(fill_to_fit_margin(0.5), 2.0)
    assert isclose(fill_to_fit_margin(0.25), 4.0)


def test_fill_to_fit_margin_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="fill"):
        fill_to_fit_margin(0.0)


def test_sample_fill_within_range() -> None:
    from random import Random

    rng = Random(0)
    for _ in range(20):
        fill = sample_fill((0.15, 0.75), rng)
        assert 0.15 <= fill <= 0.75


def test_camera_image_plane_basis_is_orthonormal() -> None:
    right, up = camera_image_plane_basis((1.0, 1.0, 1.0))
    assert isclose(right[0] ** 2 + right[1] ** 2 + right[2] ** 2, 1.0, rel_tol=1e-6)
    assert isclose(up[0] ** 2 + up[1] ** 2 + up[2] ** 2, 1.0, rel_tol=1e-6)
    dot = right[0] * up[0] + right[1] * up[1] + right[2] * up[2]
    assert isclose(dot, 0.0, abs_tol=1e-6)


def test_fitted_camera_distance_respects_lower_bound() -> None:
    distance = fitted_camera_distance(
        camera_location=(5.0, 0.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        target_radius=1.0,
        focal_length=50.0,
        resolution=(640, 640),
        fit_margin=2.0,
    )
    assert distance >= 5.0


def test_fitted_camera_distance_pushes_back_for_large_fill() -> None:
    close = fitted_camera_distance(
        camera_location=(3.0, 0.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        target_radius=1.0,
        focal_length=50.0,
        resolution=(640, 640),
        fit_margin=fill_to_fit_margin(0.75),
    )
    far = fitted_camera_distance(
        camera_location=(3.0, 0.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        target_radius=1.0,
        focal_length=50.0,
        resolution=(640, 640),
        fit_margin=fill_to_fit_margin(0.15),
    )
    assert far > close


def test_jitter_look_at_zero_jitter_is_identity() -> None:
    look_at = (0.0, 0.0, 0.0)
    result = jitter_look_at(
        look_at=look_at,
        camera_location=(5.0, 0.0, 0.0),
        fitted_distance=5.0,
        limiting_fov_rad=limiting_fov_for_focal_length(50.0, (640, 640)),
        center_jitter=0.0,
        jitter_uv=(0.9, -0.9),
    )
    assert result == look_at


def test_jitter_look_at_moves_in_image_plane() -> None:
    look_at = (0.0, 0.0, 0.0)
    limiting_fov = limiting_fov_for_focal_length(50.0, (640, 640))
    result = jitter_look_at(
        look_at=look_at,
        camera_location=(5.0, 0.0, 0.0),
        fitted_distance=5.0,
        limiting_fov_rad=limiting_fov,
        center_jitter=0.35,
        jitter_uv=(1.0, 0.0),
    )
    assert result != look_at
    assert isclose(result[2], look_at[2], abs_tol=1e-6)


def test_sample_frame_framing_deterministic_with_seed() -> None:
    kwargs = {
        "frame_index": 3,
        "camera_location": (4.0, 1.0, 2.0),
        "look_at": (0.0, 0.0, 0.0),
        "target_radius": 1.0,
        "focal_length": 50.0,
        "resolution": (640, 640),
        "center_jitter": 0.35,
        "fill_range": (0.15, 0.75),
        "seed": 42,
    }
    first = sample_frame_framing(**kwargs)
    second = sample_frame_framing(**kwargs)
    assert first == second


def test_sample_frame_framing_differs_with_frame_index() -> None:
    base = {
        "camera_location": (4.0, 1.0, 2.0),
        "look_at": (0.0, 0.0, 0.0),
        "target_radius": 1.0,
        "focal_length": 50.0,
        "resolution": (640, 640),
        "center_jitter": 0.35,
        "fill_range": (0.15, 0.75),
        "seed": 42,
    }
    first = sample_frame_framing(frame_index=0, **base)
    second = sample_frame_framing(frame_index=1, **base)
    assert first != second


def test_sample_frame_framing_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        sample_frame_framing(
            frame_index=-1,
            camera_location=(4.0, 0.0, 0.0),
            look_at=(0.0, 0.0, 0.0),
            target_radius=1.0,
            focal_length=50.0,
            resolution=(640, 640),
            center_jitter=0.0,
            fill_range=(0.5, 0.5),
            seed=0,
        )


def test_fitted_camera_distance_matches_scene_helper_for_fill() -> None:
    """Framing and scene fit math agree for fill=0.5."""
    from math import isclose, sqrt

    from rembrandt.camera.fit import fit_camera_location

    camera_location = (4.0, 1.0, 2.0)
    look_at = (0.0, 0.0, 0.0)
    target_radius = 1.2
    focal_length = 50.0
    resolution = (640, 480)
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
