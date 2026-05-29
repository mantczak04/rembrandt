"""Tests for bpy-free preview band and pose geometry."""

from __future__ import annotations

from math import asin, atan2, degrees, sqrt
from pathlib import Path

import numpy as np
import pytest

from rembrandt.camera_poses import SamplingStrategy
from rembrandt.preview.geometry import (
    PreviewPoseGeometry,
    band_display_radius,
    build_preview_pose_geometry,
    spherical_to_cartesian,
)
from rembrandt.preview.mesh import load_preview_mesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHESS_BOARD_OBJ = PROJECT_ROOT / "test-obj" / "12951_Stone_Chess_Board_v1_L3.obj"


@pytest.mark.parametrize("strategy", ["random", "fibonacci"])
def test_build_preview_pose_geometry_camera_count(strategy: SamplingStrategy) -> None:
    mesh = load_preview_mesh(CHESS_BOARD_OBJ)
    geometry = build_preview_pose_geometry(
        bbox=mesh.bbox,
        n=25,
        strategy=strategy,
        seed=7,
    )

    assert isinstance(geometry, PreviewPoseGeometry)
    assert len(geometry.cameras.locations) == 25


def test_build_preview_pose_geometry_band_within_angular_bounds() -> None:
    mesh = load_preview_mesh(CHESS_BOARD_OBJ)
    azimuth_range = (20.0, 120.0)
    elevation_range = (-15.0, 25.0)
    look_at = (1.0, -2.0, 0.5)
    geometry = build_preview_pose_geometry(
        bbox=mesh.bbox,
        n=10,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=(4.0, 6.0),
        seed=3,
        look_at=look_at,
    )

    points = np.asarray(geometry.band.surface.positions, dtype=np.float64).reshape(-1, 3)
    for point in points:
        azimuth, elevation = _recover_angles(point, look_at)
        assert azimuth_range[0] - 1e-6 <= azimuth <= azimuth_range[1] + 1e-6
        assert elevation_range[0] - 1e-6 <= elevation <= elevation_range[1] + 1e-6
        distance = _distance_from_look_at(point, look_at)
        assert distance == pytest.approx(geometry.display_radius, rel=1e-5, abs=1e-5)


def test_build_preview_pose_geometry_ground_plane_at_bbox_base() -> None:
    mesh = load_preview_mesh(CHESS_BOARD_OBJ)
    geometry = build_preview_pose_geometry(bbox=mesh.bbox, n=5, seed=1)
    z_base = mesh.bbox[0][2]
    plane_z = np.asarray(geometry.ground_plane.positions, dtype=np.float64).reshape(-1, 3)[:, 2]
    np.testing.assert_allclose(plane_z, z_base, atol=1e-12)


def test_band_display_radius_wraps_large_objects() -> None:
    mesh = load_preview_mesh(CHESS_BOARD_OBJ)
    distance_range = (3.0, 5.0)
    radius = band_display_radius(mesh.bbox, distance_range)
    assert radius >= distance_range[1]
    assert radius > 10.0


def test_preview_band_uses_display_radius_for_legibility() -> None:
    mesh = load_preview_mesh(CHESS_BOARD_OBJ)
    geometry = build_preview_pose_geometry(
        bbox=mesh.bbox,
        n=5,
        distance_range=(3.0, 5.0),
        seed=1,
    )

    assert geometry.band.surface.distance == pytest.approx(geometry.display_radius)


def test_spherical_to_cartesian_round_trip() -> None:
    point = spherical_to_cartesian(4.0, 45.0, 10.0, look_at=(0.0, 0.0, 0.0))
    azimuth, elevation = _recover_angles(np.asarray(point), (0.0, 0.0, 0.0))
    assert azimuth == pytest.approx(45.0, abs=1e-6)
    assert elevation == pytest.approx(10.0, abs=1e-6)
    assert _distance_from_look_at(np.asarray(point), (0.0, 0.0, 0.0)) == pytest.approx(4.0)


def test_preview_geometry_module_is_bpy_free() -> None:
    import rembrandt.preview.geometry as geometry_module

    source = Path(geometry_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def _recover_angles(
    location: np.ndarray,
    look_at: tuple[float, float, float],
) -> tuple[float, float]:
    offset = location - np.asarray(look_at, dtype=np.float64)
    distance = float(np.linalg.norm(offset))
    elevation = degrees(asin(offset[2] / distance))
    azimuth = degrees(atan2(offset[1], offset[0]))
    return azimuth, elevation


def _distance_from_look_at(
    location: np.ndarray,
    look_at: tuple[float, float, float],
) -> float:
    offset = location - np.asarray(look_at, dtype=np.float64)
    return float(sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2))
