"""Tests for pure per-frame light rig sampling."""

from __future__ import annotations

from math import asin, atan2, degrees, sqrt
from pathlib import Path
from random import random, seed

import pytest

from rembrandt.light_poses import (
    DEFAULT_LIGHT_ENERGY,
    SampledLight,
    sample_light_rig,
)


def test_sample_light_rig_count_within_range() -> None:
    for frame_index in range(50):
        rig = sample_light_rig(frame_index=frame_index, seed=1)
        assert 1 <= len(rig) <= 3


def test_sample_light_rig_fixed_count() -> None:
    for frame_index in range(20):
        rig = sample_light_rig(frame_index=frame_index, count_range=(2, 2), seed=2)
        assert len(rig) == 2


def test_sample_light_rig_within_angular_bounds() -> None:
    azimuth_range = (20.0, 120.0)
    elevation_range = (15.0, 75.0)
    look_at = (10.0, 20.0, 30.0)
    for frame_index in range(30):
        rig = sample_light_rig(
            frame_index=frame_index,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            distance_range=(5.0, 5.0),
            look_at=look_at,
            seed=3,
        )
        for light in rig:
            azimuth, elevation = _recover_angles(light)
            assert azimuth_range[0] - 1e-6 <= azimuth <= azimuth_range[1] + 1e-6
            assert elevation_range[0] - 1e-6 <= elevation <= elevation_range[1] + 1e-6


def test_sample_light_rig_within_distance_range() -> None:
    distance_range = (6.0, 9.0)
    look_at = (10.0, 20.0, 30.0)
    for frame_index in range(30):
        rig = sample_light_rig(
            frame_index=frame_index,
            distance_range=distance_range,
            look_at=look_at,
            seed=4,
        )
        for light in rig:
            distance = _distance_from_look_at(light)
            assert distance_range[0] - 1e-9 <= distance <= distance_range[1] + 1e-9


def test_sample_light_rig_light_types_restricted() -> None:
    for frame_index in range(30):
        rig = sample_light_rig(
            frame_index=frame_index,
            light_types=("SUN",),
            seed=5,
        )
        assert all(light.light_type == "SUN" for light in rig)


def test_sample_light_rig_light_types_drawn_from_pool() -> None:
    allowed = {"POINT", "SUN", "AREA"}
    for frame_index in range(30):
        rig = sample_light_rig(frame_index=frame_index, seed=6)
        assert all(light.light_type in allowed for light in rig)


def test_sample_light_rig_energy_scaled() -> None:
    energy_scale_range = (0.5, 2.0)
    for frame_index in range(30):
        rig = sample_light_rig(
            frame_index=frame_index,
            energy_scale_range=energy_scale_range,
            seed=7,
        )
        for light in rig:
            scale = light.energy / DEFAULT_LIGHT_ENERGY[light.light_type]
            assert energy_scale_range[0] - 1e-9 <= scale <= energy_scale_range[1] + 1e-9


def test_sample_light_rig_color_no_jitter() -> None:
    rig = sample_light_rig(frame_index=0, color_jitter=0.0, count_range=(2, 2), seed=8)
    for light in rig:
        assert light.color == (1.0, 1.0, 1.0)


def test_sample_light_rig_color_with_jitter() -> None:
    rig = sample_light_rig(frame_index=0, color_jitter=0.3, count_range=(3, 3), seed=9)
    for light in rig:
        assert all(0.7 - 1e-9 <= channel <= 1.0 + 1e-9 for channel in light.color)
        assert max(light.color) == pytest.approx(1.0, abs=1e-9)


def test_sample_light_rig_reproducible_with_same_seed() -> None:
    first = sample_light_rig(frame_index=0, seed=10)
    second = sample_light_rig(frame_index=0, seed=10)
    assert first == second


def test_sample_light_rig_differs_across_frames_with_same_seed() -> None:
    rigs = [sample_light_rig(frame_index=i, seed=11) for i in range(5)]
    assert len({tuple(rig) for rig in rigs}) > 1


def test_sample_light_rig_seed_none_works() -> None:
    rig = sample_light_rig(frame_index=0, seed=None)
    assert len(rig) >= 1


def test_sample_light_rig_does_not_mutate_global_rng() -> None:
    seed(0)
    before = random()
    sample_light_rig(frame_index=0, seed=99)
    after = random()

    seed(0)
    expected_before = random()
    expected_after = random()

    assert before == expected_before
    assert after == expected_after


@pytest.mark.parametrize(
    "light_types",
    [("POINT",), ("POINT", "SUN", "AREA")],
)
def test_sample_light_rig_self_consistent_per_configuration(
    light_types: tuple[str, ...],
) -> None:
    first = sample_light_rig(
        frame_index=0,
        light_types=light_types,
        count_range=(2, 2),
        seed=12,
    )
    second = sample_light_rig(
        frame_index=0,
        light_types=light_types,
        count_range=(2, 2),
        seed=12,
    )
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_index": -1}, "frame_index"),
        ({"frame_index": 0, "count_range": (0, 2)}, "count_range"),
        ({"frame_index": 0, "count_range": (3, 2)}, "count_range"),
        ({"frame_index": 0, "light_types": ()}, "light_types"),
        ({"frame_index": 0, "light_types": ("DISCO",)}, "light_types"),
        ({"frame_index": 0, "azimuth_range": (350.0, 10.0)}, "azimuth_range"),
        ({"frame_index": 0, "elevation_range": (-91.0, 10.0)}, "elevation_range"),
        ({"frame_index": 0, "distance_range": (0.0, 1.0)}, "distance_range"),
        ({"frame_index": 0, "energy_scale_range": (0.0, 1.0)}, "energy_scale_range"),
        ({"frame_index": 0, "energy_scale_range": (2.0, 1.0)}, "energy_scale_range"),
        ({"frame_index": 0, "area_size_range": (0.0, 1.0)}, "area_size_range"),
        ({"frame_index": 0, "color_jitter": 1.1}, "color_jitter"),
    ],
)
def test_sample_light_rig_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_light_rig(**kwargs)  # type: ignore[arg-type]


def test_light_poses_module_is_bpy_free() -> None:
    import rembrandt.light_poses as light_poses_module

    source = Path(light_poses_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def _recover_angles(light: SampledLight) -> tuple[float, float]:
    dx = light.location[0] - light.look_at[0]
    dy = light.location[1] - light.look_at[1]
    dz = light.location[2] - light.look_at[2]
    distance = _distance_from_look_at(light)

    azimuth = degrees(atan2(dy, dx))
    if azimuth < 0.0:
        azimuth += 360.0
    elevation = degrees(asin(dz / distance))
    return azimuth, elevation


def _distance_from_look_at(light: SampledLight) -> float:
    dx = light.location[0] - light.look_at[0]
    dy = light.location[1] - light.look_at[1]
    dz = light.location[2] - light.look_at[2]
    return sqrt((dx * dx) + (dy * dy) + (dz * dz))
