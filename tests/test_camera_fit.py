"""Tests for pure camera fit math."""

from __future__ import annotations

from math import sin

import pytest

from rembrandt.camera.fit import fit_distance


def test_fit_distance_known_value() -> None:
    radius = 2.0
    fov = 1.0
    margin = 1.2
    expected = (radius * margin) / sin(fov / 2)
    assert fit_distance(target_radius=radius, fov_rad=fov, margin=margin) == pytest.approx(expected)


def test_fit_distance_default_margin() -> None:
    assert fit_distance(target_radius=1.0, fov_rad=1.0) == pytest.approx(1.2 / sin(0.5))


@pytest.mark.parametrize("margin", [0.0, -1.0])
def test_fit_distance_invalid_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="fit_margin"):
        fit_distance(target_radius=1.0, fov_rad=1.0, margin=margin)
