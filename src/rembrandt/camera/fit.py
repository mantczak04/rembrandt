"""Pure camera framing math."""

from __future__ import annotations

from math import sin


def fit_distance(*, target_radius: float, fov_rad: float, margin: float = 1.2) -> float:
    """Minimum camera distance so a sphere of target_radius fits in the FOV cone."""
    if margin <= 0:
        raise ValueError("fit_margin must be greater than 0.")
    return (target_radius * margin) / sin(fov_rad / 2)
