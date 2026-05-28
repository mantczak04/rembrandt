"""Camera projection and orientation utilities."""

from rembrandt.camera.fit import fit_distance
from rembrandt.camera.intrinsics import (
    fov_from_k_matrix,
    get_sensor_size,
    get_view_fac_in_px,
    intrinsics_as_k_matrix,
    limiting_fov_from_camera,
)

__all__ = [
    "fit_distance",
    "fov_from_k_matrix",
    "get_sensor_size",
    "get_view_fac_in_px",
    "intrinsics_as_k_matrix",
    "limiting_fov_from_camera",
    "require_nonzero_direction",
    "rotation_euler_from_forward",
]


def __getattr__(name: str) -> object:
    if name in {"require_nonzero_direction", "rotation_euler_from_forward"}:
        from rembrandt.camera import orientation

        return getattr(orientation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
