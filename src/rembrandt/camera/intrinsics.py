"""Camera intrinsics and FOV math (BlenderProc CameraUtility port)."""

from __future__ import annotations

from math import atan
from typing import Literal, Protocol

import numpy as np

SensorFit = Literal["AUTO", "HORIZONTAL", "VERTICAL"]
FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


class CameraData(Protocol):
    """Blender camera data block fields used for intrinsics."""

    lens: float
    shift_x: float
    shift_y: float
    sensor_fit: SensorFit
    sensor_width: float
    sensor_height: float


def get_sensor_size(cam: CameraData) -> float:
    """Sensor size in millimeters based on sensor_fit."""
    if cam.sensor_fit == "VERTICAL":
        return cam.sensor_height
    return cam.sensor_width


def get_view_fac_in_px(
    cam: CameraData,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
    resolution_x_in_px: int,
    resolution_y_in_px: int,
) -> float:
    """Camera view extent in pixels (BlenderProc get_view_fac_in_px)."""
    if cam.sensor_fit == "AUTO":
        if pixel_aspect_x * resolution_x_in_px >= pixel_aspect_y * resolution_y_in_px:
            sensor_fit: SensorFit = "HORIZONTAL"
        else:
            sensor_fit = "VERTICAL"
    else:
        sensor_fit = cam.sensor_fit

    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == "HORIZONTAL":
        return float(resolution_x_in_px)
    return pixel_aspect_ratio * resolution_y_in_px


def intrinsics_as_k_matrix(
    *,
    cam: CameraData,
    resolution_x_in_px: int,
    resolution_y_in_px: int,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
) -> FloatMatrix:
    """3x3 K matrix for the given camera and render settings."""
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    view_fac_in_px = get_view_fac_in_px(
        cam,
        pixel_aspect_x,
        pixel_aspect_y,
        resolution_x_in_px,
        resolution_y_in_px,
    )
    sensor_size_in_mm = get_sensor_size(cam)

    fx = cam.lens / sensor_size_in_mm * view_fac_in_px
    fy = fx / pixel_aspect_ratio

    cx = (resolution_x_in_px - 1) / 2 - cam.shift_x * view_fac_in_px
    cy = (resolution_y_in_px - 1) / 2 + cam.shift_y * view_fac_in_px / pixel_aspect_ratio

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )


def fov_from_k_matrix(
    k: FloatMatrix,
    resolution_x_in_px: int,
    resolution_y_in_px: int,
) -> tuple[float, float]:
    """Horizontal and vertical FOV in radians from K and image size."""
    fov_x = 2 * atan(resolution_x_in_px / 2 / k[0, 0])
    fov_y = 2 * atan(resolution_y_in_px / 2 / k[1, 1])
    return fov_x, fov_y


def limiting_fov_from_camera(
    *,
    cam: CameraData,
    resolution_x_in_px: int,
    resolution_y_in_px: int,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
) -> float:
    """Smaller of horizontal/vertical FOV — limiting axis for framing."""
    k = intrinsics_as_k_matrix(
        cam=cam,
        resolution_x_in_px=resolution_x_in_px,
        resolution_y_in_px=resolution_y_in_px,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    fov_x, fov_y = fov_from_k_matrix(k, resolution_x_in_px, resolution_y_in_px)
    return min(fov_x, fov_y)
