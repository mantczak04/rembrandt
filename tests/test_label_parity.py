"""Parity between alpha-mask bboxes and projected vertex bboxes (bpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rembrandt.annotations import bbox_from_mask, mask_from_alpha
from rembrandt.camera.intrinsics import intrinsics_as_k_matrix
from rembrandt.scene import Scene
from tests.fixture_factories import write_two_offset_cubes_obj
from tests.test_paths import chess_board_object_path, sample_object_path, sample_object_up_axis


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return inter / (area_a + area_b - inter)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _projected_vertex_bbox(
    scene: Scene,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    bpy = pytest.importorskip("bpy")
    assert scene.camera is not None
    cam_obj = scene.camera
    cam_data = cam_obj.data
    render = bpy.context.scene.render

    k = intrinsics_as_k_matrix(
        cam=cam_data,
        resolution_x_in_px=width,
        resolution_y_in_px=height,
        pixel_aspect_x=render.pixel_aspect_x,
        pixel_aspect_y=render.pixel_aspect_y,
    )
    world_to_cam = np.array(cam_obj.matrix_world.inverted(), dtype=np.float64)

    points: list[tuple[float, float]] = []
    for target in scene.targets:
        for vertex in target.data.vertices:
            world = target.matrix_world @ vertex.co
            cam_h = world_to_cam @ np.array([world.x, world.y, world.z, 1.0], dtype=np.float64)
            cam = cam_h[:3].copy()
            cam[1] *= -1.0
            cam[2] *= -1.0
            if cam[2] <= 1e-6:
                continue
            projected = k @ cam
            u = projected[0] / projected[2]
            v = projected[1] / projected[2]
            if 0.0 <= u < width and 0.0 <= v < height:
                points.append((u, v))

    if not points:
        raise RuntimeError("no vertices projected into the image")

    us, vs = zip(*points, strict=True)
    return (
        int(np.floor(min(us))),
        int(np.floor(min(vs))),
        int(np.ceil(max(us))),
        int(np.ceil(max(vs))),
    )


def _mask_bbox_from_render(
    scene: Scene,
    frame_path: Path,
    *,
    resolution: int,
) -> tuple[int, int, int, int]:
    scene.render(frame_path, resolution=(resolution, resolution), samples=1, transparent_film=True)
    with Image.open(frame_path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    bbox = bbox_from_mask(mask_from_alpha(rgba))
    assert bbox is not None
    return bbox


def _assert_mask_bbox_matches_projected_vertices(
    scene: Scene,
    tmp_path: Path,
    *,
    resolution: int = 128,
) -> None:
    scene.add_camera()
    scene.add_light(light_type="SUN", location=(4.0, -4.0, 6.0), look_at=(0.0, 0.0, 0.0))
    scene.move_camera(location=(4.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.0))

    frame_path = tmp_path / "parity.png"
    mask_bbox = _mask_bbox_from_render(scene, frame_path, resolution=resolution)
    projected_bbox = _projected_vertex_bbox(scene, width=resolution, height=resolution)

    assert _bbox_contains(projected_bbox, mask_bbox)
    assert _bbox_iou(mask_bbox, projected_bbox) > 0.9


@pytest.mark.bpy
@pytest.mark.parametrize(
    "obj_path,up_axis",
    [
        (sample_object_path(), sample_object_up_axis()),
        (chess_board_object_path(), "Z"),
    ],
)
def test_mask_bbox_matches_projected_vertices(
    tmp_path: Path,
    obj_path: Path,
    up_axis: str,
) -> None:
    """Mask bbox should agree with projected mesh vertices for visible objects."""
    pytest.importorskip("bpy")
    if not obj_path.is_file():
        pytest.skip(f"fixture not found: {obj_path}")

    scene = Scene()
    scene.load_object(obj_path, up_axis=up_axis)  # type: ignore[arg-type]
    scene.center_target()
    _assert_mask_bbox_matches_projected_vertices(scene, tmp_path)


@pytest.mark.bpy
def test_mask_bbox_matches_projected_vertices_on_two_offset_cubes(tmp_path: Path) -> None:
    """Multi-mesh OBJ parity using an in-test generated two-cube fixture."""
    pytest.importorskip("bpy")

    obj_path = write_two_offset_cubes_obj(tmp_path / "two_offset_cubes.obj")
    scene = Scene()
    scene.load_object(obj_path, up_axis="Z")
    assert len(scene.targets) == 2
    scene.center_target()
    _assert_mask_bbox_matches_projected_vertices(scene, tmp_path)
