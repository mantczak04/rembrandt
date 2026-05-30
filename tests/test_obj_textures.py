"""Tests for OBJ texture loading in Blender."""

from __future__ import annotations

from pathlib import Path

import bpy
import pytest

from rembrandt.scene import Scene

pytestmark = pytest.mark.bpy

FIXTURE_OBJ = Path(__file__).parent / "fixtures" / "textured_cube" / "cube.obj"


def _image_texture_nodes() -> list[bpy.types.ShaderNodeTexImage]:
    nodes: list[bpy.types.ShaderNodeTexImage] = []
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                nodes.append(node)
    return nodes


def _image_has_pixels(image: bpy.types.Image) -> bool:
    return image.size[0] > 0 and len(image.pixels) > 0


def test_load_object_loads_mtl_textures() -> None:
    scene = Scene()
    scene.load_object(FIXTURE_OBJ)

    texture_nodes = _image_texture_nodes()
    assert texture_nodes, "expected at least one image texture node"
    assert all(
        node.image is not None and _image_has_pixels(node.image) for node in texture_nodes
    )
