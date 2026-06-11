"""Tests for deterministic fixture writers (bpy-free)."""

from __future__ import annotations

from pathlib import Path

from tests.fixture_factories import write_two_offset_cubes_obj


def test_write_two_offset_cubes_obj_emits_two_object_groups(tmp_path: Path) -> None:
    obj_path = write_two_offset_cubes_obj(tmp_path / "two_cubes.obj")
    text = obj_path.read_text(encoding="utf-8")
    assert text.count("o cube_") == 2
    assert text.count("\nv ") == 16
    assert text.count("\nf ") == 12
