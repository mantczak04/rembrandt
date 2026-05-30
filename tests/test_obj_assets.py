"""Tests for Wavefront OBJ/MTL asset helpers."""

from __future__ import annotations

from pathlib import Path

from rembrandt.obj_assets import (
    mtllib_names,
    normalize_mtl_line_endings,
    normalize_obj_mtllibs,
    resolve_texture_file,
    texture_map_names,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "textured_cube"
FIXTURE_OBJ = FIXTURE_DIR / "cube.obj"
FIXTURE_MTL = FIXTURE_DIR / "cube.mtl"
FIXTURE_TEX = FIXTURE_DIR / "red.png"


def test_mtllib_names_reads_obj_reference() -> None:
    assert mtllib_names(FIXTURE_OBJ) == ["cube.mtl"]


def test_texture_map_names_reads_mtl_reference() -> None:
    assert texture_map_names(FIXTURE_MTL) == ["red.png"]


def test_normalize_mtl_line_endings_rewrites_crlf_and_drops_map_ka() -> None:
    path = FIXTURE_MTL.parent / "_crlf.mtl"
    path.write_bytes(b"newmtl m\r\nmap_Ka ambient.png\r\nmap_Kd tex.png\r\n")
    try:
        assert normalize_mtl_line_endings(path) is True
        assert path.read_text(encoding="utf-8") == "newmtl m\nmap_Kd tex.png\n"
        assert normalize_mtl_line_endings(path) is False
    finally:
        path.unlink(missing_ok=True)


def test_normalize_obj_mtllibs_updates_referenced_mtl(tmp_path: Path) -> None:
    obj_path = tmp_path / "model.obj"
    mtl_path = tmp_path / "mat.mtl"
    obj_path.write_text("mtllib mat.mtl\n", encoding="utf-8")
    mtl_path.write_bytes(b"map_Kd tex.png\r\n")

    normalize_obj_mtllibs(obj_path)

    assert mtl_path.read_bytes() == b"map_Kd tex.png\n"


def test_resolve_texture_file_finds_basename_and_spaced_variant(tmp_path: Path) -> None:
    texture = tmp_path / "red.png"
    texture.write_bytes(b"png")

    assert resolve_texture_file(tmp_path, "red.png") == texture.resolve()
    assert resolve_texture_file(tmp_path, "nested/red.png") == texture.resolve()
    assert resolve_texture_file(tmp_path, "red tex.png") is None

    spaced = tmp_path / "red tex.png"
    spaced.write_bytes(b"png")
    assert resolve_texture_file(tmp_path, "red_tex.png") == spaced.resolve()


def test_textured_fixture_files_exist() -> None:
    assert FIXTURE_OBJ.is_file()
    assert FIXTURE_MTL.is_file()
    assert FIXTURE_TEX.is_file()
