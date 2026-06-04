"""Tests for the background fetch CLI."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from rembrandt.backgrounds import index_backgrounds
from rembrandt.fetch_backgrounds import (
    _parquet_image_to_pil,
    app,
    write_background_images,
)


def test_fetch_backgrounds_module_is_bpy_free() -> None:
    import rembrandt.fetch_backgrounds as fetch_module

    source = Path(fetch_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_parquet_image_to_pil_from_bytes() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(buffer, format="PNG")
    image = _parquet_image_to_pil({"bytes": buffer.getvalue(), "path": None})

    assert image.size == (3, 2)
    assert image.mode == "RGB"


def test_parquet_image_to_pil_rejects_empty_row() -> None:
    with pytest.raises(ValueError, match="no bytes or path"):
        _parquet_image_to_pil({"bytes": None, "path": None})


def test_lazy_stream_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import rembrandt.fetch_backgrounds as fetch_module

    def raise_import_error(name: str, *args: object, **kwargs: object) -> object:
        if name in {"huggingface_hub", "pyarrow"}:
            raise ImportError(f"no {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", raise_import_error)

    with pytest.raises(ImportError, match=r'pip install -e "\.\[backgrounds\]"'):
        list(fetch_module._stream_dataset_images("repo", "train"))


def test_write_background_images(tmp_path: Path) -> None:
    def images() -> list[Image.Image]:
        return [Image.new("RGBA", (4, 4), (index, 0, 0, 255)) for index in range(3)]

    written = write_background_images(iter(images()), out_dir=tmp_path, count=3)

    assert written == [
        tmp_path / "bg_00000.jpg",
        tmp_path / "bg_00001.jpg",
        tmp_path / "bg_00002.jpg",
    ]
    for path in written:
        with Image.open(path) as image:
            assert image.mode == "RGB"


def test_write_background_images_short_stream_warns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def one_image() -> list[Image.Image]:
        return [Image.new("RGB", (2, 2), (1, 2, 3))]

    written = write_background_images(iter(one_image()), out_dir=tmp_path, count=5)

    assert len(written) == 1
    assert "Warning" in capsys.readouterr().err


def test_written_directory_is_indexable(tmp_path: Path) -> None:
    images = [Image.new("RGB", (8, 8), color) for color in ((1, 0, 0), (0, 1, 0))]
    written = write_background_images(iter(images), out_dir=tmp_path / "pool", count=2)
    indexed = index_backgrounds(tmp_path / "pool")
    assert indexed == written


def test_fetch_cli_with_mocked_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rembrandt.fetch_backgrounds as fetch_module

    def fake_stream(dataset: str, split: str) -> list[Image.Image]:
        assert dataset == fetch_module.DEFAULT_DATASET
        assert split == "train"
        return [Image.new("RGB", (4, 4), (9, 9, 9)) for _ in range(2)]

    monkeypatch.setattr(fetch_module, "_stream_dataset_images", fake_stream)
    runner = CliRunner()
    out_dir = tmp_path / "backgrounds"
    result = runner.invoke(app, ["--out", str(out_dir), "--count", "2"])

    assert result.exit_code == 0, result.output
    assert len(list(out_dir.glob("bg_*.jpg"))) == 2
    assert "background.image_dir" in result.stdout
    assert re.search(r"MIT", result.stdout)


def test_lazy_import_only_inside_stream_function() -> None:
    import rembrandt.fetch_backgrounds as fetch_module

    source = Path(fetch_module.__file__).read_text(encoding="utf-8")
    stream_start = source.index("def _stream_dataset_images")
    stream_body = source[stream_start : source.index("\ndef write_background_images", stream_start)]
    assert "HfApi" in stream_body or "huggingface_hub" in stream_body
    assert "pyarrow" in stream_body
    assert "from datasets" not in stream_body
    assert "import datasets" not in stream_body
