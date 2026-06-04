"""Tests for post-render background compositing."""

from __future__ import annotations

from pathlib import Path
from random import random, seed

import numpy as np
import pytest
from PIL import Image

from rembrandt.backgrounds import (
    apply_background_to_frame,
    choose_background,
    composite_over,
    index_backgrounds,
    load_cover_resized,
)
from rembrandt.errors import BackgroundDirectoryNotFoundError


def test_backgrounds_module_is_bpy_free() -> None:
    import rembrandt.backgrounds as backgrounds_module

    source = Path(backgrounds_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_index_backgrounds_finds_nested_files(tmp_path: Path) -> None:
    nested = tmp_path / "pool" / "nested"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"png")
    (tmp_path / "pool" / "b.JPG").write_bytes(b"jpg")
    (tmp_path / "pool" / "skip.txt").write_text("nope", encoding="utf-8")

    paths = index_backgrounds(tmp_path / "pool")

    assert paths == sorted(
        [
            nested / "a.png",
            tmp_path / "pool" / "b.JPG",
        ]
    )


def test_index_backgrounds_missing_dir_raises() -> None:
    with pytest.raises(BackgroundDirectoryNotFoundError, match="missing"):
        index_backgrounds("/nonexistent/missing")


def test_index_backgrounds_empty_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no background images found"):
        index_backgrounds(empty)


def test_choose_background_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.png" for index in range(3)]
    for path in paths:
        path.touch()

    first = choose_background(paths, frame_index=0, seed=7)
    second = choose_background(paths, frame_index=0, seed=7)
    third = choose_background(paths, frame_index=1, seed=7)

    assert first == second
    assert first != third or len(paths) == 1


def test_choose_background_seed_none_works(tmp_path: Path) -> None:
    paths = [tmp_path / "only.png"]
    paths[0].touch()
    chosen = choose_background(paths, frame_index=0, seed=None)
    assert chosen == paths[0]


def test_choose_background_rejects_invalid_inputs(tmp_path: Path) -> None:
    path = tmp_path / "one.png"
    path.touch()
    with pytest.raises(ValueError, match="frame_index"):
        choose_background([path], frame_index=-1, seed=1)
    with pytest.raises(ValueError, match="empty"):
        choose_background([], frame_index=0, seed=1)


def test_choose_background_does_not_mutate_global_rng(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.png" for index in range(5)]
    for path in paths:
        path.touch()

    seed(0)
    before = random()
    choose_background(paths, frame_index=3, seed=99)
    after = random()

    seed(0)
    expected_before = random()
    expected_after = random()

    assert before == expected_before
    assert after == expected_after


def _write_solid_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_load_cover_resized_output_shape(tmp_path: Path) -> None:
    wide = tmp_path / "wide.png"
    tall = tmp_path / "tall.png"
    exact = tmp_path / "exact.png"
    _write_solid_image(wide, (200, 100), (255, 0, 0))
    _write_solid_image(tall, (100, 200), (0, 255, 0))
    _write_solid_image(exact, (80, 60), (0, 0, 255))

    for path in (wide, tall, exact):
        result = load_cover_resized(path, width=64, height=48)
        assert result.shape == (48, 64, 3)


def test_composite_over_opaque_and_transparent_pixels() -> None:
    foreground = np.zeros((2, 2, 4), dtype=np.uint8)
    foreground[0, 0] = (10, 20, 30, 255)
    foreground[0, 1] = (40, 50, 60, 0)
    foreground[1, 0] = (70, 80, 90, 128)
    background = np.full((2, 2, 3), 200, dtype=np.uint8)

    result = composite_over(foreground, background)

    assert tuple(result[0, 0]) == (10, 20, 30)
    assert tuple(result[0, 1]) == (200, 200, 200)
    blended = result[1, 0]
    for channel, fg, bg in zip(blended, (70, 80, 90), (200, 200, 200), strict=True):
        expected = round(fg * 0.5 + bg * 0.5)
        assert abs(int(channel) - expected) <= 1


def test_composite_over_shape_mismatch_raises() -> None:
    foreground = np.zeros((2, 2, 4), dtype=np.uint8)
    background = np.zeros((3, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="match"):
        composite_over(foreground, background)


def test_apply_background_to_frame_round_trip(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    background_path = tmp_path / "bg.png"
    frame = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    frame.putpixel((0, 0), (0, 0, 0, 0))
    frame.save(frame_path)
    _write_solid_image(background_path, (64, 64), (0, 255, 0))

    apply_background_to_frame(frame_path, background_path)

    with Image.open(frame_path) as result:
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (0, 255, 0)
        assert result.getpixel((16, 16)) == (255, 0, 0)
