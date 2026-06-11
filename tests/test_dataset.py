"""Tests for YOLO dataset layout and train/val splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rembrandt.dataset import split_indices, write_yolo_dataset


def test_dataset_module_is_bpy_free() -> None:
    import rembrandt.dataset as dataset_module

    source = Path(dataset_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source


def test_split_indices_single_frame_all_train() -> None:
    train, val = split_indices(1, 0.8, seed=1)
    assert train == [0]
    assert val == []


def test_split_indices_two_frames_has_validation() -> None:
    train, val = split_indices(2, 0.8, seed=1)
    assert len(train) == 1
    assert len(val) == 1
    assert sorted(train + val) == [0, 1]


def test_split_indices_deterministic_with_seed() -> None:
    first = split_indices(10, 0.8, seed=42)
    second = split_indices(10, 0.8, seed=42)
    assert first == second


def test_split_indices_differs_with_different_seed() -> None:
    first = split_indices(10, 0.8, seed=42)
    second = split_indices(10, 0.8, seed=43)
    assert first != second


def test_split_indices_high_train_fraction_still_has_val_when_n_ge_2() -> None:
    train, val = split_indices(10, 0.99, seed=0)
    assert len(val) >= 1
    assert len(train) + len(val) == 10


def test_split_indices_low_train_fraction() -> None:
    train, val = split_indices(10, 0.1, seed=0)
    assert len(val) >= 1
    assert len(train) + len(val) == 10


def test_split_indices_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n"):
        split_indices(-1, 0.8, seed=0)
    with pytest.raises(ValueError, match="train_fraction"):
        split_indices(5, 1.0, seed=0)


def _write_run_dir(run_dir: Path, count: int) -> None:
    for index in range(count):
        (run_dir / f"frame_{index:04d}.png").write_bytes(b"png")
        (run_dir / f"frame_{index:04d}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
    (run_dir / "run.json").write_text("{}", encoding="utf-8")


def test_write_yolo_dataset_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_dir(run_dir, 4)

    out_dir = tmp_path / "dataset"
    data_yaml = write_yolo_dataset(
        run_dir,
        out_dir,
        class_names={0: "pawn"},
        train_fraction=0.75,
        seed=7,
    )

    assert data_yaml == out_dir / "data.yaml"
    assert not list(run_dir.glob("frame_*.png"))
    train_images = sorted((out_dir / "images" / "train").glob("*.png"))
    val_images = sorted((out_dir / "images" / "val").glob("*.png"))
    assert len(train_images) + len(val_images) == 4
    assert val_images

    for image_path in train_images + val_images:
        label_path = out_dir / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        assert label_path.is_file()

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert payload["path"] == "."
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["names"] == {"0": "pawn"}


def test_write_yolo_dataset_creates_empty_labels_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "frame_0000.png").write_bytes(b"png")

    out_dir = tmp_path / "dataset"
    write_yolo_dataset(
        run_dir,
        out_dir,
        class_names={0: "object"},
        train_fraction=0.8,
        seed=0,
    )

    label = out_dir / "labels" / "train" / "frame_0000.txt"
    assert label.is_file()
    assert label.read_text(encoding="utf-8") == ""


def test_write_yolo_dataset_no_frames_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="no frame"):
        write_yolo_dataset(
            run_dir,
            tmp_path / "dataset",
            class_names={0: "object"},
            train_fraction=0.8,
            seed=0,
        )
