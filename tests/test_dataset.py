"""Tests for YOLO dataset layout and train/val splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rembrandt.backgrounds import BG20K_ATTRIBUTION
from rembrandt.dataset import (
    parse_yolo_label_line,
    print_label_stats,
    split_indices,
    summarize_labels,
    validate_yolo_dataset,
    write_training_handoff,
    write_yolo_dataset,
)


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
        imgsz=640,
    )

    assert data_yaml == out_dir / "data.yaml"
    assert (out_dir / "train_yolo.py").is_file()
    assert (out_dir / "README.md").is_file()
    train_script = (out_dir / "train_yolo.py").read_text(encoding="utf-8")
    assert "imgsz=640" in train_script
    validate_yolo_dataset(out_dir)
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
        imgsz=320,
    )

    label = out_dir / "labels" / "train" / "frame_0000.txt"
    assert label.is_file()
    assert label.read_text(encoding="utf-8") == ""


def test_parse_yolo_label_line_accepts_valid_box() -> None:
    parsed = parse_yolo_label_line("0 0.500000 0.500000 0.200000 0.300000")
    assert parsed == (0, 0.5, 0.5, 0.2, 0.3)


def test_parse_yolo_label_line_rejects_invalid() -> None:
    assert parse_yolo_label_line("") is None
    assert parse_yolo_label_line("0 0.5 0.5 0.2") is None
    assert parse_yolo_label_line("0 1.5 0.5 0.2 0.2") is None


def test_summarize_labels_from_yolo_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for split, lines in {
        "train": ["0 0.2 0.5 0.1 0.4\n", "0 0.8 0.5 0.1 0.6\n"],
        "val": ["0 0.5 0.3 0.2 0.2\n", ""],
    }.items():
        label_dir = dataset_dir / "labels" / split
        label_dir.mkdir(parents=True)
        for index, line in enumerate(lines):
            (label_dir / f"frame_{index:04d}.txt").write_text(line, encoding="utf-8")

    summary = summarize_labels(dataset_dir)
    assert summary.total_files == 4
    assert summary.labeled_count == 3
    assert summary.empty_count == 1
    assert min(summary.centers_x) == pytest.approx(0.2)
    assert max(summary.centers_x) == pytest.approx(0.8)
    assert min(summary.heights) == pytest.approx(0.2)
    assert max(summary.heights) == pytest.approx(0.6)


def test_print_label_stats_writes_summary(capsys: pytest.CaptureFixture[str]) -> None:
    from rembrandt.dataset import LabelSummary

    print_label_stats(
        LabelSummary(
            total_files=2,
            labeled_count=1,
            empty_count=1,
            centers_x=(0.25, 0.75),
            centers_y=(0.5, 0.5),
            heights=(0.2, 0.8),
            widths=(0.1, 0.1),
        )
    )
    output = capsys.readouterr().out
    assert "Label files: 2 total" in output
    assert "height:" in output


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
            imgsz=640,
        )


def test_write_training_handoff_includes_bg20k_when_image_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_training_handoff(dataset_dir, imgsz=512, background_mode="image")

    readme = (dataset_dir / "README.md").read_text(encoding="utf-8")
    assert BG20K_ATTRIBUTION in readme
    assert "pip install ultralytics" in readme
    assert "python train_yolo.py" in readme

    script = (dataset_dir / "train_yolo.py").read_text(encoding="utf-8")
    assert "imgsz=512" in script
    assert 'data="data.yaml"' in script


def test_write_training_handoff_omits_bg20k_when_none_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_training_handoff(dataset_dir, imgsz=640, background_mode="none")

    readme = (dataset_dir / "README.md").read_text(encoding="utf-8")
    assert BG20K_ATTRIBUTION not in readme
    assert "Ultralytics applies its own train-time augmentations" in readme


def test_validate_yolo_dataset_rejects_invalid_label_line(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_valid_dataset_layout(dataset_dir)
    bad_label = dataset_dir / "labels" / "train" / "frame_0000.txt"
    bad_label.write_text("0 0.5 0.5 1.5 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid label line"):
        validate_yolo_dataset(dataset_dir)


def test_validate_yolo_dataset_rejects_missing_label(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_valid_dataset_layout(dataset_dir)
    (dataset_dir / "labels" / "train" / "frame_0000.txt").unlink()

    with pytest.raises(ValueError, match="pairing mismatch"):
        validate_yolo_dataset(dataset_dir)


def test_validate_yolo_dataset_rejects_bad_data_yaml(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "data.yaml").write_text("path: .\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key"):
        validate_yolo_dataset(dataset_dir)


def _write_valid_dataset_layout(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: object",
            ]
        ),
        encoding="utf-8",
    )
    for split, label_line in {
        "train": "0 0.5 0.5 0.2 0.2\n",
        "val": "",
    }.items():
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "frame_0000.png").write_bytes(b"png")
        (label_dir / "frame_0000.txt").write_text(label_line, encoding="utf-8")
