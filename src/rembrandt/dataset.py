"""YOLO dataset layout, train/val splitting, and label analysis. NO bpy."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, pstdev

import yaml


def split_indices(
    n: int,
    train_fraction: float,
    *,
    seed: int | None,
) -> tuple[list[int], list[int]]:
    """Shuffle frame indices into train and validation splits.

    Args:
        n: Total number of frames (indices ``0 .. n-1``).
        train_fraction: Fraction of frames assigned to training (``(0, 1)``).
        seed: RNG seed for reproducible shuffling; ``None`` is non-deterministic.

    Returns:
        Two lists of frame indices: ``(train_indices, val_indices)``. When
        ``n < 2``, all indices go to train and validation is empty. When
        ``n >= 2``, validation receives at least one frame.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    indices = list(range(n))
    rng = Random(seed)
    rng.shuffle(indices)

    if n < 2:
        return indices, []

    n_val = max(1, round(n * (1.0 - train_fraction)))
    n_val = min(n_val, n - 1)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def write_yolo_dataset(
    run_dir: Path,
    out_dir: Path,
    *,
    class_names: dict[int, str],
    train_fraction: float,
    seed: int | None,
) -> Path:
    """Organize flat run frames and labels into Ultralytics YOLO layout.

    Moves ``frame_XXXX.png`` and matching ``frame_XXXX.txt`` from ``run_dir``
    into ``out_dir/images/{train,val}`` and ``out_dir/labels/{train,val}``,
    then writes ``out_dir/data.yaml``.

    Args:
        run_dir: Directory containing flat ``frame_*.png`` and ``frame_*.txt``
            files plus ``run.json``.
        out_dir: Destination root for the YOLO dataset (typically
            ``run_dir / "dataset"``).
        class_names: Mapping of class index to human-readable name for
            ``data.yaml``.
        train_fraction: Train split fraction (see ``split_indices``).
        seed: Split shuffle seed (independent of render seeds).

    Returns:
        Path to the written ``data.yaml`` file.

    Raises:
        ValueError: If no frame PNGs are found in ``run_dir``.
    """
    frame_paths = sorted(run_dir.glob("frame_*.png"))
    if not frame_paths:
        msg = f"no frame_*.png files found in {run_dir}"
        raise ValueError(msg)

    n = len(frame_paths)
    train_indices, val_indices = split_indices(n, train_fraction, seed=seed)
    index_to_split = {index: "train" for index in train_indices}
    index_to_split.update({index: "val" for index in val_indices})

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for frame_index, frame_path in enumerate(frame_paths):
        split = index_to_split[frame_index]
        stem = frame_path.stem
        label_path = run_dir / f"{stem}.txt"
        shutil.move(str(frame_path), str(out_dir / "images" / split / frame_path.name))
        if label_path.is_file():
            shutil.move(str(label_path), str(out_dir / "labels" / split / label_path.name))
        else:
            (out_dir / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")

    names_payload = {str(class_id): name for class_id, name in sorted(class_names.items())}
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": names_payload,
    }
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, default_flow_style=False, sort_keys=False)

    return yaml_path


@dataclass(frozen=True)
class LabelSummary:
    """Distribution stats for YOLO labels in a dataset directory."""

    total_files: int
    labeled_count: int
    empty_count: int
    centers_x: tuple[float, ...]
    centers_y: tuple[float, ...]
    heights: tuple[float, ...]
    widths: tuple[float, ...]


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO detection line into class id and normalized box.

    Args:
        line: A single label line ``"<class> <cx> <cy> <w> <h>"``.

    Returns:
        ``(class_id, cx, cy, w, h)`` when the line is valid, else ``None``.
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) != 5:
        return None

    try:
        class_id = int(parts[0])
        cx, cy, width, height = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None

    if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
        return None

    return class_id, cx, cy, width, height


def _iter_label_files(dataset_dir: Path) -> list[Path]:
    labels_root = dataset_dir / "labels"
    if labels_root.is_dir():
        return sorted(labels_root.rglob("*.txt"))
    return sorted(dataset_dir.glob("frame_*.txt"))


def summarize_labels(dataset_dir: Path) -> LabelSummary:
    """Scan YOLO label files and collect bbox distribution statistics.

    Args:
        dataset_dir: A flat run directory or a YOLO ``dataset/`` root with
            ``labels/{train,val}`` subdirectories.

    Returns:
        Summary stats for non-empty labels in the directory tree.
    """
    label_paths = _iter_label_files(dataset_dir)
    centers_x: list[float] = []
    centers_y: list[float] = []
    heights: list[float] = []
    widths: list[float] = []
    labeled_count = 0
    empty_count = 0

    for label_path in label_paths:
        parsed_any = False
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_label_line(line)
            if parsed is None:
                continue
            parsed_any = True
            _, cx, cy, width, height = parsed
            centers_x.append(cx)
            centers_y.append(cy)
            widths.append(width)
            heights.append(height)

        if parsed_any:
            labeled_count += 1
        else:
            empty_count += 1

    return LabelSummary(
        total_files=len(label_paths),
        labeled_count=labeled_count,
        empty_count=empty_count,
        centers_x=tuple(centers_x),
        centers_y=tuple(centers_y),
        heights=tuple(heights),
        widths=tuple(widths),
    )


def _format_range(values: tuple[float, ...]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    spread = pstdev(values) if len(values) > 1 else 0.0
    return f"min={min(values):.3f} max={max(values):.3f} mean={mean(values):.3f} std={spread:.3f}"


def print_label_stats(summary: LabelSummary) -> None:
    """Print a human-readable label distribution summary to stdout.

    Args:
        summary: Output from ``summarize_labels``.
    """
    print(f"Label files: {summary.total_files} total")
    print(f"  with boxes: {summary.labeled_count}")
    print(f"  empty: {summary.empty_count}")
    if summary.centers_x:
        print(f"  cx: {_format_range(summary.centers_x)}")
        print(f"  cy: {_format_range(summary.centers_y)}")
        print(f"  height: {_format_range(summary.heights)}")
        print(f"  width: {_format_range(summary.widths)}")
