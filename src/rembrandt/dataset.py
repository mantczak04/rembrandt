"""YOLO dataset layout and train/val splitting. NO bpy."""

from __future__ import annotations

import shutil
from pathlib import Path
from random import Random

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
