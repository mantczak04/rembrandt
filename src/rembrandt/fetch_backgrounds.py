"""CLI to fetch BG-20k background images from the Hugging Face Hub."""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

DEFAULT_DATASET = "unography/BG-20k-1200px"

app = typer.Typer(
    name="rembrandt-fetch-backgrounds",
    help="Download background images for post-render compositing.",
    add_completion=False,
)


def _parquet_image_to_pil(image_cell: object) -> Image.Image:
    """Decode a Hugging Face parquet ``Image`` struct row to a PIL image.

    Args:
        image_cell: Dict-like value with ``bytes`` and/or ``path`` keys.

    Returns:
        The decoded PIL image.

    Raises:
        TypeError: If ``image_cell`` is not a mapping.
        ValueError: If neither ``bytes`` nor ``path`` is populated.
    """
    if not isinstance(image_cell, dict):
        msg = f"expected image dict, got {type(image_cell).__name__}"
        raise TypeError(msg)

    raw_bytes = image_cell.get("bytes")
    if raw_bytes:
        return Image.open(io.BytesIO(raw_bytes))

    path = image_cell.get("path")
    if path:
        return Image.open(path)

    raise ValueError("parquet image row has no bytes or path")


def _stream_dataset_images(dataset: str, split: str) -> Iterator[Image.Image]:
    """Stream PIL images from parquet shards on the Hugging Face Hub.

    Args:
        dataset: Hugging Face dataset repo id.
        split: Dataset split name (used as the parquet filename prefix).

    Yields:
        PIL images from the dataset ``image`` column.

    Raises:
        ImportError: If optional ``huggingface_hub`` / ``pyarrow`` are missing.
        ValueError: If no parquet shards match ``split``.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem
    except ImportError as exc:
        msg = 'optional background fetch dependencies are required: pip install -e ".[backgrounds]"'
        raise ImportError(msg) from exc

    shards = sorted(
        path
        for path in HfApi().list_repo_files(dataset, repo_type="dataset")
        if path.startswith(f"data/{split}-") and path.endswith(".parquet")
    )
    if not shards:
        msg = f"no parquet shards found for dataset {dataset!r} split {split!r}"
        raise ValueError(msg)

    fs = HfFileSystem()
    for shard in shards:
        hub_path = f"datasets/{dataset}/{shard}"
        with fs.open(hub_path, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(batch_size=1, columns=["image"]):
                image_cell = batch.to_pydict()["image"][0]
                yield _parquet_image_to_pil(image_cell)


def write_background_images(
    images: Iterable[Image.Image],
    *,
    out_dir: Path,
    count: int,
) -> list[Path]:
    """Write up to ``count`` RGB JPEG backgrounds into ``out_dir``.

    Args:
        images: Iterable of source images.
        out_dir: Destination directory (created if missing).
        count: Maximum number of images to write.

    Returns:
        Paths of written JPEG files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, image in enumerate(islice(images, count)):
        path = out_dir / f"bg_{index:05d}.jpg"
        image.convert("RGB").save(path, format="JPEG", quality=90)
        written.append(path)
    if len(written) < count:
        typer.echo(
            f"Warning: stream yielded {len(written)} images, fewer than requested {count}.",
            err=True,
        )
    return written


@app.command()
def fetch_command(
    out_dir: Annotated[
        Path,
        typer.Option("--out", help="Destination directory."),
    ] = Path("backgrounds"),
    count: Annotated[
        int,
        typer.Option(min=1, help="Number of images to fetch."),
    ] = 2000,
    dataset: Annotated[
        str,
        typer.Option(help="Hugging Face dataset repo id."),
    ] = DEFAULT_DATASET,
    split: Annotated[
        str,
        typer.Option(help="Dataset split to stream from."),
    ] = "train",
) -> None:
    """Download background images for use with ``background.image_dir``."""
    images = _stream_dataset_images(dataset, split)
    written = write_background_images(images, out_dir=out_dir, count=count)
    typer.echo(
        f"Wrote {len(written)} background images to {out_dir.resolve()}. "
        f"Set background.image_dir: {out_dir} in your render config."
    )
    typer.echo(
        "BG-20k (MIT): Bridging Composite and Real: Towards End-to-End Deep "
        "Image Matting (IJCV 2021)."
    )


def main() -> None:
    """Console script entry point for ``rembrandt-fetch-backgrounds``."""
    app()


if __name__ == "__main__":
    main()
