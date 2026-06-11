"""Config-driven rendering entry point for Rembrandt."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import numpy.typing as npt
import typer
from PIL import Image

from rembrandt.annotations import (
    bbox_from_mask,
    mask_from_alpha,
    visible_pixel_count,
    yolo_line,
)
from rembrandt.backgrounds import (
    choose_background,
    composite_over,
    composite_over_color,
    index_backgrounds,
    load_cover_resized,
)
from rembrandt.camera_poses import sample_camera_poses
from rembrandt.config import RembrandtConfig, load_config
from rembrandt.dataset import print_label_stats, summarize_labels, write_yolo_dataset
from rembrandt.errors import WorkerRenderError
from rembrandt.framing import sample_frame_framing
from rembrandt.light_poses import sample_light_rig
from rembrandt.postfx import apply_postfx, sample_frame_postfx
from rembrandt.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(
    name="rembrandt-render",
    help="Render synthetic dataset frames from a YAML config.",
    add_completion=False,
)


def resolve_object_path(config_path: Path, object_path: str) -> Path:
    """Resolve an object path relative to the config file or working directory.

    Args:
        config_path: Path to the YAML config file.
        object_path: Object path from the config (absolute or relative).

    Returns:
        Resolved filesystem path to the ``.obj`` file.
    """
    path = Path(object_path)
    if path.is_absolute():
        return path.resolve()

    relative_to_config = (config_path.parent / path).resolve()
    if relative_to_config.is_file():
        return relative_to_config

    relative_to_cwd = (Path.cwd() / path).resolve()
    if relative_to_cwd.is_file():
        return relative_to_cwd

    return path.resolve()


def resolve_output_dir(config_path: Path, output_dir: str) -> Path:
    """Resolve an output directory relative to the config file or working directory.

    Args:
        config_path: Path to the YAML config file.
        output_dir: Output directory from the config (absolute or relative).

    Returns:
        Resolved filesystem path to the output root directory.
    """
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()

    relative_to_config = (config_path.parent / path).resolve()
    if relative_to_config.is_dir():
        return relative_to_config

    relative_to_cwd = (Path.cwd() / path).resolve()
    if relative_to_cwd.is_dir():
        return relative_to_cwd

    return relative_to_config


def resolve_background_dir(config_path: Path, image_dir: str) -> Path:
    """Resolve a background image directory relative to the config or CWD.

    Args:
        config_path: Path to the YAML config file.
        image_dir: Background directory from the config (absolute or relative).

    Returns:
        Resolved filesystem path to the background image directory.
    """
    path = Path(image_dir)
    if path.is_absolute():
        return path.resolve()

    relative_to_config = (config_path.parent / path).resolve()
    if relative_to_config.is_dir():
        return relative_to_config

    relative_to_cwd = (Path.cwd() / path).resolve()
    if relative_to_cwd.is_dir():
        return relative_to_cwd

    return path.resolve()


def parse_frame_range(frame_range: str) -> tuple[int, int]:
    """Parse a half-open frame range ``start:end`` from CLI input.

    Args:
        frame_range: Range string with a single colon separator.

    Returns:
        ``(start, end)`` where frame indices satisfy ``start <= index < end``.

    Raises:
        ValueError: If the string is malformed or bounds are invalid.
    """
    if frame_range.count(":") != 1:
        msg = f"frame range must be start:end, got {frame_range!r}"
        raise ValueError(msg)
    start_text, end_text = frame_range.split(":", 1)
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        msg = f"frame range bounds must be integers, got {frame_range!r}"
        raise ValueError(msg) from exc
    if start < 0 or end < 0:
        msg = f"frame range bounds must be >= 0, got {frame_range!r}"
        raise ValueError(msg)
    if start >= end:
        msg = f"frame range start must be < end, got {frame_range!r}"
        raise ValueError(msg)
    return start, end


def worker_frame_indices(
    *,
    n_frames: int,
    worker_index: int,
    num_workers: int,
    frame_range: tuple[int, int] | None = None,
) -> list[int]:
    """Return the frame indices assigned to one parallel worker.

    Worker ``k`` of ``N`` renders indices ``k, k+N, k+2N, ...`` optionally
    limited to a half-open ``frame_range``.

    Args:
        n_frames: Total number of frames in the run.
        worker_index: Zero-based worker id in ``[0, num_workers)``.
        num_workers: Number of parallel workers.
        frame_range: Optional half-open ``(start, end)`` limit on indices.

    Returns:
        Sorted frame indices for this worker.

    Raises:
        ValueError: If worker or frame counts are invalid.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index must be in [0, {num_workers}), got {worker_index}")

    start, end = frame_range if frame_range is not None else (0, n_frames)
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"invalid frame_range {(start, end)}")
    end = min(end, n_frames)
    return [index for index in range(start, end) if index % num_workers == worker_index]


def _run_metadata_payload(
    cfg: RembrandtConfig,
    *,
    resolved_object_path: Path,
    frame_records: list[dict[str, Any]],
    normalization_scale: float | None = None,
) -> dict[str, Any]:
    """Build the JSON payload written to ``run.json``."""
    return {
        "config": cfg.model_dump(mode="json"),
        "resolved_object_path": str(resolved_object_path),
        "normalization_scale": normalization_scale,
        "frames": frame_records,
    }


def merge_run_metadata(
    run_dir: Path,
    *,
    cfg: RembrandtConfig,
    resolved_object_path: Path,
) -> None:
    """Merge per-worker frame metadata into ``run.json``.

    Args:
        run_dir: Run directory containing ``run.frames.worker_*.json`` files.
        cfg: Validated render configuration.
        resolved_object_path: Resolved path to the source ``.obj`` file.
    """
    partial_paths = sorted(run_dir.glob("run.frames.worker_*.json"))
    frame_records: list[dict[str, Any]] = []
    normalization_scales: list[float | None] = []
    for partial_path in partial_paths:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        frame_records.extend(payload["frames"])
        normalization_scales.append(payload.get("normalization_scale"))
    frame_records.sort(key=lambda record: record["frame"])

    if normalization_scales:
        first_scale = normalization_scales[0]
        if not all(scale == first_scale for scale in normalization_scales):
            msg = "worker partial metadata disagrees on normalization_scale"
            raise ValueError(msg)
        normalization_scale = first_scale
    else:
        normalization_scale = None

    (run_dir / "run.json").write_text(
        json.dumps(
            _run_metadata_payload(
                cfg,
                resolved_object_path=resolved_object_path,
                frame_records=frame_records,
                normalization_scale=normalization_scale,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    for partial_path in partial_paths:
        partial_path.unlink()


def _write_label_file(
    label_path: Path,
    *,
    class_id: int,
    bbox_px: tuple[int, int, int, int] | None,
    visible_pixels: int,
    min_visible_pixels: int,
    width: int,
    height: int,
    frame_index: int,
) -> None:
    """Write a YOLO label file, using an empty file below the visibility floor."""
    if bbox_px is None or visible_pixels < min_visible_pixels:
        label_path.write_text("", encoding="utf-8")
        if bbox_px is not None and visible_pixels < min_visible_pixels:
            print(
                f"Frame {frame_index}: only {visible_pixels} visible pixels "
                f"(min {min_visible_pixels}); wrote empty label"
            )
        return

    label_path.write_text(
        yolo_line(class_id, bbox_px, width=width, height=height) + "\n",
        encoding="utf-8",
    )


def render_from_config(
    cfg: RembrandtConfig,
    *,
    config_path: Path,
    scene_factory: Callable[[], Scene] | None = None,
    stamp: str | None = None,
    output_dir: Path | None = None,
    frame_indices: Sequence[int] | None = None,
    frames_only: bool = False,
    write_run_metadata: bool = True,
    worker_partial_metadata_path: Path | None = None,
) -> Path:
    """Render frames for a validated config.

    Args:
        cfg: Validated render configuration.
        config_path: Path to the YAML file (used to resolve relative object paths).
        scene_factory: Optional factory for tests; defaults to ``Scene``.
        stamp: Optional output subdirectory name; defaults to a timestamp.
        output_dir: Optional pre-created run directory (parallel workers).
        frame_indices: Optional subset of frame indices to render.
        frames_only: When True, skip label files and dataset layout (debugging).
        write_run_metadata: When True, write ``run.json`` after rendering.
        worker_partial_metadata_path: When set, write this worker's frame records
            to the given JSON file instead of ``run.json``.

    Returns:
        The directory containing rendered frame PNGs (flat layout when
        ``frames_only`` is True; otherwise frames are moved into
        ``dataset/`` by the caller).
    """
    object_path = resolve_object_path(config_path, cfg.object.path)
    poses = sample_camera_poses(**cfg.camera.model_dump())
    if output_dir is None:
        run_stamp = stamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_root = resolve_output_dir(config_path, cfg.output.dir)
        output_dir = output_root / run_stamp
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(frame_indices) if frame_indices is not None else list(range(len(poses)))
    if not indices:
        if write_run_metadata and worker_partial_metadata_path is None:
            (output_dir / "run.json").write_text(
                json.dumps(
                    _run_metadata_payload(
                        cfg,
                        resolved_object_path=object_path,
                        frame_records=[],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
        return output_dir

    for index in indices:
        if index < 0 or index >= len(poses):
            msg = f"frame index {index} out of range for {len(poses)} poses"
            raise ValueError(msg)

    frame_records: list[dict[str, Any]] = []

    scene = scene_factory() if scene_factory is not None else Scene()
    scene.load_object(object_path, up_axis=cfg.object.up_axis)
    scene.center_target()
    normalization_scale: float | None = None
    if cfg.object.normalize:
        normalization_scale = scene.normalize_target()

    randomize_lights = cfg.light_randomization.mode == "random"

    if not randomize_lights:
        for light in cfg.lights:
            scene.add_light(
                light_type=light.light_type,
                location=light.location,
                look_at=light.look_at,
                energy=light.energy,
                color=light.color,
                size=light.size,
            )

    scene.add_camera(focal_length=cfg.render.focal_length)

    use_background = cfg.background.mode == "image"
    use_labels = cfg.labels.enabled and not frames_only
    transparent_film = use_labels or use_background
    width, height = cfg.render.resolution

    background_pool: list[Path] = []
    if use_background:
        assert cfg.background.image_dir is not None
        bg_dir = resolve_background_dir(config_path, cfg.background.image_dir)
        background_pool = index_backgrounds(bg_dir)

    for index in indices:
        pose = poses[index]
        rig_summary = ""
        light_rig_record: list[dict[str, Any]] | None = None
        if randomize_lights:
            scene.clear_lights()
            rig = sample_light_rig(
                frame_index=index,
                **cfg.light_randomization.model_dump(exclude={"mode"}),
            )
            light_rig_record = [asdict(sampled) for sampled in rig]
            rig_summary = f" lights=[{', '.join(sampled.light_type for sampled in rig)}]"
            for sampled in rig:
                scene.add_light(
                    light_type=sampled.light_type,
                    location=sampled.location,
                    look_at=sampled.look_at,
                    energy=sampled.energy,
                    color=sampled.color,
                    size=sampled.size,
                )

        framing = sample_frame_framing(
            frame_index=index,
            camera_location=pose.location,
            look_at=pose.look_at,
            target_radius=scene.target_radius_about(pose.look_at),
            focal_length=cfg.render.focal_length,
            resolution=cfg.render.resolution,
            center_jitter=cfg.framing.center_jitter,
            fill_range=cfg.framing.fill_range,
            seed=cfg.framing.seed,
        )
        scene.move_camera(
            location=pose.location,
            look_at=framing.look_at,
            fit_margin=framing.fit_margin,
            fit_about=pose.look_at,
        )
        frame_path = output_dir / f"frame_{index:04d}.png"
        rendered = scene.render(
            frame_path,
            resolution=cfg.render.resolution,
            engine=cfg.render.engine,
            samples=cfg.render.samples,
            transparent_film=transparent_film,
        )

        background_record: str | None = None
        postfx_record: dict[str, float | int] | None = None
        if transparent_film:
            with Image.open(rendered) as frame_image:
                foreground = np.asarray(frame_image.convert("RGBA"), dtype=np.uint8)

            mask = mask_from_alpha(foreground) if use_labels else None
            bbox_px = bbox_from_mask(mask) if mask is not None else None
            visible_pixels = visible_pixel_count(mask) if mask is not None else 0

            composited_rgb: npt.NDArray[np.uint8] | None = None
            if use_background:
                background_path = choose_background(
                    background_pool,
                    frame_index=index,
                    seed=cfg.background.seed,
                )
                background_record = background_path.name
                frame_height, frame_width = foreground.shape[:2]
                background_rgb = load_cover_resized(
                    background_path,
                    width=frame_width,
                    height=frame_height,
                )
                composited_rgb = composite_over(foreground, background_rgb)
                print(
                    f"Rendered frame {index} to {rendered}"
                    f" (background: {background_path.name}){rig_summary}"
                )
            elif use_labels:
                composited_rgb = composite_over_color(foreground, cfg.background.color)
                print(f"Rendered frame {index} to {rendered}{rig_summary}")
            else:
                print(f"Rendered frame {index} to {rendered}{rig_summary}")

            if composited_rgb is not None:
                if cfg.postfx.mode == "random":
                    postfx_params = sample_frame_postfx(
                        frame_index=index,
                        **cfg.postfx.model_dump(exclude={"mode"}),
                    )
                    postfx_rng = (
                        Random(cfg.postfx.seed + index) if cfg.postfx.seed is not None else Random()
                    )
                    composited_rgb = apply_postfx(
                        composited_rgb,
                        postfx_params,
                        rng=postfx_rng,
                    )
                    postfx_record = {
                        "gaussian_noise_sigma": postfx_params.gaussian_noise_sigma,
                        "blur_radius": postfx_params.blur_radius,
                        "jpeg_quality": postfx_params.jpeg_quality,
                        "exposure_ev": postfx_params.exposure_ev,
                    }
                Image.fromarray(composited_rgb, mode="RGB").save(rendered, format="PNG")

            if use_labels:
                label_path = output_dir / f"frame_{index:04d}.txt"
                _write_label_file(
                    label_path,
                    class_id=cfg.object.class_id,
                    bbox_px=bbox_px,
                    visible_pixels=visible_pixels,
                    min_visible_pixels=cfg.labels.min_visible_pixels,
                    width=width,
                    height=height,
                    frame_index=index,
                )
        else:
            print(f"Rendered frame {index} to {rendered}{rig_summary}")

        frame_record: dict[str, Any] = {
            "frame": index,
            "camera_pose": {
                "location": list(pose.location),
                "look_at": list(pose.look_at),
            },
            "framing": {
                "fill": framing.fill,
                "fit_margin": framing.fit_margin,
                "jitter_uv": list(framing.jitter_uv),
                "look_at": list(framing.look_at),
            },
        }
        if light_rig_record is not None:
            frame_record["light_rig"] = light_rig_record
        if background_record is not None:
            frame_record["background"] = background_record
        if postfx_record is not None:
            frame_record["postfx"] = postfx_record
        frame_records.append(frame_record)

    if worker_partial_metadata_path is not None:
        worker_partial_metadata_path.write_text(
            json.dumps(
                {
                    "frames": frame_records,
                    "normalization_scale": normalization_scale,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif write_run_metadata:
        (output_dir / "run.json").write_text(
            json.dumps(
                _run_metadata_payload(
                    cfg,
                    resolved_object_path=object_path,
                    frame_records=frame_records,
                    normalization_scale=normalization_scale,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    return output_dir


def _worker_command(
    config_path: Path,
    *,
    run_dir: Path,
    worker_index: int,
    num_workers: int,
    frame_range: tuple[int, int] | None,
    frames_only: bool,
) -> list[str]:
    """Build the subprocess argv for one parallel render worker."""
    command = [
        sys.executable,
        "-m",
        "rembrandt.render",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--worker-index",
        str(worker_index),
        "--workers-total",
        str(num_workers),
    ]
    if frame_range is not None:
        command.extend(["--frame-range", f"{frame_range[0]}:{frame_range[1]}"])
    if frames_only:
        command.append("--frames-only")
    return command


def _wait_for_workers(
    processes: list[tuple[int, subprocess.Popen[bytes]]],
) -> list[int]:
    """Poll parallel workers until all exit; terminate siblings on first failure.

    Args:
        processes: ``(worker_index, popen)`` pairs launched by the coordinator.

    Returns:
        Worker indices that exited with a non-zero status.
    """
    remaining = dict(processes)
    failed: list[int] = []
    while remaining:
        for index, process in list(remaining.items()):
            code = process.poll()
            if code is None:
                continue
            del remaining[index]
            if code != 0:
                failed.append(index)
        if failed and remaining:
            for process in remaining.values():
                process.terminate()
            for process in remaining.values():
                process.wait()
            remaining.clear()
        if remaining:
            time.sleep(0.2)
    return failed


def render(
    config_path: Path,
    *,
    frames_only: bool = False,
    stats: bool = False,
    workers: int = 1,
    run_dir: Path | None = None,
    worker_index: int | None = None,
    num_workers: int | None = None,
    frame_range: tuple[int, int] | None = None,
) -> tuple[Path, Path | None]:
    """Load a YAML config and render frames (and optionally a YOLO dataset).

    Args:
        config_path: Path to the render config YAML file.
        frames_only: When True, skip YOLO dataset layout.
        stats: When True, print label distribution stats after dataset layout.
        workers: Number of parallel worker processes (coordinator mode when > 1).
        run_dir: Pre-created run directory for worker subprocesses.
        worker_index: Zero-based worker id for subprocess rendering.
        num_workers: Total worker count for subprocess rendering.
        frame_range: Optional half-open frame index limit ``(start, end)``.

    Returns:
        ``(run_dir, data_yaml_path)`` where ``data_yaml_path`` is ``None`` when
        labeling or dataset layout was skipped.
    """
    path = Path(config_path)
    cfg = load_config(path)
    object_path = resolve_object_path(path, cfg.object.path)
    n_frames = cfg.camera.n

    if workers > 1 and worker_index is not None:
        msg = "pass either workers > 1 (coordinator) or worker_index (worker), not both"
        raise ValueError(msg)
    if worker_index is not None and num_workers is None:
        msg = "num_workers is required when worker_index is set"
        raise ValueError(msg)
    if worker_index is not None and run_dir is None:
        msg = "run_dir is required for worker subprocess rendering"
        raise ValueError(msg)

    if workers > 1:
        workers = min(workers, cfg.camera.n)
        run_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_root = resolve_output_dir(path, cfg.output.dir)
        output_dir = run_dir or (output_root / run_stamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        worker_processes: list[tuple[int, subprocess.Popen[bytes]]] = []
        for index in range(workers):
            command = _worker_command(
                path,
                run_dir=output_dir,
                worker_index=index,
                num_workers=workers,
                frame_range=frame_range,
                frames_only=frames_only,
            )
            worker_processes.append((index, subprocess.Popen(command)))

        failed_workers = _wait_for_workers(worker_processes)
        if failed_workers:
            raise WorkerRenderError(failed_workers)

        merge_run_metadata(
            output_dir,
            cfg=cfg,
            resolved_object_path=object_path,
        )
    elif worker_index is not None:
        assert run_dir is not None
        assert num_workers is not None
        indices = worker_frame_indices(
            n_frames=n_frames,
            worker_index=worker_index,
            num_workers=num_workers,
            frame_range=frame_range,
        )
        output_dir = render_from_config(
            cfg,
            config_path=path,
            output_dir=run_dir,
            frame_indices=indices,
            frames_only=frames_only,
            write_run_metadata=False,
            worker_partial_metadata_path=run_dir / f"run.frames.worker_{worker_index:04d}.json",
        )
        return output_dir, None

    else:
        output_dir = render_from_config(cfg, config_path=path, frames_only=frames_only)

    if frames_only or not cfg.labels.enabled:
        return output_dir, None

    dataset_dir = output_dir / "dataset"
    data_yaml = write_yolo_dataset(
        output_dir,
        dataset_dir,
        class_names={cfg.object.class_id: cfg.object.class_name},
        train_fraction=cfg.output.train_val_split,
        seed=cfg.output.split_seed,
        imgsz=cfg.render.resolution[0],
        background_mode=cfg.background.mode,
    )
    if stats:
        print_label_stats(summarize_labels(dataset_dir))
    return output_dir, data_yaml


@app.command()
def render_command(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the render config YAML file.",
        ),
    ],
    frames_only: Annotated[
        bool,
        typer.Option(
            "--frames-only",
            help="Render PNG frames only; skip label files and YOLO dataset layout.",
        ),
    ] = False,
    stats: Annotated[
        bool,
        typer.Option(
            "--stats",
            help="Print bbox center and height distributions from generated labels.",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            min=1,
            help="Render frames in parallel using this many worker processes.",
        ),
    ] = 1,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            help="Pre-created run directory (internal worker mode).",
            hidden=True,
        ),
    ] = None,
    frame_range: Annotated[
        str | None,
        typer.Option(
            "--frame-range",
            help="Half-open frame index range start:end (internal worker mode).",
            hidden=True,
        ),
    ] = None,
    worker_index: Annotated[
        int | None,
        typer.Option(
            "--worker-index",
            min=0,
            help="Zero-based worker id (internal worker mode).",
            hidden=True,
        ),
    ] = None,
    num_workers: Annotated[
        int | None,
        typer.Option(
            "--workers-total",
            min=1,
            help="Total worker count (internal worker mode).",
            hidden=True,
        ),
    ] = None,
) -> None:
    """Render dataset frames from a YAML configuration file."""
    parsed_frame_range = parse_frame_range(frame_range) if frame_range is not None else None
    output_dir, data_yaml = render(
        config_path,
        frames_only=frames_only,
        stats=stats,
        workers=workers,
        run_dir=run_dir,
        worker_index=worker_index,
        num_workers=num_workers,
        frame_range=parsed_frame_range,
    )
    if data_yaml is not None:
        typer.echo(f"Finished rendering dataset to {data_yaml.parent}")
        typer.echo(str(data_yaml))
    elif worker_index is None:
        typer.echo(f"Finished rendering to {output_dir}")


def main() -> None:
    """Console script entry point for ``rembrandt-render``."""
    app()


if __name__ == "__main__":
    main()
