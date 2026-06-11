"""Config-driven rendering entry point for Rembrandt."""

from __future__ import annotations

import datetime
import json
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
    frames_only: bool = False,
) -> Path:
    """Render frames for a validated config.

    Args:
        cfg: Validated render configuration.
        config_path: Path to the YAML file (used to resolve relative object paths).
        scene_factory: Optional factory for tests; defaults to ``Scene``.
        stamp: Optional output subdirectory name; defaults to a timestamp.
        frames_only: When True, skip label files and dataset layout (debugging).

    Returns:
        The directory containing rendered frame PNGs (flat layout when
        ``frames_only`` is True; otherwise frames are moved into
        ``dataset/`` by the caller).
    """
    object_path = resolve_object_path(config_path, cfg.object.path)
    poses = sample_camera_poses(**cfg.camera.model_dump())
    run_stamp = stamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = resolve_output_dir(config_path, cfg.output.dir)
    output_dir = output_root / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, Any]] = []

    scene = scene_factory() if scene_factory is not None else Scene()
    scene.load_object(object_path, up_axis=cfg.object.up_axis)
    scene.center_target()

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

    for index, pose in enumerate(poses):
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

    run_metadata = {
        "config": cfg.model_dump(mode="json"),
        "resolved_object_path": str(object_path),
        "frames": frame_records,
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    return output_dir


def render(
    config_path: Path,
    *,
    frames_only: bool = False,
    stats: bool = False,
) -> tuple[Path, Path | None]:
    """Load a YAML config and render frames (and optionally a YOLO dataset).

    Args:
        config_path: Path to the render config YAML file.
        frames_only: When True, skip YOLO dataset layout.
        stats: When True, print label distribution stats after dataset layout.

    Returns:
        ``(run_dir, data_yaml_path)`` where ``data_yaml_path`` is ``None`` when
        labeling or dataset layout was skipped.
    """
    path = Path(config_path)
    cfg = load_config(path)
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
) -> None:
    """Render dataset frames from a YAML configuration file."""
    output_dir, data_yaml = render(config_path, frames_only=frames_only, stats=stats)
    if data_yaml is not None:
        typer.echo(f"Finished rendering dataset to {data_yaml.parent}")
        typer.echo(str(data_yaml))
    else:
        typer.echo(f"Finished rendering to {output_dir}")


def main() -> None:
    """Console script entry point for ``rembrandt-render``."""
    app()


if __name__ == "__main__":
    main()
