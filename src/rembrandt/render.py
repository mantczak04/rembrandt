"""Config-driven rendering entry point for Rembrandt."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rembrandt.camera_poses import sample_camera_poses
from rembrandt.config import RembrandtConfig, load_config
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


def render_from_config(
    cfg: RembrandtConfig,
    *,
    config_path: Path,
    scene_factory: Callable[[], Scene] | None = None,
    stamp: str | None = None,
) -> Path:
    """Render frames for a validated config.

    Args:
        cfg: Validated render configuration.
        config_path: Path to the YAML file (used to resolve relative object paths).
        scene_factory: Optional factory for tests; defaults to ``Scene``.
        stamp: Optional output subdirectory name; defaults to a timestamp.

    Returns:
        The directory containing rendered frame PNGs.
    """
    object_path = resolve_object_path(config_path, cfg.object.path)
    poses = sample_camera_poses(**cfg.camera.model_dump())
    run_stamp = stamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(cfg.output.dir) / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = scene_factory() if scene_factory is not None else Scene()
    scene.load_object(object_path)
    scene.center_target()

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

    for index, pose in enumerate(poses):
        scene.move_camera(location=pose.location, look_at=pose.look_at)
        frame_path = output_dir / f"frame_{index:04d}.png"
        rendered = scene.render(
            frame_path,
            resolution=cfg.render.resolution,
            engine=cfg.render.engine,
            samples=cfg.render.samples,
        )
        print(f"Rendered frame {index} to {rendered}")

    return output_dir


def render(config_path: Path) -> Path:
    """Load a YAML config and render frames.

    Args:
        config_path: Path to the render config YAML file.

    Returns:
        The directory containing rendered frame PNGs.
    """
    path = Path(config_path)
    cfg = load_config(path)
    return render_from_config(cfg, config_path=path)


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
) -> None:
    """Render dataset frames from a YAML configuration file."""
    output_dir = render(config_path)
    typer.echo(f"Finished rendering to {output_dir}")


def main() -> None:
    """Console script entry point for ``rembrandt-render``."""
    app()


if __name__ == "__main__":
    main()
