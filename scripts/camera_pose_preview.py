"""Interactive camera pose preview for Rembrandt.

Run with:
    streamlit run scripts/camera_pose_preview.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from rembrandt.camera_poses import CameraPose, SamplingStrategy, sample_camera_poses

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT = PROJECT_ROOT / "test-obj" / "12951_Stone_Chess_Board_v1_L3.obj"
LOOK_AT = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MeshPreview:
    """Centered mesh data for Plotly preview."""

    vertices: np.ndarray
    triangles: list[tuple[int, int, int]]


def main() -> None:
    """Launch the Streamlit preview app."""
    st.set_page_config(page_title="Rembrandt Camera Pose Preview", layout="wide")
    st.title("Camera Pose Preview")
    st.write(
        "Tune `sample_camera_poses` visually before rendering. The object is centered at "
        "`look_at=(0, 0, 0)`, matching the current render setup."
    )

    controls = _read_controls()
    mesh = _load_mesh(Path(controls.object_path))
    poses = sample_camera_poses(
        n=controls.n,
        azimuth_range=controls.azimuth_range,
        elevation_range=controls.elevation_range,
        distance_range=controls.distance_range,
        strategy=controls.strategy,
        seed=controls.seed,
        look_at=LOOK_AT,
    )

    figure = _build_figure(
        mesh=mesh,
        poses=poses,
        azimuth_range=controls.azimuth_range,
        elevation_range=controls.elevation_range,
        distance_range=controls.distance_range,
        show_rays=controls.show_rays,
    )

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Sampler Snippet")
        st.code(_snippet(controls), language="python")
        st.caption("Copy these values into `src/rembrandt/main.py` or your future config.")


@dataclass(frozen=True)
class PreviewControls:
    """UI values used to generate the preview."""

    object_path: str
    n: int
    azimuth_range: tuple[float, float]
    elevation_range: tuple[float, float]
    distance_range: tuple[float, float]
    strategy: SamplingStrategy
    seed: int | None
    show_rays: bool


def _read_controls() -> PreviewControls:
    with st.sidebar:
        st.header("Object")
        object_path = st.text_input("OBJ path", value=str(DEFAULT_OBJECT))

        st.header("Camera Sampling")
        n = st.slider("Number of poses", min_value=1, max_value=500, value=10, step=1)
        azimuth_range = st.slider(
            "Azimuth range (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=(0.0, 360.0),
            step=1.0,
        )
        elevation_range = st.slider(
            "Elevation range (degrees)",
            min_value=-90.0,
            max_value=90.0,
            value=(-10.0, 30.0),
            step=1.0,
        )
        distance_range = st.slider(
            "Distance range",
            min_value=0.1,
            max_value=20.0,
            value=(3.0, 5.0),
            step=0.1,
        )
        strategy = st.selectbox(
            "Strategy",
            options=("random", "fibonacci"),
            index=0,
        )
        use_seed = st.checkbox("Use seed", value=True)
        seed = st.number_input("Seed", min_value=0, value=42, step=1) if use_seed else None
        show_rays = st.checkbox("Show look-at rays", value=True)

    return PreviewControls(
        object_path=object_path,
        n=n,
        azimuth_range=(float(azimuth_range[0]), float(azimuth_range[1])),
        elevation_range=(float(elevation_range[0]), float(elevation_range[1])),
        distance_range=(float(distance_range[0]), float(distance_range[1])),
        strategy=strategy,
        seed=int(seed) if seed is not None else None,
        show_rays=show_rays,
    )


@st.cache_data(show_spinner=False)
def _load_mesh(path: Path) -> MeshPreview:
    if not path.exists():
        st.error(f"OBJ file not found: {path}")
        st.stop()

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    with path.open(encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                indices = [_parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                triangles.extend(_triangulate(indices))

    if not vertices:
        st.error(f"OBJ file contains no vertices: {path}")
        st.stop()

    vertex_array = np.asarray(vertices, dtype=float)
    center = (vertex_array.min(axis=0) + vertex_array.max(axis=0)) / 2
    return MeshPreview(vertices=vertex_array - center, triangles=triangles)


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw_index = int(token.split("/")[0])
    if raw_index < 0:
        return vertex_count + raw_index
    return raw_index - 1


def _triangulate(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []

    first = indices[0]
    return [(first, indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def _build_figure(
    *,
    mesh: MeshPreview,
    poses: list[CameraPose],
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
    show_rays: bool,
) -> go.Figure:
    fig = go.Figure()
    _add_mesh(fig, mesh)
    _add_sphere_band(
        fig,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        distance_range=distance_range,
    )
    _add_camera_points(fig, poses)
    if show_rays:
        _add_camera_rays(fig, poses)

    radius = max(float(np.linalg.norm(mesh.vertices, axis=1).max()), distance_range[1])
    axis_limit = radius * 1.15
    fig.update_layout(
        scene={
            "aspectmode": "cube",
            "xaxis": {"range": [-axis_limit, axis_limit], "title": "X"},
            "yaxis": {"range": [-axis_limit, axis_limit], "title": "Y"},
            "zaxis": {"range": [-axis_limit, axis_limit], "title": "Z"},
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
    return fig


def _add_mesh(fig: go.Figure, mesh: MeshPreview) -> None:
    x, y, z = mesh.vertices.T
    if mesh.triangles:
        i, j, k = zip(*mesh.triangles, strict=True)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                name="Object",
                color="lightgray",
                opacity=0.75,
            )
        )
        return

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="Object vertices",
            marker={"size": 2, "color": "lightgray"},
        )
    )


def _add_sphere_band(
    fig: go.Figure,
    *,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance_range: tuple[float, float],
) -> None:
    for distance, opacity, name in (
        (distance_range[1], 0.18, "Max distance"),
        (distance_range[0], 0.07, "Min distance"),
    ):
        x, y, z = _sphere_surface(
            distance=distance,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
        )
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                name=name,
                showscale=False,
                opacity=opacity,
                colorscale=[[0, "royalblue"], [1, "royalblue"]],
            )
        )

    _add_band_edges(fig, azimuth_range, elevation_range, distance_range[1])


def _sphere_surface(
    *,
    distance: float,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuths = np.radians(np.linspace(azimuth_range[0], azimuth_range[1], 80))
    elevations = np.radians(np.linspace(elevation_range[0], elevation_range[1], 40))
    az_grid, el_grid = np.meshgrid(azimuths, elevations)
    horizontal = distance * np.cos(el_grid)
    x = horizontal * np.cos(az_grid)
    y = horizontal * np.sin(az_grid)
    z = distance * np.sin(el_grid)
    return x, y, z


def _add_band_edges(
    fig: go.Figure,
    azimuth_range: tuple[float, float],
    elevation_range: tuple[float, float],
    distance: float,
) -> None:
    elevations = np.linspace(elevation_range[0], elevation_range[1], 80)
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], 120)

    for azimuth in azimuth_range:
        points = [_spherical_to_cartesian(distance, azimuth, elevation) for elevation in elevations]
        _add_line(fig, points, name=f"Azimuth {azimuth:.0f} deg")

    for elevation in elevation_range:
        points = [_spherical_to_cartesian(distance, azimuth, elevation) for azimuth in azimuths]
        _add_line(fig, points, name=f"Elevation {elevation:.0f} deg")


def _add_camera_points(fig: go.Figure, poses: list[CameraPose]) -> None:
    x = [pose.location[0] for pose in poses]
    y = [pose.location[1] for pose in poses]
    z = [pose.location[2] for pose in poses]
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="Sampled cameras",
            marker={"size": 5, "color": "crimson"},
        )
    )


def _add_camera_rays(fig: go.Figure, poses: list[CameraPose]) -> None:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for pose in poses:
        x.extend([pose.location[0], pose.look_at[0], None])
        y.extend([pose.location[1], pose.look_at[1], None])
        z.extend([pose.location[2], pose.look_at[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name="Look-at rays",
            line={"color": "crimson", "width": 2},
            opacity=0.35,
        )
    )


def _spherical_to_cartesian(
    distance: float, azimuth_deg: float, elevation_deg: float
) -> tuple[
    float,
    float,
    float,
]:
    azimuth = radians(azimuth_deg)
    elevation = radians(elevation_deg)
    horizontal = distance * cos(elevation)
    return (
        horizontal * cos(azimuth),
        horizontal * sin(azimuth),
        distance * sin(elevation),
    )


def _add_line(fig: go.Figure, points: list[tuple[float, float, float]], *, name: str) -> None:
    x, y, z = zip(*points, strict=True)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name=name,
            line={"color": "royalblue", "width": 4},
            showlegend=False,
        )
    )


def _snippet(controls: PreviewControls) -> str:
    return "\n".join(
        [
            "poses = sample_camera_poses(",
            f"    n={controls.n},",
            f"    azimuth_range={controls.azimuth_range},",
            f"    elevation_range={controls.elevation_range},",
            f"    distance_range={controls.distance_range},",
            f'    strategy="{controls.strategy}",',
            f"    seed={controls.seed},",
            ")",
        ]
    )


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.web import cli as streamlit_cli

    if get_script_run_ctx(suppress_warning=True) is not None:
        main()
    else:
        passthrough_args = sys.argv[1:]
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.runOnSave=false",
            "--server.fileWatcherType=none",
            *passthrough_args,
        ]
        raise SystemExit(streamlit_cli.main())
