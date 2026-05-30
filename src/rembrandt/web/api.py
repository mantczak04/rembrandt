"""Preview and config API routes for the Rembrandt SPA."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rembrandt.config import RembrandtConfig, dump_config
from rembrandt.convention import SourceUpAxis
from rembrandt.errors import ModelFileNotFoundError
from rembrandt.preview.geometry import (
    PreviewPoseGeometry,
    build_preview_pose_geometry,
)
from rembrandt.preview.mesh import PreviewMesh, load_preview_mesh

router = APIRouter()
Point3 = tuple[float, float, float]
BBox = tuple[Point3, Point3]


class PreviewMeshRequest(BaseModel):
    """Request body for mesh preview."""

    path: str
    up_axis: SourceUpAxis = "Z"


class PreviewMeshResponse(BaseModel):
    """Oriented mesh geometry for Three.js."""

    positions: list[float]
    indices: list[int]
    bbox: list[list[float]]


class PreviewPosesRequest(BaseModel):
    """Camera sampling parameters plus mesh bounds for preview geometry."""

    bbox: BBox = Field(
        description="Axis-aligned bounds [[min], [max]] from ``POST /preview/mesh``.",
    )
    n: int
    azimuth_range: tuple[float, float] = (0.0, 360.0)
    elevation_range: tuple[float, float] = (-10.0, 30.0)
    distance_range: tuple[float, float] = (3.0, 5.0)
    strategy: Literal["random", "fibonacci"] = "random"
    seed: int | None = None
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)


class BandShellResponse(BaseModel):
    distance: float
    positions: list[float]
    azimuth_count: int
    elevation_count: int


class BandEdgeLineResponse(BaseModel):
    kind: Literal["azimuth", "elevation"]
    value_deg: float
    positions: list[float]


class PreviewBandResponse(BaseModel):
    surface: BandShellResponse
    edges: list[BandEdgeLineResponse]


class PreviewCamerasResponse(BaseModel):
    locations: list[list[float]]
    look_at: list[float]
    rays: list[list[list[float]]]


class PreviewGroundPlaneResponse(BaseModel):
    positions: list[float]
    indices: list[int]


class PreviewPosesResponse(BaseModel):
    """Band, camera points, and ground plane for the SPA 3D view."""

    band: PreviewBandResponse
    cameras: PreviewCamerasResponse
    ground_plane: PreviewGroundPlaneResponse
    display_radius: float


class SaveConfigRequest(BaseModel):
    """Validated render config and destination filename."""

    config: RembrandtConfig
    filename: str


class SaveConfigResponse(BaseModel):
    """Path to the written YAML file."""

    path: str


@router.post("/preview/mesh", response_model=PreviewMeshResponse)
def preview_mesh(body: PreviewMeshRequest) -> PreviewMeshResponse:
    """Load and orient an ``.obj`` file for the SPA preview."""
    try:
        mesh = load_preview_mesh(body.path, up_axis=body.up_axis)
    except ModelFileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _mesh_to_response(mesh)


@router.post("/preview/poses", response_model=PreviewPosesResponse)
def preview_poses(body: PreviewPosesRequest) -> PreviewPosesResponse:
    """Build band, camera, and ground-plane geometry for the SPA preview."""
    try:
        geometry = build_preview_pose_geometry(
            bbox=body.bbox,
            n=body.n,
            azimuth_range=body.azimuth_range,
            elevation_range=body.elevation_range,
            distance_range=body.distance_range,
            strategy=body.strategy,
            seed=body.seed,
            look_at=body.look_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _poses_to_response(geometry)


@router.post("/config/save", response_model=SaveConfigResponse)
def save_config(body: SaveConfigRequest) -> SaveConfigResponse:
    """Validate and write a render config YAML under ``./configs/``."""
    try:
        filename = _validate_config_filename(body.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    configs_dir = Path.cwd() / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    destination = configs_dir / filename
    dump_config(body.config, destination)
    return SaveConfigResponse(path=str(destination))


def _validate_config_filename(filename: str) -> str:
    if not filename:
        msg = "filename must not be empty"
        raise ValueError(msg)
    if filename != filename.strip():
        msg = "filename must not have leading or trailing whitespace"
        raise ValueError(msg)
    if Path(filename).is_absolute():
        msg = "filename must be relative"
        raise ValueError(msg)
    if "/" in filename or "\\" in filename:
        msg = "filename must not contain path separators"
        raise ValueError(msg)
    if filename != Path(filename).name:
        msg = "filename must not contain path separators"
        raise ValueError(msg)
    if filename in {".", ".."}:
        msg = "filename is not allowed"
        raise ValueError(msg)
    return filename


def _mesh_to_response(mesh: PreviewMesh) -> PreviewMeshResponse:
    return PreviewMeshResponse(
        positions=mesh.positions,
        indices=mesh.indices,
        bbox=mesh.bbox,
    )


def _poses_to_response(geometry: PreviewPoseGeometry) -> PreviewPosesResponse:
    return PreviewPosesResponse(
        band=PreviewBandResponse(
            surface=BandShellResponse(
                distance=geometry.band.surface.distance,
                positions=geometry.band.surface.positions,
                azimuth_count=geometry.band.surface.azimuth_count,
                elevation_count=geometry.band.surface.elevation_count,
            ),
            edges=[
                BandEdgeLineResponse(
                    kind=edge.kind,
                    value_deg=edge.value_deg,
                    positions=edge.positions,
                )
                for edge in geometry.band.edges
            ],
        ),
        cameras=PreviewCamerasResponse(
            locations=geometry.cameras.locations,
            look_at=geometry.cameras.look_at,
            rays=geometry.cameras.rays,
        ),
        ground_plane=PreviewGroundPlaneResponse(
            positions=geometry.ground_plane.positions,
            indices=geometry.ground_plane.indices,
        ),
        display_radius=geometry.display_radius,
    )
