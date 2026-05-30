"""Tests for preview and config API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rembrandt.config import RembrandtConfig, load_config
from rembrandt.preview.mesh import load_preview_mesh
from rembrandt.web.app import create_app
from tests.test_paths import SAMPLE_OBJECT_PATH, sample_object_path, sample_object_up_axis


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_preview_mesh_returns_geometry(client: TestClient) -> None:
    response = client.post(
        "/api/preview/mesh",
        json={"path": str(sample_object_path()), "up_axis": sample_object_up_axis()},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["positions"]) > 0
    assert len(payload["indices"]) > 0
    assert len(payload["bbox"]) == 2


def test_preview_mesh_missing_path_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/preview/mesh",
        json={"path": "/tmp/does-not-exist.obj"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_preview_poses_returns_n_camera_points(client: TestClient) -> None:
    mesh = load_preview_mesh(sample_object_path(), up_axis=sample_object_up_axis())
    response = client.post(
        "/api/preview/poses",
        json={
            "bbox": mesh.bbox,
            "n": 12,
            "azimuth_range": [0.0, 360.0],
            "elevation_range": [-10.0, 30.0],
            "distance_range": [3.0, 5.0],
            "strategy": "random",
            "seed": 11,
            "look_at": [0.0, 0.0, 0.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["cameras"]["locations"]) == 12
    assert len(payload["band"]["surface"]["positions"]) > 0
    assert len(payload["ground_plane"]["positions"]) == 12


def test_preview_poses_invalid_camera_params_returns_400(client: TestClient) -> None:
    mesh = load_preview_mesh(sample_object_path(), up_axis=sample_object_up_axis())
    response = client.post(
        "/api/preview/poses",
        json={
            "bbox": mesh.bbox,
            "n": 0,
            "strategy": "random",
        },
    )

    assert response.status_code == 400


def test_save_config_writes_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())
    config = RembrandtConfig(
        object={"path": SAMPLE_OBJECT_PATH},
        camera={"n": 8, "seed": 3},
    )

    response = client.post(
        "/api/config/save",
        json={
            "filename": "sample.yaml",
            "config": json.loads(config.model_dump_json()),
        },
    )

    assert response.status_code == 200
    written_path = Path(response.json()["path"])
    assert written_path == tmp_path / "configs" / "sample.yaml"
    assert written_path.is_file()
    loaded = load_config(written_path)
    assert loaded == config


def test_save_config_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())
    config = RembrandtConfig(
        object={"path": SAMPLE_OBJECT_PATH},
        camera={"n": 5},
    )

    response = client.post(
        "/api/config/save",
        json={
            "filename": "../escape.yaml",
            "config": json.loads(config.model_dump_json()),
        },
    )

    assert response.status_code == 400
    assert "separator" in response.json()["detail"].lower()


def test_web_api_module_is_bpy_free() -> None:
    import rembrandt.web.api as api_module

    source = Path(api_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source
