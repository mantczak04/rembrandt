"""Tests for the bpy-free FastAPI app skeleton."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rembrandt.web.app import create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_spa_static_files_fall_back_to_index(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    index.write_text("<!doctype html><title>Rembrandt</title>", encoding="utf-8")
    client = TestClient(create_app(frontend_dist=tmp_path))

    response = client.get("/camera/preview")

    assert response.status_code == 200
    assert "Rembrandt" in response.text


def test_web_app_module_is_bpy_free() -> None:
    import rembrandt.web.app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source
