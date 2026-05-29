"""Tests for the Rembrandt web server entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rembrandt.web.serve import serve, server_url


def test_server_url_maps_bind_all_to_localhost() -> None:
    assert server_url(host="0.0.0.0", port=8000) == "http://127.0.0.1:8000/"


def test_server_url_uses_configured_host() -> None:
    assert server_url(host="127.0.0.1", port=9000) == "http://127.0.0.1:9000/"


def test_serve_runs_uvicorn_with_create_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    opened: list[str] = []

    def fake_run(app: object, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    import rembrandt.web.serve as serve_module

    monkeypatch.setattr(serve_module.uvicorn, "run", fake_run)
    monkeypatch.setattr(serve_module.webbrowser, "open", opened.append)

    serve(host="127.0.0.1", port=8765, open_browser=True)

    assert captured["kwargs"] == {"host": "127.0.0.1", "port": 8765}
    assert captured["app"].title == "Rembrandt"
    assert opened == ["http://127.0.0.1:8765/"]


def test_serve_skips_browser_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    import rembrandt.web.serve as serve_module

    monkeypatch.setattr(serve_module.uvicorn, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(serve_module.webbrowser, "open", opened.append)

    serve(open_browser=False)

    assert opened == []


def test_main_invokes_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_serve(**_kwargs: object) -> None:
        nonlocal called
        called = True

    import rembrandt.web.serve as serve_module

    monkeypatch.setattr(serve_module, "serve", fake_serve)
    serve_module.main()
    assert called


def test_web_serve_module_is_bpy_free() -> None:
    import rembrandt.web.serve as serve_module

    source = Path(serve_module.__file__).read_text(encoding="utf-8")
    assert "import bpy" not in source
    assert "from bpy" not in source
