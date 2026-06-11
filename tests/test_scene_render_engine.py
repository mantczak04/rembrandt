"""Tests for render-engine availability guardrails."""

from __future__ import annotations

import pytest

from rembrandt.errors import RenderEngineUnavailableError
from rembrandt.scene import eevee_failure_is_gpu_context


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Failed to initialize OpenGL context", True),
        ("EGL: unable to create context", True),
        ("No GPU device found", True),
        ("Out of memory", False),
    ],
)
def test_eevee_failure_is_gpu_context(message: str, expected: bool) -> None:
    assert eevee_failure_is_gpu_context(RuntimeError(message)) is expected


def test_render_engine_unavailable_error_default_message() -> None:
    assert "EEVEE requires a GPU" in str(RenderEngineUnavailableError())
