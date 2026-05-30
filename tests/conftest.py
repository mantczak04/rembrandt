"""Pytest hooks and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-bpy",
        action="store_true",
        default=False,
        help="Fail tests that skip because bpy is missing (use in the bpy CI lane).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bpy: requires the Blender Python runtime (bpy)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """In the bpy CI lane, fail fast when bpy is missing instead of skipping."""
    if not item.config.getoption("--require-bpy"):
        return
    if "bpy" not in item.keywords:
        return
    try:
        import bpy  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"bpy is required in this lane but is not importable: {exc}")
