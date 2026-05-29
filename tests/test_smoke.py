"""Smoke tests to confirm that the package and key dependencies are importable."""


def test_rembrandt_imports() -> None:
    import rembrandt

    assert rembrandt.__version__ == "0.1.0"


def test_bpy_imports() -> None:
    import bpy

    assert bpy.app.version_string  # truthy
