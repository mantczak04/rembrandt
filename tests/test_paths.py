"""Shared filesystem paths for tests."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer the full test asset when present locally (gitignored); otherwise the
# committed asymmetric fixture used for orientation regression.
CHESS_BOARD_OBJ = PROJECT_ROOT / "test-obj" / "12951_Stone_Chess_Board_v1_L3.obj"
CHESS_OBJ = PROJECT_ROOT / "test-obj" / "chess.obj"
FIXTURE_OBJ = PROJECT_ROOT / "tests" / "fixtures" / "asymmetric_y_up.obj"
SAMPLE_OBJECT_PATH = "test-obj/12951_Stone_Chess_Board_v1_L3.obj"


def sample_object_path() -> Path:
    """Return the best available sample ``.obj`` for integration tests."""
    if CHESS_BOARD_OBJ.is_file():
        return CHESS_BOARD_OBJ
    if CHESS_OBJ.is_file():
        return CHESS_OBJ
    return FIXTURE_OBJ


def chess_board_object_path() -> Path:
    """Path to the OBJ that originally showed preview/render orientation drift."""
    return CHESS_BOARD_OBJ
