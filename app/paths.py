"""Shared filesystem paths for the application package."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def static_file_path(filename: str) -> Path:
    return STATIC_DIR / filename
