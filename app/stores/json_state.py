"""Shared helpers for JSON-backed state stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import read_json_file, utc_now_iso, write_json_file


class JsonStateStore:
    def __init__(self, cache_path: Path, *, log_label: str, compact: bool = False) -> None:
        self.cache_path = cache_path
        self._log_label = log_label
        self._compact = compact

    def _read_state_dict(self) -> dict[str, Any] | None:
        payload = read_json_file(self.cache_path)
        return payload if isinstance(payload, dict) else None

    def _touch_and_write_state(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = utc_now_iso()
        self._write_state(payload)

    def _write_state(self, payload: dict[str, Any]) -> None:
        try:
            write_json_file(self.cache_path, payload, compact=self._compact)
        except Exception as exc:
            LOGGER.warning("Failed to write %s: %s", self._log_label, exc)
