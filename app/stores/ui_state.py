"""Persistent UI state for watchlist symbols and commentary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LOGGER, MAX_BASIC_SYMBOLS
from ..utils import normalize_symbols, read_json_file, utc_now_iso, write_json_file


class UiStateStore:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._state: dict[str, Any] = {
            "symbols": [],
            "watchlist_commentary": None,
            "updated_at": None,
        }
        self._load_from_disk()

    def get_symbols(self) -> list[str]:
        raw = self._state.get("symbols")
        if not isinstance(raw, list):
            return []
        return normalize_symbols(raw, max_items=MAX_BASIC_SYMBOLS)

    def set_symbols(self, symbols: list[str]) -> None:
        self._state["symbols"] = normalize_symbols(symbols, max_items=MAX_BASIC_SYMBOLS)
        self._touch_and_write()

    def get_watchlist_commentary(self) -> dict[str, Any] | None:
        item = self._state.get("watchlist_commentary")
        return dict(item) if isinstance(item, dict) else None

    def set_watchlist_commentary(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._state["watchlist_commentary"] = dict(payload)
        self._touch_and_write()

    def _load_from_disk(self) -> None:
        payload = read_json_file(self.cache_path)
        if not isinstance(payload, dict):
            return
        symbols = payload.get("symbols")
        commentary = payload.get("watchlist_commentary")
        if isinstance(symbols, list):
            self._state["symbols"] = symbols
        if isinstance(commentary, dict):
            self._state["watchlist_commentary"] = commentary
        updated_at = payload.get("updated_at")
        if isinstance(updated_at, str):
            self._state["updated_at"] = updated_at

    def _touch_and_write(self) -> None:
        self._state["updated_at"] = utc_now_iso()
        self._write_to_disk()

    def _write_to_disk(self) -> None:
        try:
            write_json_file(self.cache_path, self._state)
        except Exception as exc:
            LOGGER.warning("Failed to write UI state cache: %s", exc)
