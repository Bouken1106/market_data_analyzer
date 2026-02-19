"""Persistent UI state for watchlist symbols and commentary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import LOGGER, MAX_BASIC_SYMBOLS, SYMBOL_PATTERN


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
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            symbol = str(item or "").upper().strip()
            if not symbol or symbol in seen:
                continue
            if not SYMBOL_PATTERN.match(symbol):
                continue
            seen.add(symbol)
            out.append(symbol)
            if len(out) >= MAX_BASIC_SYMBOLS:
                break
        return out

    def set_symbols(self, symbols: list[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in symbols:
            symbol = str(item or "").upper().strip()
            if not symbol or symbol in seen:
                continue
            if not SYMBOL_PATTERN.match(symbol):
                continue
            seen.add(symbol)
            cleaned.append(symbol)
            if len(cleaned) >= MAX_BASIC_SYMBOLS:
                break
        self._state["symbols"] = cleaned
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
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
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
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_to_disk()

    def _write_to_disk(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._state, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            LOGGER.warning("Failed to write UI state cache: %s", exc)
