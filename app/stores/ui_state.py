"""Persistent UI state for watchlist symbols and commentary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import settings
from ..services.watchlist_state import DEFAULT_WATCHLIST_NAMESPACE, normalize_watchlist_namespace
from ..utils import normalize_symbols
from .json_state import JsonStateStore


class UiStateStore(JsonStateStore):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path, log_label="UI state cache")
        self._state: dict[str, Any] = {
            "symbols": [],
            "watchlists": {},
            "watchlist_commentary": None,
            "updated_at": None,
        }
        self._load_from_disk()

    @staticmethod
    def _normalize_namespace(namespace: str | None) -> str:
        return normalize_watchlist_namespace(namespace)

    def get_symbols(self, namespace: str | None = None) -> list[str]:
        normalized_namespace = self._normalize_namespace(namespace)
        watchlists = self._state.get("watchlists")
        if isinstance(watchlists, dict):
            raw = watchlists.get(normalized_namespace)
            if isinstance(raw, list):
                return normalize_symbols(raw, max_items=settings.provider.max_basic_symbols)

        if normalized_namespace != DEFAULT_WATCHLIST_NAMESPACE:
            return []

        raw = self._state.get("symbols")
        if not isinstance(raw, list):
            return []
        return normalize_symbols(raw, max_items=settings.provider.max_basic_symbols)

    def set_symbols(self, symbols: list[str], namespace: str | None = None) -> None:
        normalized_symbols = normalize_symbols(symbols, max_items=settings.provider.max_basic_symbols)
        normalized_namespace = self._normalize_namespace(namespace)
        watchlists = self._state.get("watchlists")
        if not isinstance(watchlists, dict):
            watchlists = {}
            self._state["watchlists"] = watchlists
        watchlists[normalized_namespace] = list(normalized_symbols)
        if normalized_namespace == DEFAULT_WATCHLIST_NAMESPACE:
            self._state["symbols"] = list(normalized_symbols)
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
        payload = self._read_state_dict()
        if payload is None:
            return
        symbols = payload.get("symbols")
        watchlists = payload.get("watchlists")
        commentary = payload.get("watchlist_commentary")
        if isinstance(symbols, list):
            self._state["symbols"] = normalize_symbols(symbols, max_items=settings.provider.max_basic_symbols)
        if isinstance(watchlists, dict):
            normalized_watchlists: dict[str, list[str]] = {}
            for namespace, raw_symbols in watchlists.items():
                if not isinstance(raw_symbols, list):
                    continue
                normalized_watchlists[self._normalize_namespace(str(namespace))] = normalize_symbols(
                    raw_symbols,
                    max_items=settings.provider.max_basic_symbols,
                )
            self._state["watchlists"] = normalized_watchlists
        if (
            self._state["symbols"]
            and DEFAULT_WATCHLIST_NAMESPACE not in self._state["watchlists"]
        ):
            self._state["watchlists"][DEFAULT_WATCHLIST_NAMESPACE] = list(self._state["symbols"])
        if isinstance(commentary, dict):
            self._state["watchlist_commentary"] = commentary
        updated_at = payload.get("updated_at")
        if isinstance(updated_at, str):
            self._state["updated_at"] = updated_at

    def _touch_and_write(self) -> None:
        self._touch_and_write_state(self._state)
