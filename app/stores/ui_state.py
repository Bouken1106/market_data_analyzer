"""Persistent UI state for watchlist symbols and commentary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import MAX_BASIC_SYMBOLS
from ..utils import clone_json_like, normalize_symbol, normalize_symbols
from .json_state import JsonStateStore

_MARKET_DATA_LAB_CHART_INTERVALS = {"1min", "5min", "1day"}


class UiStateStore(JsonStateStore):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path, log_label="UI state cache")
        self._state: dict[str, Any] = {
            "symbols": [],
            "watchlist_commentary": None,
            "market_data_lab": {
                "onboarding_dismissed": False,
                "watchlist_symbols": [],
                "last_viewed_symbol": "",
                "chart_interval": "1day",
            },
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

    def get_market_data_lab_state(self) -> dict[str, Any]:
        return clone_json_like(self._sanitize_market_data_lab_state(self._state.get("market_data_lab")))

    def set_market_data_lab_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self._sanitize_market_data_lab_state(self._state.get("market_data_lab"))
        if isinstance(payload, dict):
            if "onboarding_dismissed" in payload:
                current["onboarding_dismissed"] = bool(payload.get("onboarding_dismissed"))
            if "watchlist_symbols" in payload:
                current["watchlist_symbols"] = normalize_symbols(
                    payload.get("watchlist_symbols", []),
                    max_items=MAX_BASIC_SYMBOLS,
                )
            if "last_viewed_symbol" in payload:
                current["last_viewed_symbol"] = normalize_symbol(payload.get("last_viewed_symbol"))
            if "chart_interval" in payload:
                interval = str(payload.get("chart_interval") or "").strip().lower()
                current["chart_interval"] = interval if interval in _MARKET_DATA_LAB_CHART_INTERVALS else "1day"
        current = self._sanitize_market_data_lab_state(current)
        self._state["market_data_lab"] = current
        self._touch_and_write()
        return clone_json_like(current)

    def get_market_data_lab_onboarding(self) -> dict[str, bool]:
        state = self.get_market_data_lab_state()
        dismissed = bool(state.get("onboarding_dismissed"))
        return {
            "dismissed": dismissed,
            "enabled": not dismissed,
        }

    def set_market_data_lab_onboarding(self, dismissed: bool) -> dict[str, bool]:
        self.set_market_data_lab_state({"onboarding_dismissed": bool(dismissed)})
        return self.get_market_data_lab_onboarding()

    def _sanitize_market_data_lab_state(self, payload: Any) -> dict[str, Any]:
        raw = payload if isinstance(payload, dict) else {}
        watchlist_symbols = normalize_symbols(raw.get("watchlist_symbols", []), max_items=MAX_BASIC_SYMBOLS)
        last_viewed_symbol = normalize_symbol(raw.get("last_viewed_symbol"))
        if last_viewed_symbol and last_viewed_symbol not in watchlist_symbols:
            last_viewed_symbol = watchlist_symbols[0] if watchlist_symbols else ""
        if not last_viewed_symbol and watchlist_symbols:
            last_viewed_symbol = watchlist_symbols[0]
        chart_interval = str(raw.get("chart_interval") or "").strip().lower()
        if chart_interval not in _MARKET_DATA_LAB_CHART_INTERVALS:
            chart_interval = "1day"
        return {
            "onboarding_dismissed": bool(raw.get("onboarding_dismissed")),
            "watchlist_symbols": watchlist_symbols,
            "last_viewed_symbol": last_viewed_symbol,
            "chart_interval": chart_interval,
        }

    def _load_from_disk(self) -> None:
        payload = self._read_state_dict()
        if payload is None:
            return
        symbols = payload.get("symbols")
        commentary = payload.get("watchlist_commentary")
        if isinstance(symbols, list):
            self._state["symbols"] = symbols
        if isinstance(commentary, dict):
            self._state["watchlist_commentary"] = commentary
        self._state["market_data_lab"] = self._sanitize_market_data_lab_state(payload.get("market_data_lab"))
        updated_at = payload.get("updated_at")
        if isinstance(updated_at, str):
            self._state["updated_at"] = updated_at

    def _touch_and_write(self) -> None:
        self._touch_and_write_state(self._state)
