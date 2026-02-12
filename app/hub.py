"""MarketDataHub – central orchestrator for real-time and historical prices."""

from __future__ import annotations

import asyncio
from typing import Any

from .config import (
    API_LIMIT_PER_DAY,
    DATA_PROVIDER,
    SYMBOL_CATALOG_COUNTRY,
    SYMBOL_COUNTRY_MAP_RAW,
)
from .market_session import (
    DEFAULT_MARKET_SESSIONS,
    _normalize_country_key,
    parse_symbol_country_map,
)
from .services import (
    MarketDataQueriesMixin,
    MarketDataRealtimeMixin,
    MarketDataStateMixin,
)
from .stores import FmpReferenceStore, FullDailyHistoryStore, LastPriceStore


class MarketDataHub(MarketDataRealtimeMixin, MarketDataQueriesMixin, MarketDataStateMixin):
    def __init__(
        self,
        provider: str,
        twelvedata_api_key: str,
        fmp_api_key: str,
        symbols: list[str],
        last_price_store: LastPriceStore,
        full_daily_history_store: FullDailyHistoryStore,
        fmp_reference_store: FmpReferenceStore,
    ) -> None:
        self.provider = str(provider or DATA_PROVIDER).strip().lower()
        self.twelvedata_api_key = str(twelvedata_api_key or "").strip()
        self.fmp_api_key = str(fmp_api_key or "").strip()
        self.symbols: list[str] = symbols
        self.default_country_key = _normalize_country_key(SYMBOL_CATALOG_COUNTRY)
        self.symbol_country_map = parse_symbol_country_map(SYMBOL_COUNTRY_MAP_RAW)
        self.market_sessions = DEFAULT_MARKET_SESSIONS
        self.prices: dict[str, dict[str, Any]] = {}
        self.last_price_store = last_price_store
        self.full_daily_history_store = full_daily_history_store
        self.fmp_reference_store = fmp_reference_store
        self.ws_connected = False
        self.last_ws_message_at = 0.0
        self.mode = "starting"
        self.daily_credits_left: int | None = None
        self.daily_credits_used: int | None = None
        self.daily_credits_limit: int | None = API_LIMIT_PER_DAY
        self.daily_credits_updated_at: str | None = None
        self.daily_credits_source: str | None = None
        self.daily_credits_is_estimated = False
        self.minute_credits_left: int | None = None
        self.minute_credits_used: int | None = None

        self._listeners: set[asyncio.Queue[dict[str, Any]]] = set()
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()
        self._restart_ws_event = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._credits_lock = asyncio.Lock()
        self._historical_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._historical_lock = asyncio.Lock()
        self._sparkline_cache: dict[str, dict[str, Any]] = {}
        self._sparkline_lock = asyncio.Lock()
        self._overview_cache: dict[tuple[str, bool, bool, bool], dict[str, Any]] = {}
        self._overview_lock = asyncio.Lock()
        self._fmp_reference_cache: dict[str, dict[str, Any]] = {}
        self._fmp_reference_lock = asyncio.Lock()

    def _uses_twelvedata(self) -> bool:
        return self.provider in {"twelvedata", "both"}

    def _uses_fmp(self) -> bool:
        return self.provider in {"fmp", "both"}
