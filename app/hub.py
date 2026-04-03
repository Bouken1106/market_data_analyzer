"""MarketDataHub – central orchestrator for real-time and historical prices."""

from __future__ import annotations

from typing import Any

from .config import settings
from .hub_state import CacheState, ProviderState, RuntimeState, StoreState
from .market_session import (
    DEFAULT_MARKET_SESSIONS,
    _normalize_country_key,
    parse_symbol_country_map,
)
from .services.market_data_queries import MarketDataQueriesMixin
from .services.market_data_historical_service import MarketDataHistoricalQueryService
from .services.market_data_overview_service import MarketDataOverviewQueryService
from .services.market_data_realtime import MarketDataRealtimeMixin
from .services.market_data_state import MarketDataStateMixin
from .stores import FmpReferenceStore, FullDailyHistoryStore, LastPriceStore


def _state_property(state_attr: str, field_name: str) -> property:
    def getter(self):
        return getattr(getattr(self, state_attr), field_name)

    def setter(self, value):
        setattr(getattr(self, state_attr), field_name, value)

    return property(getter, setter)


def _readonly_state_property(state_attr: str, field_name: str) -> property:
    def getter(self):
        return getattr(getattr(self, state_attr), field_name)

    return property(getter)


class MarketDataHub(MarketDataRealtimeMixin, MarketDataQueriesMixin, MarketDataStateMixin):
    provider = _state_property("_provider_state", "provider")
    twelvedata_api_key = _readonly_state_property("_provider_state", "twelvedata_api_key")
    fmp_api_key = _readonly_state_property("_provider_state", "fmp_api_key")
    symbols = _state_property("_provider_state", "symbols")
    default_country_key = _readonly_state_property("_provider_state", "default_country_key")
    symbol_country_map = _readonly_state_property("_provider_state", "symbol_country_map")
    market_sessions = _readonly_state_property("_provider_state", "market_sessions")

    last_price_store = _readonly_state_property("_store_state", "last_price_store")
    full_daily_history_store = _readonly_state_property("_store_state", "full_daily_history_store")
    fmp_reference_store = _readonly_state_property("_store_state", "fmp_reference_store")
    ui_state_store = _readonly_state_property("_store_state", "ui_state_store")

    prices = _state_property("_runtime_state", "prices")
    ws_connected = _state_property("_runtime_state", "ws_connected")
    last_ws_message_at = _state_property("_runtime_state", "last_ws_message_at")
    mode = _state_property("_runtime_state", "mode")
    daily_credits_left = _state_property("_runtime_state", "daily_credits_left")
    daily_credits_used = _state_property("_runtime_state", "daily_credits_used")
    daily_credits_limit = _state_property("_runtime_state", "daily_credits_limit")
    daily_credits_updated_at = _state_property("_runtime_state", "daily_credits_updated_at")
    daily_credits_source = _state_property("_runtime_state", "daily_credits_source")
    daily_credits_is_estimated = _state_property("_runtime_state", "daily_credits_is_estimated")
    minute_credits_left = _state_property("_runtime_state", "minute_credits_left")
    minute_credits_used = _state_property("_runtime_state", "minute_credits_used")
    _listeners = _readonly_state_property("_runtime_state", "_listeners")
    _worker_tasks = _state_property("_runtime_state", "_worker_tasks")
    _stop_event = _readonly_state_property("_runtime_state", "_stop_event")
    _restart_ws_event = _readonly_state_property("_runtime_state", "_restart_ws_event")
    _state_lock = _readonly_state_property("_runtime_state", "_state_lock")
    _credits_lock = _readonly_state_property("_runtime_state", "_credits_lock")

    _historical_cache = _readonly_state_property("_cache_state", "_historical_cache")
    _historical_lock = _readonly_state_property("_cache_state", "_historical_lock")
    _sparkline_cache = _readonly_state_property("_cache_state", "_sparkline_cache")
    _sparkline_lock = _readonly_state_property("_cache_state", "_sparkline_lock")
    _overview_cache = _readonly_state_property("_cache_state", "_overview_cache")
    _overview_lock = _readonly_state_property("_cache_state", "_overview_lock")
    _fmp_reference_cache = _readonly_state_property("_cache_state", "_fmp_reference_cache")
    _fmp_reference_lock = _readonly_state_property("_cache_state", "_fmp_reference_lock")

    def __init__(
        self,
        provider: str,
        twelvedata_api_key: str,
        fmp_api_key: str,
        symbols: list[str],
        last_price_store: LastPriceStore,
        full_daily_history_store: FullDailyHistoryStore,
        fmp_reference_store: FmpReferenceStore,
        ui_state_store: Any | None = None,
    ) -> None:
        self._init_provider_state(
            provider=provider,
            twelvedata_api_key=twelvedata_api_key,
            fmp_api_key=fmp_api_key,
            symbols=symbols,
        )
        self._init_store_state(
            last_price_store=last_price_store,
            full_daily_history_store=full_daily_history_store,
            fmp_reference_store=fmp_reference_store,
            ui_state_store=ui_state_store,
        )
        self._init_runtime_state()
        self._init_cache_state()
        self._init_query_services()

    def _init_provider_state(
        self,
        *,
        provider: str,
        twelvedata_api_key: str,
        fmp_api_key: str,
        symbols: list[str],
    ) -> None:
        self._provider_state = ProviderState(
            provider=str(provider or settings.provider.data_provider).strip().lower(),
            twelvedata_api_key=str(twelvedata_api_key or "").strip(),
            fmp_api_key=str(fmp_api_key or "").strip(),
            symbols=symbols,
            default_country_key=_normalize_country_key(settings.storage.symbol_catalog_country),
            symbol_country_map=parse_symbol_country_map(settings.behavior.symbol_country_map_raw),
            market_sessions=DEFAULT_MARKET_SESSIONS,
        )

    def _init_store_state(
        self,
        *,
        last_price_store: LastPriceStore,
        full_daily_history_store: FullDailyHistoryStore,
        fmp_reference_store: FmpReferenceStore,
        ui_state_store: Any | None,
    ) -> None:
        self._store_state = StoreState(
            last_price_store=last_price_store,
            full_daily_history_store=full_daily_history_store,
            fmp_reference_store=fmp_reference_store,
            ui_state_store=ui_state_store,
        )

    def _init_runtime_state(self) -> None:
        self._runtime_state = RuntimeState(daily_credits_limit=settings.budget.api_limit_per_day)

    def _init_cache_state(self) -> None:
        self._cache_state = CacheState()

    def _init_query_services(self) -> None:
        self.historical_query_service = MarketDataHistoricalQueryService(
            context=self._historical_query_context(),
            dependencies=self._historical_query_dependencies(),
        )
        self.overview_query_service = MarketDataOverviewQueryService(
            context=self._overview_query_context(),
            dependencies=self._overview_query_dependencies(),
        )

    def _uses_twelvedata(self) -> bool:
        return self.provider in {"twelvedata", "both"}

    def _uses_fmp(self) -> bool:
        return self.provider in {"fmp", "both"}
