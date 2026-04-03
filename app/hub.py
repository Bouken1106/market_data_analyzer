"""MarketDataHub – central orchestrator for real-time and historical prices."""

from __future__ import annotations

from typing import Any

from .config import settings
from .hub_state import CacheState, ProviderState, RuntimeState, StoreState, assign_state_fields
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
            provider=str(provider or settings.data_provider).strip().lower(),
            twelvedata_api_key=str(twelvedata_api_key or "").strip(),
            fmp_api_key=str(fmp_api_key or "").strip(),
            symbols=symbols,
            default_country_key=_normalize_country_key(settings.symbol_catalog_country),
            symbol_country_map=parse_symbol_country_map(settings.symbol_country_map_raw),
            market_sessions=DEFAULT_MARKET_SESSIONS,
        )
        assign_state_fields(self, self._provider_state)

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
        assign_state_fields(self, self._store_state)

    def _init_runtime_state(self) -> None:
        self._runtime_state = RuntimeState(daily_credits_limit=settings.api_limit_per_day)
        assign_state_fields(self, self._runtime_state)

    def _init_cache_state(self) -> None:
        self._cache_state = CacheState()
        assign_state_fields(self, self._cache_state)

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
