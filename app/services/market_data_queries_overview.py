"""Overview, quote, and sparkline helpers for MarketData query mixins."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import settings
from ..utils import normalize_symbols
from .market_data_overview_ops import MarketDataOverviewOps
from .market_data_overview_service import (
    MarketDataOverviewQueryService,
    OverviewQueryContext,
    OverviewQueryDependencies,
)
from .market_data_queries_overview_support import (
    OverviewInputs,
    OverviewRequest,
    build_market_section,
    build_overview_payload,
    build_overview_request,
    build_overview_source_detail,
    build_price_context,
)


class MarketDataOverviewMixin:
    def _overview_query_service(self) -> MarketDataOverviewQueryService:
        service = getattr(self, "overview_query_service", None)
        if service is None:
            service = getattr(self, "_overview_query_service_instance", None)
        if service is None:
            service = MarketDataOverviewQueryService(
                context=self._overview_query_context(),
                dependencies=self._overview_query_dependencies(),
            )
            setattr(self, "_overview_query_service_instance", service)
        return service

    def _overview_query_context(self) -> OverviewQueryContext:
        return OverviewQueryContext(
            provider=self.provider,
            overview_cache=self._overview_cache,
            overview_lock=self._overview_lock,
            sparkline_cache=self._sparkline_cache,
            sparkline_lock=self._sparkline_lock,
            historical_cache=self._historical_cache,
            historical_lock=self._historical_lock,
            full_daily_history_store=self.full_daily_history_store,
            symbol_pattern=settings.provider.symbol_pattern,
        )

    def _overview_query_dependencies(self) -> OverviewQueryDependencies:
        return OverviewQueryDependencies(
            build_request=self._build_overview_request,
            fetch_inputs=self._fetch_overview_inputs,
            build_payload=self._compose_overview_payload,
            fetch_sparkline_item=self._fetch_sparkline_item,
            normalize_symbols=normalize_symbols,
        )

    def _overview_ops(self) -> MarketDataOverviewOps:
        ops = getattr(self, "_overview_ops_service", None)
        if ops is None:
            ops = MarketDataOverviewOps(self)
            setattr(self, "_overview_ops_service", ops)
        return ops

    async def security_overview_payload(
        self,
        symbol: str,
        refresh: bool = False,
        include_intraday: bool = True,
        include_market: bool = True,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        return await self._overview_query_service().security_overview_payload(
            symbol=symbol,
            refresh=refresh,
            include_intraday=include_intraday,
            include_market=include_market,
            include_qqq=include_qqq,
        )

    async def _fetch_overview_inputs(
        self,
        *,
        client: httpx.AsyncClient,
        request: OverviewRequest,
        refresh: bool,
    ) -> OverviewInputs:
        return await self._overview_ops().fetch_overview_inputs(
            client=client,
            request=request,
            refresh=refresh,
        )

    def _build_overview_technicals(
        self,
        *,
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
        m5_points: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        return {
            "vwap_1m": self._intraday_vwap(m1_points),
            "vwap_5m": self._intraday_vwap(m5_points),
            "ma_short_20": self._moving_average(day_points, window=20),
            "ma_mid_50": self._moving_average(day_points, window=50),
            "atr_14": self._atr(day_points, window=14),
        }

    def _compose_overview_payload(
        self,
        *,
        request: OverviewRequest,
        inputs: OverviewInputs,
        provider: str,
    ) -> dict[str, Any]:
        price_context = self._build_price_context(
            quote=inputs.quote,
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
        )
        technical = self._build_overview_technicals(
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
            m5_points=inputs.m5_points,
        )
        market = self._build_overview_market_section(
            day_points=inputs.day_points,
            market_context=inputs.market_context,
            include_market=request.include_market,
            include_qqq=request.include_qqq,
        )
        source_detail = build_overview_source_detail(
            quote=inputs.quote,
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
            m5_points=inputs.m5_points,
            spy_points=market["spy_points"],
            qqq_points=market["qqq_points"],
            price_context=price_context,
            series_source_descriptor=self._series_source_descriptor,
        )
        return build_overview_payload(
            request=request,
            inputs=inputs,
            provider=provider,
            price_context=price_context,
            technical=technical,
            market_payload=market["payload"],
            source_detail=source_detail,
            pick_string=self._pick_string,
            best_updated_at=self._best_updated_at,
            delay_note=self._delay_note,
        )

    def _build_overview_market_section(
        self,
        *,
        day_points: list[dict[str, Any]],
        market_context: dict[str, Any] | None,
        include_market: bool,
        include_qqq: bool,
    ) -> dict[str, Any]:
        market_section = build_market_section(
            day_points=day_points,
            market_context=market_context,
            include_market=include_market,
            include_qqq=include_qqq,
            beta_and_corr_60d=self._beta_and_corr_60d,
            build_market_item=self._build_market_item,
        )
        return {
            "spy_points": market_section.spy_points,
            "qqq_points": market_section.qqq_points,
            "payload": market_section.payload,
        }

    @staticmethod
    def _quote_source_detail(provider: str) -> dict[str, str]:
        return {
            "symbol": provider,
            "name": provider,
            "instrument_name": provider,
            "exchange": provider,
            "price": provider,
            "close": provider,
            "previous_close": provider,
            "prev_close": provider,
            "open": provider,
            "high": provider,
            "low": provider,
            "volume": provider,
            "bid": provider,
            "ask": provider,
            "timestamp": provider,
            "datetime": provider,
        }

    def _build_overview_request(
        self,
        *,
        symbol: str,
        include_intraday: bool,
        include_market: bool,
        include_qqq: bool,
    ) -> OverviewRequest:
        return build_overview_request(
            symbol=symbol,
            include_intraday=include_intraday,
            include_market=include_market,
            include_qqq=include_qqq,
        )

    def _build_price_context(
        self,
        *,
        quote: dict[str, Any],
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_price_context(
            quote=quote,
            day_points=day_points,
            m1_points=m1_points,
            pick_float=self._pick_float,
            series_source_descriptor=self._series_source_descriptor,
            extract_latest_session_points=self._extract_latest_session_points,
        )

    async def clear_symbol_overview_cache(self, symbol: str) -> dict[str, Any]:
        return await self._overview_query_service().clear_symbol_overview_cache(symbol)

    async def _fetch_quote(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        return await self._overview_ops().fetch_quote(client, symbol)

    async def _fetch_quote_twelvedata(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        return await self._overview_ops().fetch_quote_twelvedata(client, symbol)

    async def _fetch_quote_both(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        return await self._overview_ops().fetch_quote_both(client, symbol)

    async def _fetch_quote_fmp(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        return await self._overview_ops().fetch_quote_fmp(client, symbol)

    async def sparkline_payload(self, symbols: list[str], refresh: bool = False) -> list[dict[str, Any]]:
        return await self._overview_query_service().sparkline_payload(symbols, refresh=refresh)

    async def _fetch_sparkline_item(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None:
        return await self._overview_ops().fetch_sparkline_item(client, symbol)
