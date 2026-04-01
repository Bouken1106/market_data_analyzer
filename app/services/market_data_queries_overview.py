"""Overview, quote, and sparkline helpers for MarketData query mixins."""

from __future__ import annotations

from typing import Any

import httpx

from .market_data_overview_ops import MarketDataOverviewOps
from .market_data_overview_service import MarketDataOverviewQueryService
from .market_data_queries_overview_support import (
    OverviewInputs,
    OverviewRequest,
    build_market_section,
    build_overview_request,
    build_price_context,
    compute_change_metrics,
    compute_spread_metrics,
    compute_volume_metrics,
    fill_day_fields_from_daily_series,
    fill_day_fields_from_intraday,
    support_status_payload,
)


class MarketDataOverviewMixin:
    def _overview_query_service(self) -> MarketDataOverviewQueryService:
        service = getattr(self, "overview_query_service", None)
        if service is None:
            service = getattr(self, "_overview_query_service_instance", None)
        if service is None:
            service = MarketDataOverviewQueryService(self)
            setattr(self, "_overview_query_service_instance", service)
        return service

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

    def _build_overview_payload(
        self,
        *,
        request: OverviewRequest,
        inputs: OverviewInputs,
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
        source_detail = self._build_overview_source_detail(
            quote=inputs.quote,
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
            m5_points=inputs.m5_points,
            spy_points=market["spy_points"],
            qqq_points=market["qqq_points"],
            price_context=price_context,
        )

        return {
            "symbol": request.symbol,
            "name": self._pick_string(inputs.quote, "name", "instrument_name"),
            "exchange": self._pick_string(inputs.quote, "exchange"),
            "price": {
                "current": price_context["current_price"],
                "previous_close": price_context["previous_close"],
                "change_abs": price_context["change_abs"],
                "change_pct": price_context["change_pct"],
                "day_open": price_context["day_open"],
                "day_high": price_context["day_high"],
                "day_low": price_context["day_low"],
                "gap_abs": price_context["gap_abs"],
                "gap_pct": price_context["gap_pct"],
                "updated_at": self._best_updated_at(inputs.quote, inputs.m1_points, inputs.day_points),
                "delay_note": self._delay_note(),
            },
            "volume": {
                "today": price_context["day_volume"],
                "avg20": price_context["avg_volume_20"],
                "avg_ratio": price_context["avg_volume_ratio"],
                "turnover": price_context["turnover"],
            },
            "spread": {
                "bid": price_context["bid"],
                "ask": price_context["ask"],
                "spread_abs": price_context["spread_abs"],
                "spread_pct": price_context["spread_pct"],
            },
            "technical": technical,
            "market": market["payload"],
            "charts": {
                "1min": inputs.m1_points,
                "5min": inputs.m5_points,
                "1day": inputs.day_points,
            },
            "support_status": {
                **support_status_payload(),
            },
            "source": f"{self.provider}-live",
            "source_detail": source_detail,
        }

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

    def _fill_day_fields_from_intraday(
        self,
        field_values: dict[str, float | None],
        field_sources: dict[str, str | None],
        m1_points: list[dict[str, Any]],
    ) -> None:
        fill_day_fields_from_intraday(
            field_values=field_values,
            field_sources=field_sources,
            m1_points=m1_points,
            extract_latest_session_points=self._extract_latest_session_points,
        )

    @staticmethod
    def _fill_day_fields_from_daily_series(
        field_values: dict[str, float | None],
        field_sources: dict[str, str | None],
        latest_day: dict[str, Any],
        day_series_source: str,
    ) -> None:
        fill_day_fields_from_daily_series(
            field_values=field_values,
            field_sources=field_sources,
            latest_day=latest_day,
            day_series_source=day_series_source,
        )

    @staticmethod
    def _compute_change_metrics(current: float | None, previous: float | None) -> tuple[float | None, float | None]:
        return compute_change_metrics(current=current, previous=previous)

    @staticmethod
    def _compute_volume_metrics(
        *,
        day_points: list[dict[str, Any]],
        day_volume: float | None,
        current_price: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        return compute_volume_metrics(
            day_points=day_points,
            day_volume=day_volume,
            current_price=current_price,
        )

    @staticmethod
    def _compute_spread_metrics(
        *,
        bid: float | None,
        ask: float | None,
        current_price: float | None,
    ) -> tuple[float | None, float | None]:
        return compute_spread_metrics(
            bid=bid,
            ask=ask,
            current_price=current_price,
        )

    def _build_overview_source_detail(
        self,
        *,
        quote: dict[str, Any],
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
        m5_points: list[dict[str, Any]],
        spy_points: list[dict[str, Any]],
        qqq_points: list[dict[str, Any]],
        price_context: dict[str, Any],
    ) -> dict[str, Any]:
        quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
        if not isinstance(quote_source_detail, dict):
            quote_source_detail = {}
        return {
            "quote_provider": quote.get("_source_provider") if isinstance(quote, dict) else None,
            "chart_sources": {
                "1min": self._series_source_descriptor(m1_points),
                "5min": self._series_source_descriptor(m5_points),
                "1day": self._series_source_descriptor(day_points),
                "SPY": self._series_source_descriptor(spy_points),
                "QQQ": self._series_source_descriptor(qqq_points),
            },
            "fields": {
                "price.current": price_context["current_price_source"] or "unknown",
                "price.previous_close": price_context["previous_close_source"] or "unknown",
                "price.day_open": price_context["day_open_source"] or "unknown",
                "price.day_high": price_context["day_high_source"] or "unknown",
                "price.day_low": price_context["day_low_source"] or "unknown",
                "volume.today": price_context["day_volume_source"] or "unknown",
                "spread.bid": quote_source_detail.get("bid") or "unknown",
                "spread.ask": quote_source_detail.get("ask") or "unknown",
            },
        }

    async def clear_symbol_overview_cache(self, symbol: str) -> dict[str, Any]:
        return await self._overview_query_service().clear_symbol_overview_cache(symbol)

    async def _fetch_market_context(
        self,
        client: httpx.AsyncClient,
        refresh: bool = False,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        return await self._overview_ops().fetch_market_context(
            client,
            refresh=refresh,
            include_qqq=include_qqq,
        )

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
