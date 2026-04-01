"""High-level overview and sparkline orchestration."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from fastapi import HTTPException

from ..config import OVERVIEW_CACHE_TTL_SEC, SPARKLINE_CACHE_TTL_SEC, SYMBOL_PATTERN
from ..utils import normalize_symbols
from .market_data_queries_overview_support import OverviewInputs, OverviewRequest, support_status_payload
from .ttl_cache import ttl_cache_lookup_response, ttl_cache_pop_matching, ttl_cache_store


class OverviewQueryOwner(Protocol):
    provider: str
    _overview_cache: dict[Any, Any]
    _overview_lock: Any
    _sparkline_cache: dict[Any, Any]
    _sparkline_lock: Any
    _historical_cache: dict[Any, Any]
    _historical_lock: Any
    full_daily_history_store: Any

    def _build_overview_request(
        self,
        *,
        symbol: str,
        include_intraday: bool,
        include_market: bool,
        include_qqq: bool,
    ) -> OverviewRequest: ...

    async def _fetch_overview_inputs(
        self,
        *,
        client: httpx.AsyncClient,
        request: OverviewRequest,
        refresh: bool,
    ) -> OverviewInputs: ...

    def _pick_string(self, payload: dict[str, Any], *keys: str) -> str | None: ...
    def _build_price_context(
        self,
        *,
        quote: dict[str, Any],
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
    def _build_overview_technicals(
        self,
        *,
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
        m5_points: list[dict[str, Any]],
    ) -> dict[str, float | None]: ...
    def _build_overview_market_section(
        self,
        *,
        day_points: list[dict[str, Any]],
        market_context: dict[str, Any] | None,
        include_market: bool,
        include_qqq: bool,
    ) -> dict[str, Any]: ...
    def _best_updated_at(
        self,
        quote_payload: dict[str, Any],
        intraday_points: list[dict[str, Any]],
        day_points: list[dict[str, Any]],
    ) -> str | None: ...
    def _delay_note(self) -> str: ...
    def _series_source_descriptor(self, points: list[dict[str, Any]]) -> str: ...
    async def _fetch_sparkline_item(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None: ...


class MarketDataOverviewQueryService:
    def __init__(self, owner: OverviewQueryOwner) -> None:
        self.owner = owner

    async def security_overview_payload(
        self,
        *,
        symbol: str,
        refresh: bool,
        include_intraday: bool,
        include_market: bool,
        include_qqq: bool,
    ) -> dict[str, Any]:
        request = self.owner._build_overview_request(
            symbol=symbol,
            include_intraday=include_intraday,
            include_market=include_market,
            include_qqq=include_qqq,
        )

        if not refresh:
            cached_payload = await ttl_cache_lookup_response(
                self.owner._overview_cache,
                self.owner._overview_lock,
                request.cache_key,
                ttl_sec=OVERVIEW_CACHE_TTL_SEC,
                copy_fn=dict,
            )
            if cached_payload is not None:
                return cached_payload

        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            inputs = await self.owner._fetch_overview_inputs(client=client, request=request, refresh=refresh)

        if not inputs.day_points:
            raise HTTPException(status_code=404, detail="No overview data found for this symbol.")

        payload = self._build_overview_payload(request=request, inputs=inputs)
        await ttl_cache_store(
            self.owner._overview_cache,
            self.owner._overview_lock,
            request.cache_key,
            payload,
        )
        return payload

    async def sparkline_payload(self, symbols: list[str], *, refresh: bool) -> list[dict[str, Any]]:
        target_symbols = normalize_symbols(symbols)
        if not target_symbols:
            return []

        items_by_symbol: dict[str, dict[str, Any]] = {}
        missing_symbols: list[str] = []
        if not refresh:
            for symbol in target_symbols:
                cached_payload = await ttl_cache_lookup_response(
                    self.owner._sparkline_cache,
                    self.owner._sparkline_lock,
                    symbol,
                    ttl_sec=SPARKLINE_CACHE_TTL_SEC,
                    copy_fn=dict,
                )
                if cached_payload is None:
                    missing_symbols.append(symbol)
                    continue
                items_by_symbol[symbol] = cached_payload
        else:
            missing_symbols = list(target_symbols)

        if missing_symbols:
            timeout = httpx.Timeout(20.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for symbol in missing_symbols:
                    item = await self.owner._fetch_sparkline_item(client, symbol)
                    if not item:
                        continue
                    items_by_symbol[symbol] = item
                    await ttl_cache_store(
                        self.owner._sparkline_cache,
                        self.owner._sparkline_lock,
                        symbol,
                        item,
                    )

        return [items_by_symbol[symbol] for symbol in target_symbols if symbol in items_by_symbol]

    async def clear_symbol_overview_cache(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")

        removed_overview = await ttl_cache_pop_matching(
            self.owner._overview_cache,
            self.owner._overview_lock,
            lambda key: key[0] == normalized,
        )
        removed_historical = await ttl_cache_pop_matching(
            self.owner._historical_cache,
            self.owner._historical_lock,
            lambda key: key[0] == normalized,
        )
        removed_daily_files = await self.owner.full_daily_history_store.clear(normalized)
        return {
            "symbol": normalized,
            "removed_overview_entries": removed_overview,
            "removed_historical_entries": removed_historical,
            "removed_daily_history_files": removed_daily_files,
        }

    def _build_overview_payload(
        self,
        *,
        request: OverviewRequest,
        inputs: OverviewInputs,
    ) -> dict[str, Any]:
        price_context = self.owner._build_price_context(
            quote=inputs.quote,
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
        )
        technical = self.owner._build_overview_technicals(
            day_points=inputs.day_points,
            m1_points=inputs.m1_points,
            m5_points=inputs.m5_points,
        )
        market = self.owner._build_overview_market_section(
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
            "name": self.owner._pick_string(inputs.quote, "name", "instrument_name"),
            "exchange": self.owner._pick_string(inputs.quote, "exchange"),
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
                "updated_at": self.owner._best_updated_at(inputs.quote, inputs.m1_points, inputs.day_points),
                "delay_note": self.owner._delay_note(),
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
            "source": f"{self.owner.provider}-live",
            "source_detail": source_detail,
        }

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
                "1min": self.owner._series_source_descriptor(m1_points),
                "5min": self.owner._series_source_descriptor(m5_points),
                "1day": self.owner._series_source_descriptor(day_points),
                "SPY": self.owner._series_source_descriptor(spy_points),
                "QQQ": self.owner._series_source_descriptor(qqq_points),
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
