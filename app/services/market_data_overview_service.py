"""High-level overview and sparkline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Pattern

import httpx
from fastapi import HTTPException

from ..config import OVERVIEW_CACHE_TTL_SEC, SPARKLINE_CACHE_TTL_SEC
from .market_data_queries_overview_support import (
    OverviewInputs,
    OverviewRequest,
)
from .ttl_cache import ttl_cache_lookup_response, ttl_cache_pop_matching, ttl_cache_store


@dataclass(frozen=True)
class OverviewQueryContext:
    provider: str
    overview_cache: dict[Any, Any]
    overview_lock: Any
    sparkline_cache: dict[Any, Any]
    sparkline_lock: Any
    historical_cache: dict[Any, Any]
    historical_lock: Any
    full_daily_history_store: Any
    symbol_pattern: Pattern[str]


@dataclass(frozen=True)
class OverviewQueryDependencies:
    build_request: Callable[..., OverviewRequest]
    fetch_inputs: Callable[..., Awaitable[OverviewInputs]]
    build_payload: Callable[..., dict[str, Any]]
    fetch_sparkline_item: Callable[[httpx.AsyncClient, str], Awaitable[dict[str, Any] | None]]
    normalize_symbols: Callable[[list[str]], list[str]]


class MarketDataOverviewQueryService:
    def __init__(
        self,
        context: OverviewQueryContext,
        dependencies: OverviewQueryDependencies,
    ) -> None:
        self.context = context
        self.dependencies = dependencies

    async def security_overview_payload(
        self,
        *,
        symbol: str,
        refresh: bool,
        include_intraday: bool,
        include_market: bool,
        include_qqq: bool,
    ) -> dict[str, Any]:
        request = self.dependencies.build_request(
            symbol=symbol,
            include_intraday=include_intraday,
            include_market=include_market,
            include_qqq=include_qqq,
        )

        if not refresh:
            cached_payload = await ttl_cache_lookup_response(
                self.context.overview_cache,
                self.context.overview_lock,
                request.cache_key,
                ttl_sec=OVERVIEW_CACHE_TTL_SEC,
                copy_fn=dict,
            )
            if cached_payload is not None:
                return cached_payload

        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            inputs = await self.dependencies.fetch_inputs(client=client, request=request, refresh=refresh)

        if not inputs.day_points:
            raise HTTPException(status_code=404, detail="No overview data found for this symbol.")

        payload = self._build_overview_payload(request=request, inputs=inputs)
        await ttl_cache_store(
            self.context.overview_cache,
            self.context.overview_lock,
            request.cache_key,
            payload,
        )
        return payload

    async def sparkline_payload(self, symbols: list[str], *, refresh: bool) -> list[dict[str, Any]]:
        target_symbols = self.dependencies.normalize_symbols(symbols)
        if not target_symbols:
            return []

        items_by_symbol: dict[str, dict[str, Any]] = {}
        missing_symbols: list[str] = []
        if not refresh:
            for symbol in target_symbols:
                cached_payload = await ttl_cache_lookup_response(
                    self.context.sparkline_cache,
                    self.context.sparkline_lock,
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
                    item = await self.dependencies.fetch_sparkline_item(client, symbol)
                    if not item:
                        continue
                    items_by_symbol[symbol] = item
                    await ttl_cache_store(
                        self.context.sparkline_cache,
                        self.context.sparkline_lock,
                        symbol,
                        item,
                    )

        return [items_by_symbol[symbol] for symbol in target_symbols if symbol in items_by_symbol]

    async def clear_symbol_overview_cache(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not self.context.symbol_pattern.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")

        removed_overview = await ttl_cache_pop_matching(
            self.context.overview_cache,
            self.context.overview_lock,
            lambda key: key[0] == normalized,
        )
        removed_historical = await ttl_cache_pop_matching(
            self.context.historical_cache,
            self.context.historical_lock,
            lambda key: key[0] == normalized,
        )
        removed_daily_files = await self.context.full_daily_history_store.clear(normalized)
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
        return self.dependencies.build_payload(
            request=request,
            inputs=inputs,
            provider=self.context.provider,
        )
