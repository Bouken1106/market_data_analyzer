"""High-level historical query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from fastapi import HTTPException

from .market_data_queries_historical_support import (
    HistoricalRequest,
    build_historical_payload,
    build_no_historical_data_detail,
    is_daily_interval,
)
from .ttl_cache import ttl_cache_lookup_response, ttl_cache_store


@dataclass(frozen=True)
class HistoricalQueryContext:
    provider: str
    historical_cache: dict[Any, Any]
    historical_lock: Any
    historical_cache_ttl_sec: int
    historical_interval: str
    historical_max_points: int


@dataclass(frozen=True)
class HistoricalQueryDependencies:
    build_request: Callable[..., HistoricalRequest]
    fetch_stooq_daily_points_with_detail: Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]
    fetch_historical_points_with_detail: Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]
    fetch_full_daily_series: Callable[..., Awaitable[list[dict[str, Any]]]]


class MarketDataHistoricalQueryService:
    def __init__(
        self,
        context: HistoricalQueryContext,
        dependencies: HistoricalQueryDependencies,
    ) -> None:
        self.context = context
        self.dependencies = dependencies

    async def historical_payload(
        self,
        *,
        symbol: str,
        years: int,
        months: int | None,
        refresh: bool,
        source_preference: str | None,
        allow_api_fallback: bool,
    ) -> dict[str, Any]:
        request = self.dependencies.build_request(
            symbol=symbol,
            years=years,
            months=months,
            source_preference=source_preference,
        )
        if not refresh:
            cached_payload = await ttl_cache_lookup_response(
                self.context.historical_cache,
                self.context.historical_lock,
                request.cache_key,
                ttl_sec=self.context.historical_cache_ttl_sec,
                copy_fn=dict,
            )
            if cached_payload is not None:
                return cached_payload

        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            points, source_detail = await self._resolve_historical_points(
                client=client,
                request=request,
                refresh=refresh,
                allow_api_fallback=allow_api_fallback,
            )

        if not points:
            raise HTTPException(
                status_code=404,
                detail=build_no_historical_data_detail(
                    symbol=request.symbol,
                    source_mode=request.source_mode,
                    source_detail=source_detail,
                    allow_api_fallback=allow_api_fallback,
                ),
            )

        payload = build_historical_payload(
            request=request,
            points=points,
            source_detail=source_detail,
            provider=self.context.provider,
            interval=self.context.historical_interval,
        )
        await ttl_cache_store(
            self.context.historical_cache,
            self.context.historical_lock,
            request.cache_key,
            payload,
        )
        return payload

    async def _resolve_historical_points(
        self,
        *,
        client: httpx.AsyncClient,
        request: HistoricalRequest,
        refresh: bool,
        allow_api_fallback: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        source_detail: dict[str, Any] = {
            "provider": request.source_mode or self.context.provider,
            "mode": "uninitialized",
        }
        if self._should_use_stooq_source(request):
            points, source_detail = await self.dependencies.fetch_stooq_daily_points_with_detail(
                client,
                symbol=request.symbol,
                outputsize=request.outputsize,
                start_date=request.start_date_iso,
                end_date=request.end_date_iso,
                refresh=refresh,
            )
            if points or not allow_api_fallback:
                return points, source_detail
            return await self._fetch_provider_historical_points_for_request(
                client=client,
                request=request,
                refresh=refresh,
                outputsize=max(self.context.historical_max_points, request.outputsize),
            )

        return await self._fetch_provider_historical_points_for_request(
            client=client,
            request=request,
            refresh=refresh,
            outputsize=request.outputsize,
        )

    async def _fetch_provider_historical_points_for_request(
        self,
        *,
        client: httpx.AsyncClient,
        request: HistoricalRequest,
        refresh: bool,
        outputsize: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if request.fetch_full_history and is_daily_interval(self.context.historical_interval):
            points = await self.dependencies.fetch_full_daily_series(client, symbol=request.symbol, refresh=refresh)
            return points, {
                "provider": self.context.provider,
                "mode": "full_daily_history",
            }

        return await self.dependencies.fetch_historical_points_with_detail(
            client,
            symbol=request.symbol,
            interval=self.context.historical_interval,
            outputsize=outputsize,
            start_date=request.start_date_iso,
            end_date=request.end_date_iso,
        )

    def _should_use_stooq_source(self, request: HistoricalRequest) -> bool:
        return (
            request.months is None
            and is_daily_interval(self.context.historical_interval)
            and request.source_mode == "stooq"
        )
