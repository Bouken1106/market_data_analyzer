"""High-level historical query orchestration."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from fastapi import HTTPException

from ..config import HISTORICAL_CACHE_TTL_SEC, HISTORICAL_INTERVAL, HISTORICAL_MAX_POINTS
from .market_data_queries_historical_support import (
    HistoricalRequest,
    build_historical_payload,
    build_no_historical_data_detail,
    is_daily_interval,
)
from .ttl_cache import ttl_cache_lookup_response, ttl_cache_store


class HistoricalQueryOwner(Protocol):
    provider: str
    _historical_cache: dict[Any, Any]
    _historical_lock: Any

    def _build_historical_request(
        self,
        *,
        symbol: str,
        years: int,
        months: int | None,
        source_preference: str | None,
    ) -> HistoricalRequest: ...

    async def _fetch_stooq_daily_points_with_detail(
        self,
        client: httpx.AsyncClient,
        *,
        symbol: str,
        outputsize: int,
        start_date: str | None,
        end_date: str | None,
        refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    async def _fetch_historical_points_with_detail(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    async def _fetch_full_daily_series(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        refresh: bool = False,
        min_recheck_sec: int | None = None,
    ) -> list[dict[str, Any]]: ...


class MarketDataHistoricalQueryService:
    def __init__(self, owner: HistoricalQueryOwner) -> None:
        self.owner = owner

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
        request = self.owner._build_historical_request(
            symbol=symbol,
            years=years,
            months=months,
            source_preference=source_preference,
        )
        if not refresh:
            cached_payload = await ttl_cache_lookup_response(
                self.owner._historical_cache,
                self.owner._historical_lock,
                request.cache_key,
                ttl_sec=HISTORICAL_CACHE_TTL_SEC,
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
            provider=self.owner.provider,
            interval=HISTORICAL_INTERVAL,
        )
        await ttl_cache_store(
            self.owner._historical_cache,
            self.owner._historical_lock,
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
            "provider": request.source_mode or self.owner.provider,
            "mode": "uninitialized",
        }
        if self._should_use_stooq_source(request):
            points, source_detail = await self.owner._fetch_stooq_daily_points_with_detail(
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
                outputsize=max(HISTORICAL_MAX_POINTS, request.outputsize),
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
        if request.fetch_full_history and is_daily_interval(HISTORICAL_INTERVAL):
            points = await self.owner._fetch_full_daily_series(client, symbol=request.symbol, refresh=refresh)
            return points, {
                "provider": self.owner.provider,
                "mode": "full_daily_history",
            }

        return await self.owner._fetch_historical_points_with_detail(
            client,
            symbol=request.symbol,
            interval=HISTORICAL_INTERVAL,
            outputsize=outputsize,
            start_date=request.start_date_iso,
            end_date=request.end_date_iso,
        )

    @staticmethod
    def _should_use_stooq_source(request: HistoricalRequest) -> bool:
        return (
            request.months is None
            and is_daily_interval(HISTORICAL_INTERVAL)
            and request.source_mode == "stooq"
        )
