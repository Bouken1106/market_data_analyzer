"""Historical-data helpers for MarketData query mixins."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from ..config import (
    JQUANTS_API_KEY as DEFAULT_JQUANTS_API_KEY,
    JQUANTS_MIN_REQUEST_INTERVAL_SEC as DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
    JQUANTS_RATE_LIMIT_BACKOFF_SEC as DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
    settings,
)
from ..utils import cached_attr
from .market_data_historical_ops import MarketDataHistoricalOps
from .market_data_historical_service import (
    HistoricalQueryContext,
    HistoricalQueryDependencies,
    MarketDataHistoricalQueryService,
)
from .market_data_queries_historical_runtime import (
    bound_jquants_request_dates,
    build_standard_historical_detail,
    clamp_jquants_request_dates,
    extract_jquants_coverage_window,
    is_jquants_invalid_api_key_message,
    is_jquants_rate_limit_message,
    normalize_jquants_code,
    runtime_value,
    should_use_jquants_for_symbol,
)
from .market_data_queries_historical_support import (
    HistoricalRequest,
    build_historical_request,
    build_historical_payload,
    is_daily_interval,
    slice_daily_points,
)


class MarketDataHistoricalMixin:
    def _historical_query_service(self) -> MarketDataHistoricalQueryService:
        service = getattr(self, "historical_query_service", None)
        if service is not None:
            return service
        return cached_attr(
            self,
            "_historical_query_service_instance",
            lambda: MarketDataHistoricalQueryService(
                context=self._historical_query_context(),
                dependencies=self._historical_query_dependencies(),
            ),
        )

    def _historical_query_context(self) -> HistoricalQueryContext:
        return HistoricalQueryContext(
            provider=self.provider,
            historical_cache=self._historical_cache,
            historical_lock=self._historical_lock,
            historical_cache_ttl_sec=settings.historical.historical_cache_ttl_sec,
            historical_interval=settings.historical.historical_interval,
            historical_max_points=settings.historical.historical_max_points,
        )

    def _historical_query_dependencies(self) -> HistoricalQueryDependencies:
        return HistoricalQueryDependencies(
            build_request=self._build_historical_request,
            fetch_stooq_daily_points_with_detail=self._fetch_stooq_daily_points_with_detail,
            fetch_historical_points_with_detail=self._fetch_historical_points_with_detail,
            fetch_full_daily_series=self._fetch_full_daily_series,
        )

    def _historical_ops(self) -> MarketDataHistoricalOps:
        return cached_attr(self, "_historical_ops_service", lambda: MarketDataHistoricalOps(self))

    async def historical_payload(
        self,
        symbol: str,
        years: int = settings.historical.historical_default_years,
        months: int | None = None,
        refresh: bool = False,
        source_preference: str | None = None,
        allow_api_fallback: bool = True,
    ) -> dict[str, Any]:
        return await self._historical_query_service().historical_payload(
            symbol=symbol,
            years=years,
            months=months,
            refresh=refresh,
            source_preference=source_preference,
            allow_api_fallback=allow_api_fallback,
        )

    @staticmethod
    def _is_daily_interval(interval: str) -> bool:
        return is_daily_interval(interval)

    def _build_historical_request(
        self,
        *,
        symbol: str,
        years: int,
        months: int | None,
        source_preference: str | None,
    ) -> HistoricalRequest:
        return build_historical_request(
            symbol=symbol,
            years=years,
            months=months,
            source_preference=source_preference,
        )

    def _should_use_stooq_source(self, request: HistoricalRequest) -> bool:
        return self._historical_query_service()._should_use_stooq_source(request)

    async def _resolve_historical_points(
        self,
        *,
        client: httpx.AsyncClient,
        request: HistoricalRequest,
        refresh: bool,
        allow_api_fallback: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_query_service()._resolve_historical_points(
            client=client,
            request=request,
            refresh=refresh,
            allow_api_fallback=allow_api_fallback,
        )

    async def _fetch_provider_historical_points_for_request(
        self,
        *,
        client: httpx.AsyncClient,
        request: HistoricalRequest,
        refresh: bool,
        outputsize: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_query_service()._fetch_provider_historical_points_for_request(
            client=client,
            request=request,
            refresh=refresh,
            outputsize=outputsize,
        )

    async def _fetch_full_history_with_detail(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        refresh: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        points = await self._fetch_full_daily_series(client, symbol=symbol, refresh=refresh)
        return points, {
            "provider": self.provider,
            "mode": "full_daily_history",
        }

    def _build_historical_payload(
        self,
        *,
        request: HistoricalRequest,
        points: list[dict[str, Any]],
        source_detail: dict[str, Any],
    ) -> dict[str, Any]:
        return build_historical_payload(
            request=request,
            points=points,
            source_detail=source_detail,
            provider=self.provider,
            interval=settings.historical.historical_interval,
        )

    @staticmethod
    def _slice_daily_points(
        points: list[dict[str, Any]],
        *,
        start_date: str | None,
        end_date: str | None,
        outputsize: int,
    ) -> list[dict[str, Any]]:
        return slice_daily_points(
            points,
            start_date=start_date,
            end_date=end_date,
            outputsize=outputsize,
        )

    async def _fetch_stooq_daily_points_with_detail(
        self,
        client: httpx.AsyncClient,
        *,
        symbol: str,
        outputsize: int,
        start_date: str | None,
        end_date: str | None,
        refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_ops().fetch_stooq_daily_points_with_detail(
            client,
            symbol=symbol,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
            refresh=refresh,
        )

    async def _fetch_historical_points_with_detail(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_ops().fetch_historical_points_with_detail(
            client,
            symbol,
            interval,
            outputsize,
            start_date,
            end_date,
        )

    def _should_try_fmp_daily_fallback(
        self,
        *,
        points: list[dict[str, Any]],
        interval: str,
    ) -> bool:
        return (
            not points
            and self.provider == "twelvedata"
            and bool(self.fmp_api_key)
            and self._is_daily_interval(interval)
        )

    def _build_standard_historical_detail(self, *, points: list[dict[str, Any]]) -> dict[str, Any]:
        return build_standard_historical_detail(provider=self.provider, points=len(points))

    async def _fetch_series(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._should_use_jquants_for_symbol(symbol, interval):
            return await self._fetch_series_jquants(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        return await self._fetch_primary_provider_series(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_primary_provider_series(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.provider == "both":
            return await self._fetch_series_both(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        if self.provider == "fmp":
            return await self._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        return await self._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    def _should_use_jquants_for_symbol(self, symbol: str, interval: str) -> bool:
        if bool(getattr(self, "_jquants_api_key_invalid", False)):
            return False
        api_key = str(runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        return should_use_jquants_for_symbol(symbol, interval, api_key=api_key)

    @staticmethod
    def _normalize_jquants_code(symbol: str) -> str | None:
        return normalize_jquants_code(symbol)

    @classmethod
    def _extract_jquants_coverage_window(cls, message: Any) -> tuple[date, date] | None:
        del cls
        return extract_jquants_coverage_window(message)

    @staticmethod
    def _bound_jquants_request_dates(
        *,
        start_date: str | None,
        end_date: str | None,
        coverage_window: tuple[date, date],
    ) -> tuple[str, str] | None:
        return bound_jquants_request_dates(
            start_date=start_date,
            end_date=end_date,
            coverage_window=coverage_window,
        )

    @classmethod
    def _clamp_jquants_request_dates(
        cls,
        *,
        start_date: str | None,
        end_date: str | None,
        coverage_message: Any,
    ) -> tuple[str, str] | None:
        del cls
        return clamp_jquants_request_dates(
            start_date=start_date,
            end_date=end_date,
            coverage_message=coverage_message,
        )

    @staticmethod
    def _is_jquants_rate_limit_message(message: Any) -> bool:
        return is_jquants_rate_limit_message(message)

    @staticmethod
    def _is_jquants_invalid_api_key_message(message: Any) -> bool:
        return is_jquants_invalid_api_key_message(message)

    async def _await_jquants_request_slot(self) -> None:
        await self._historical_ops().await_jquants_request_slot()

    async def _delay_future_jquants_requests(self, delay_sec: float) -> None:
        await self._historical_ops().delay_future_jquants_requests(delay_sec)

    async def _fetch_series_jquants(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_series_jquants(
            client,
            symbol,
            interval,
            outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_series_twelvedata(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_series_twelvedata(
            client,
            symbol,
            interval,
            outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_series_both(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_series_both(
            client,
            symbol,
            interval,
            outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_series_fmp(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_series_fmp(
            client,
            symbol,
            interval,
            outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_full_daily_series(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        refresh: bool = False,
        min_recheck_sec: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_full_daily_series(
            client,
            symbol,
            refresh=refresh,
            min_recheck_sec=min_recheck_sec,
        )

    async def _fetch_earliest_date(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        interval: str,
    ) -> date | None:
        return await self._historical_ops().fetch_earliest_date(client, symbol, interval)
