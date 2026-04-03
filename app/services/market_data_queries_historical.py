"""Historical-data helpers for MarketData query mixins."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from ..config import (
    HISTORICAL_DEFAULT_YEARS,
    HISTORICAL_INTERVAL,
    JQUANTS_API_KEY as DEFAULT_JQUANTS_API_KEY,
    JQUANTS_MIN_REQUEST_INTERVAL_SEC as DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
    JQUANTS_RATE_LIMIT_BACKOFF_SEC as DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
)
from .market_data_historical_ops import MarketDataHistoricalOps
from .market_data_historical_service import MarketDataHistoricalQueryService
from .market_data_queries_historical_runtime import (
    bound_jquants_request_dates,
    build_standard_historical_detail,
    clamp_jquants_request_dates,
    extract_jquants_coverage_window,
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
        if service is None:
            service = getattr(self, "_historical_query_service_instance", None)
        if service is None:
            service = MarketDataHistoricalQueryService(self)
            setattr(self, "_historical_query_service_instance", service)
        return service

    def _historical_ops(self) -> MarketDataHistoricalOps:
        ops = getattr(self, "_historical_ops_service", None)
        if ops is None:
            ops = MarketDataHistoricalOps(self)
            setattr(self, "_historical_ops_service", ops)
        return ops

    async def historical_payload(
        self,
        symbol: str,
        years: int = HISTORICAL_DEFAULT_YEARS,
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
            interval=HISTORICAL_INTERVAL,
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

    async def _fetch_jquants_historical_points_with_detail(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_ops().fetch_jquants_historical_points_with_detail(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_combined_historical_points_with_detail(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._historical_ops().fetch_combined_historical_points_with_detail(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
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

    async def _load_cached_full_daily_series(self, *, symbol: str, refresh: bool) -> list[dict[str, Any]]:
        return await self._historical_ops().load_cached_full_daily_series(symbol=symbol, refresh=refresh)

    async def _should_recheck_cached_full_daily_series(
        self,
        *,
        symbol: str,
        min_recheck_sec: int | None,
    ) -> bool:
        return await self._historical_ops().should_recheck_cached_full_daily_series(
            symbol=symbol,
            min_recheck_sec=min_recheck_sec,
        )

    async def _refresh_cached_full_daily_series(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        cached_points: list[dict[str, Any]],
        last_date: date | None,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().refresh_cached_full_daily_series(
            client=client,
            symbol=symbol,
            cached_points=cached_points,
            last_date=last_date,
            today=today,
        )

    async def _fetch_uncached_full_daily_series(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_uncached_full_daily_series(
            client=client,
            symbol=symbol,
            today=today,
        )

    async def _fetch_daily_series_fallback(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_daily_series_fallback(
            client=client,
            symbol=symbol,
        )

    async def _fetch_full_daily_series_from_earliest(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        start_cursor: date,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self._historical_ops().fetch_full_daily_series_from_earliest(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
        )

    async def _extend_daily_history_in_chunks(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        start_cursor: date,
        today: date,
        point_groups: list[list[dict[str, Any]]] | None = None,
        merged_points: list[dict[str, Any]] | None = None,
    ) -> date:
        return await self._historical_ops().extend_daily_history_in_chunks(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
            point_groups=point_groups,
            merged_points=merged_points,
        )

    async def _fetch_earliest_date(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        interval: str,
    ) -> date | None:
        return await self._historical_ops().fetch_earliest_date(client, symbol, interval)
