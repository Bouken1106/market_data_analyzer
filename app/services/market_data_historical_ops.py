"""Implementation helpers for historical market-data queries."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx

from ..config import (
    FMP_HISTORICAL_EOD_URL,
    HISTORICAL_CACHE_TTL_SEC,
    LOGGER,
    TIME_SERIES_MAX_OUTPUTSIZE,
    TIME_SERIES_URL,
)
from ..ohlcv import normalize_ohlcv_point
from ..stooq import fetch_stooq_daily_history as default_fetch_stooq_daily_history
from .market_data_full_history import FullDailyHistoryLoader
from .market_data_historical_jquants import JQuantsHistoricalClient
from .market_data_provider_clients import owner_fmp_client, owner_twelvedata_client
from .market_data_queries_historical_runtime import (
    build_combined_historical_detail,
    build_jquants_historical_detail,
    build_stooq_historical_detail,
    resolve_runtime_fetcher,
)


class MarketDataHistoricalOps:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.jquants = JQuantsHistoricalClient(owner)
        self.full_history = FullDailyHistoryLoader(owner)

    async def fetch_stooq_daily_points_with_detail(
        self,
        client: httpx.AsyncClient,
        *,
        symbol: str,
        outputsize: int,
        start_date: str | None,
        end_date: str | None,
        refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cached_full_points, cached_updated_epoch = await self._read_stooq_cache(symbol)
        cached_result = self._fresh_stooq_cache_result(
            cached_full_points=cached_full_points,
            cached_updated_epoch=cached_updated_epoch,
            refresh=refresh,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        if cached_result is not None:
            return cached_result

        full_points, error_detail = await self._fetch_stooq_full_points(client=client, symbol=symbol)
        if error_detail is not None:
            return [], error_detail

        if full_points:
            await self.owner.full_daily_history_store.upsert(symbol, full_points)
            fetched_result = self._stooq_slice_result(
                full_points=full_points,
                mode="stooq_live",
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            if fetched_result is not None:
                return fetched_result

        if cached_full_points:
            stale_result = self._stooq_slice_result(
                full_points=cached_full_points,
                mode="stooq_cached_stale",
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            if stale_result is not None:
                return stale_result

        return [], build_stooq_historical_detail(
            mode="stooq_empty_range" if (start_date or end_date) else "stooq_empty",
            points=0,
        )

    async def _read_stooq_cache(self, symbol: str) -> tuple[list[dict[str, Any]], float | None]:
        cached_full_points = await self.owner.full_daily_history_store.get(symbol, copy=False)
        cached_updated_epoch = await self.owner.full_daily_history_store.last_updated_epoch(symbol)
        return cached_full_points, cached_updated_epoch

    def _fresh_stooq_cache_result(
        self,
        *,
        cached_full_points: list[dict[str, Any]],
        cached_updated_epoch: float | None,
        refresh: bool,
        outputsize: int,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        if (
            not cached_full_points
            or refresh
            or cached_updated_epoch is None
            or not self.owner._is_cache_fresh(cached_updated_epoch, HISTORICAL_CACHE_TTL_SEC)
        ):
            return None
        return self._stooq_slice_result(
            full_points=cached_full_points,
            mode="stooq_cached",
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def _fetch_stooq_full_points(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        fetch_stooq_daily_history = resolve_runtime_fetcher(
            "fetch_stooq_daily_history",
            default_fetch_stooq_daily_history,
        )
        try:
            return await fetch_stooq_daily_history(symbol, client=client), None
        except Exception as exc:
            LOGGER.warning("Stooq daily CSV fetch failed for %s: %s", symbol, exc)
            return [], build_stooq_historical_detail(
                mode="stooq_fetch_failed",
                points=0,
                error=str(exc).strip(),
            )

    def _stooq_slice_result(
        self,
        *,
        full_points: list[dict[str, Any]],
        mode: str,
        outputsize: int,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        sliced_points = self.owner._slice_daily_points(
            full_points,
            start_date=start_date,
            end_date=end_date,
            outputsize=outputsize,
        )
        if not sliced_points:
            return None
        return sliced_points, build_stooq_historical_detail(
            mode=mode,
            points=len(sliced_points),
        )

    async def fetch_historical_points_with_detail(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.owner._should_use_jquants_for_symbol(symbol, interval):
            return await self.fetch_jquants_historical_points_with_detail(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )

        if self.owner.provider == "both":
            return await self.fetch_combined_historical_points_with_detail(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )

        points = await self.owner._fetch_primary_provider_series(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        if self.owner._should_try_fmp_daily_fallback(points=points, interval=interval):
            fmp_points = await self.owner._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            if fmp_points:
                return fmp_points, {
                    "mode": "twelvedata_with_fmp_fallback",
                    "dataset": "historical_daily",
                    "provider": "fmp",
                    "points": len(fmp_points),
                }

        return points, self.owner._build_standard_historical_detail(points=points)

    async def fetch_jquants_historical_points_with_detail(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        points = await self.owner._fetch_series_jquants(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        return points, build_jquants_historical_detail(points=len(points))

    async def fetch_combined_historical_points_with_detail(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        td_task = self.owner._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        fmp_task = self.owner._fetch_series_fmp(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        td_points, fmp_points = await asyncio.gather(td_task, fmp_task)
        merged = self.owner._merge_points_by_timestamp(fmp_points, td_points)
        return merged, build_combined_historical_detail(
            td_points=td_points,
            fmp_points=fmp_points,
            merged_points=merged,
        )

    async def await_jquants_request_slot(self) -> None:
        await self.jquants.await_request_slot()

    async def delay_future_jquants_requests(self, delay_sec: float) -> None:
        await self.jquants.delay_future_requests(delay_sec)

    async def fetch_series_jquants(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self.jquants.fetch_series(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    async def fetch_series_twelvedata(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        provider_symbol = self.owner._format_twelvedata_symbol(symbol)
        try:
            api_response = await owner_twelvedata_client(self.owner, client).get_time_series(
                TIME_SERIES_URL,
                symbol=provider_symbol,
                interval=interval,
                outputsize=min(max(1, int(outputsize)), TIME_SERIES_MAX_OUTPUTSIZE),
                start_date=start_date,
                end_date=end_date,
            )
            async with self.owner._credits_lock:
                await self.owner._update_minute_credits_from_response(api_response.response)
                await self.owner._consume_daily_credit_estimate(1, source=f"series:{symbol}:{interval}")
            payload = api_response.payload
        except Exception as exc:
            LOGGER.warning("Time series fetch failed for %s (%s) %s: %s", symbol, provider_symbol, interval, exc)
            return []

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Time series API error for %s (%s) %s: %s", symbol, provider_symbol, interval, payload.get("message"))
            return []

        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return []

        points: list[dict[str, Any]] = []
        for item in values:
            point = normalize_ohlcv_point(
                item,
                timestamp_keys=("datetime",),
                open_keys=("open",),
                high_keys=("high",),
                low_keys=("low",),
                close_keys=("close",),
                volume_keys=("volume",),
                source="twelvedata",
            )
            if point is not None:
                points.append(point)

        return points

    async def fetch_series_both(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_interval = str(interval or "").strip().lower()
        if normalized_interval not in {"1day", "1d", "day"}:
            return await self.owner._fetch_series_twelvedata(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )

        td_task = self.owner._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        fmp_task = self.owner._fetch_series_fmp(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
        td_points = td_result if isinstance(td_result, list) else []
        fmp_points = fmp_result if isinstance(fmp_result, list) else []

        if isinstance(td_result, Exception):
            LOGGER.warning("Time series fetch failed (TD) for %s %s: %s", symbol, interval, td_result)
        if isinstance(fmp_result, Exception):
            LOGGER.warning("Time series fetch failed (FMP) for %s %s: %s", symbol, interval, fmp_result)

        return self.owner._merge_points_by_timestamp(fmp_points, td_points)

    async def fetch_series_fmp(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return []

        try:
            payload = await owner_fmp_client(self.owner, client).get_historical_eod(
                FMP_HISTORICAL_EOD_URL,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            LOGGER.warning("FMP time series fetch failed for %s %s: %s", symbol, interval, exc)
            return []

        if self.owner._is_fmp_error(payload):
            LOGGER.warning("FMP time series API error for %s %s: %s", symbol, interval, payload.get("Error Message"))
            return []

        if isinstance(payload, dict):
            values = payload.get("historical") if isinstance(payload.get("historical"), list) else payload.get("data")
        elif isinstance(payload, list):
            values = payload
        else:
            values = None
        if not isinstance(values, list):
            return []

        points: list[dict[str, Any]] = []
        for item in values:
            point = normalize_ohlcv_point(
                item,
                timestamp_keys=("date", "datetime"),
                open_keys=("open",),
                high_keys=("high",),
                low_keys=("low",),
                close_keys=("close",),
                volume_keys=("volume",),
                source="fmp",
            )
            if point is not None:
                points.append(point)

        points.sort(key=lambda item: str(item.get("t", "")))
        if outputsize > 0 and len(points) > outputsize:
            points = points[-outputsize:]
        return points

    async def fetch_full_daily_series(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        refresh: bool = False,
        min_recheck_sec: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.full_history.fetch_full_daily_series(
            client=client,
            symbol=symbol,
            refresh=refresh,
            min_recheck_sec=min_recheck_sec,
        )

    async def load_cached_full_daily_series(self, *, symbol: str, refresh: bool) -> list[dict[str, Any]]:
        return await self.full_history.load_cached_full_daily_series(symbol=symbol, refresh=refresh)

    async def should_recheck_cached_full_daily_series(
        self,
        *,
        symbol: str,
        min_recheck_sec: int | None,
    ) -> bool:
        return await self.full_history.should_recheck_cached_full_daily_series(
            symbol=symbol,
            min_recheck_sec=min_recheck_sec,
        )

    async def refresh_cached_full_daily_series(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        cached_points: list[dict[str, Any]],
        last_date: date | None,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self.full_history.refresh_cached_full_daily_series(
            client=client,
            symbol=symbol,
            cached_points=cached_points,
            last_date=last_date,
            today=today,
        )

    async def fetch_uncached_full_daily_series(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self.full_history.fetch_uncached_full_daily_series(
            client=client,
            symbol=symbol,
            today=today,
        )

    async def fetch_daily_series_fallback(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
    ) -> list[dict[str, Any]]:
        return await self.full_history.fetch_daily_series_fallback(client=client, symbol=symbol)

    async def fetch_full_daily_series_from_earliest(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        start_cursor: date,
        today: date,
    ) -> list[dict[str, Any]]:
        return await self.full_history.fetch_full_daily_series_from_earliest(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
        )

    async def extend_daily_history_in_chunks(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        start_cursor: date,
        today: date,
        point_groups: list[list[dict[str, Any]]] | None = None,
        merged_points: list[dict[str, Any]] | None = None,
    ) -> date:
        return await self.full_history.extend_daily_history_in_chunks(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
            point_groups=point_groups,
            merged_points=merged_points,
        )

    async def fetch_earliest_date(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        interval: str,
    ) -> date | None:
        return await self.full_history.fetch_earliest_date(
            client=client,
            symbol=symbol,
            interval=interval,
        )
