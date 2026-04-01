"""Implementation helpers for historical market-data queries."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import (
    DAILY_DIFF_MIN_RECHECK_SEC,
    EARLIEST_TIMESTAMP_URL,
    FMP_HISTORICAL_EOD_URL,
    FULL_HISTORY_CHUNK_YEARS,
    FULL_HISTORY_MAX_CHUNKS,
    HISTORICAL_CACHE_TTL_SEC,
    HISTORICAL_MAX_POINTS,
    JQUANTS_API_KEY as DEFAULT_JQUANTS_API_KEY,
    JQUANTS_DAILY_BARS_URL,
    JQUANTS_MIN_REQUEST_INTERVAL_SEC as DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
    JQUANTS_RATE_LIMIT_BACKOFF_SEC as DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
    LOGGER,
    TIME_SERIES_MAX_OUTPUTSIZE,
    TIME_SERIES_URL,
)
from ..ohlcv import merge_points_by_timestamp as merge_ohlcv_points, normalize_ohlcv_point
from ..stooq import fetch_stooq_daily_history as default_fetch_stooq_daily_history
from .market_data_provider_clients import FmpClient, TwelveDataClient
from .market_data_queries_historical_runtime import (
    build_combined_historical_detail,
    build_jquants_historical_detail,
    build_stooq_historical_detail,
    resolve_runtime_fetcher,
    runtime_value,
)


class MarketDataHistoricalOps:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _td_client(self, client: httpx.AsyncClient) -> TwelveDataClient:
        return TwelveDataClient(client, getattr(self.owner, "twelvedata_api_key", ""))

    def _fmp_client(self, client: httpx.AsyncClient) -> FmpClient:
        return FmpClient(client, getattr(self.owner, "fmp_api_key", ""))

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
        cached_full_points = await self.owner.full_daily_history_store.get(symbol, copy=False)
        cached_updated_epoch = await self.owner.full_daily_history_store.last_updated_epoch(symbol)
        if (
            cached_full_points
            and not refresh
            and cached_updated_epoch is not None
            and self.owner._is_cache_fresh(cached_updated_epoch, HISTORICAL_CACHE_TTL_SEC)
        ):
            cached_slice = self.owner._slice_daily_points(
                cached_full_points,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
            )
            if cached_slice:
                return cached_slice, {
                    "mode": "stooq_cached",
                    "dataset": "historical_daily",
                    "provider": "stooq",
                    "points": len(cached_slice),
                }

        fetch_stooq_daily_history = resolve_runtime_fetcher(
            "fetch_stooq_daily_history",
            default_fetch_stooq_daily_history,
        )
        try:
            full_points = await fetch_stooq_daily_history(symbol, client=client)
        except Exception as exc:
            LOGGER.warning("Stooq daily CSV fetch failed for %s: %s", symbol, exc)
            return [], build_stooq_historical_detail(
                mode="stooq_fetch_failed",
                points=0,
                error=str(exc).strip(),
            )

        if full_points:
            await self.owner.full_daily_history_store.upsert(symbol, full_points)
            fetched_slice = self.owner._slice_daily_points(
                full_points,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
            )
            if fetched_slice:
                return fetched_slice, build_stooq_historical_detail(
                    mode="stooq_live",
                    points=len(fetched_slice),
                )

        if cached_full_points:
            stale_slice = self.owner._slice_daily_points(
                cached_full_points,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
            )
            if stale_slice:
                return stale_slice, build_stooq_historical_detail(
                    mode="stooq_cached_stale",
                    points=len(stale_slice),
                )

        return [], build_stooq_historical_detail(
            mode="stooq_empty_range" if (start_date or end_date) else "stooq_empty",
            points=0,
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
        spacing = max(
            0.0,
            float(
                runtime_value(
                    "JQUANTS_MIN_REQUEST_INTERVAL_SEC",
                    DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
                )
            ),
        )
        if spacing <= 0.0:
            return

        lock = getattr(self.owner, "_jquants_request_lock", None)
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            setattr(self.owner, "_jquants_request_lock", lock)

        async with lock:
            now = time.monotonic()
            next_request_at = float(getattr(self.owner, "_jquants_next_request_at", 0.0) or 0.0)
            if next_request_at > now:
                await asyncio.sleep(next_request_at - now)
                now = time.monotonic()
            setattr(self.owner, "_jquants_next_request_at", now + spacing)

    async def delay_future_jquants_requests(self, delay_sec: float) -> None:
        lock = getattr(self.owner, "_jquants_request_lock", None)
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            setattr(self.owner, "_jquants_request_lock", lock)

        async with lock:
            now = time.monotonic()
            next_request_at = float(getattr(self.owner, "_jquants_next_request_at", 0.0) or 0.0)
            setattr(self.owner, "_jquants_next_request_at", max(next_request_at, now + max(0.0, delay_sec)))

    async def fetch_series_jquants(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        del outputsize
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return []

        code = self.owner._normalize_jquants_code(symbol)
        api_key = str(runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        if not code or not api_key:
            return []

        headers = {"x-api-key": api_key}
        cached_coverage = getattr(self.owner, "_jquants_coverage_window", None)
        bounded_dates = None
        if (
            isinstance(cached_coverage, tuple)
            and len(cached_coverage) == 2
            and isinstance(cached_coverage[0], date)
            and isinstance(cached_coverage[1], date)
        ):
            bounded_dates = self.owner._bound_jquants_request_dates(
                start_date=start_date,
                end_date=end_date,
                coverage_window=(cached_coverage[0], cached_coverage[1]),
            )
        request_start, request_end = bounded_dates if bounded_dates is not None else (start_date, end_date)
        adjusted_to_coverage = False
        rate_limit_attempts = 0

        while True:
            params: dict[str, Any] = {"code": code}
            if request_start:
                params["from"] = request_start
            if request_end:
                params["to"] = request_end

            points: list[dict[str, Any]] = []
            pagination_key: str | None = None
            should_retry = False

            while True:
                request_params = dict(params)
                if pagination_key:
                    request_params["pagination_key"] = pagination_key

                try:
                    await self.await_jquants_request_slot()
                    response = await client.get(JQUANTS_DAILY_BARS_URL, params=request_params, headers=headers)
                    payload = response.json()
                except Exception as exc:
                    LOGGER.warning("J-Quants daily bars fetch failed for %s: %s", symbol, exc)
                    return []

                if response.status_code >= 400:
                    message = payload.get("message") if isinstance(payload, dict) else payload
                    coverage_window = self.owner._extract_jquants_coverage_window(message)
                    if coverage_window is not None:
                        setattr(self.owner, "_jquants_coverage_window", coverage_window)
                    clamped_dates = None
                    if not adjusted_to_coverage:
                        clamped_dates = self.owner._clamp_jquants_request_dates(
                            start_date=request_start,
                            end_date=request_end,
                            coverage_message=message,
                        )
                    if clamped_dates is not None:
                        request_start, request_end = clamped_dates
                        adjusted_to_coverage = True
                        should_retry = True
                        break

                    if self.owner._is_jquants_rate_limit_message(message) and rate_limit_attempts < 3:
                        rate_limit_attempts += 1
                        backoff_sec = float(
                            runtime_value(
                                "JQUANTS_RATE_LIMIT_BACKOFF_SEC",
                                DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
                            )
                        )
                        await self.delay_future_jquants_requests(backoff_sec * rate_limit_attempts)
                        should_retry = True
                        break

                    LOGGER.warning("J-Quants daily bars API error for %s: %s", symbol, payload)
                    return []

                values = None
                if isinstance(payload, dict):
                    for key in ("daily_quotes", "quotes", "bars", "dailyBars"):
                        candidate = payload.get(key)
                        if isinstance(candidate, list):
                            values = candidate
                            break
                if not isinstance(values, list):
                    return []

                for item in values:
                    point = normalize_ohlcv_point(
                        item,
                        timestamp_keys=("Date", "date"),
                        open_keys=("Open", "open", "AdjustmentOpen", "adjustment_open"),
                        high_keys=("High", "high", "AdjustmentHigh", "adjustment_high"),
                        low_keys=("Low", "low", "AdjustmentLow", "adjustment_low"),
                        close_keys=("Close", "close", "AdjustmentClose", "adjustment_close"),
                        volume_keys=("Volume", "volume", "AdjustmentVolume", "adjustment_volume"),
                        source="jquants",
                    )
                    if point is not None:
                        points.append(point)

                pagination_key = payload.get("pagination_key") if isinstance(payload, dict) else None
                if not pagination_key:
                    break

            if should_retry:
                continue
            return sorted(points, key=lambda item: str(item.get("t") or ""))

    async def fetch_series_twelvedata(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            api_response = await self._td_client(client).get_time_series(
                TIME_SERIES_URL,
                symbol=symbol,
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
            LOGGER.warning("Time series fetch failed for %s %s: %s", symbol, interval, exc)
            return []

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Time series API error for %s %s: %s", symbol, interval, payload.get("message"))
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
            payload = await self._fmp_client(client).get_historical_eod(
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
        today = date.today()
        cached_points = await self.load_cached_full_daily_series(symbol=symbol, refresh=refresh)
        if cached_points:
            last_date = self.owner._point_date(cached_points[-1])
            if last_date and last_date >= today:
                return cached_points

            if not await self.should_recheck_cached_full_daily_series(
                symbol=symbol,
                min_recheck_sec=min_recheck_sec,
            ):
                return cached_points

            return await self.refresh_cached_full_daily_series(
                client=client,
                symbol=symbol,
                cached_points=cached_points,
                last_date=last_date,
                today=today,
            )

        return await self.fetch_uncached_full_daily_series(
            client=client,
            symbol=symbol,
            today=today,
        )

    async def load_cached_full_daily_series(self, *, symbol: str, refresh: bool) -> list[dict[str, Any]]:
        if refresh:
            await self.owner.full_daily_history_store.clear(symbol)
            return []
        return await self.owner.full_daily_history_store.get(symbol, copy=False)

    async def should_recheck_cached_full_daily_series(
        self,
        *,
        symbol: str,
        min_recheck_sec: int | None,
    ) -> bool:
        last_cache_update_epoch = await self.owner.full_daily_history_store.last_updated_epoch(symbol)
        if last_cache_update_epoch is None:
            return True
        last_cache_update_dt = datetime.fromtimestamp(last_cache_update_epoch, tz=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        recheck_sec = DAILY_DIFF_MIN_RECHECK_SEC if min_recheck_sec is None else max(60, int(min_recheck_sec))
        return not (
            last_cache_update_dt.date() == now_utc.date()
            and (now_utc.timestamp() - last_cache_update_epoch) < recheck_sec
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
        point_groups: list[list[dict[str, Any]]] = [cached_points]
        start_cursor = (last_date - timedelta(days=5)) if last_date else (today - timedelta(days=10))
        start_cursor = await self.extend_daily_history_in_chunks(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
            point_groups=point_groups,
        )
        if start_cursor <= today:
            LOGGER.warning(
                "Daily cache catch-up truncated for %s: reached chunk limit (%s).",
                symbol,
                FULL_HISTORY_MAX_CHUNKS,
            )
        merged_cached = merge_ohlcv_points(*point_groups)
        await self.owner.full_daily_history_store.upsert(symbol, merged_cached)
        return merged_cached

    async def fetch_uncached_full_daily_series(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        today: date,
    ) -> list[dict[str, Any]]:
        fallback_points = await self.fetch_daily_series_fallback(client=client, symbol=symbol)
        if self.owner._should_use_jquants_for_symbol(symbol, "1day"):
            if fallback_points:
                await self.owner.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        earliest = await self.fetch_earliest_date(client, symbol=symbol, interval="1day")
        if earliest is None:
            if fallback_points:
                await self.owner.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        merged = await self.fetch_full_daily_series_from_earliest(
            client=client,
            symbol=symbol,
            start_cursor=earliest,
            today=today,
        )
        if not merged:
            if fallback_points:
                await self.owner.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        deduped = merge_ohlcv_points(merged)
        await self.owner.full_daily_history_store.upsert(symbol, deduped)
        return deduped

    async def fetch_daily_series_fallback(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
    ) -> list[dict[str, Any]]:
        return await self.owner._fetch_series(
            client,
            symbol=symbol,
            interval="1day",
            outputsize=max(1300, HISTORICAL_MAX_POINTS),
        )

    async def fetch_full_daily_series_from_earliest(
        self,
        *,
        client: httpx.AsyncClient | None,
        symbol: str,
        start_cursor: date,
        today: date,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        end_cursor = await self.extend_daily_history_in_chunks(
            client=client,
            symbol=symbol,
            start_cursor=start_cursor,
            today=today,
            merged_points=merged,
        )
        if end_cursor <= today:
            LOGGER.warning(
                "Daily full history truncated for %s: reached chunk limit (%s).",
                symbol,
                FULL_HISTORY_MAX_CHUNKS,
            )
        return merged

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
        chunks = 0
        while start_cursor <= today and chunks < FULL_HISTORY_MAX_CHUNKS:
            chunk_end = min(
                today,
                start_cursor + timedelta(days=(366 * FULL_HISTORY_CHUNK_YEARS) - 1),
            )
            points = await self.owner._fetch_series(
                client,
                symbol=symbol,
                interval="1day",
                outputsize=TIME_SERIES_MAX_OUTPUTSIZE,
                start_date=start_cursor.isoformat(),
                end_date=chunk_end.isoformat(),
            )
            if points:
                if point_groups is not None:
                    point_groups.append(points)
                if merged_points is not None:
                    merged_points.extend(points)
            start_cursor = chunk_end + timedelta(days=1)
            chunks += 1
        return start_cursor

    async def fetch_earliest_date(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        interval: str,
    ) -> date | None:
        if not self.owner._uses_twelvedata() or client is None:
            return None
        try:
            api_response = await self._td_client(client).get_earliest_timestamp(
                EARLIEST_TIMESTAMP_URL,
                symbol=symbol,
                interval=interval,
            )
            async with self.owner._credits_lock:
                await self.owner._update_minute_credits_from_response(api_response.response)
                await self.owner._consume_daily_credit_estimate(1, source=f"earliest:{symbol}:{interval}")
            payload = api_response.payload
        except Exception as exc:
            LOGGER.warning("Earliest timestamp fetch failed for %s %s: %s", symbol, interval, exc)
            return None

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Earliest timestamp API error for %s %s: %s", symbol, interval, payload.get("message"))
            return None

        raw_value = None
        if isinstance(payload, dict):
            raw_value = payload.get("datetime") or payload.get("timestamp")
        if raw_value is None:
            return None

        parsed_iso = self.owner._parse_timestamp(raw_value)
        if not parsed_iso:
            text = str(raw_value).strip()
            if text:
                try:
                    return date.fromisoformat(text.split(" ")[0])
                except ValueError:
                    return None
            return None

        try:
            return date.fromisoformat(parsed_iso[:10])
        except ValueError:
            return None
