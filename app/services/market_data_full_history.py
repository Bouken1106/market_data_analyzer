"""Helpers for full daily-history expansion and cache refresh."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import (
    DAILY_DIFF_MIN_RECHECK_SEC,
    EARLIEST_TIMESTAMP_URL,
    FULL_HISTORY_CHUNK_YEARS,
    FULL_HISTORY_MAX_CHUNKS,
    HISTORICAL_MAX_POINTS,
    LOGGER,
    TIME_SERIES_MAX_OUTPUTSIZE,
)
from ..ohlcv import merge_points_by_timestamp as merge_ohlcv_points
from ..utils import date_or_none
from .market_data_provider_clients import owner_twelvedata_client


class FullDailyHistoryLoader:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

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
        provider_symbol = self.owner._format_twelvedata_symbol(symbol)
        try:
            api_response = await owner_twelvedata_client(self.owner, client).get_earliest_timestamp(
                EARLIEST_TIMESTAMP_URL,
                symbol=provider_symbol,
                interval=interval,
            )
            async with self.owner._credits_lock:
                await self.owner._update_minute_credits_from_response(api_response.response)
                await self.owner._consume_daily_credit_estimate(1, source=f"earliest:{symbol}:{interval}")
            payload = api_response.payload
        except Exception as exc:
            LOGGER.warning("Earliest timestamp fetch failed for %s (%s) %s: %s", symbol, provider_symbol, interval, exc)
            return None

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Earliest timestamp API error for %s (%s) %s: %s", symbol, provider_symbol, interval, payload.get("message"))
            return None

        raw_value = None
        if isinstance(payload, dict):
            raw_value = payload.get("datetime") or payload.get("timestamp")
        if raw_value is None:
            return None

        parsed_iso = self.owner._parse_timestamp(raw_value)
        return date_or_none(parsed_iso or raw_value)
