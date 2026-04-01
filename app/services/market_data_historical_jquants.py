"""Focused J-Quants helpers for historical market-data queries."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

import httpx

from ..config import (
    JQUANTS_API_KEY as DEFAULT_JQUANTS_API_KEY,
    JQUANTS_DAILY_BARS_URL,
    JQUANTS_MIN_REQUEST_INTERVAL_SEC as DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
    JQUANTS_RATE_LIMIT_BACKOFF_SEC as DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
    LOGGER,
)
from ..ohlcv import normalize_ohlcv_point
from .market_data_queries_historical_runtime import (
    bound_jquants_request_dates,
    clamp_jquants_request_dates,
    extract_jquants_coverage_window,
    is_jquants_rate_limit_message,
    normalize_jquants_code,
    runtime_value,
)


class JQuantsHistoricalClient:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _request_lock(self) -> asyncio.Lock:
        lock = getattr(self.owner, "_jquants_request_lock", None)
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            setattr(self.owner, "_jquants_request_lock", lock)
        return lock

    async def await_request_slot(self) -> None:
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

        async with self._request_lock():
            now = time.monotonic()
            next_request_at = float(getattr(self.owner, "_jquants_next_request_at", 0.0) or 0.0)
            if next_request_at > now:
                await asyncio.sleep(next_request_at - now)
                now = time.monotonic()
            setattr(self.owner, "_jquants_next_request_at", now + spacing)

    async def delay_future_requests(self, delay_sec: float) -> None:
        async with self._request_lock():
            now = time.monotonic()
            next_request_at = float(getattr(self.owner, "_jquants_next_request_at", 0.0) or 0.0)
            setattr(self.owner, "_jquants_next_request_at", max(next_request_at, now + max(0.0, delay_sec)))

    def _resolve_request_dates(
        self,
        *,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[str | None, str | None]:
        cached_coverage = getattr(self.owner, "_jquants_coverage_window", None)
        if (
            isinstance(cached_coverage, tuple)
            and len(cached_coverage) == 2
            and isinstance(cached_coverage[0], date)
            and isinstance(cached_coverage[1], date)
        ):
            bounded_dates = bound_jquants_request_dates(
                start_date=start_date,
                end_date=end_date,
                coverage_window=(cached_coverage[0], cached_coverage[1]),
            )
            if bounded_dates is not None:
                return bounded_dates
        return start_date, end_date

    async def fetch_series(
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

        code = normalize_jquants_code(symbol)
        api_key = str(runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        if not code or not api_key:
            return []

        headers = {"x-api-key": api_key}
        request_start, request_end = self._resolve_request_dates(
            start_date=start_date,
            end_date=end_date,
        )
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
                    await self.await_request_slot()
                    response = await client.get(JQUANTS_DAILY_BARS_URL, params=request_params, headers=headers)
                    payload = response.json()
                except Exception as exc:
                    LOGGER.warning("J-Quants daily bars fetch failed for %s: %s", symbol, exc)
                    return []

                if response.status_code >= 400:
                    message = payload.get("message") if isinstance(payload, dict) else payload
                    coverage_window = extract_jquants_coverage_window(message)
                    if coverage_window is not None:
                        setattr(self.owner, "_jquants_coverage_window", coverage_window)
                    clamped_dates = None
                    if not adjusted_to_coverage:
                        clamped_dates = clamp_jquants_request_dates(
                            start_date=request_start,
                            end_date=request_end,
                            coverage_message=message,
                        )
                    if clamped_dates is not None:
                        request_start, request_end = clamped_dates
                        adjusted_to_coverage = True
                        should_retry = True
                        break

                    if is_jquants_rate_limit_message(message) and rate_limit_attempts < 3:
                        rate_limit_attempts += 1
                        backoff_sec = float(
                            runtime_value(
                                "JQUANTS_RATE_LIMIT_BACKOFF_SEC",
                                DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
                            )
                        )
                        await self.delay_future_requests(backoff_sec * rate_limit_attempts)
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
