"""Focused J-Quants helpers for historical market-data queries."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
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
from .market_data_intervals import is_daily_interval
from .market_data_queries_historical_runtime import (
    bound_jquants_request_dates,
    clamp_jquants_request_dates,
    extract_jquants_coverage_window,
    is_jquants_rate_limit_message,
    normalize_jquants_code,
    runtime_value,
)


@dataclass(frozen=True)
class JQuantsErrorResolution:
    retry: bool
    request_start: str | None
    request_end: str | None
    adjusted_to_coverage: bool
    rate_limit_attempts: int


@dataclass(frozen=True)
class JQuantsPageFetchResult:
    points: list[dict[str, Any]]
    retry: bool
    failed: bool
    request_start: str | None
    request_end: str | None
    adjusted_to_coverage: bool
    rate_limit_attempts: int


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

    @staticmethod
    def _is_supported_interval(interval: str) -> bool:
        return is_daily_interval(interval)

    @staticmethod
    def _build_headers(api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key}

    @staticmethod
    def _build_request_params(
        *,
        code: str,
        start_date: str | None,
        end_date: str | None,
        pagination_key: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"code": code}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        if pagination_key:
            params["pagination_key"] = pagination_key
        return params

    async def _fetch_page(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        headers: dict[str, str],
        request_params: dict[str, Any],
    ) -> tuple[httpx.Response | None, Any]:
        try:
            await self.await_request_slot()
            response = await client.get(JQUANTS_DAILY_BARS_URL, params=request_params, headers=headers)
            return response, response.json()
        except Exception as exc:
            LOGGER.warning("J-Quants daily bars fetch failed for %s: %s", symbol, exc)
            return None, None

    @staticmethod
    def _extract_error_message(payload: Any) -> Any:
        if isinstance(payload, dict):
            return payload.get("message")
        return payload

    def _update_coverage_window(self, message: Any) -> None:
        coverage_window = extract_jquants_coverage_window(message)
        if coverage_window is not None:
            setattr(self.owner, "_jquants_coverage_window", coverage_window)

    async def _resolve_error_response(
        self,
        *,
        message: Any,
        request_start: str | None,
        request_end: str | None,
        adjusted_to_coverage: bool,
        rate_limit_attempts: int,
    ) -> JQuantsErrorResolution:
        self._update_coverage_window(message)

        if self.owner._is_jquants_invalid_api_key_message(message):
            setattr(self.owner, "_jquants_api_key_invalid", True)
            LOGGER.warning("Disabling J-Quants daily bars requests because the API key is invalid or expired.")
            return JQuantsErrorResolution(
                retry=False,
                request_start=request_start,
                request_end=request_end,
                adjusted_to_coverage=adjusted_to_coverage,
                rate_limit_attempts=rate_limit_attempts,
            )

        if not adjusted_to_coverage:
            clamped_dates = clamp_jquants_request_dates(
                start_date=request_start,
                end_date=request_end,
                coverage_message=message,
            )
            if clamped_dates is not None:
                return JQuantsErrorResolution(
                    retry=True,
                    request_start=clamped_dates[0],
                    request_end=clamped_dates[1],
                    adjusted_to_coverage=True,
                    rate_limit_attempts=rate_limit_attempts,
                )

        if is_jquants_rate_limit_message(message) and rate_limit_attempts < 3:
            next_attempts = rate_limit_attempts + 1
            backoff_sec = float(
                runtime_value(
                    "JQUANTS_RATE_LIMIT_BACKOFF_SEC",
                    DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
                )
            )
            await self.delay_future_requests(backoff_sec * next_attempts)
            return JQuantsErrorResolution(
                retry=True,
                request_start=request_start,
                request_end=request_end,
                adjusted_to_coverage=adjusted_to_coverage,
                rate_limit_attempts=next_attempts,
            )

        return JQuantsErrorResolution(
            retry=False,
            request_start=request_start,
            request_end=request_end,
            adjusted_to_coverage=adjusted_to_coverage,
            rate_limit_attempts=rate_limit_attempts,
        )

    @staticmethod
    def _extract_daily_quote_values(payload: Any) -> list[dict[str, Any]] | None:
        if not isinstance(payload, dict):
            return None
        for key in ("daily_quotes", "quotes", "bars", "dailyBars", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate
        return None

    @staticmethod
    def _normalize_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for item in values:
            point = normalize_ohlcv_point(
                item,
                timestamp_keys=("Date", "date"),
                open_keys=("AdjO", "AdjustmentOpen", "adjustment_open", "O", "Open", "open"),
                high_keys=("AdjH", "AdjustmentHigh", "adjustment_high", "H", "High", "high"),
                low_keys=("AdjL", "AdjustmentLow", "adjustment_low", "L", "Low", "low"),
                close_keys=("AdjC", "AdjustmentClose", "adjustment_close", "C", "Close", "close"),
                volume_keys=("AdjVo", "AdjustmentVolume", "adjustment_volume", "Vo", "Volume", "volume"),
                source="jquants",
            )
            if point is not None:
                points.append(point)
        return points

    async def _fetch_pages_for_request(
        self,
        *,
        client: httpx.AsyncClient,
        symbol: str,
        code: str,
        headers: dict[str, str],
        request_start: str | None,
        request_end: str | None,
        adjusted_to_coverage: bool,
        rate_limit_attempts: int,
    ) -> JQuantsPageFetchResult:
        points: list[dict[str, Any]] = []
        pagination_key: str | None = None

        while True:
            request_params = self._build_request_params(
                code=code,
                start_date=request_start,
                end_date=request_end,
                pagination_key=pagination_key,
            )
            response, payload = await self._fetch_page(
                client=client,
                symbol=symbol,
                headers=headers,
                request_params=request_params,
            )
            if response is None:
                return self._page_fetch_failure(
                    request_start=request_start,
                    request_end=request_end,
                    adjusted_to_coverage=adjusted_to_coverage,
                    rate_limit_attempts=rate_limit_attempts,
                )

            if response.status_code >= 400:
                return await self._page_fetch_error_result(
                    payload=payload,
                    symbol=symbol,
                    request_start=request_start,
                    request_end=request_end,
                    adjusted_to_coverage=adjusted_to_coverage,
                    rate_limit_attempts=rate_limit_attempts,
                )

            values = self._extract_daily_quote_values(payload)
            if not isinstance(values, list):
                return self._page_fetch_failure(
                    request_start=request_start,
                    request_end=request_end,
                    adjusted_to_coverage=adjusted_to_coverage,
                    rate_limit_attempts=rate_limit_attempts,
                )

            points.extend(self._normalize_values(values))

            pagination_key = payload.get("pagination_key") if isinstance(payload, dict) else None
            if not pagination_key:
                return JQuantsPageFetchResult(
                    points=points,
                    retry=False,
                    failed=False,
                    request_start=request_start,
                    request_end=request_end,
                    adjusted_to_coverage=adjusted_to_coverage,
                    rate_limit_attempts=rate_limit_attempts,
                )

    async def _page_fetch_error_result(
        self,
        *,
        payload: Any,
        symbol: str,
        request_start: str | None,
        request_end: str | None,
        adjusted_to_coverage: bool,
        rate_limit_attempts: int,
    ) -> JQuantsPageFetchResult:
        message = self._extract_error_message(payload)
        resolution = await self._resolve_error_response(
            message=message,
            request_start=request_start,
            request_end=request_end,
            adjusted_to_coverage=adjusted_to_coverage,
            rate_limit_attempts=rate_limit_attempts,
        )
        if not resolution.retry:
            LOGGER.warning("J-Quants daily bars API error for %s: %s", symbol, payload)
        return JQuantsPageFetchResult(
            points=[],
            retry=resolution.retry,
            failed=not resolution.retry,
            request_start=resolution.request_start,
            request_end=resolution.request_end,
            adjusted_to_coverage=resolution.adjusted_to_coverage,
            rate_limit_attempts=resolution.rate_limit_attempts,
        )

    @staticmethod
    def _page_fetch_failure(
        *,
        request_start: str | None,
        request_end: str | None,
        adjusted_to_coverage: bool,
        rate_limit_attempts: int,
    ) -> JQuantsPageFetchResult:
        return JQuantsPageFetchResult(
            points=[],
            retry=False,
            failed=True,
            request_start=request_start,
            request_end=request_end,
            adjusted_to_coverage=adjusted_to_coverage,
            rate_limit_attempts=rate_limit_attempts,
        )

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
        if not self._is_supported_interval(interval):
            return []

        code = normalize_jquants_code(symbol)
        api_key = str(runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        if not code or not api_key:
            return []

        headers = self._build_headers(api_key)
        request_start, request_end = self._resolve_request_dates(
            start_date=start_date,
            end_date=end_date,
        )
        adjusted_to_coverage = False
        rate_limit_attempts = 0

        while True:
            result = await self._fetch_pages_for_request(
                client=client,
                symbol=symbol,
                code=code,
                headers=headers,
                request_start=request_start,
                request_end=request_end,
                adjusted_to_coverage=adjusted_to_coverage,
                rate_limit_attempts=rate_limit_attempts,
            )
            if result.failed:
                return []
            if result.retry:
                request_start = result.request_start
                request_end = result.request_end
                adjusted_to_coverage = result.adjusted_to_coverage
                rate_limit_attempts = result.rate_limit_attempts
                continue
            return sorted(result.points, key=lambda item: str(item.get("t") or ""))
