"""Historical-data helpers for MarketData query mixins."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    DAILY_DIFF_MIN_RECHECK_SEC,
    EARLIEST_TIMESTAMP_URL,
    FMP_HISTORICAL_EOD_URL,
    FULL_HISTORY_CHUNK_YEARS,
    FULL_HISTORY_MAX_CHUNKS,
    HISTORICAL_CACHE_TTL_SEC,
    HISTORICAL_DEFAULT_YEARS,
    HISTORICAL_INTERVAL,
    HISTORICAL_MAX_POINTS,
    HISTORICAL_MAX_YEARS,
    JQUANTS_API_KEY as DEFAULT_JQUANTS_API_KEY,
    JQUANTS_DAILY_BARS_URL,
    JQUANTS_MIN_REQUEST_INTERVAL_SEC as DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
    JQUANTS_RATE_LIMIT_BACKOFF_SEC as DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
    LOGGER,
    ML_HISTORY_MAX_MONTHS,
    SYMBOL_PATTERN,
    TIME_SERIES_MAX_OUTPUTSIZE,
    TIME_SERIES_URL,
)
from ..market_session import infer_country_from_symbol
from ..ohlcv import merge_points_by_timestamp as merge_ohlcv_points, normalize_ohlcv_point
from ..stooq import fetch_stooq_daily_history as default_fetch_stooq_daily_history


def _queries_module():
    from . import market_data_queries as module

    return module


def _runtime_value(name: str, default: Any) -> Any:
    return getattr(_queries_module(), name, default)


class MarketDataHistoricalMixin:
    _JQUANTS_COVERAGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})")

    async def historical_payload(
        self,
        symbol: str,
        years: int = HISTORICAL_DEFAULT_YEARS,
        months: int | None = None,
        refresh: bool = False,
        source_preference: str | None = None,
        allow_api_fallback: bool = True,
    ) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        source_mode = str(source_preference or "").strip().lower() or "provider"
        requested_years = max(1, int(years))
        fetch_full_history = months is None and requested_years > HISTORICAL_MAX_YEARS
        years = requested_years if fetch_full_history else max(1, min(requested_years, HISTORICAL_MAX_YEARS))
        months = None if months is None else max(1, min(int(months), ML_HISTORY_MAX_MONTHS))

        if months is None:
            cache_key = (
                (normalized, "years:max", f"source:{source_mode}")
                if fetch_full_history
                else (normalized, f"years:{years}", f"source:{source_mode}")
            )
        else:
            cache_key = (normalized, f"months:{months}", f"source:{source_mode}")
        async with self._historical_lock:
            cached = self._historical_cache.get(cache_key)
            if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), HISTORICAL_CACHE_TTL_SEC):
                payload = dict(cached["payload"])
                payload["source"] = "cache"
                return payload

        end_date = date.today()
        if months is None:
            start_date = end_date - timedelta(days=(365 * years) + (years // 4))
            requested_days = (365 * years) + (years // 4)
        else:
            start_date = end_date - timedelta(days=(31 * months) + 7)
            requested_days = (31 * months) + 7
        estimated_points = max(200, int(requested_days * 0.8))
        outputsize = (
            0
            if fetch_full_history
            else min(
                TIME_SERIES_MAX_OUTPUTSIZE,
                max(HISTORICAL_MAX_POINTS, estimated_points),
            )
        )

        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            use_stooq = (
                months is None
                and str(HISTORICAL_INTERVAL).strip().lower() in {"1day", "1d", "day"}
                and source_mode == "stooq"
            )
            if use_stooq:
                points, source_detail = await self._fetch_stooq_daily_points_with_detail(
                    client,
                    symbol=normalized,
                    outputsize=outputsize,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    refresh=refresh,
                )
                if not points and allow_api_fallback:
                    if fetch_full_history:
                        points = await self._fetch_full_daily_series(client, symbol=normalized, refresh=refresh)
                        source_detail = {
                            "provider": self.provider,
                            "mode": "full_daily_history",
                        }
                    else:
                        points, source_detail = await self._fetch_historical_points_with_detail(
                            client,
                            symbol=normalized,
                            interval=HISTORICAL_INTERVAL,
                            outputsize=max(HISTORICAL_MAX_POINTS, outputsize),
                            start_date=start_date.isoformat(),
                            end_date=end_date.isoformat(),
                        )
            elif fetch_full_history and str(HISTORICAL_INTERVAL).strip().lower() in {"1day", "1d", "day"}:
                points = await self._fetch_full_daily_series(client, symbol=normalized, refresh=refresh)
                source_detail = {
                    "provider": self.provider,
                    "mode": "full_daily_history",
                }
            else:
                points, source_detail = await self._fetch_historical_points_with_detail(
                    client,
                    symbol=normalized,
                    interval=HISTORICAL_INTERVAL,
                    outputsize=outputsize,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )

        if not points:
            raise HTTPException(status_code=404, detail="No historical data found for this symbol.")

        historical_payload = {
            "symbol": normalized,
            "years": years,
            "months": months,
            "interval": HISTORICAL_INTERVAL,
            "from": points[0]["t"],
            "to": points[-1]["t"],
            "count": len(points),
            "points": points,
            "source": (
                "stooq-live"
                if str(source_detail.get("provider") or "").strip().lower() == "stooq"
                else f"{self.provider}-live"
            ),
            "source_detail": source_detail,
        }

        async with self._historical_lock:
            self._historical_cache[cache_key] = {
                "cached_epoch": time.time(),
                "payload": historical_payload,
            }

        return historical_payload

    @staticmethod
    def _slice_daily_points(
        points: list[dict[str, Any]],
        *,
        start_date: str | None,
        end_date: str | None,
        outputsize: int,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for item in points:
            point_date = str(item.get("t") or "").split(" ")[0]
            if not point_date:
                continue
            if start_date and point_date < start_date:
                continue
            if end_date and point_date > end_date:
                continue
            filtered.append(dict(item))
        if outputsize > 0 and len(filtered) > outputsize:
            filtered = filtered[-outputsize:]
        return filtered

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
        cached_full_points = await self.full_daily_history_store.get(symbol, copy=False)
        cached_updated_epoch = await self.full_daily_history_store.last_updated_epoch(symbol)
        if (
            cached_full_points
            and not refresh
            and cached_updated_epoch is not None
            and self._is_cache_fresh(cached_updated_epoch, HISTORICAL_CACHE_TTL_SEC)
        ):
            cached_slice = self._slice_daily_points(
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

        fetch_stooq_daily_history = _runtime_value(
            "fetch_stooq_daily_history",
            default_fetch_stooq_daily_history,
        )
        try:
            full_points = await fetch_stooq_daily_history(symbol, client=client)
        except Exception as exc:
            LOGGER.warning("Stooq daily CSV fetch failed for %s: %s", symbol, exc)
            full_points = []

        if full_points:
            await self.full_daily_history_store.upsert(symbol, full_points)
            fetched_slice = self._slice_daily_points(
                full_points,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
            )
            if fetched_slice:
                return fetched_slice, {
                    "mode": "stooq_live",
                    "dataset": "historical_daily",
                    "provider": "stooq",
                    "points": len(fetched_slice),
                }

        if cached_full_points:
            stale_slice = self._slice_daily_points(
                cached_full_points,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
            )
            if stale_slice:
                return stale_slice, {
                    "mode": "stooq_cached_stale",
                    "dataset": "historical_daily",
                    "provider": "stooq",
                    "points": len(stale_slice),
                }

        return [], {
            "mode": "stooq_empty",
            "dataset": "historical_daily",
            "provider": "stooq",
            "points": 0,
        }

    async def _fetch_historical_points_with_detail(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._should_use_jquants_for_symbol(symbol, interval):
            points = await self._fetch_series_jquants(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            detail = {
                "mode": "jquants",
                "dataset": "historical_daily",
                "provider": "jquants",
                "points": len(points),
            }
            return points, detail

        if self.provider == "both":
            td_task = self._fetch_series_twelvedata(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            fmp_task = self._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            td_points, fmp_points = await asyncio.gather(td_task, fmp_task)
            merged = self._merge_points_by_timestamp(fmp_points, td_points)
            detail = {
                "mode": "both",
                "dataset": "historical_daily",
                "merge_policy": "twelvedata_overrides_fmp_on_same_timestamp",
                "providers": {
                    "twelvedata_points": len(td_points),
                    "fmp_points": len(fmp_points),
                    "merged_points": len(merged),
                },
            }
            return merged, detail

        points = await self._fetch_series(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        if (
            not points
            and self.provider == "twelvedata"
            and self.fmp_api_key
            and str(interval or "").strip().lower() in {"1day", "1d", "day"}
        ):
            fmp_points = await self._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            if fmp_points:
                detail = {
                    "mode": "twelvedata_with_fmp_fallback",
                    "dataset": "historical_daily",
                    "provider": "fmp",
                    "points": len(fmp_points),
                }
                return fmp_points, detail

        provider_name = "fmp" if self.provider == "fmp" else "twelvedata"
        detail = {
            "mode": self.provider,
            "dataset": "historical_daily",
            "provider": provider_name,
            "points": len(points),
        }
        return points, detail

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
        api_key = str(_runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        if not api_key:
            return False
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return False
        return infer_country_from_symbol(symbol) == "JAPAN" and self._normalize_jquants_code(symbol) is not None

    @staticmethod
    def _normalize_jquants_code(symbol: str) -> str | None:
        normalized = str(symbol or "").strip().upper()
        if normalized.endswith(".T"):
            normalized = normalized[:-2]
        normalized = normalized.strip()
        if normalized.isdigit() and len(normalized) in {4, 5}:
            return normalized
        return None

    @classmethod
    def _extract_jquants_coverage_window(cls, message: Any) -> tuple[date, date] | None:
        matched = cls._JQUANTS_COVERAGE_RE.search(str(message or ""))
        if matched is None:
            return None
        try:
            return date.fromisoformat(matched.group(1)), date.fromisoformat(matched.group(2))
        except ValueError:
            return None

    @staticmethod
    def _bound_jquants_request_dates(
        *,
        start_date: str | None,
        end_date: str | None,
        coverage_window: tuple[date, date],
    ) -> tuple[str, str] | None:
        coverage_start, coverage_end = coverage_window
        try:
            requested_start = date.fromisoformat(start_date) if start_date else coverage_start
            requested_end = date.fromisoformat(end_date) if end_date else coverage_end
        except ValueError:
            return None

        bounded_start = max(requested_start, coverage_start)
        bounded_end = min(requested_end, coverage_end)
        if bounded_start > bounded_end:
            return None
        return bounded_start.isoformat(), bounded_end.isoformat()

    @classmethod
    def _clamp_jquants_request_dates(
        cls,
        *,
        start_date: str | None,
        end_date: str | None,
        coverage_message: Any,
    ) -> tuple[str, str] | None:
        coverage_window = cls._extract_jquants_coverage_window(coverage_message)
        if coverage_window is None:
            return None

        bounded_dates = cls._bound_jquants_request_dates(
            start_date=start_date,
            end_date=end_date,
            coverage_window=coverage_window,
        )
        if bounded_dates is None:
            return None

        clamped_start, clamped_end = bounded_dates
        if clamped_start == str(start_date or "") and clamped_end == str(end_date or ""):
            return None
        return clamped_start, clamped_end

    @staticmethod
    def _is_jquants_rate_limit_message(message: Any) -> bool:
        return "rate limit exceeded" in str(message or "").strip().lower()

    async def _await_jquants_request_slot(self) -> None:
        spacing = max(
            0.0,
            float(
                _runtime_value(
                    "JQUANTS_MIN_REQUEST_INTERVAL_SEC",
                    DEFAULT_JQUANTS_MIN_REQUEST_INTERVAL_SEC,
                )
            ),
        )
        if spacing <= 0.0:
            return

        lock = getattr(self, "_jquants_request_lock", None)
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            setattr(self, "_jquants_request_lock", lock)

        async with lock:
            now = time.monotonic()
            next_request_at = float(getattr(self, "_jquants_next_request_at", 0.0) or 0.0)
            if next_request_at > now:
                await asyncio.sleep(next_request_at - now)
                now = time.monotonic()
            setattr(self, "_jquants_next_request_at", now + spacing)

    async def _delay_future_jquants_requests(self, delay_sec: float) -> None:
        lock = getattr(self, "_jquants_request_lock", None)
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            setattr(self, "_jquants_request_lock", lock)

        async with lock:
            now = time.monotonic()
            next_request_at = float(getattr(self, "_jquants_next_request_at", 0.0) or 0.0)
            setattr(self, "_jquants_next_request_at", max(next_request_at, now + max(0.0, delay_sec)))

    async def _fetch_series_jquants(
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

        code = self._normalize_jquants_code(symbol)
        api_key = str(_runtime_value("JQUANTS_API_KEY", DEFAULT_JQUANTS_API_KEY) or "").strip()
        if not code or not api_key:
            return []

        headers = {"x-api-key": api_key}
        cached_coverage = getattr(self, "_jquants_coverage_window", None)
        bounded_dates = None
        if (
            isinstance(cached_coverage, tuple)
            and len(cached_coverage) == 2
            and isinstance(cached_coverage[0], date)
            and isinstance(cached_coverage[1], date)
        ):
            bounded_dates = self._bound_jquants_request_dates(
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
                    await self._await_jquants_request_slot()
                    response = await client.get(JQUANTS_DAILY_BARS_URL, params=request_params, headers=headers)
                    payload = response.json()
                except Exception as exc:
                    LOGGER.warning("J-Quants daily bars fetch failed for %s: %s", symbol, exc)
                    return []

                if response.status_code >= 400:
                    message = payload.get("message") if isinstance(payload, dict) else payload
                    coverage_window = self._extract_jquants_coverage_window(message)
                    if coverage_window is not None:
                        setattr(self, "_jquants_coverage_window", coverage_window)
                    clamped_dates = None
                    if not adjusted_to_coverage:
                        clamped_dates = self._clamp_jquants_request_dates(
                            start_date=request_start,
                            end_date=request_end,
                            coverage_message=message,
                        )
                    if clamped_dates is not None:
                        request_start, request_end = clamped_dates
                        adjusted_to_coverage = True
                        should_retry = True
                        break

                    if self._is_jquants_rate_limit_message(message) and rate_limit_attempts < 3:
                        rate_limit_attempts += 1
                        backoff_sec = float(
                            _runtime_value(
                                "JQUANTS_RATE_LIMIT_BACKOFF_SEC",
                                DEFAULT_JQUANTS_RATE_LIMIT_BACKOFF_SEC,
                            )
                        )
                        await self._delay_future_jquants_requests(backoff_sec * rate_limit_attempts)
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

    async def _fetch_series_twelvedata(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            params: dict[str, Any] = {
                "apikey": self.twelvedata_api_key,
                "symbol": symbol,
                "interval": interval,
                "order": "ASC",
                "outputsize": min(max(1, int(outputsize)), TIME_SERIES_MAX_OUTPUTSIZE),
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            response = await client.get(TIME_SERIES_URL, params=params)
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"series:{symbol}:{interval}")
            payload = response.json()
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

    async def _fetch_series_both(
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
            return await self._fetch_series_twelvedata(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )

        td_task = self._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        fmp_task = self._fetch_series_fmp(
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

        return self._merge_points_by_timestamp(fmp_points, td_points)

    async def _fetch_series_fmp(
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

        params: dict[str, Any] = {
            "apikey": self.fmp_api_key,
            "symbol": symbol,
        }
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        try:
            response = await client.get(FMP_HISTORICAL_EOD_URL, params=params)
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("FMP time series fetch failed for %s %s: %s", symbol, interval, exc)
            return []

        if self._is_fmp_error(payload):
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

    async def _fetch_full_daily_series(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        refresh: bool = False,
        min_recheck_sec: int | None = None,
    ) -> list[dict[str, Any]]:
        today = date.today()
        if refresh:
            await self.full_daily_history_store.clear(symbol)
            cached_points: list[dict[str, Any]] = []
        else:
            cached_points = await self.full_daily_history_store.get(symbol, copy=False)
        if cached_points:
            last_date = self._point_date(cached_points[-1])
            if last_date and last_date >= today:
                return cached_points

            last_cache_update_epoch = await self.full_daily_history_store.last_updated_epoch(symbol)
            if last_cache_update_epoch is not None:
                last_cache_update_dt = datetime.fromtimestamp(last_cache_update_epoch, tz=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                recheck_sec = DAILY_DIFF_MIN_RECHECK_SEC if min_recheck_sec is None else max(60, int(min_recheck_sec))
                if (
                    last_cache_update_dt.date() == now_utc.date()
                    and (now_utc.timestamp() - last_cache_update_epoch) < recheck_sec
                ):
                    return cached_points

            point_groups: list[list[dict[str, Any]]] = [cached_points]
            start_cursor = (last_date - timedelta(days=5)) if last_date else (today - timedelta(days=10))
            chunks = 0
            while start_cursor <= today and chunks < FULL_HISTORY_MAX_CHUNKS:
                chunk_end = min(
                    today,
                    start_cursor + timedelta(days=(366 * FULL_HISTORY_CHUNK_YEARS) - 1),
                )
                incremental_points = await self._fetch_series(
                    client,
                    symbol=symbol,
                    interval="1day",
                    outputsize=TIME_SERIES_MAX_OUTPUTSIZE,
                    start_date=start_cursor.isoformat(),
                    end_date=chunk_end.isoformat(),
                )
                if incremental_points:
                    point_groups.append(incremental_points)
                start_cursor = chunk_end + timedelta(days=1)
                chunks += 1

            if start_cursor <= today:
                LOGGER.warning(
                    "Daily cache catch-up truncated for %s: reached chunk limit (%s).",
                    symbol,
                    FULL_HISTORY_MAX_CHUNKS,
                )

            merged_cached = merge_ohlcv_points(*point_groups)
            await self.full_daily_history_store.upsert(symbol, merged_cached)
            return merged_cached

        fallback_points = await self._fetch_series(
            client,
            symbol=symbol,
            interval="1day",
            outputsize=max(1300, HISTORICAL_MAX_POINTS),
        )
        if self._should_use_jquants_for_symbol(symbol, "1day"):
            if fallback_points:
                await self.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        earliest = await self._fetch_earliest_date(client, symbol=symbol, interval="1day")
        if earliest is None:
            if fallback_points:
                await self.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        start_cursor = earliest
        chunks = 0
        merged: list[dict[str, Any]] = []

        while start_cursor <= today and chunks < FULL_HISTORY_MAX_CHUNKS:
            chunk_end = min(
                today,
                start_cursor + timedelta(days=(366 * FULL_HISTORY_CHUNK_YEARS) - 1),
            )
            points = await self._fetch_series(
                client,
                symbol=symbol,
                interval="1day",
                outputsize=TIME_SERIES_MAX_OUTPUTSIZE,
                start_date=start_cursor.isoformat(),
                end_date=chunk_end.isoformat(),
            )
            if points:
                merged.extend(points)
            start_cursor = chunk_end + timedelta(days=1)
            chunks += 1

        if not merged:
            if fallback_points:
                await self.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        if start_cursor <= today:
            LOGGER.warning(
                "Daily full history truncated for %s: reached chunk limit (%s).",
                symbol,
                FULL_HISTORY_MAX_CHUNKS,
            )

        deduped = merge_ohlcv_points(merged)
        await self.full_daily_history_store.upsert(symbol, deduped)
        return deduped

    async def _fetch_earliest_date(
        self,
        client: httpx.AsyncClient | None,
        symbol: str,
        interval: str,
    ) -> date | None:
        if not self._uses_twelvedata() or client is None:
            return None
        try:
            response = await client.get(
                EARLIEST_TIMESTAMP_URL,
                params={
                    "apikey": self.twelvedata_api_key,
                    "symbol": symbol,
                    "interval": interval,
                },
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"earliest:{symbol}:{interval}")
            payload = response.json()
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

        parsed_iso = self._parse_timestamp(raw_value)
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
