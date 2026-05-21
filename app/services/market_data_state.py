"""State, lifecycle, and event-stream mixin for ``MarketDataHub``."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import LOGGER, settings
from ..utils import fallback_interval_seconds, finite_float_or_none
from .market_data_state_support import (
    build_empty_price_row,
    build_price_record,
    build_snapshot_payload,
    build_status_payload,
    iter_fresh_cached_price_rows,
    normalize_price_record,
)


class MarketDataStateMixin:
    @staticmethod
    def _is_cache_fresh(cached_epoch: Any, ttl_sec: int) -> bool:
        parsed_epoch = finite_float_or_none(cached_epoch)
        return parsed_epoch is not None and (time.time() - parsed_epoch) <= ttl_sec

    @staticmethod
    def _build_price_record(
        symbol: str,
        price: Any,
        source: str,
        timestamp: Any = None,
        source_detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_price_record(
            symbol=symbol,
            price=price,
            source=source,
            timestamp=timestamp,
            source_detail=source_detail,
        )

    async def _store_and_publish_price(self, record: dict[str, Any]) -> None:
        normalized_record = normalize_price_record(record)
        if normalized_record is None:
            return
        symbol, normalized = normalized_record
        async with self._state_lock:
            self.prices[symbol] = normalized
        await self.last_price_store.upsert(normalized)
        await self.publish({"type": "price", "data": normalized})

    async def start(self) -> None:
        await self._hydrate_prices_from_store(self.symbols)
        realtime_enabled = settings.storage.auto_refresh_on_startup or settings.storage.realtime_on_market_open
        if not realtime_enabled and not settings.storage.eod_cache_auto_refresh:
            await self._set_mode("cached-only", False)
            return

        self._worker_tasks = []
        if realtime_enabled:
            self._worker_tasks.extend(
                [
                    asyncio.create_task(self._websocket_worker(), name="ws-worker"),
                    asyncio.create_task(self._fallback_rest_worker(), name="rest-fallback-worker"),
                ]
            )
        else:
            await self._set_mode("cached-only", False)

        if settings.storage.eod_cache_auto_refresh:
            self._worker_tasks.append(asyncio.create_task(self._eod_cache_worker(), name="eod-cache-worker"))

        if settings.storage.auto_refresh_on_startup and self._uses_twelvedata():
            try:
                await self.refresh_api_credits()
            except Exception as exc:
                LOGGER.warning("Failed to initialize daily credits from /api_usage: %s", exc)

    async def stop(self) -> None:
        self._stop_event.set()
        self._restart_ws_event.set()
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        await self.last_price_store.flush(force=True)

    def register_listener(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._listeners.add(queue)
        return queue

    def unregister_listener(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._listeners.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._listeners):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def set_symbols(self, new_symbols: list[str]) -> None:
        max_symbols = settings.provider.max_basic_symbols
        if not new_symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")
        if len(new_symbols) > max_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Basic plan supports up to {max_symbols} symbols for websocket streaming.",
            )

        async with self._state_lock:
            self.symbols = new_symbols

        ui_state_store = getattr(self, "ui_state_store", None)
        if ui_state_store is not None:
            try:
                ui_state_store.set_symbols(new_symbols)
            except Exception as exc:
                LOGGER.warning("Failed to persist symbols: %s", exc)

        await self._hydrate_prices_from_store(new_symbols)
        self._restart_ws_event.set()
        rows = await self.current_rows(new_symbols)
        await self.publish(
            {
                "type": "symbols",
                "data": {
                    "symbols": self.symbols,
                    "poll_interval_sec": fallback_interval_seconds(len(self.symbols)),
                    "rows": rows,
                },
            }
        )

    async def status_payload(self) -> dict[str, Any]:
        open_symbols = self._open_symbols(self.symbols)
        return build_status_payload(
            provider=self.provider,
            mode=self.mode,
            ws_connected=self.ws_connected,
            last_ws_message_at=self.last_ws_message_at,
            symbols=self.symbols,
            open_symbols=open_symbols,
            fallback_poll_interval_sec=fallback_interval_seconds(len(self.symbols)),
            daily_credits_left=self.daily_credits_left,
            daily_credits_used=self.daily_credits_used,
            daily_credits_limit=self.daily_credits_limit,
            daily_credits_updated_at=self.daily_credits_updated_at,
            daily_credits_source=self.daily_credits_source,
            daily_credits_is_estimated=self.daily_credits_is_estimated,
        )

    async def snapshot_payload(self) -> dict[str, Any]:
        rows = await self.current_rows()
        return build_snapshot_payload(status=await self.status_payload(), rows=rows)

    async def current_rows(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        async with self._state_lock:
            target_symbols = list(symbols) if symbols is not None else list(self.symbols)
            for symbol in target_symbols:
                row = self.prices.get(symbol)
                if row:
                    rows.append(dict(row))
                else:
                    rows.append(build_empty_price_row(symbol))
        return rows

    async def _hydrate_prices_from_store(self, symbols: list[str]) -> None:
        if not symbols:
            return
        now = datetime.now(timezone.utc).timestamp()
        async with self._state_lock:
            hydrated_rows = iter_fresh_cached_price_rows(
                symbols=symbols,
                prices=self.prices,
                last_price_store=self.last_price_store,
                now_epoch=now,
                logger=LOGGER,
            )
            for symbol, row in hydrated_rows:
                self.prices[symbol] = row

    def _eod_cache_target_symbols(self) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        for symbol in list(self.symbols):
            normalized = str(symbol or "").strip().upper()
            if not normalized or normalized in seen:
                continue
            if self._resolve_symbol_country_key(normalized) != "UNITED STATES":
                continue
            seen.add(normalized)
            targets.append(normalized)
        return targets

    def _eligible_us_eod_session_date(self, now_utc: datetime | None = None) -> date | None:
        session = self.market_sessions.get("UNITED STATES")
        if session is None:
            return (now_utc or datetime.now(timezone.utc)).date()

        now = now_utc or datetime.now(timezone.utc)
        local_now = now.astimezone(session.tz)
        current_minutes = (local_now.hour * 60) + local_now.minute
        delay_minutes = settings.storage.eod_cache_refresh_delay_min
        ready_minutes = session.close_minutes + delay_minutes
        is_session_weekday = local_now.weekday() in session.weekdays

        if is_session_weekday and current_minutes >= ready_minutes:
            return local_now.date()
        if is_session_weekday and current_minutes < session.open_minutes:
            return self._previous_us_session_date(local_now.date())
        if not is_session_weekday:
            return self._previous_us_session_date(local_now.date())
        return None

    def _previous_us_session_date(self, current_date: date) -> date:
        session = self.market_sessions.get("UNITED STATES")
        weekdays = session.weekdays if session is not None else frozenset({0, 1, 2, 3, 4})
        candidate = current_date - timedelta(days=1)
        while candidate.weekday() not in weekdays:
            candidate -= timedelta(days=1)
        return candidate

    async def _eod_cache_worker(self) -> None:
        refreshed_by_symbol: dict[str, date] = {}
        while not self._stop_event.is_set():
            try:
                session_date = self._eligible_us_eod_session_date()
                if session_date is not None:
                    targets: list[str] = []
                    for symbol in self._eod_cache_target_symbols():
                        if refreshed_by_symbol.get(symbol) == session_date:
                            continue
                        if await self._cached_eod_session_is_present(symbol, session_date):
                            refreshed_by_symbol[symbol] = session_date
                            continue
                        targets.append(symbol)
                    refreshed_symbols = await self.refresh_eod_cache_for_symbols(targets)
                    for symbol in refreshed_symbols:
                        refreshed_by_symbol[symbol] = session_date
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("EOD cache worker failed: %s", exc)
            await asyncio.sleep(settings.storage.eod_cache_refresh_check_sec)

    async def _cached_eod_session_is_present(self, symbol: str, session_date: date) -> bool:
        try:
            points = await self.full_daily_history_store.get(symbol, copy=True)
        except Exception as exc:
            LOGGER.warning("EOD cache read failed for %s: %s", symbol, exc)
            return False

        for point in points:
            if not isinstance(point, dict) or self._point_date(point) != session_date:
                continue
            close_value = finite_float_or_none(point.get("c"), minimum=0.0, strict_minimum=True)
            if close_value is not None:
                return True
        return False

    async def refresh_eod_cache_for_symbols(self, symbols: list[str]) -> list[str]:
        targets = [symbol for symbol in symbols if symbol]
        if not targets:
            return []

        refreshed_symbols: list[str] = []
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for index, symbol in enumerate(targets):
                if self._stop_event.is_set():
                    break
                try:
                    points = await self._fetch_series(
                        client=client,
                        symbol=symbol,
                        interval="1day",
                        outputsize=max(settings.overview.sparkline_points + 2, 64),
                    )
                except Exception as exc:
                    LOGGER.warning("EOD cache refresh failed for %s: %s", symbol, exc)
                    points = []

                if points:
                    cached_points = await self.full_daily_history_store.get(symbol, copy=True)
                    merged_points = self._merge_points_by_timestamp(cached_points, points)
                    await self.full_daily_history_store.upsert(symbol, merged_points)
                    refreshed_symbols.append(symbol)

                if index < len(targets) - 1:
                    await asyncio.sleep(self._eod_cache_request_spacing_seconds())
        return refreshed_symbols

    @staticmethod
    def _eod_cache_request_spacing_seconds() -> int:
        effective_rpm = settings.budget.api_limit_per_min * settings.budget.per_min_limit_utilization
        return max(1, math.ceil(60 / max(0.1, effective_rpm)))
