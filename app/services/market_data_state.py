"""State, lifecycle, and event-stream mixin for ``MarketDataHub``."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ..config import LOGGER, settings
from ..utils import fallback_interval_seconds
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
        try:
            return (time.time() - float(cached_epoch)) <= ttl_sec
        except (TypeError, ValueError):
            return False

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
        if not settings.storage.auto_refresh_on_startup:
            await self._set_mode("cached-only", False)
            return

        self._worker_tasks = [
            asyncio.create_task(self._websocket_worker(), name="ws-worker"),
            asyncio.create_task(self._fallback_rest_worker(), name="rest-fallback-worker"),
        ]
        if self._uses_twelvedata():
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
