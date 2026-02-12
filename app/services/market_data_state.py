"""State, lifecycle, and event-stream mixin for ``MarketDataHub``."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ..config import LOGGER, MAX_BASIC_SYMBOLS
from ..utils import fallback_interval_seconds, to_iso8601


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
        record = {
            "symbol": symbol.upper().strip(),
            "price": str(price),
            "timestamp": to_iso8601(timestamp),
            "source": source,
        }
        if isinstance(source_detail, dict) and source_detail:
            record["source_detail"] = dict(source_detail)
        return record

    async def _store_and_publish_price(self, record: dict[str, Any]) -> None:
        symbol = str(record.get("symbol", "")).upper().strip()
        if not symbol:
            return
        normalized = dict(record)
        normalized["symbol"] = symbol
        async with self._state_lock:
            self.prices[symbol] = normalized
        await self.last_price_store.upsert(normalized)
        await self.publish({"type": "price", "data": normalized})

    async def start(self) -> None:
        await self._hydrate_prices_from_store(self.symbols)
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
        if not new_symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")
        if len(new_symbols) > MAX_BASIC_SYMBOLS:
            raise HTTPException(
                status_code=400,
                detail=f"Basic plan supports up to {MAX_BASIC_SYMBOLS} symbols for websocket streaming.",
            )

        async with self._state_lock:
            self.symbols = new_symbols

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
        last_seen = None
        if self.last_ws_message_at:
            last_seen = datetime.fromtimestamp(self.last_ws_message_at, tz=timezone.utc).isoformat()
        open_symbols = self._open_symbols(self.symbols)
        return {
            "provider": self.provider,
            "mode": self.mode,
            "ws_connected": self.ws_connected,
            "last_ws_message_at": last_seen,
            "symbols": self.symbols,
            "open_symbols": open_symbols,
            "fallback_poll_interval_sec": fallback_interval_seconds(len(self.symbols)),
            "daily_credits_left": self.daily_credits_left,
            "daily_credits_used": self.daily_credits_used,
            "daily_credits_limit": self.daily_credits_limit,
            "daily_credits_updated_at": self.daily_credits_updated_at,
            "daily_credits_source": self.daily_credits_source,
            "daily_credits_is_estimated": self.daily_credits_is_estimated,
            # Backward compatibility for older UI field name.
            "api_credits_left": self.daily_credits_left,
        }

    async def snapshot_payload(self) -> dict[str, Any]:
        rows = await self.current_rows()
        return {
            "type": "snapshot",
            "data": {
                "status": await self.status_payload(),
                "rows": rows,
            },
        }

    async def current_rows(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        async with self._state_lock:
            target_symbols = list(symbols) if symbols is not None else list(self.symbols)
            for symbol in target_symbols:
                row = self.prices.get(symbol)
                if row:
                    rows.append(dict(row))
                else:
                    rows.append(
                        {
                            "symbol": symbol,
                            "price": None,
                            "timestamp": None,
                            "source": None,
                        }
                    )
        return rows

    async def _hydrate_prices_from_store(self, symbols: list[str]) -> None:
        if not symbols:
            return
        async with self._state_lock:
            for symbol in symbols:
                if symbol in self.prices:
                    continue
                cached = self.last_price_store.get(symbol)
                if not cached:
                    continue
                self.prices[symbol] = {
                    "symbol": symbol,
                    "price": cached.get("price"),
                    "timestamp": to_iso8601(cached.get("timestamp")),
                    "source": "stored",
                }
