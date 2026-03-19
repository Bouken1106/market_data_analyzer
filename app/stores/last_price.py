"""Persistent store for the last known price of each symbol."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, to_iso8601, utc_now_iso, write_json_file


class LastPriceStore:
    def __init__(self, cache_path: Path, flush_interval_sec: int = 5) -> None:
        self.cache_path = cache_path
        self.flush_interval_sec = max(1, flush_interval_sec)
        self._data: dict[str, dict[str, Any]] = {}
        self._last_flush_at = 0.0
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def get(self, symbol: str) -> dict[str, Any] | None:
        item = self._data.get(normalize_symbol(symbol))
        if not item:
            return None
        return dict(item)

    async def upsert(self, record: dict[str, Any]) -> None:
        symbol = normalize_symbol(record.get("symbol"))
        if not symbol:
            return

        normalized = {
            "symbol": symbol,
            "price": str(record.get("price")) if record.get("price") is not None else None,
            "timestamp": to_iso8601(record.get("timestamp")),
            "source": str(record.get("source") or "unknown"),
        }

        async with self._lock:
            self._data[symbol] = normalized
            now = time.time()
            if (now - self._last_flush_at) >= self.flush_interval_sec:
                self._write_no_lock()
                self._last_flush_at = now

    async def flush(self, force: bool = False) -> None:
        async with self._lock:
            now = time.time()
            if force or (now - self._last_flush_at) >= self.flush_interval_sec:
                self._write_no_lock()
                self._last_flush_at = now

    def _load_from_disk(self) -> None:
        payload = read_json_file(self.cache_path)
        if not isinstance(payload, dict):
            return

        rows = payload.get("prices") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return

        loaded: dict[str, dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            if not is_valid_symbol(symbol):
                continue
            price = item.get("price")
            timestamp = item.get("timestamp")
            source = item.get("source")
            if price is None:
                continue
            loaded[symbol] = {
                "symbol": symbol,
                "price": str(price),
                "timestamp": to_iso8601(timestamp),
                "source": str(source or "stored"),
            }

        self._data = loaded

    def _write_no_lock(self) -> None:
        try:
            payload = {
                "updated_at": utc_now_iso(),
                "prices": sorted(self._data.values(), key=lambda item: item["symbol"]),
            }
            write_json_file(self.cache_path, payload)
        except Exception as exc:
            LOGGER.warning("Failed to write last price cache: %s", exc)
