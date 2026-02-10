"""Cached catalog of available stock symbols from Twelve Data."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    LOGGER,
    STOCKS_LIST_URL,
    SYMBOL_CATALOG_COUNTRY,
    SYMBOL_CATALOG_MAX_ITEMS,
    SYMBOL_PATTERN,
)


class SymbolCatalogStore:
    def __init__(self, api_key: str, cache_path: Path, ttl_sec: int) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self.ttl_sec = ttl_sec
        self._symbols: list[dict[str, str]] = []
        self._updated_at: str | None = None
        self._loaded_from = "none"
        self._loaded_epoch = 0.0
        self._lock = asyncio.Lock()

    async def get_catalog(self, refresh: bool = False) -> dict[str, Any]:
        async with self._lock:
            if not refresh and self._symbols and self._is_memory_fresh():
                return self._payload()

            if not refresh:
                cached = self._load_from_cache(require_fresh=True)
                if cached:
                    self._apply_state(cached["symbols"], cached["updated_at"], source="cache")
                    return self._payload()

            try:
                symbols = await self._fetch_from_api()
                updated_at = datetime.now(timezone.utc).isoformat()
                self._apply_state(symbols, updated_at, source="twelvedata-live")
                self._write_cache()
            except Exception as exc:
                LOGGER.warning("Failed to fetch symbol catalog from Twelve Data: %s", exc)
                cached = self._load_from_cache(require_fresh=False)
                if cached:
                    self._apply_state(cached["symbols"], cached["updated_at"], source="cache-stale")
                elif self._symbols:
                    self._loaded_from = "memory-stale"
                else:
                    if isinstance(exc, HTTPException):
                        raise
                    raise HTTPException(status_code=502, detail="Failed to load symbol catalog.")

            return self._payload()

    def _is_memory_fresh(self) -> bool:
        return (time.time() - self._loaded_epoch) <= self.ttl_sec

    def _apply_state(self, symbols: list[dict[str, str]], updated_at: str, source: str) -> None:
        self._symbols = symbols
        self._updated_at = updated_at
        self._loaded_from = source
        self._loaded_epoch = time.time()

    def _payload(self) -> dict[str, Any]:
        return {
            "source": self._loaded_from,
            "updated_at": self._updated_at,
            "count": len(self._symbols),
            "symbols": self._symbols,
        }

    async def _fetch_from_api(self) -> list[dict[str, str]]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                STOCKS_LIST_URL,
                params={
                    "apikey": self.api_key,
                    "country": SYMBOL_CATALOG_COUNTRY,
                },
            )
            payload = response.json()

        if isinstance(payload, dict) and payload.get("status") == "error":
            message = payload.get("message", "Failed to fetch symbol catalog.")
            raise HTTPException(status_code=400, detail=message)

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise HTTPException(status_code=502, detail="Unexpected symbol catalog format from Twelve Data.")

        seen: set[str] = set()
        symbols: list[dict[str, str]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            if not SYMBOL_PATTERN.match(symbol):
                continue

            seen.add(symbol)
            symbols.append(
                {
                    "symbol": symbol,
                    "name": str(item.get("name", "")).strip(),
                    "exchange": str(item.get("exchange", "")).strip(),
                    "type": str(item.get("type", "")).strip(),
                }
            )

        symbols.sort(key=lambda value: value["symbol"])
        return symbols[:SYMBOL_CATALOG_MAX_ITEMS]

    def _load_from_cache(self, require_fresh: bool) -> dict[str, Any] | None:
        if not self.cache_path.exists():
            return None
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        symbols = raw.get("symbols")
        updated_at = raw.get("updated_at")
        cached_epoch = raw.get("cached_epoch")
        if not isinstance(symbols, list) or not isinstance(updated_at, str):
            return None

        if require_fresh:
            if not isinstance(cached_epoch, (int, float)):
                return None
            if (time.time() - float(cached_epoch)) > self.ttl_sec:
                return None

        normalized: list[dict[str, str]] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            normalized.append(
                {
                    "symbol": symbol,
                    "name": str(item.get("name", "")).strip(),
                    "exchange": str(item.get("exchange", "")).strip(),
                    "type": str(item.get("type", "")).strip(),
                }
            )

        if not normalized:
            return None

        return {
            "symbols": normalized[:SYMBOL_CATALOG_MAX_ITEMS],
            "updated_at": updated_at,
        }

    def _write_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": self._updated_at,
                "cached_epoch": time.time(),
                "symbols": self._symbols,
            }
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to write symbol catalog cache: %s", exc)
