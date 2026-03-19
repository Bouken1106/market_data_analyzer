"""Cached catalog of available stock symbols from configured data provider."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    FMP_STOCK_LIST_LEGACY_URL,
    FMP_STOCK_LIST_URL,
    LOGGER,
    STOCKS_LIST_URL,
    SYMBOL_CATALOG_COUNTRY,
    SYMBOL_CATALOG_MAX_ITEMS,
)
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, write_json_file


class SymbolCatalogStore:
    def __init__(
        self,
        provider: str,
        twelvedata_api_key: str,
        fmp_api_key: str,
        cache_path: Path,
        ttl_sec: int,
    ) -> None:
        self.provider = str(provider or "twelvedata").strip().lower()
        self.twelvedata_api_key = str(twelvedata_api_key or "").strip()
        self.fmp_api_key = str(fmp_api_key or "").strip()
        self.cache_path = cache_path
        self.ttl_sec = ttl_sec
        self._symbols: list[dict[str, str]] = []
        self._updated_at: str | None = None
        self._loaded_from = "none"
        self._loaded_epoch = 0.0
        self._lock = asyncio.Lock()

    async def get_catalog(self, refresh: bool = False, cache_only: bool = False) -> dict[str, Any]:
        async with self._lock:
            if cache_only:
                if self._symbols:
                    self._loaded_from = "memory-cache"
                    return self._payload()
                cached = self._load_from_cache(require_fresh=False)
                if cached:
                    self._apply_state(cached["symbols"], cached["updated_at"], source="cache-only")
                    return self._payload()
                self._symbols = []
                self._updated_at = None
                self._loaded_from = "cache-miss"
                self._loaded_epoch = time.time()
                return self._payload()

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
                self._apply_state(symbols, updated_at, source=f"{self.provider}-live")
                self._write_cache()
            except Exception as exc:
                LOGGER.warning("Failed to fetch symbol catalog from %s: %s", self.provider, exc)
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

    @staticmethod
    def _catalog_row(
        *,
        symbol: Any,
        name: Any = "",
        exchange: Any = "",
        security_type: Any = "",
    ) -> dict[str, str] | None:
        normalized_symbol = normalize_symbol(symbol)
        if not is_valid_symbol(normalized_symbol):
            return None
        return {
            "symbol": normalized_symbol,
            "name": str(name or "").strip(),
            "exchange": str(exchange or "").strip(),
            "type": str(security_type or "").strip(),
        }

    async def _fetch_from_api(self) -> list[dict[str, str]]:
        if self.provider == "both":
            td_task = self._fetch_from_twelvedata_api()
            fmp_task = self._fetch_from_fmp_api()
            td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
            if isinstance(td_result, Exception):
                LOGGER.warning("Symbol catalog fetch failed (TD): %s", td_result)
            if isinstance(fmp_result, Exception):
                LOGGER.warning("Symbol catalog fetch failed (FMP): %s", fmp_result)
            if isinstance(td_result, Exception) and isinstance(fmp_result, Exception):
                if isinstance(td_result, HTTPException):
                    raise td_result
                if isinstance(fmp_result, HTTPException):
                    raise fmp_result
                raise HTTPException(status_code=502, detail="Failed to fetch symbol catalog from both providers.")
            td_rows = td_result if isinstance(td_result, list) else []
            fmp_rows = fmp_result if isinstance(fmp_result, list) else []
            merged = self._merge_catalog_rows(td_rows, fmp_rows)
            if merged:
                return merged[:SYMBOL_CATALOG_MAX_ITEMS]
            raise HTTPException(status_code=502, detail="Failed to fetch symbol catalog from both providers.")
        if self.provider == "fmp":
            return await self._fetch_from_fmp_api()
        return await self._fetch_from_twelvedata_api()

    async def _fetch_from_twelvedata_api(self) -> list[dict[str, str]]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                STOCKS_LIST_URL,
                params={
                    "apikey": self.twelvedata_api_key,
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
            row = self._catalog_row(
                symbol=item.get("symbol"),
                name=item.get("name"),
                exchange=item.get("exchange"),
                security_type=item.get("type"),
            )
            if row is None or row["symbol"] in seen:
                continue
            seen.add(row["symbol"])
            symbols.append(row)

        symbols.sort(key=lambda value: value["symbol"])
        return symbols[:SYMBOL_CATALOG_MAX_ITEMS]

    async def _fetch_from_fmp_api(self) -> list[dict[str, str]]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(FMP_STOCK_LIST_URL, params={"apikey": self.fmp_api_key})
            payload = response.json()
            if self._is_fmp_error_payload(payload):
                legacy_response = await client.get(FMP_STOCK_LIST_LEGACY_URL, params={"apikey": self.fmp_api_key})
                payload = legacy_response.json()

        rows = payload if isinstance(payload, list) else payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise HTTPException(status_code=502, detail="Unexpected symbol catalog format from FMP.")

        seen: set[str] = set()
        symbols: list[dict[str, str]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue

            row = self._catalog_row(
                symbol=item.get("symbol"),
                name=item.get("name"),
                exchange=item.get("exchangeShortName") or item.get("exchange"),
                security_type=item.get("type"),
            )
            if row is None or row["symbol"] in seen:
                continue
            if not self._is_us_equity_exchange(row["exchange"]):
                continue

            seen.add(row["symbol"])
            symbols.append(row)

        symbols.sort(key=lambda value: value["symbol"])
        return symbols[:SYMBOL_CATALOG_MAX_ITEMS]

    @staticmethod
    def _is_fmp_error_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("status") == "error":
            return True
        message = str(payload.get("Error Message", "")).strip().lower()
        return bool(message)

    @staticmethod
    def _is_us_equity_exchange(exchange: str) -> bool:
        code = str(exchange or "").strip().upper()
        if not code:
            return False
        return code in {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}

    @staticmethod
    def _merge_catalog_rows(
        primary_rows: list[dict[str, str]],
        secondary_rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        merged: dict[str, dict[str, str]] = {}
        for row in secondary_rows:
            normalized_row = SymbolCatalogStore._catalog_row(
                symbol=row.get("symbol"),
                name=row.get("name"),
                exchange=row.get("exchange"),
                security_type=row.get("type"),
            )
            if normalized_row is None:
                continue
            merged[normalized_row["symbol"]] = normalized_row
        for row in primary_rows:
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            base = merged.get(symbol, {})
            normalized_row = SymbolCatalogStore._catalog_row(
                symbol=symbol,
                name=row.get("name") or base.get("name"),
                exchange=row.get("exchange") or base.get("exchange"),
                security_type=row.get("type") or base.get("type"),
            )
            if normalized_row is None:
                continue
            merged[symbol] = normalized_row
        out = [merged[key] for key in sorted(merged.keys())]
        return out

    def _load_from_cache(self, require_fresh: bool) -> dict[str, Any] | None:
        raw = read_json_file(self.cache_path)
        if not isinstance(raw, dict):
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
            row = self._catalog_row(
                symbol=item.get("symbol"),
                name=item.get("name"),
                exchange=item.get("exchange"),
                security_type=item.get("type"),
            )
            if row is None:
                continue
            normalized.append(row)

        if not normalized:
            return None

        return {
            "symbols": normalized[:SYMBOL_CATALOG_MAX_ITEMS],
            "updated_at": updated_at,
        }

    def _write_cache(self) -> None:
        try:
            payload = {
                "updated_at": self._updated_at,
                "cached_epoch": time.time(),
                "symbols": self._symbols,
            }
            write_json_file(self.cache_path, payload)
        except Exception as exc:
            LOGGER.warning("Failed to write symbol catalog cache: %s", exc)
