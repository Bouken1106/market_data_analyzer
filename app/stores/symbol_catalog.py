"""Cached catalog of available stock symbols from configured data provider."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..config import LOGGER
from ..services.ttl_cache import ttl_cache_is_fresh
from ..utils import read_json_file, write_json_file
from .symbol_catalog_sources import (
    SymbolCatalogRowNormalizer,
    build_symbol_catalog_fetcher,
)


class SymbolCatalogStore:
    def __init__(
        self,
        provider: str,
        twelvedata_api_key: str,
        fmp_api_key: str,
        cache_path: Path,
        ttl_sec: int,
        *,
        country: str,
        max_items: int,
    ) -> None:
        self.provider = str(provider or "twelvedata").strip().lower()
        self.twelvedata_api_key = str(twelvedata_api_key or "").strip()
        self.fmp_api_key = str(fmp_api_key or "").strip()
        self.cache_path = cache_path
        self.ttl_sec = ttl_sec
        self.country = str(country or "").strip()
        self.max_items = max(1, int(max_items))
        self._row_normalizer = SymbolCatalogRowNormalizer(max_items=self.max_items, country=self.country)
        self._fetcher = build_symbol_catalog_fetcher(
            provider=self.provider,
            twelvedata_api_key=self.twelvedata_api_key,
            fmp_api_key=self.fmp_api_key,
            country=self.country,
            max_items=self.max_items,
        )
        self._symbols: list[dict[str, str]] = []
        self._updated_at: str | None = None
        self._loaded_from = "none"
        self._loaded_epoch = 0.0
        self._lock = asyncio.Lock()
        self._country_stores: dict[str, SymbolCatalogStore] = {}

    async def get_catalog(
        self,
        refresh: bool = False,
        cache_only: bool = False,
        *,
        country: str | None = None,
    ) -> dict[str, Any]:
        resolved_country = str(country or self.country or "").strip() or self.country
        if resolved_country != self.country:
            delegated_store = self._country_stores.get(resolved_country)
            if delegated_store is None:
                delegated_store = SymbolCatalogStore(
                    provider=self.provider,
                    twelvedata_api_key=self.twelvedata_api_key,
                    fmp_api_key=self.fmp_api_key,
                    cache_path=self._cache_path_for_country(resolved_country),
                    ttl_sec=self.ttl_sec,
                    country=resolved_country,
                    max_items=self.max_items,
                )
                self._country_stores[resolved_country] = delegated_store
            return await delegated_store.get_catalog(refresh=refresh, cache_only=cache_only)

        return await self._get_catalog_for_bound_country(refresh=refresh, cache_only=cache_only)

    async def _get_catalog_for_bound_country(self, refresh: bool = False, cache_only: bool = False) -> dict[str, Any]:
        async with self._lock:
            if cache_only:
                if self._try_use_cache_only_payload():
                    return self._payload()
                self._mark_cache_miss()
                return self._payload()

            if not refresh and self._symbols and self._is_memory_fresh():
                return self._payload()

            if not refresh and self._try_apply_cached_payload(require_fresh=True, source="cache"):
                return self._payload()

            try:
                await self._refresh_live_catalog()
            except Exception as exc:
                LOGGER.warning("Failed to fetch symbol catalog from %s: %s", self.provider, exc)
                if self._try_apply_cached_payload(require_fresh=False, source="cache-stale"):
                    pass
                elif self._symbols:
                    self._loaded_from = "memory-stale"
                else:
                    if isinstance(exc, HTTPException):
                        raise
                    raise HTTPException(status_code=502, detail="Failed to load symbol catalog.")

            return self._payload()

    def _cache_path_for_country(self, country: str) -> Path:
        normalized_country = re.sub(r"[^a-z0-9]+", "_", str(country or "").strip().lower()).strip("_") or "default"
        default_country = re.sub(r"[^a-z0-9]+", "_", str(self.country or "").strip().lower()).strip("_") or "default"
        if normalized_country == default_country:
            return self.cache_path
        return self.cache_path.with_name(f"{self.cache_path.stem}_{normalized_country}{self.cache_path.suffix}")

    def _is_memory_fresh(self) -> bool:
        return ttl_cache_is_fresh(self._loaded_epoch, self.ttl_sec)

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

    def _mark_cache_miss(self) -> None:
        self._symbols = []
        self._updated_at = None
        self._loaded_from = "cache-miss"
        self._loaded_epoch = time.time()

    def _try_use_cache_only_payload(self) -> bool:
        if self._symbols:
            self._loaded_from = "memory-cache"
            return True
        return self._try_apply_cached_payload(require_fresh=False, source="cache-only")

    def _try_apply_cached_payload(self, *, require_fresh: bool, source: str) -> bool:
        cached = self._load_from_cache(require_fresh=require_fresh)
        if not cached:
            return False
        self._apply_state(cached["symbols"], cached["updated_at"], source=source)
        return True

    async def _refresh_live_catalog(self) -> None:
        symbols = await self._fetch_from_api()
        updated_at = datetime.now(timezone.utc).isoformat()
        self._apply_state(symbols, updated_at, source=f"{self.provider}-live")
        self._write_cache()

    async def _fetch_from_api(self) -> list[dict[str, str]]:
        return await self._fetcher.fetch()

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
            if not ttl_cache_is_fresh(cached_epoch, self.ttl_sec):
                return None

        normalized: list[dict[str, str]] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            row = self._row_normalizer.build_row(
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
            "symbols": normalized[: self.max_items],
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
