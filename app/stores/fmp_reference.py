"""Persistent cache store for FMP reference/fundamental payloads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, utc_now_iso, write_json_file


class FmpReferenceStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._lock = asyncio.Lock()

    def _path_for(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.json"

    async def get(self, symbol: str) -> dict[str, Any] | None:
        normalized = normalize_symbol(symbol)
        if not is_valid_symbol(normalized):
            return None
        async with self._lock:
            payload = read_json_file(self._path_for(normalized))
            return payload if isinstance(payload, dict) else None

    async def upsert(self, symbol: str, payload: dict[str, Any]) -> None:
        normalized = normalize_symbol(symbol)
        if not is_valid_symbol(normalized):
            return
        if not isinstance(payload, dict):
            return
        async with self._lock:
            try:
                out = dict(payload)
                out["symbol"] = normalized
                out["cached_at"] = utc_now_iso()
                write_json_file(self._path_for(normalized), out)
            except Exception as exc:
                LOGGER.warning("Failed to write FMP reference cache for %s: %s", normalized, exc)

    async def clear(self, symbol: str) -> bool:
        normalized = normalize_symbol(symbol)
        if not is_valid_symbol(normalized):
            return False
        async with self._lock:
            path = self._path_for(normalized)
            if not path.exists():
                return False
            try:
                path.unlink()
                return True
            except Exception as exc:
                LOGGER.warning("Failed to remove FMP reference cache for %s: %s", normalized, exc)
                return False
