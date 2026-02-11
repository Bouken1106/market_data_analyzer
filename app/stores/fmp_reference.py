"""Persistent cache store for FMP reference/fundamental payloads."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import LOGGER, SYMBOL_PATTERN


class FmpReferenceStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol or "").upper().strip()

    def _path_for(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.json"

    async def get(self, symbol: str) -> dict[str, Any] | None:
        normalized = self._normalize_symbol(symbol)
        if not normalized or not SYMBOL_PATTERN.match(normalized):
            return None
        async with self._lock:
            path = self._path_for(normalized)
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            return payload if isinstance(payload, dict) else None

    async def upsert(self, symbol: str, payload: dict[str, Any]) -> None:
        normalized = self._normalize_symbol(symbol)
        if not normalized or not SYMBOL_PATTERN.match(normalized):
            return
        if not isinstance(payload, dict):
            return
        async with self._lock:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                out = dict(payload)
                out["symbol"] = normalized
                out["cached_at"] = datetime.now(timezone.utc).isoformat()
                self._path_for(normalized).write_text(
                    json.dumps(out, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                LOGGER.warning("Failed to write FMP reference cache for %s: %s", normalized, exc)

    async def clear(self, symbol: str) -> bool:
        normalized = self._normalize_symbol(symbol)
        if not normalized or not SYMBOL_PATTERN.match(normalized):
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
