"""Persistent store for full daily OHLCV history, with disk caching."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import LOGGER, SYMBOL_PATTERN


class FullDailyHistoryStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._memory: dict[str, list[dict[str, Any]]] = {}
        self._updated_at_epoch: dict[str, float] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol or "").upper().strip()

    def _path_for(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.json"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_from_disk_no_lock(self, symbol: str) -> list[dict[str, Any]]:
        path = self._path_for(symbol)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        updated_raw = payload.get("updated_at") if isinstance(payload, dict) else None
        if isinstance(updated_raw, str):
            try:
                parsed = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                self._updated_at_epoch[symbol] = parsed.astimezone(timezone.utc).timestamp()
            except Exception:
                pass
        points = payload.get("points") if isinstance(payload, dict) else None
        if not isinstance(points, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in points:
            if not isinstance(item, dict):
                continue
            t = str(item.get("t", "")).strip()
            c = item.get("c")
            if not t:
                continue
            close = self._to_float(c)
            if close is None or close <= 0:
                continue
            o = self._to_float(item.get("o"))
            h = self._to_float(item.get("h"))
            l = self._to_float(item.get("l"))
            v = self._to_float(item.get("v"))
            open_value = o if o is not None and o > 0 else close
            high_value = h if h is not None and h > 0 else max(open_value, close)
            low_value = l if l is not None and l > 0 else min(open_value, close)
            normalized.append(
                {
                    "t": t,
                    "o": open_value,
                    "h": max(high_value, open_value, close),
                    "l": min(low_value, open_value, close),
                    "c": close,
                    "v": v,
                }
            )
        return normalized

    async def get(self, symbol: str) -> list[dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol or not SYMBOL_PATTERN.match(normalized_symbol):
            return []
        async with self._lock:
            cached = self._memory.get(normalized_symbol)
            if cached is not None:
                return [dict(item) for item in cached]
            loaded = self._load_from_disk_no_lock(normalized_symbol)
            self._memory[normalized_symbol] = loaded
            return [dict(item) for item in loaded]

    async def upsert(self, symbol: str, points: list[dict[str, Any]]) -> None:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol or not SYMBOL_PATTERN.match(normalized_symbol):
            return
        safe_points = [dict(item) for item in points if isinstance(item, dict)]
        async with self._lock:
            self._memory[normalized_symbol] = safe_points
            try:
                now_epoch = time.time()
                self._updated_at_epoch[normalized_symbol] = now_epoch
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "symbol": normalized_symbol,
                    "updated_at": datetime.fromtimestamp(now_epoch, tz=timezone.utc).isoformat(),
                    "count": len(safe_points),
                    "points": safe_points,
                }
                self._path_for(normalized_symbol).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                LOGGER.warning("Failed to write full daily history cache for %s: %s", normalized_symbol, exc)

    async def clear(self, symbol: str | None = None) -> int:
        async with self._lock:
            if symbol:
                normalized_symbol = self._normalize_symbol(symbol)
                self._memory.pop(normalized_symbol, None)
                self._updated_at_epoch.pop(normalized_symbol, None)
                removed = 0
                path = self._path_for(normalized_symbol)
                if path.exists():
                    try:
                        path.unlink()
                        removed = 1
                    except Exception as exc:
                        LOGGER.warning("Failed to remove daily history cache for %s: %s", normalized_symbol, exc)
                return removed

            removed = 0
            self._memory.clear()
            self._updated_at_epoch.clear()
            if not self.cache_dir.exists():
                return 0
            for path in self.cache_dir.glob("*.json"):
                try:
                    path.unlink()
                    removed += 1
                except Exception as exc:
                    LOGGER.warning("Failed to remove daily history cache file %s: %s", path, exc)
            return removed

    async def last_updated_epoch(self, symbol: str) -> float | None:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol or not SYMBOL_PATTERN.match(normalized_symbol):
            return None
        async with self._lock:
            value = self._updated_at_epoch.get(normalized_symbol)
            if value is not None:
                return float(value)
            _ = self._load_from_disk_no_lock(normalized_symbol)
            value = self._updated_at_epoch.get(normalized_symbol)
            return float(value) if value is not None else None
