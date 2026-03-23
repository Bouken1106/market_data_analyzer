"""Helpers for loading public daily OHLCV CSVs from Stooq."""

from __future__ import annotations

import csv
import io
from typing import Any

import httpx

from .ohlcv import normalize_ohlcv_point
from .utils import normalize_symbol

_STOOQ_TIMEOUT_SEC = 25.0
_STOOQ_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
}


def resolve_stooq_daily_symbol(symbol: str) -> str | None:
    normalized = normalize_symbol(symbol)
    if not normalized:
        return None

    if normalized.endswith(".T") or normalized.endswith(".JP"):
        base = normalized.rsplit(".", 1)[0].strip()
        if base.isdigit() and len(base) in {4, 5}:
            return f"{base.lower()}.jp"
        return None

    if normalized.endswith(".US"):
        base = normalized[:-3].strip()
        return f"{base.lower()}.us" if base else None

    if normalized.isdigit() and len(normalized) in {4, 5}:
        return f"{normalized.lower()}.jp"

    return f"{normalized.lower()}.us"


def parse_stooq_daily_csv(payload: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(str(payload or "")))
    points: list[dict[str, Any]] = []
    for row in reader:
        point = normalize_ohlcv_point(
            row,
            timestamp_keys=("Date",),
            open_keys=("Open",),
            high_keys=("High",),
            low_keys=("Low",),
            close_keys=("Close",),
            volume_keys=("Volume",),
            source="stooq",
        )
        if point is None:
            continue
        if point.get("v") is None:
            point["v"] = 0.0
        points.append(point)
    points.sort(key=lambda item: str(item.get("t") or ""))
    return points


async def fetch_stooq_daily_history(
    symbol: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_sec: float = _STOOQ_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    resolved_symbol = resolve_stooq_daily_symbol(symbol)
    if not resolved_symbol:
        return []

    url = f"https://stooq.com/q/d/l/?s={resolved_symbol}&i=d"
    if client is None:
        async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as owned_client:
            response = await owned_client.get(url, headers=_STOOQ_HEADERS)
    else:
        response = await client.get(url, headers=_STOOQ_HEADERS)
    response.raise_for_status()
    return parse_stooq_daily_csv(response.text)
