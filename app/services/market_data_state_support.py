"""Support helpers for market data state lifecycle and payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..utils import epoch_from_iso8601, to_iso8601

MAX_CACHED_PRICE_AGE_SEC = 7 * 24 * 3600


def build_price_record(
    *,
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


def normalize_price_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    symbol = str(record.get("symbol", "")).upper().strip()
    if not symbol:
        return None
    normalized = dict(record)
    normalized["symbol"] = symbol
    return symbol, normalized


def build_empty_price_row(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "price": None,
        "timestamp": None,
        "source": None,
    }


def build_status_payload(
    *,
    provider: str,
    mode: str,
    ws_connected: bool,
    last_ws_message_at: float,
    symbols: list[str],
    open_symbols: list[str],
    fallback_poll_interval_sec: int,
    daily_credits_left: int | None,
    daily_credits_used: int | None,
    daily_credits_limit: int | None,
    daily_credits_updated_at: str | None,
    daily_credits_source: str | None,
    daily_credits_is_estimated: bool,
) -> dict[str, Any]:
    last_seen = None
    if last_ws_message_at:
        last_seen = datetime.fromtimestamp(last_ws_message_at, tz=timezone.utc).isoformat()
    return {
        "provider": provider,
        "mode": mode,
        "ws_connected": ws_connected,
        "last_ws_message_at": last_seen,
        "symbols": symbols,
        "open_symbols": open_symbols,
        "fallback_poll_interval_sec": fallback_poll_interval_sec,
        "daily_credits_left": daily_credits_left,
        "daily_credits_used": daily_credits_used,
        "daily_credits_limit": daily_credits_limit,
        "daily_credits_updated_at": daily_credits_updated_at,
        "daily_credits_source": daily_credits_source,
        "daily_credits_is_estimated": daily_credits_is_estimated,
    }


def build_snapshot_payload(
    *,
    status: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "snapshot",
        "data": {
            "status": status,
            "rows": rows,
        },
    }


def iter_fresh_cached_price_rows(
    *,
    symbols: list[str],
    prices: dict[str, dict[str, Any]],
    last_price_store: Any,
    now_epoch: float,
    logger: Any,
) -> list[tuple[str, dict[str, Any]]]:
    hydrated: list[tuple[str, dict[str, Any]]] = []
    for symbol in symbols:
        if symbol in prices:
            continue
        cached = last_price_store.get(symbol)
        if not cached:
            continue
        raw_ts = cached.get("timestamp")
        if raw_ts:
            ts_epoch = epoch_from_iso8601(raw_ts)
            if ts_epoch is not None:
                if (now_epoch - ts_epoch) > MAX_CACHED_PRICE_AGE_SEC:
                    logger.info(
                        "Skipping stale cached price for %s (age %.0fh > %dh)",
                        symbol,
                        (now_epoch - ts_epoch) / 3600,
                        MAX_CACHED_PRICE_AGE_SEC // 3600,
                    )
                    continue
        hydrated.append(
            (
                symbol,
                {
                    "symbol": symbol,
                    "price": cached.get("price"),
                    "timestamp": to_iso8601(cached.get("timestamp")),
                    "source": "stored",
                },
            )
        )
    return hydrated
