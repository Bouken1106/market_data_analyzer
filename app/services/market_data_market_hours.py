"""Market-hours helpers shared across state and query services."""

from __future__ import annotations

from datetime import datetime, timezone

from ..market_session import infer_country_from_symbol


def resolve_symbol_country_key(
    symbol: str,
    *,
    symbol_country_map: dict[str, str],
    default_country_key: str,
) -> str:
    normalized_symbol = symbol.upper().strip()
    mapped_country = symbol_country_map.get(normalized_symbol)
    if mapped_country:
        return mapped_country
    inferred_country = infer_country_from_symbol(normalized_symbol)
    if inferred_country:
        return inferred_country
    return default_country_key


def is_country_market_open(country_key: str, *, market_sessions: dict[str, object], now_utc: datetime) -> bool:
    session = market_sessions.get(country_key)
    if session is None:
        return True

    local_now = now_utc.astimezone(session.tz)
    if local_now.weekday() not in session.weekdays:
        return False
    current_minutes = (local_now.hour * 60) + local_now.minute

    if session.open_minutes <= session.close_minutes:
        return session.open_minutes <= current_minutes < session.close_minutes
    return current_minutes >= session.open_minutes or current_minutes < session.close_minutes


def is_symbol_market_open(
    symbol: str,
    *,
    symbol_country_map: dict[str, str],
    default_country_key: str,
    market_sessions: dict[str, object],
    now_utc: datetime | None = None,
) -> bool:
    utc_now = now_utc or datetime.now(timezone.utc)
    country_key = resolve_symbol_country_key(
        symbol,
        symbol_country_map=symbol_country_map,
        default_country_key=default_country_key,
    )
    return is_country_market_open(
        country_key,
        market_sessions=market_sessions,
        now_utc=utc_now,
    )


def open_symbols(
    symbols: list[str],
    *,
    symbol_country_map: dict[str, str],
    default_country_key: str,
    market_sessions: dict[str, object],
) -> list[str]:
    now_utc = datetime.now(timezone.utc)
    return [
        symbol
        for symbol in symbols
        if is_symbol_market_open(
            symbol,
            symbol_country_map=symbol_country_map,
            default_country_key=default_country_key,
            market_sessions=market_sessions,
            now_utc=now_utc,
        )
    ]
