"""Market session helpers — country/timezone mapping and symbol inference."""

from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import SYMBOL_CATALOG_COUNTRY, SYMBOL_PATTERN
from .models import MarketSession


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_country_key(value: str) -> str:
    return " ".join(value.upper().split())


def _hhmm_to_minutes(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value: {value}")
    return hour * 60 + minute


def _build_default_market_sessions() -> dict[str, MarketSession]:
    # Regular sessions only (no holidays, no lunch breaks).
    definitions: list[tuple[str, str, str, str]] = [
        ("United States", "America/New_York", "09:30", "16:00"),
        ("Canada", "America/Toronto", "09:30", "16:00"),
        ("United Kingdom", "Europe/London", "08:00", "16:30"),
        ("Germany", "Europe/Berlin", "09:00", "17:30"),
        ("France", "Europe/Paris", "09:00", "17:30"),
        ("Japan", "Asia/Tokyo", "09:00", "15:00"),
        ("Hong Kong", "Asia/Hong_Kong", "09:30", "16:00"),
        ("Singapore", "Asia/Singapore", "09:00", "17:00"),
        ("India", "Asia/Kolkata", "09:15", "15:30"),
        ("Australia", "Australia/Sydney", "10:00", "16:00"),
        ("South Korea", "Asia/Seoul", "09:00", "15:30"),
        ("Taiwan", "Asia/Taipei", "09:00", "13:30"),
        ("China", "Asia/Shanghai", "09:30", "15:00"),
    ]
    weekdays = frozenset({0, 1, 2, 3, 4})
    sessions: dict[str, MarketSession] = {}
    for country, tz_name, open_at, close_at in definitions:
        try:
            tzinfo = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            tzinfo = ZoneInfo("UTC")
        sessions[_normalize_country_key(country)] = MarketSession(
            tz=tzinfo,
            open_minutes=_hhmm_to_minutes(open_at),
            close_minutes=_hhmm_to_minutes(close_at),
            weekdays=weekdays,
        )
    return sessions


DEFAULT_MARKET_SESSIONS = _build_default_market_sessions()

DEFAULT_COUNTRY_KEY = _normalize_country_key(SYMBOL_CATALOG_COUNTRY)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def parse_symbol_country_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        symbol_raw, country_raw = item.split(":", 1)
        symbol = symbol_raw.strip().upper()
        country = country_raw.strip()
        if not symbol or not country:
            continue
        if not SYMBOL_PATTERN.match(symbol):
            continue
        mapping[symbol] = _normalize_country_key(country)
    return mapping


def infer_country_from_symbol(symbol: str) -> str | None:
    suffix_map: list[tuple[str, str]] = [
        (".T", "JAPAN"),
        (".HK", "HONG KONG"),
        (".L", "UNITED KINGDOM"),
        (".PA", "FRANCE"),
        (".F", "GERMANY"),
        (".DE", "GERMANY"),
        (".TO", "CANADA"),
        (".AX", "AUSTRALIA"),
        (".NS", "INDIA"),
        (".BO", "INDIA"),
        (".SS", "CHINA"),
        (".SZ", "CHINA"),
        (".KS", "SOUTH KOREA"),
        (".KQ", "SOUTH KOREA"),
        (".TW", "TAIWAN"),
        (".SI", "SINGAPORE"),
    ]
    symbol_upper = symbol.upper()
    for suffix, country in suffix_map:
        if symbol_upper.endswith(suffix):
            return country
    return None
