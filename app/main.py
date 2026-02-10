import asyncio
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .patchtst_quantile import run_patchtst_forecast
from .quantile_lstm import run_quantile_lstm_forecast

load_dotenv()

LOGGER = logging.getLogger("market-data-analyzer")

MAX_BASIC_SYMBOLS = 8
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,15}$")
WS_URL_TEMPLATE = "wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"
REST_PRICE_URL = "https://api.twelvedata.com/price"
QUOTE_URL = "https://api.twelvedata.com/quote"
API_USAGE_URL = "https://api.twelvedata.com/api_usage"
STOCKS_LIST_URL = "https://api.twelvedata.com/stocks"
TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
EARLIEST_TIMESTAMP_URL = "https://api.twelvedata.com/earliest_timestamp"


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


API_LIMIT_PER_MIN = _int_env("API_LIMIT_PER_MIN", default=8, minimum=1)
API_LIMIT_PER_DAY = _int_env("API_LIMIT_PER_DAY", default=800, minimum=1)
DAILY_BUDGET_UTILIZATION = _float_env("DAILY_BUDGET_UTILIZATION", default=0.75, minimum=0.1)
PER_MIN_LIMIT_UTILIZATION = _float_env("PER_MIN_LIMIT_UTILIZATION", default=0.9, minimum=0.1)
REST_MIN_POLL_INTERVAL_SEC = _int_env("REST_MIN_POLL_INTERVAL_SEC", default=30, minimum=10)
SYMBOL_CATALOG_COUNTRY = os.getenv("SYMBOL_CATALOG_COUNTRY", "United States").strip() or "United States"
SYMBOL_CATALOG_TTL_SEC = _int_env("SYMBOL_CATALOG_TTL_SEC", default=86400, minimum=60)
SYMBOL_CATALOG_MAX_ITEMS = _int_env("SYMBOL_CATALOG_MAX_ITEMS", default=25000, minimum=1000)
SYMBOL_CATALOG_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "us_stock_symbol_catalog.json"
LAST_PRICE_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "last_prices.json"
FULL_DAILY_HISTORY_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "daily_history"
HISTORICAL_DEFAULT_YEARS = _int_env("HISTORICAL_DEFAULT_YEARS", default=5, minimum=1)
HISTORICAL_MAX_YEARS = _int_env("HISTORICAL_MAX_YEARS", default=10, minimum=1)
ML_HISTORY_DEFAULT_MONTHS = 60
ML_HISTORY_MIN_MONTHS = 3
ML_HISTORY_MAX_MONTHS = 60
ML_EVAL_MONTHS = 2
ML_SPLIT_EVAL_DAYS = ML_EVAL_MONTHS * 31
ML_SPLIT_TRAIN_VAL_RATIO = 0.8
HISTORICAL_CACHE_TTL_SEC = _int_env("HISTORICAL_CACHE_TTL_SEC", default=43200, minimum=60)
HISTORICAL_INTERVAL = os.getenv("HISTORICAL_INTERVAL", "1day").strip() or "1day"
HISTORICAL_MAX_POINTS = _int_env("HISTORICAL_MAX_POINTS", default=2000, minimum=100)
SPARKLINE_CACHE_TTL_SEC = _int_env("SPARKLINE_CACHE_TTL_SEC", default=21600, minimum=300)
SPARKLINE_POINTS = _int_env("SPARKLINE_POINTS", default=30, minimum=10)
OVERVIEW_CACHE_TTL_SEC = _int_env("OVERVIEW_CACHE_TTL_SEC", default=120, minimum=10)
TIME_SERIES_MAX_OUTPUTSIZE = _int_env("TIME_SERIES_MAX_OUTPUTSIZE", default=5000, minimum=100)
FULL_HISTORY_CHUNK_YEARS = _int_env("FULL_HISTORY_CHUNK_YEARS", default=15, minimum=1)
FULL_HISTORY_MAX_CHUNKS = _int_env("FULL_HISTORY_MAX_CHUNKS", default=20, minimum=1)
DAILY_DIFF_MIN_RECHECK_SEC = _int_env("DAILY_DIFF_MIN_RECHECK_SEC", default=21600, minimum=60)
BETA_MARKET_RECHECK_SEC = _int_env("BETA_MARKET_RECHECK_SEC", default=86400, minimum=300)
MARKET_CLOSED_SLEEP_SEC = _int_env("MARKET_CLOSED_SLEEP_SEC", default=60, minimum=10)
SYMBOL_COUNTRY_MAP_RAW = os.getenv("SYMBOL_COUNTRY_MAP", "")


@dataclass(frozen=True)
class MarketSession:
    tz: ZoneInfo
    open_minutes: int
    close_minutes: int
    weekdays: frozenset[int]


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


def normalize_symbols(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        tokens = [item.strip().upper() for item in raw.split(",")]
    else:
        tokens = [str(item).strip().upper() for item in raw]

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in tokens:
        if not symbol:
            continue
        if not SYMBOL_PATTERN.match(symbol):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)

    return normalized


def to_iso8601(value: Any) -> str:
    if isinstance(value, (int, float)):
        parsed = _datetime_from_unix(value)
        if parsed is not None:
            return parsed.isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _datetime_from_unix(value: Any) -> datetime | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None

    # Support unix values in second/ms/us/ns order of magnitude.
    abs_value = abs(numeric)
    if abs_value >= 1e18:
        numeric /= 1_000_000_000
    elif abs_value >= 1e15:
        numeric /= 1_000_000
    elif abs_value >= 1e12:
        numeric /= 1_000

    try:
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def fallback_interval_seconds(symbol_count: int) -> int:
    spacing = rest_request_spacing_seconds()
    if symbol_count <= 0:
        return spacing
    # One full cycle means each tracked symbol is refreshed once.
    return symbol_count * spacing


def effective_rest_requests_per_minute() -> float:
    # Respect both per-minute and per-day limits, then keep a safety margin.
    minute_cap = API_LIMIT_PER_MIN * PER_MIN_LIMIT_UTILIZATION
    day_cap_as_rpm = (API_LIMIT_PER_DAY * DAILY_BUDGET_UTILIZATION) / (24 * 60)
    return max(0.05, min(minute_cap, day_cap_as_rpm))


def rest_request_spacing_seconds() -> int:
    rpm = effective_rest_requests_per_minute()
    return max(REST_MIN_POLL_INTERVAL_SEC, math.ceil(60 / rpm))


def ok_json_response(**payload: Any) -> JSONResponse:
    return JSONResponse({"ok": True, **payload})


class SymbolUpdateRequest(BaseModel):
    symbols: str


class QuantileLstmJobRequest(BaseModel):
    symbol: str
    months: int = ML_HISTORY_DEFAULT_MONTHS
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    representative_days: int = 5
    seed: int = 42
    refresh: bool = False


class MlComparisonJobRequest(BaseModel):
    symbols: str = "AAPL,MSFT,GOOG,JPM,XOM,UNH,WMT,META,LLY,BRK.B,NVDA,HD"
    models: str = "quantile_lstm,patchtst_quantile"
    months: int = ML_HISTORY_DEFAULT_MONTHS
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    seed: int = 42
    refresh: bool = False


class MlJobCancelledError(Exception):
    pass


def _normalize_ml_history_months(months: int) -> int:
    value = int(months)
    if value < ML_HISTORY_MIN_MONTHS or value > ML_HISTORY_MAX_MONTHS:
        raise HTTPException(
            status_code=400,
            detail=f"months は {ML_HISTORY_MIN_MONTHS}〜{ML_HISTORY_MAX_MONTHS} の範囲で指定してください。",
        )
    return value


class LastPriceStore:
    def __init__(self, cache_path: Path, flush_interval_sec: int = 5) -> None:
        self.cache_path = cache_path
        self.flush_interval_sec = max(1, flush_interval_sec)
        self._data: dict[str, dict[str, Any]] = {}
        self._last_flush_at = 0.0
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def get(self, symbol: str) -> dict[str, Any] | None:
        item = self._data.get(symbol.upper())
        if not item:
            return None
        return dict(item)

    async def upsert(self, record: dict[str, Any]) -> None:
        symbol = str(record.get("symbol", "")).upper().strip()
        if not symbol:
            return

        normalized = {
            "symbol": symbol,
            "price": str(record.get("price")) if record.get("price") is not None else None,
            "timestamp": to_iso8601(record.get("timestamp")),
            "source": str(record.get("source") or "unknown"),
        }

        async with self._lock:
            self._data[symbol] = normalized
            now = time.time()
            if (now - self._last_flush_at) >= self.flush_interval_sec:
                self._write_no_lock()
                self._last_flush_at = now

    async def flush(self, force: bool = False) -> None:
        async with self._lock:
            now = time.time()
            if force or (now - self._last_flush_at) >= self.flush_interval_sec:
                self._write_no_lock()
                self._last_flush_at = now

    def _load_from_disk(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return

        rows = payload.get("prices") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return

        loaded: dict[str, dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            if not symbol or not SYMBOL_PATTERN.match(symbol):
                continue
            price = item.get("price")
            timestamp = item.get("timestamp")
            source = item.get("source")
            if price is None:
                continue
            loaded[symbol] = {
                "symbol": symbol,
                "price": str(price),
                "timestamp": to_iso8601(timestamp),
                "source": str(source or "stored"),
            }

        self._data = loaded

    def _write_no_lock(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "prices": sorted(self._data.values(), key=lambda item: item["symbol"]),
            }
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to write last price cache: %s", exc)


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


class MarketDataHub:
    def __init__(
        self,
        api_key: str,
        symbols: list[str],
        last_price_store: LastPriceStore,
        full_daily_history_store: FullDailyHistoryStore,
    ) -> None:
        self.api_key = api_key
        self.symbols: list[str] = symbols
        self.default_country_key = _normalize_country_key(SYMBOL_CATALOG_COUNTRY)
        self.symbol_country_map = parse_symbol_country_map(SYMBOL_COUNTRY_MAP_RAW)
        self.market_sessions = DEFAULT_MARKET_SESSIONS
        self.prices: dict[str, dict[str, Any]] = {}
        self.last_price_store = last_price_store
        self.full_daily_history_store = full_daily_history_store
        self.ws_connected = False
        self.last_ws_message_at = 0.0
        self.mode = "starting"
        self.daily_credits_left: int | None = None
        self.daily_credits_used: int | None = None
        self.daily_credits_limit: int | None = API_LIMIT_PER_DAY
        self.daily_credits_updated_at: str | None = None
        self.daily_credits_source: str | None = None
        self.daily_credits_is_estimated = False
        self.minute_credits_left: int | None = None
        self.minute_credits_used: int | None = None

        self._listeners: set[asyncio.Queue[dict[str, Any]]] = set()
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()
        self._restart_ws_event = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._credits_lock = asyncio.Lock()
        self._historical_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._historical_lock = asyncio.Lock()
        self._sparkline_cache: dict[str, dict[str, Any]] = {}
        self._sparkline_lock = asyncio.Lock()
        self._overview_cache: dict[tuple[str, bool, bool, bool], dict[str, Any]] = {}
        self._overview_lock = asyncio.Lock()

    @staticmethod
    def _is_cache_fresh(cached_epoch: Any, ttl_sec: int) -> bool:
        try:
            return (time.time() - float(cached_epoch)) <= ttl_sec
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _build_price_record(symbol: str, price: Any, source: str, timestamp: Any = None) -> dict[str, Any]:
        return {
            "symbol": symbol.upper().strip(),
            "price": str(price),
            "timestamp": to_iso8601(timestamp),
            "source": source,
        }

    async def _store_and_publish_price(self, record: dict[str, Any]) -> None:
        symbol = str(record.get("symbol", "")).upper().strip()
        if not symbol:
            return
        normalized = dict(record)
        normalized["symbol"] = symbol
        async with self._state_lock:
            self.prices[symbol] = normalized
        await self.last_price_store.upsert(normalized)
        await self.publish({"type": "price", "data": normalized})

    async def start(self) -> None:
        await self._hydrate_prices_from_store(self.symbols)
        self._worker_tasks = [
            asyncio.create_task(self._websocket_worker(), name="ws-worker"),
            asyncio.create_task(self._fallback_rest_worker(), name="rest-fallback-worker"),
        ]
        try:
            await self.refresh_api_credits()
        except Exception as exc:
            LOGGER.warning("Failed to initialize daily credits from /api_usage: %s", exc)

    async def stop(self) -> None:
        self._stop_event.set()
        self._restart_ws_event.set()
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        await self.last_price_store.flush(force=True)

    def register_listener(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._listeners.add(queue)
        return queue

    def unregister_listener(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._listeners.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._listeners):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def set_symbols(self, new_symbols: list[str]) -> None:
        if not new_symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")
        if len(new_symbols) > MAX_BASIC_SYMBOLS:
            raise HTTPException(
                status_code=400,
                detail=f"Basic plan supports up to {MAX_BASIC_SYMBOLS} symbols for websocket streaming.",
            )

        async with self._state_lock:
            self.symbols = new_symbols

        await self._hydrate_prices_from_store(new_symbols)
        self._restart_ws_event.set()
        rows = await self.current_rows(new_symbols)
        await self.publish(
            {
                "type": "symbols",
                "data": {
                    "symbols": self.symbols,
                    "poll_interval_sec": fallback_interval_seconds(len(self.symbols)),
                    "rows": rows,
                },
            }
        )

    async def status_payload(self) -> dict[str, Any]:
        last_seen = None
        if self.last_ws_message_at:
            last_seen = datetime.fromtimestamp(self.last_ws_message_at, tz=timezone.utc).isoformat()
        open_symbols = self._open_symbols(self.symbols)
        return {
            "mode": self.mode,
            "ws_connected": self.ws_connected,
            "last_ws_message_at": last_seen,
            "symbols": self.symbols,
            "open_symbols": open_symbols,
            "fallback_poll_interval_sec": fallback_interval_seconds(len(self.symbols)),
            "daily_credits_left": self.daily_credits_left,
            "daily_credits_used": self.daily_credits_used,
            "daily_credits_limit": self.daily_credits_limit,
            "daily_credits_updated_at": self.daily_credits_updated_at,
            "daily_credits_source": self.daily_credits_source,
            "daily_credits_is_estimated": self.daily_credits_is_estimated,
            # Backward compatibility for older UI field name.
            "api_credits_left": self.daily_credits_left,
        }

    async def snapshot_payload(self) -> dict[str, Any]:
        rows = await self.current_rows()
        return {
            "type": "snapshot",
            "data": {
                "status": await self.status_payload(),
                "rows": rows,
            },
        }

    async def current_rows(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        async with self._state_lock:
            target_symbols = list(symbols) if symbols is not None else list(self.symbols)
            for symbol in target_symbols:
                row = self.prices.get(symbol)
                if row:
                    rows.append(dict(row))
                else:
                    rows.append(
                        {
                            "symbol": symbol,
                            "price": None,
                            "timestamp": None,
                            "source": None,
                        }
                    )
        return rows

    async def _hydrate_prices_from_store(self, symbols: list[str]) -> None:
        if not symbols:
            return
        async with self._state_lock:
            for symbol in symbols:
                if symbol in self.prices:
                    continue
                cached = self.last_price_store.get(symbol)
                if not cached:
                    continue
                self.prices[symbol] = {
                    "symbol": symbol,
                    "price": cached.get("price"),
                    "timestamp": to_iso8601(cached.get("timestamp")),
                    "source": "stored",
                }

    async def _set_mode(self, mode: str, ws_connected: bool) -> None:
        changed = self.mode != mode or self.ws_connected != ws_connected
        self.mode = mode
        self.ws_connected = ws_connected
        if changed:
            await self.publish({"type": "status", "data": await self.status_payload()})

    async def _websocket_worker(self) -> None:
        backoff = 1
        while not self._stop_event.is_set():
            symbols = self.symbols
            if not symbols:
                await asyncio.sleep(1)
                continue

            active_symbols = self._open_symbols(symbols)
            if not active_symbols:
                await self._set_mode("market-closed", False)
                await asyncio.sleep(MARKET_CLOSED_SLEEP_SEC)
                continue

            ws_url = WS_URL_TEMPLATE.format(api_key=self.api_key)
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1000,
                ) as ws:
                    await self._set_mode("websocket", True)
                    self.last_ws_message_at = time.time()

                    await ws.send(
                        json.dumps({"action": "subscribe", "params": {"symbols": ",".join(active_symbols)}})
                    )

                    backoff = 1
                    last_market_check = time.time()
                    while not self._stop_event.is_set():
                        if self._restart_ws_event.is_set():
                            self._restart_ws_event.clear()
                            break

                        if (time.time() - last_market_check) >= 60:
                            last_market_check = time.time()
                            current_open = self._open_symbols(self.symbols)
                            if set(current_open) != set(active_symbols):
                                break

                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        self.last_ws_message_at = time.time()
                        await self._handle_ws_message(raw_message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Websocket worker error: %s", exc)

            await self._set_mode("rest-fallback", False)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _handle_ws_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        event_type = payload.get("event")
        if event_type and event_type not in {"price", "subscribe-status"}:
            return

        symbol = payload.get("symbol")
        price = payload.get("price")
        if not symbol or price is None:
            return

        record = self._build_price_record(
            symbol=str(symbol),
            price=price,
            source="websocket",
            timestamp=payload.get("timestamp"),
        )
        if not self._is_symbol_market_open(record["symbol"]):
            return

        await self._store_and_publish_price(record)

    async def _fallback_rest_worker(self) -> None:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while not self._stop_event.is_set():
                symbols = self.symbols
                if not symbols:
                    await asyncio.sleep(1)
                    continue

                active_symbols = self._open_symbols(symbols)
                if not active_symbols:
                    await self._set_mode("market-closed", False)
                    await asyncio.sleep(MARKET_CLOSED_SLEEP_SEC)
                    continue

                ws_stale = self.ws_connected and (time.time() - self.last_ws_message_at) > 25
                should_poll = (not self.ws_connected) or ws_stale

                if not should_poll:
                    await asyncio.sleep(2)
                    continue

                if self.ws_connected and ws_stale:
                    await self._set_mode("websocket+rest-fallback", True)
                else:
                    await self._set_mode("rest-fallback", False)

                for index, symbol in enumerate(active_symbols):
                    if self._stop_event.is_set():
                        break
                    await self._poll_one_symbol(client, symbol)
                    if index < len(active_symbols) - 1:
                        await asyncio.sleep(rest_request_spacing_seconds())

                await asyncio.sleep(rest_request_spacing_seconds())

    async def _poll_one_symbol(self, client: httpx.AsyncClient, symbol: str) -> None:
        try:
            response = await client.get(
                REST_PRICE_URL,
                params={
                    "apikey": self.api_key,
                    "symbol": symbol,
                },
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"rest:{symbol}")
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("REST fallback failed for %s: %s", symbol, exc)
            return

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("REST API error for %s: %s", symbol, payload.get("message"))
            return

        price = payload.get("price") if isinstance(payload, dict) else None
        if price is None:
            return

        record = self._build_price_record(symbol=symbol, price=price, source="rest")
        await self._store_and_publish_price(record)

    async def refresh_api_credits(self) -> dict[str, Any]:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with self._credits_lock, httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(API_USAGE_URL, params={"apikey": self.api_key})
            await self._update_minute_credits_from_response(response)
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "error":
                message = payload.get("message", "Failed to fetch API usage.")
                raise HTTPException(status_code=400, detail=message)
            await self._update_daily_credits_from_api_usage(payload)
            return await self.status_payload()

    async def historical_payload(
        self,
        symbol: str,
        years: int = HISTORICAL_DEFAULT_YEARS,
        months: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        years = max(1, min(years, HISTORICAL_MAX_YEARS))
        months = None if months is None else max(1, min(int(months), ML_HISTORY_MAX_MONTHS))

        cache_key = (normalized, f"years:{years}") if months is None else (normalized, f"months:{months}")
        async with self._historical_lock:
            cached = self._historical_cache.get(cache_key)
            if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), HISTORICAL_CACHE_TTL_SEC):
                payload = dict(cached["payload"])
                payload["source"] = "cache"
                return payload

        end_date = date.today()
        if months is None:
            start_date = end_date - timedelta(days=(365 * years) + (years // 4))
        else:
            start_date = end_date - timedelta(days=(31 * months) + 7)

        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                TIME_SERIES_URL,
                params={
                    "apikey": self.api_key,
                    "symbol": normalized,
                    "interval": HISTORICAL_INTERVAL,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "order": "ASC",
                    "outputsize": HISTORICAL_MAX_POINTS,
                },
            )

        async with self._credits_lock:
            await self._update_minute_credits_from_response(response)
            await self._consume_daily_credit_estimate(1, source=f"historical:{normalized}")

        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") == "error":
            message = payload.get("message", "Failed to fetch historical data.")
            raise HTTPException(status_code=400, detail=message)

        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            raise HTTPException(status_code=502, detail="Unexpected historical data format.")

        points: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            dt = str(item.get("datetime", "")).strip()
            close_raw = item.get("close")
            if not dt or close_raw is None:
                continue
            close_value = self._try_parse_float(close_raw)
            if close_value is None or close_value <= 0:
                continue

            open_value = self._try_parse_float(item.get("open"))
            high_value = self._try_parse_float(item.get("high"))
            low_value = self._try_parse_float(item.get("low"))
            volume_value = self._try_parse_float(item.get("volume"))

            if open_value is None or open_value <= 0:
                open_value = close_value
            if high_value is None or high_value <= 0:
                high_value = max(open_value, close_value)
            if low_value is None or low_value <= 0:
                low_value = min(open_value, close_value)

            high_value = max(high_value, open_value, close_value)
            low_value = min(low_value, open_value, close_value)
            if low_value <= 0:
                low_value = min(open_value, close_value)
                if low_value <= 0:
                    low_value = close_value

            points.append(
                {
                    "t": dt,
                    "o": open_value,
                    "h": high_value,
                    "l": low_value,
                    "c": close_value,
                    "v": volume_value,
                }
            )

        if not points:
            raise HTTPException(status_code=404, detail="No historical data found for this symbol.")

        historical_payload = {
            "symbol": normalized,
            "years": years,
            "months": months,
            "interval": HISTORICAL_INTERVAL,
            "from": points[0]["t"],
            "to": points[-1]["t"],
            "count": len(points),
            "points": points,
            "source": "twelvedata-live",
        }

        async with self._historical_lock:
            self._historical_cache[cache_key] = {
                "cached_epoch": time.time(),
                "payload": historical_payload,
            }

        return historical_payload

    async def security_overview_payload(
        self,
        symbol: str,
        refresh: bool = False,
        include_intraday: bool = True,
        include_market: bool = True,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        cache_key = (normalized, bool(include_intraday), bool(include_market), bool(include_qqq))

        async with self._overview_lock:
            cached = self._overview_cache.get(cache_key)
            if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), OVERVIEW_CACHE_TTL_SEC):
                payload = dict(cached["payload"])
                payload["source"] = "cache"
                return payload

        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            quote_task = self._fetch_quote(client, normalized)
            day_task = self._fetch_full_daily_series(client, normalized, refresh=refresh)
            quote, day_points = await asyncio.gather(quote_task, day_task)

            m1_points: list[dict[str, Any]] = []
            m5_points: list[dict[str, Any]] = []
            if include_intraday:
                m1_points, m5_points = await asyncio.gather(
                    self._fetch_series(client, normalized, "1min", outputsize=390),
                    self._fetch_series(client, normalized, "5min", outputsize=390),
                )

            market_context: dict[str, Any] | None = None
            if include_market:
                market_context = await self._fetch_market_context(
                    client,
                    refresh=refresh,
                    include_qqq=include_qqq,
                )

        if not day_points:
            raise HTTPException(status_code=404, detail="No overview data found for this symbol.")

        latest_day = day_points[-1]
        previous_day = day_points[-2] if len(day_points) >= 2 else None

        quote_price = self._pick_float(quote, "close", "price")
        current_price = quote_price if quote_price is not None else latest_day["c"]
        previous_close = (
            self._pick_float(quote, "previous_close", "prev_close")
            or (previous_day["c"] if previous_day else None)
        )
        day_open = self._pick_float(quote, "open")
        day_high = self._pick_float(quote, "high")
        day_low = self._pick_float(quote, "low")
        day_volume = self._pick_float(quote, "volume")
        bid = self._pick_float(quote, "bid")
        ask = self._pick_float(quote, "ask")

        if m1_points and (day_high is None or day_low is None or day_open is None):
            latest_session = self._extract_latest_session_points(m1_points)
            if latest_session:
                if day_open is None:
                    day_open = latest_session[0]["o"]
                if day_high is None:
                    day_high = max((item["h"] for item in latest_session), default=None)
                if day_low is None:
                    day_low = min((item["l"] for item in latest_session), default=None)
                if day_volume is None:
                    day_volume = sum((item["v"] or 0.0) for item in latest_session)

        if day_open is None:
            day_open = latest_day["o"]
        if day_high is None:
            day_high = latest_day["h"]
        if day_low is None:
            day_low = latest_day["l"]
        if day_volume is None:
            day_volume = latest_day["v"]

        change_abs = None
        change_pct = None
        if current_price is not None and previous_close is not None and previous_close > 0:
            change_abs = current_price - previous_close
            change_pct = (change_abs / previous_close) * 100

        recent_daily_volumes = [p["v"] for p in day_points[-21:-1] if p.get("v") is not None and p["v"] > 0]
        avg_volume_20 = (
            sum(recent_daily_volumes) / len(recent_daily_volumes)
            if recent_daily_volumes
            else None
        )
        avg_volume_ratio = (
            (day_volume / avg_volume_20)
            if day_volume is not None and avg_volume_20 is not None and avg_volume_20 > 0
            else None
        )

        turnover = (
            current_price * day_volume
            if current_price is not None and day_volume is not None
            else None
        )

        spread_abs = (
            ask - bid
            if ask is not None and bid is not None
            else None
        )
        spread_pct = (
            (spread_abs / current_price) * 100
            if spread_abs is not None and current_price is not None and current_price > 0
            else None
        )

        ma_short = self._moving_average(day_points, window=20)
        ma_mid = self._moving_average(day_points, window=50)
        atr_14 = self._atr(day_points, window=14)
        intraday_vwap_1m = self._intraday_vwap(m1_points)
        intraday_vwap_5m = self._intraday_vwap(m5_points)

        gap_abs = None
        gap_pct = None
        if day_open is not None and previous_close is not None and previous_close > 0:
            gap_abs = day_open - previous_close
            gap_pct = (gap_abs / previous_close) * 100

        spy_points = market_context.get("spy_points", []) if isinstance(market_context, dict) else []
        qqq_points = market_context.get("qqq_points", []) if isinstance(market_context, dict) else []
        beta_60, corr_60 = self._beta_and_corr_60d(day_points, spy_points) if include_market else (None, None)

        spy_latest = spy_points[-1]["c"] if spy_points else None
        spy_prev = spy_points[-2]["c"] if len(spy_points) >= 2 else None
        qqq_latest = qqq_points[-1]["c"] if qqq_points else None
        qqq_prev = qqq_points[-2]["c"] if len(qqq_points) >= 2 else None

        overview_payload = {
            "symbol": normalized,
            "name": self._pick_string(quote, "name", "instrument_name"),
            "exchange": self._pick_string(quote, "exchange"),
            "price": {
                "current": current_price,
                "previous_close": previous_close,
                "change_abs": change_abs,
                "change_pct": change_pct,
                "day_open": day_open,
                "day_high": day_high,
                "day_low": day_low,
                "gap_abs": gap_abs,
                "gap_pct": gap_pct,
                "updated_at": self._best_updated_at(quote, m1_points, day_points),
                "delay_note": "Twelve Data Basic plan (delayed feed may apply).",
            },
            "volume": {
                "today": day_volume,
                "avg20": avg_volume_20,
                "avg_ratio": avg_volume_ratio,
                "turnover": turnover,
            },
            "spread": {
                "bid": bid,
                "ask": ask,
                "spread_abs": spread_abs,
                "spread_pct": spread_pct,
            },
            "technical": {
                "vwap_1m": intraday_vwap_1m,
                "vwap_5m": intraday_vwap_5m,
                "ma_short_20": ma_short,
                "ma_mid_50": ma_mid,
                "atr_14": atr_14,
            },
            "market": {
                "sp500_proxy": self._build_market_item("SPY", spy_latest, spy_prev),
                "nasdaq_proxy": self._build_market_item("QQQ", qqq_latest, qqq_prev) if include_qqq else None,
                "beta_60d_vs_spy": beta_60,
                "corr_60d_vs_spy": corr_60,
            },
            "charts": {
                "1min": m1_points,
                "5min": m5_points,
                "1day": day_points,
            },
            "support_status": {
                "order_book": "not_supported_on_current_data_source",
                "corporate_events": "not_supported_on_current_data_source",
                "earnings_calendar": "not_supported_on_current_data_source",
                "news_headlines": "not_supported_on_current_data_source",
                "sector_etf": "not_supported_on_current_data_source",
            },
            "source": "twelvedata-live",
        }

        async with self._overview_lock:
            self._overview_cache[cache_key] = {
                "cached_epoch": time.time(),
                "payload": overview_payload,
            }

        return overview_payload

    async def clear_symbol_overview_cache(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")

        removed_overview = 0
        async with self._overview_lock:
            keys = [key for key in self._overview_cache.keys() if key[0] == normalized]
            for key in keys:
                self._overview_cache.pop(key, None)
                removed_overview += 1

        removed_historical = 0
        async with self._historical_lock:
            keys = [key for key in self._historical_cache.keys() if key[0] == normalized]
            for key in keys:
                self._historical_cache.pop(key, None)
                removed_historical += 1

        removed_daily_files = await self.full_daily_history_store.clear(normalized)
        return {
            "symbol": normalized,
            "removed_overview_entries": removed_overview,
            "removed_historical_entries": removed_historical,
            "removed_daily_history_files": removed_daily_files,
        }

    async def _fetch_market_context(
        self,
        client: httpx.AsyncClient,
        refresh: bool = False,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        spy_points = await self._fetch_full_daily_series(
            client,
            "SPY",
            refresh=refresh,
            min_recheck_sec=BETA_MARKET_RECHECK_SEC,
        )
        qqq_points: list[dict[str, Any]] = []
        if include_qqq:
            qqq_points = await self._fetch_full_daily_series(
                client,
                "QQQ",
                refresh=refresh,
                min_recheck_sec=BETA_MARKET_RECHECK_SEC,
            )
        return {
            "spy_points": spy_points[-90:] if len(spy_points) > 90 else spy_points,
            "qqq_points": qqq_points[-90:] if len(qqq_points) > 90 else qqq_points,
        }

    async def _fetch_quote(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        try:
            response = await client.get(
                QUOTE_URL,
                params={
                    "apikey": self.api_key,
                    "symbol": symbol,
                },
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"quote:{symbol}")
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Quote fetch failed for %s: %s", symbol, exc)
            return {}
        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Quote API error for %s: %s", symbol, payload.get("message"))
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _fetch_series(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            params: dict[str, Any] = {
                "apikey": self.api_key,
                "symbol": symbol,
                "interval": interval,
                "order": "ASC",
                "outputsize": min(max(1, int(outputsize)), TIME_SERIES_MAX_OUTPUTSIZE),
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            response = await client.get(
                TIME_SERIES_URL,
                params=params,
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"series:{symbol}:{interval}")
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Time series fetch failed for %s %s: %s", symbol, interval, exc)
            return []

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Time series API error for %s %s: %s", symbol, interval, payload.get("message"))
            return []

        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return []

        points: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            dt = str(item.get("datetime", "")).strip()
            close_value = self._try_parse_float(item.get("close"))
            if not dt or close_value is None or close_value <= 0:
                continue

            open_value = self._try_parse_float(item.get("open"))
            high_value = self._try_parse_float(item.get("high"))
            low_value = self._try_parse_float(item.get("low"))
            volume_value = self._try_parse_float(item.get("volume"))

            if open_value is None or open_value <= 0:
                open_value = close_value
            if high_value is None or high_value <= 0:
                high_value = max(open_value, close_value)
            if low_value is None or low_value <= 0:
                low_value = min(open_value, close_value)

            points.append(
                {
                    "t": dt,
                    "o": open_value,
                    "h": max(high_value, open_value, close_value),
                    "l": min(low_value, open_value, close_value),
                    "c": close_value,
                    "v": volume_value,
                }
            )

        return points

    async def _fetch_full_daily_series(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        refresh: bool = False,
        min_recheck_sec: int | None = None,
    ) -> list[dict[str, Any]]:
        today = date.today()
        if refresh:
            await self.full_daily_history_store.clear(symbol)
            cached_points: list[dict[str, Any]] = []
        else:
            cached_points = await self.full_daily_history_store.get(symbol)
        if cached_points:
            last_date = self._point_date(cached_points[-1])
            if last_date and last_date >= today:
                return cached_points

            last_cache_update_epoch = await self.full_daily_history_store.last_updated_epoch(symbol)
            if last_cache_update_epoch is not None:
                last_cache_update_dt = datetime.fromtimestamp(last_cache_update_epoch, tz=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                recheck_sec = DAILY_DIFF_MIN_RECHECK_SEC if min_recheck_sec is None else max(60, int(min_recheck_sec))
                if (
                    last_cache_update_dt.date() == now_utc.date()
                    and (now_utc.timestamp() - last_cache_update_epoch) < recheck_sec
                ):
                    return cached_points

            # Catch up in chunks to avoid truncation when the cached range is old.
            merged_cached = [dict(item) for item in cached_points]
            start_cursor = (last_date - timedelta(days=5)) if last_date else (today - timedelta(days=10))
            chunks = 0
            while start_cursor <= today and chunks < FULL_HISTORY_MAX_CHUNKS:
                chunk_end = min(
                    today,
                    start_cursor + timedelta(days=(366 * FULL_HISTORY_CHUNK_YEARS) - 1),
                )
                incremental_points = await self._fetch_series(
                    client,
                    symbol=symbol,
                    interval="1day",
                    outputsize=TIME_SERIES_MAX_OUTPUTSIZE,
                    start_date=start_cursor.isoformat(),
                    end_date=chunk_end.isoformat(),
                )
                if incremental_points:
                    merged_cached = self._merge_points_by_timestamp(merged_cached, incremental_points)
                start_cursor = chunk_end + timedelta(days=1)
                chunks += 1

            if start_cursor <= today:
                LOGGER.warning(
                    "Daily cache catch-up truncated for %s: reached chunk limit (%s).",
                    symbol,
                    FULL_HISTORY_MAX_CHUNKS,
                )

            await self.full_daily_history_store.upsert(symbol, merged_cached)
            return merged_cached

        fallback_points = await self._fetch_series(
            client,
            symbol=symbol,
            interval="1day",
            outputsize=max(1300, HISTORICAL_MAX_POINTS),
        )
        earliest = await self._fetch_earliest_date(client, symbol=symbol, interval="1day")
        if earliest is None:
            if fallback_points:
                await self.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        start_cursor = earliest
        chunks = 0
        merged: list[dict[str, Any]] = []

        while start_cursor <= today and chunks < FULL_HISTORY_MAX_CHUNKS:
            chunk_end = min(
                today,
                start_cursor + timedelta(days=(366 * FULL_HISTORY_CHUNK_YEARS) - 1),
            )
            points = await self._fetch_series(
                client,
                symbol=symbol,
                interval="1day",
                outputsize=TIME_SERIES_MAX_OUTPUTSIZE,
                start_date=start_cursor.isoformat(),
                end_date=chunk_end.isoformat(),
            )
            if points:
                merged.extend(points)
            start_cursor = chunk_end + timedelta(days=1)
            chunks += 1

        if not merged:
            if fallback_points:
                await self.full_daily_history_store.upsert(symbol, fallback_points)
            return fallback_points

        if start_cursor <= today:
            LOGGER.warning(
                "Daily full history truncated for %s: reached chunk limit (%s).",
                symbol,
                FULL_HISTORY_MAX_CHUNKS,
            )

        deduped = self._merge_points_by_timestamp([], merged)
        await self.full_daily_history_store.upsert(symbol, deduped)
        return deduped

    async def _fetch_earliest_date(self, client: httpx.AsyncClient, symbol: str, interval: str) -> date | None:
        try:
            response = await client.get(
                EARLIEST_TIMESTAMP_URL,
                params={
                    "apikey": self.api_key,
                    "symbol": symbol,
                    "interval": interval,
                },
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"earliest:{symbol}:{interval}")
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Earliest timestamp fetch failed for %s %s: %s", symbol, interval, exc)
            return None

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Earliest timestamp API error for %s %s: %s", symbol, interval, payload.get("message"))
            return None

        raw_value = None
        if isinstance(payload, dict):
            raw_value = payload.get("datetime") or payload.get("timestamp")
        if raw_value is None:
            return None

        parsed_iso = self._parse_timestamp(raw_value)
        if not parsed_iso:
            text = str(raw_value).strip()
            if text:
                try:
                    return date.fromisoformat(text.split(" ")[0])
                except ValueError:
                    return None
            return None

        try:
            return date.fromisoformat(parsed_iso[:10])
        except ValueError:
            return None

    @staticmethod
    def _pick_float(payload: dict[str, Any], *keys: str) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                return num
        return None

    @staticmethod
    def _pick_string(payload: dict[str, Any], *keys: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = str(payload.get(key, "")).strip()
            if value:
                return value
        return None

    @staticmethod
    def _extract_latest_session_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not points:
            return []
        latest_date = str(points[-1].get("t", "")).split(" ")[0]
        if not latest_date:
            return points
        return [item for item in points if str(item.get("t", "")).startswith(latest_date)]

    @staticmethod
    def _point_date(point: dict[str, Any]) -> date | None:
        raw_t = str(point.get("t", "")).strip()
        if not raw_t:
            return None
        date_text = raw_t.split(" ")[0]
        try:
            return date.fromisoformat(date_text)
        except ValueError:
            return None

    @staticmethod
    def _merge_points_by_timestamp(
        base_points: list[dict[str, Any]],
        incoming_points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in base_points:
            key = str(item.get("t", "")).strip()
            if not key:
                continue
            merged[key] = dict(item)
        for item in incoming_points:
            key = str(item.get("t", "")).strip()
            if not key:
                continue
            merged[key] = dict(item)
        return [merged[key] for key in sorted(merged.keys())]

    @staticmethod
    def _moving_average(points: list[dict[str, Any]], window: int) -> float | None:
        closes = [item["c"] for item in points if isinstance(item.get("c"), (int, float))]
        if len(closes) < window or window <= 0:
            return None
        sample = closes[-window:]
        return sum(sample) / window

    @staticmethod
    def _atr(points: list[dict[str, Any]], window: int = 14) -> float | None:
        if len(points) < window + 1:
            return None
        trs: list[float] = []
        prev_close = points[0]["c"]
        for item in points[1:]:
            high = item.get("h")
            low = item.get("l")
            close = item.get("c")
            if not isinstance(high, (int, float)) or not isinstance(low, (int, float)) or not isinstance(close, (int, float)):
                prev_close = close if isinstance(close, (int, float)) else prev_close
                continue
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            if tr >= 0:
                trs.append(tr)
            prev_close = close
        if len(trs) < window:
            return None
        sample = trs[-window:]
        return sum(sample) / window

    @staticmethod
    def _intraday_vwap(points: list[dict[str, Any]]) -> float | None:
        if not points:
            return None
        latest_session = MarketDataHub._extract_latest_session_points(points)
        if not latest_session:
            return None
        pv_sum = 0.0
        v_sum = 0.0
        for item in latest_session:
            close = item.get("c")
            volume = item.get("v")
            if not isinstance(close, (int, float)) or not isinstance(volume, (int, float)) or volume <= 0:
                continue
            pv_sum += close * volume
            v_sum += volume
        if v_sum <= 0:
            return None
        return pv_sum / v_sum

    @staticmethod
    def _daily_returns(points: list[dict[str, Any]], max_len: int) -> dict[str, float]:
        closes: list[tuple[str, float]] = []
        for item in points:
            raw_t = str(item.get("t", "")).strip()
            close = item.get("c")
            if not raw_t or not isinstance(close, (int, float)) or close <= 0:
                continue
            closes.append((raw_t.split(" ")[0], close))
        if len(closes) < 2:
            return {}
        target = closes[-(max_len + 1):]
        out: dict[str, float] = {}
        for idx in range(1, len(target)):
            date_key, close_value = target[idx]
            prev_close = target[idx - 1][1]
            if prev_close <= 0:
                continue
            out[date_key] = (close_value / prev_close) - 1
        return out

    @staticmethod
    def _beta_and_corr_60d(symbol_points: list[dict[str, Any]], benchmark_points: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        symbol_returns = MarketDataHub._daily_returns(symbol_points, max_len=60)
        benchmark_returns = MarketDataHub._daily_returns(benchmark_points, max_len=60)
        common_dates = sorted(set(symbol_returns.keys()) & set(benchmark_returns.keys()))
        if len(common_dates) < 20:
            return None, None

        x = [benchmark_returns[d] for d in common_dates]
        y = [symbol_returns[d] for d in common_dates]
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        cov = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y)) / max(1, len(x) - 1)
        var_x = sum((xv - mean_x) ** 2 for xv in x) / max(1, len(x) - 1)
        var_y = sum((yv - mean_y) ** 2 for yv in y) / max(1, len(y) - 1)
        if var_x <= 0 or var_y <= 0:
            return None, None
        beta = cov / var_x
        corr = cov / math.sqrt(var_x * var_y)
        return beta, corr

    @staticmethod
    def _parse_timestamp(raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            parsed = _datetime_from_unix(raw)
            return parsed.isoformat() if parsed is not None else None
        text = str(raw).strip()
        if not text:
            return None
        if text.isdigit():
            parsed = _datetime_from_unix(text)
            return parsed.isoformat() if parsed is not None else None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _best_updated_at(
        self,
        quote_payload: dict[str, Any],
        intraday_points: list[dict[str, Any]],
        day_points: list[dict[str, Any]],
    ) -> str | None:
        candidates = [
            self._parse_timestamp(quote_payload.get("timestamp")) if isinstance(quote_payload, dict) else None,
            self._parse_timestamp(quote_payload.get("datetime")) if isinstance(quote_payload, dict) else None,
            self._parse_timestamp(intraday_points[-1]["t"]) if intraday_points else None,
            self._parse_timestamp(day_points[-1]["t"]) if day_points else None,
        ]
        for item in candidates:
            if item:
                return item
        return None

    @staticmethod
    def _build_market_item(symbol: str, latest: float | None, previous: float | None) -> dict[str, Any]:
        change_abs = None
        change_pct = None
        if latest is not None and previous is not None and previous > 0:
            change_abs = latest - previous
            change_pct = (change_abs / previous) * 100
        return {
            "symbol": symbol,
            "price": latest,
            "change_abs": change_abs,
            "change_pct": change_pct,
        }

    async def sparkline_payload(self, symbols: list[str], refresh: bool = False) -> list[dict[str, Any]]:
        target_symbols = normalize_symbols(symbols)
        if not target_symbols:
            return []

        items_by_symbol: dict[str, dict[str, Any]] = {}
        missing_symbols: list[str] = []

        async with self._sparkline_lock:
            for symbol in target_symbols:
                cached = self._sparkline_cache.get(symbol)
                if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), SPARKLINE_CACHE_TTL_SEC):
                    items_by_symbol[symbol] = dict(cached["payload"])
                else:
                    missing_symbols.append(symbol)

        if missing_symbols:
            timeout = httpx.Timeout(20.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for symbol in missing_symbols:
                    item = await self._fetch_sparkline_item(client, symbol)
                    if not item:
                        continue
                    items_by_symbol[symbol] = item
                    async with self._sparkline_lock:
                        self._sparkline_cache[symbol] = {
                            "cached_epoch": time.time(),
                            "payload": item,
                        }

        return [items_by_symbol[symbol] for symbol in target_symbols if symbol in items_by_symbol]

    async def _fetch_sparkline_item(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None:
        try:
            response = await client.get(
                TIME_SERIES_URL,
                params={
                    "apikey": self.api_key,
                    "symbol": symbol,
                    "interval": "1day",
                    "order": "DESC",
                    "outputsize": max(SPARKLINE_POINTS + 2, 32),
                },
            )
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Sparkline fetch failed for %s: %s", symbol, exc)
            return None

        async with self._credits_lock:
            await self._update_minute_credits_from_response(response)
            await self._consume_daily_credit_estimate(1, source=f"sparkline:{symbol}")

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Sparkline API error for %s: %s", symbol, payload.get("message"))
            return None

        raw_values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(raw_values, list):
            return None

        values: list[tuple[str, float]] = []
        for item in raw_values:
            if not isinstance(item, dict):
                continue
            dt = str(item.get("datetime", "")).strip()
            close_value = self._try_parse_float(item.get("close"))
            if not dt or close_value is None:
                continue
            values.append((dt, close_value))

        if len(values) < 2:
            return None

        today_iso = date.today().isoformat()
        start_index = 1 if values[0][0].startswith(today_iso) and len(values) >= 2 else 0
        completed = values[start_index:]
        if len(completed) < 2:
            return None

        latest_completed_close = completed[0][1]
        previous_completed_close = completed[1][1] if len(completed) >= 2 else None
        recent_desc = completed[:SPARKLINE_POINTS]
        recent_asc = list(reversed(recent_desc))

        trend_values = [point[1] for point in recent_asc]
        return {
            "symbol": symbol,
            "latest_close": latest_completed_close,
            "latest_close_date": completed[0][0],
            "previous_close": previous_completed_close,
            "previous_close_date": completed[1][0] if len(completed) >= 2 else None,
            "trend_30d": trend_values,
            "trend_from": recent_asc[0][0],
            "trend_to": recent_asc[-1][0],
            "points": len(trend_values),
            "source": "twelvedata-live",
        }

    async def _update_minute_credits_from_response(self, response: httpx.Response) -> None:
        used_value = self._try_parse_int(response.headers.get("api-credits-used"))
        left_value = self._try_parse_int(response.headers.get("api-credits-left"))
        if used_value is None and left_value is None:
            return

        if used_value is not None:
            self.minute_credits_used = used_value
        if left_value is not None:
            self.minute_credits_left = left_value

    async def _update_daily_credits_from_api_usage(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        daily_usage = self._try_parse_int(payload.get("daily_usage"))
        plan_daily_limit = self._try_parse_int(payload.get("plan_daily_limit"))
        if daily_usage is None and plan_daily_limit is None:
            return

        if plan_daily_limit is not None:
            self.daily_credits_limit = plan_daily_limit
        if daily_usage is not None:
            self.daily_credits_used = max(0, daily_usage)
        if self.daily_credits_limit is not None and self.daily_credits_used is not None:
            self.daily_credits_left = max(0, self.daily_credits_limit - self.daily_credits_used)

        self.daily_credits_source = "api_usage"
        self.daily_credits_is_estimated = False
        self.daily_credits_updated_at = datetime.now(timezone.utc).isoformat()
        await self.publish({"type": "status", "data": await self.status_payload()})

    async def _consume_daily_credit_estimate(self, amount: int, source: str) -> None:
        if amount <= 0:
            return
        if self.daily_credits_limit is None or self.daily_credits_used is None:
            return

        self.daily_credits_used = max(0, self.daily_credits_used + amount)
        self.daily_credits_left = max(0, self.daily_credits_limit - self.daily_credits_used)
        self.daily_credits_source = source
        self.daily_credits_is_estimated = True
        self.daily_credits_updated_at = datetime.now(timezone.utc).isoformat()

        await self.publish({"type": "status", "data": await self.status_payload()})

    def _resolve_symbol_country_key(self, symbol: str) -> str:
        normalized_symbol = symbol.upper().strip()
        mapped_country = self.symbol_country_map.get(normalized_symbol)
        if mapped_country:
            return mapped_country
        inferred_country = infer_country_from_symbol(normalized_symbol)
        if inferred_country:
            return inferred_country
        return self.default_country_key

    def _is_country_market_open(self, country_key: str, now_utc: datetime) -> bool:
        session = self.market_sessions.get(country_key)
        if session is None:
            return True

        local_now = now_utc.astimezone(session.tz)
        if local_now.weekday() not in session.weekdays:
            return False
        current_minutes = (local_now.hour * 60) + local_now.minute

        if session.open_minutes <= session.close_minutes:
            return session.open_minutes <= current_minutes < session.close_minutes
        return current_minutes >= session.open_minutes or current_minutes < session.close_minutes

    def _is_symbol_market_open(self, symbol: str, now_utc: datetime | None = None) -> bool:
        utc_now = now_utc or datetime.now(timezone.utc)
        country_key = self._resolve_symbol_country_key(symbol)
        return self._is_country_market_open(country_key, utc_now)

    def _open_symbols(self, symbols: list[str]) -> list[str]:
        now_utc = datetime.now(timezone.utc)
        return [symbol for symbol in symbols if self._is_symbol_market_open(symbol, now_utc=now_utc)]

    @staticmethod
    def _try_parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_parse_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class MlJobStore:
    def __init__(self, max_jobs: int = 100) -> None:
        self.max_jobs = max(20, max_jobs)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._terminal_statuses = frozenset({"completed", "failed", "cancelled"})

    def create(self, kind: str, symbol: str) -> str:
        job_id = uuid.uuid4().hex
        now_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "job_id": job_id,
            "kind": kind,
            "symbol": symbol.upper().strip(),
            "status": "queued",
            "progress": 0,
            "message": "ジョブを作成しました。",
            "result": None,
            "error": None,
            "cancel_requested": False,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        with self._lock:
            self._jobs[job_id] = payload
            self._trim_no_lock()
        return job_id

    def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            item.update(changes)
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(item)

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return
            if item.get("cancel_requested"):
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    result=None,
                    error=None,
                )
            else:
                item.update(
                    status="completed",
                    progress=100,
                    message="完了しました。",
                    result=result,
                    error=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return
            if item.get("cancel_requested"):
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    error=None,
                    result=None,
                )
            else:
                item.update(
                    status="failed",
                    message="失敗しました。",
                    error=error,
                    result=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return False
            return bool(item.get("cancel_requested"))

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            status = str(item.get("status") or "")
            if status in self._terminal_statuses:
                return dict(item)

            item["cancel_requested"] = True
            if status == "queued":
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    result=None,
                    error=None,
                )
            else:
                item.update(
                    status="cancelling",
                    progress=0,
                    message="停止を要求しました。",
                    result=None,
                    error=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(item)

    def mark_cancelled(self, job_id: str, message: str = "停止しました。") -> None:
        self.update(
            job_id,
            status="cancelled",
            progress=0,
            message=message,
            result=None,
            error=None,
            cancel_requested=True,
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            return dict(item)

    def _trim_no_lock(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        sorted_items = sorted(self._jobs.items(), key=lambda pair: pair[1].get("updated_at", ""))
        remove_count = len(self._jobs) - self.max_jobs
        for idx in range(remove_count):
            self._jobs.pop(sorted_items[idx][0], None)


API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DEFAULT_SYMBOLS = normalize_symbols(
    os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
)
if not DEFAULT_SYMBOLS:
    DEFAULT_SYMBOLS = ["AAPL"]
DEFAULT_SYMBOLS = DEFAULT_SYMBOLS[:MAX_BASIC_SYMBOLS]

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is required. Set it in your environment or .env file.")

last_price_store = LastPriceStore(cache_path=LAST_PRICE_CACHE_PATH)
full_daily_history_store = FullDailyHistoryStore(cache_dir=FULL_DAILY_HISTORY_CACHE_DIR)
hub = MarketDataHub(
    api_key=API_KEY,
    symbols=DEFAULT_SYMBOLS,
    last_price_store=last_price_store,
    full_daily_history_store=full_daily_history_store,
)
symbol_catalog_store = SymbolCatalogStore(
    api_key=API_KEY,
    cache_path=SYMBOL_CATALOG_CACHE_PATH,
    ttl_sec=SYMBOL_CATALOG_TTL_SEC,
)

ML_MODEL_CATALOG = [
    {
        "id": "quantile_lstm",
        "name": "Quantile LSTM",
        "short_description": "翌営業日の分位点分布を推定（現在利用可能）",
        "status": "ready",
        "status_label": "Ready",
        "run_label": "Run Quantile LSTM",
        "api_path": "/api/ml/quantile-lstm",
    },
    {
        "id": "patchtst_quantile",
        "name": "PatchTST Quantile",
        "short_description": "PatchTSTで翌営業日の分位点分布を推定（現在利用可能）",
        "status": "ready",
        "status_label": "Ready",
        "run_label": "Run PatchTST Quantile",
        "api_path": "/api/ml/patchtst",
    },
    {
        "id": "quantile_gru",
        "name": "Quantile GRU",
        "short_description": "LSTMより軽量な系列モデル（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run Quantile GRU",
        "api_path": "",
    },
    {
        "id": "temporal_transformer",
        "name": "Temporal Transformer",
        "short_description": "注意機構ベースの時系列モデル（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run Temporal Transformer",
        "api_path": "",
    },
    {
        "id": "xgboost_quantile",
        "name": "XGBoost Quantile",
        "short_description": "勾配ブースティングの分位点回帰（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run XGBoost Quantile",
        "api_path": "",
    },
]
ML_COMPARE_DEFAULT_SYMBOLS = normalize_symbols(
    "AAPL,MSFT,GOOG,JPM,XOM,UNH,WMT,META,LLY,BRK.B,NVDA,HD"
)
ML_COMPARE_ALLOWED_MODELS = {"quantile_lstm", "patchtst_quantile"}
ml_job_store = MlJobStore(max_jobs=120)


async def _run_quantile_lstm_pipeline(
    *,
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 80,
    patience: int = 10,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
    split_eval_days: int | None = None,
    split_train_val_ratio: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    return await _run_ml_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
        split_eval_days=split_eval_days,
        split_train_val_ratio=split_train_val_ratio,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        trainer=run_quantile_lstm_forecast,
        training_start_message="モデル学習を開始します。",
        error_log_message="Quantile LSTM training failed for %s",
        error_detail="Quantile LSTM training failed.",
    )


async def _run_ml_pipeline(
    *,
    symbol: str,
    months: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    representative_days: int,
    seed: int,
    refresh: bool,
    split_eval_days: int | None,
    split_train_val_ratio: float | None,
    progress_callback: Callable[[float, str], None] | None,
    cancel_check: Callable[[], None] | None,
    trainer: Callable[
        [list[dict[str, Any]], dict[str, Any], Callable[[float, str], None] | None, Callable[[], None] | None],
        dict[str, Any],
    ],
    training_start_message: str,
    error_log_message: str,
    error_detail: str,
) -> dict[str, Any]:
    if cancel_check is not None:
        cancel_check()

    if progress_callback is not None:
        progress_callback(2, "ヒストリカルデータを取得しています。")

    effective_months = _normalize_ml_history_months(months)
    historical_data = await hub.historical_payload(symbol=symbol, months=effective_months, refresh=refresh)
    points = historical_data.get("points")
    if not isinstance(points, list):
        raise HTTPException(status_code=502, detail="Unexpected historical payload format.")

    config_payload = {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "representative_days": representative_days,
        "seed": seed,
    }
    effective_split_eval_days = int(split_eval_days) if split_eval_days is not None else ML_SPLIT_EVAL_DAYS
    effective_split_train_val_ratio = (
        float(split_train_val_ratio) if split_train_val_ratio is not None else ML_SPLIT_TRAIN_VAL_RATIO
    )
    config_payload["split_eval_days"] = max(1, effective_split_eval_days)
    config_payload["split_train_val_ratio"] = max(0.5, min(0.95, effective_split_train_val_ratio))

    if progress_callback is not None:
        progress_callback(5, training_start_message)

    try:
        model_payload = await asyncio.to_thread(
            trainer,
            points,
            config_payload,
            progress_callback,
            cancel_check,
        )
    except MlJobCancelledError:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(error_log_message, symbol, exc_info=exc)
        raise HTTPException(status_code=500, detail=error_detail) from exc

    if progress_callback is not None:
        progress_callback(100, "結果を整形しました。")

    return {
        "symbol": historical_data.get("symbol"),
        "months": effective_months,
        "historical_source": historical_data.get("source"),
        **model_payload,
    }


async def _run_patchtst_pipeline(
    *,
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 256,
    hidden_size: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 40,
    patience: int = 8,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
    split_eval_days: int | None = None,
    split_train_val_ratio: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    return await _run_ml_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
        split_eval_days=split_eval_days,
        split_train_val_ratio=split_train_val_ratio,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        trainer=run_patchtst_forecast,
        training_start_message="PatchTST学習を開始します。",
        error_log_message="PatchTST training failed for %s",
        error_detail="PatchTST training failed.",
    )


def _progress_callback_for_job(job_id: str):
    def _cb(progress: float, message: str) -> None:
        if ml_job_store.is_cancel_requested(job_id):
            raise MlJobCancelledError("ML job cancelled.")
        current = ml_job_store.get(job_id)
        if current is None:
            return
        status = str(current.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            return
        next_status = "cancelling" if status == "cancelling" else "running"
        ml_job_store.update(
            job_id,
            status=next_status,
            progress=max(0.0, min(100.0, float(progress))),
            message=message,
        )

    return _cb


def _cancel_check_for_job(job_id: str):
    def _check() -> None:
        if ml_job_store.is_cancel_requested(job_id):
            raise MlJobCancelledError("ML job cancelled.")

    return _check


def _parse_compare_models(raw_models: str) -> list[str]:
    tokens = [item.strip().lower() for item in str(raw_models or "").split(",")]
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        if token not in ML_COMPARE_ALLOWED_MODELS:
            continue
        selected.append(token)
        seen.add(token)
    return selected


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _as_aligned_arrays(left: list[Any], right: list[Any]) -> tuple[list[float], list[float]]:
    n = min(len(left), len(right))
    if n <= 0:
        return [], []
    out_left: list[float] = []
    out_right: list[float] = []
    for idx in range(n):
        try:
            l_value = float(left[idx])
            r_value = float(right[idx])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(l_value) or not math.isfinite(r_value):
            continue
        out_left.append(l_value)
        out_right.append(r_value)
    return out_left, out_right


def _compute_loss_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    fan_chart = payload.get("fan_chart") if isinstance(payload, dict) else None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    splits = payload.get("splits") if isinstance(payload, dict) else None
    training = payload.get("training") if isinstance(payload, dict) else None

    actual_returns = fan_chart.get("actual_returns") if isinstance(fan_chart, dict) else []
    q50_returns = fan_chart.get("q50_returns") if isinstance(fan_chart, dict) else []
    actual_prices = fan_chart.get("actual_prices") if isinstance(fan_chart, dict) else []
    q50_prices = fan_chart.get("q50_prices") if isinstance(fan_chart, dict) else []

    y_ret, yhat_ret = _as_aligned_arrays(actual_returns if isinstance(actual_returns, list) else [], q50_returns if isinstance(q50_returns, list) else [])
    y_price, yhat_price = _as_aligned_arrays(actual_prices if isinstance(actual_prices, list) else [], q50_prices if isinstance(q50_prices, list) else [])

    mae_return = float(sum(abs(a - b) for a, b in zip(y_ret, yhat_ret)) / len(y_ret)) if y_ret else None
    rmse_return = (
        float(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_ret, yhat_ret)) / len(y_ret)))
        if y_ret else None
    )
    mae_price = float(sum(abs(a - b) for a, b in zip(y_price, yhat_price)) / len(y_price)) if y_price else None
    rmse_price = (
        float(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_price, yhat_price)) / len(y_price)))
        if y_price else None
    )

    mape_terms = [
        abs((actual - pred) / actual) * 100.0
        for actual, pred in zip(y_price, yhat_price)
        if abs(actual) > 1e-12
    ]
    smape_terms = [
        (2.0 * abs(actual - pred) / (abs(actual) + abs(pred))) * 100.0
        for actual, pred in zip(y_price, yhat_price)
        if (abs(actual) + abs(pred)) > 1e-12
    ]

    test_split = splits.get("test") if isinstance(splits, dict) else {}
    return {
        "test_count": _safe_int(test_split.get("count")),
        "test_from": test_split.get("from"),
        "test_to": test_split.get("to"),
        "epochs_trained": _safe_int(training.get("epochs_trained")) if isinstance(training, dict) else 0,
        "best_val_pinball_loss": _safe_float(training.get("best_val_pinball_loss")) if isinstance(training, dict) else None,
        "mean_pinball_loss": _safe_float(metrics.get("mean_pinball_loss")) if isinstance(metrics, dict) else None,
        "coverage_90": _safe_float(metrics.get("coverage_90")) if isinstance(metrics, dict) else None,
        "coverage_50": _safe_float(metrics.get("coverage_50")) if isinstance(metrics, dict) else None,
        "mae_return": mae_return,
        "rmse_return": rmse_return,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mape_price_pct": float(sum(mape_terms) / len(mape_terms)) if mape_terms else None,
        "smape_price_pct": float(sum(smape_terms) / len(smape_terms)) if smape_terms else None,
    }


async def _run_ml_comparison_job(job_id: str, req: MlComparisonJobRequest) -> None:
    try:
        raw_symbols = normalize_symbols(req.symbols) if str(req.symbols or "").strip() else []
        symbols = raw_symbols or ML_COMPARE_DEFAULT_SYMBOLS
        if not symbols:
            raise HTTPException(status_code=400, detail="比較対象シンボルが空です。")

        selected_models = _parse_compare_models(req.models)
        if not selected_models:
            raise HTTPException(status_code=400, detail="比較対象モデルが空です。")

        split_eval_days = ML_SPLIT_EVAL_DAYS
        split_train_val_ratio = ML_SPLIT_TRAIN_VAL_RATIO

        ml_job_store.update(job_id, status="running", progress=1, message="比較ジョブを開始しました。")
        cancel_check = _cancel_check_for_job(job_id)

        rows: list[dict[str, Any]] = []
        task_items = [(symbol, model_id) for symbol in symbols for model_id in selected_models]
        task_count = len(task_items)

        for task_idx, (symbol, model_id) in enumerate(task_items):
            cancel_check()
            progress_start = 5 + int((task_idx * 88) / max(1, task_count))
            progress_end = 5 + int(((task_idx + 1) * 88) / max(1, task_count))

            def _task_progress(progress: float, message: str) -> None:
                cancel_check()
                clamped = max(0.0, min(100.0, float(progress)))
                mapped = progress_start + (((progress_end - progress_start) * clamped) / 100.0)
                ml_job_store.update(
                    job_id,
                    status="running",
                    progress=max(1.0, min(99.0, mapped)),
                    message=f"[{task_idx + 1}/{task_count}] {symbol} | {model_id}: {message}",
                )

            try:
                if model_id == "quantile_lstm":
                    payload = await _run_quantile_lstm_pipeline(
                        symbol=symbol,
                        months=req.months,
                        sequence_length=req.sequence_length,
                        hidden_size=req.hidden_size,
                        num_layers=req.num_layers,
                        dropout=req.dropout,
                        learning_rate=req.learning_rate,
                        batch_size=req.batch_size,
                        max_epochs=req.max_epochs,
                        patience=req.patience,
                        representative_days=3,
                        seed=req.seed,
                        refresh=req.refresh,
                        split_eval_days=split_eval_days,
                        split_train_val_ratio=split_train_val_ratio,
                        progress_callback=_task_progress,
                        cancel_check=cancel_check,
                    )
                elif model_id == "patchtst_quantile":
                    payload = await _run_patchtst_pipeline(
                        symbol=symbol,
                        months=req.months,
                        sequence_length=req.sequence_length,
                        hidden_size=req.hidden_size,
                        num_layers=req.num_layers,
                        dropout=req.dropout,
                        learning_rate=req.learning_rate,
                        batch_size=req.batch_size,
                        max_epochs=req.max_epochs,
                        patience=req.patience,
                        representative_days=3,
                        seed=req.seed,
                        refresh=req.refresh,
                        split_eval_days=split_eval_days,
                        split_train_val_ratio=split_train_val_ratio,
                        progress_callback=_task_progress,
                        cancel_check=cancel_check,
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "ok",
                        "error": None,
                        "metrics": _compute_loss_metrics(payload),
                    }
                )
            except MlJobCancelledError:
                raise
            except HTTPException as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(exc.detail),
                        "metrics": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(exc),
                        "metrics": None,
                    }
                )

        cancel_check()
        summary_by_model: list[dict[str, Any]] = []
        for model_id in selected_models:
            model_rows = [row for row in rows if row.get("model_id") == model_id and row.get("status") == "ok"]
            metrics_list = [row.get("metrics") for row in model_rows if isinstance(row.get("metrics"), dict)]
            summary_by_model.append(
                {
                    "model_id": model_id,
                    "success_count": len(model_rows),
                    "mean_pinball_loss": _mean_or_none([item.get("mean_pinball_loss") for item in metrics_list]),
                    "mean_mae_return": _mean_or_none([item.get("mae_return") for item in metrics_list]),
                    "mean_rmse_return": _mean_or_none([item.get("rmse_return") for item in metrics_list]),
                    "mean_mae_price": _mean_or_none([item.get("mae_price") for item in metrics_list]),
                    "mean_rmse_price": _mean_or_none([item.get("rmse_price") for item in metrics_list]),
                    "mean_mape_price_pct": _mean_or_none([item.get("mape_price_pct") for item in metrics_list]),
                    "mean_smape_price_pct": _mean_or_none([item.get("smape_price_pct") for item in metrics_list]),
                    "mean_coverage_90": _mean_or_none([item.get("coverage_90") for item in metrics_list]),
                    "mean_coverage_50": _mean_or_none([item.get("coverage_50") for item in metrics_list]),
                }
            )

        success_count = len([row for row in rows if row.get("status") == "ok"])
        failed_count = len(rows) - success_count
        result = {
            "symbols": symbols,
            "models": selected_models,
            "config": {
                "months": int(req.months),
                "sequence_length": int(req.sequence_length),
                "hidden_size": int(req.hidden_size),
                "num_layers": int(req.num_layers),
                "dropout": float(req.dropout),
                "learning_rate": float(req.learning_rate),
                "batch_size": int(req.batch_size),
                "max_epochs": int(req.max_epochs),
                "patience": int(req.patience),
                "seed": int(req.seed),
                "refresh": bool(req.refresh),
            },
            "evaluation_policy": {
                "eval_months": ML_EVAL_MONTHS,
                "eval_days_approx": ML_SPLIT_EVAL_DAYS,
                "test_window": "latest 2 months (relative to the latest available trading date)",
                "train_val_split": "remaining history split by 4:1",
                "train_ratio": split_train_val_ratio,
                "val_ratio": 1.0 - split_train_val_ratio,
            },
            "summary_by_model": summary_by_model,
            "rows": rows,
            "success_count": success_count,
            "failed_count": failed_count,
        }
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


async def _run_quantile_lstm_job(job_id: str, req: QuantileLstmJobRequest) -> None:
    try:
        if ml_job_store.is_cancel_requested(job_id):
            ml_job_store.mark_cancelled(job_id, message="ジョブ開始前に停止しました。")
            return
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_quantile_lstm_pipeline(
            symbol=req.symbol,
            months=req.months,
            sequence_length=req.sequence_length,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            dropout=req.dropout,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            max_epochs=req.max_epochs,
            patience=req.patience,
            representative_days=req.representative_days,
            seed=req.seed,
            refresh=req.refresh,
            progress_callback=_progress_callback_for_job(job_id),
            cancel_check=_cancel_check_for_job(job_id),
        )
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


async def _run_patchtst_job(job_id: str, req: QuantileLstmJobRequest) -> None:
    try:
        if ml_job_store.is_cancel_requested(job_id):
            ml_job_store.mark_cancelled(job_id, message="ジョブ開始前に停止しました。")
            return
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_patchtst_pipeline(
            symbol=req.symbol,
            months=req.months,
            sequence_length=req.sequence_length,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            dropout=req.dropout,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            max_epochs=req.max_epochs,
            patience=req.patience,
            representative_days=req.representative_days,
            seed=req.seed,
            refresh=req.refresh,
            progress_callback=_progress_callback_for_job(job_id),
            cancel_check=_cancel_check_for_job(job_id),
        )
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


@asynccontextmanager
async def lifespan(_: FastAPI):
    await hub.start()
    try:
        yield
    finally:
        await hub.stop()


app = FastAPI(title="Market Data Analyzer", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ml-lab", include_in_schema=False)
async def ml_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ml_lab.html")


@app.get("/strategy-lab", include_in_schema=False)
async def strategy_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "strategy_lab.html")


@app.get("/compare-lab", include_in_schema=False)
async def compare_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "compare_lab.html")


@app.get("/historical/{symbol}", include_in_schema=False)
async def historical_page(symbol: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "historical.html")


@app.get("/api/snapshot")
async def snapshot() -> JSONResponse:
    payload = await hub.snapshot_payload()
    return JSONResponse(payload)


@app.post("/api/symbols")
async def update_symbols(req: SymbolUpdateRequest) -> JSONResponse:
    symbols = normalize_symbols(req.symbols)
    await hub.set_symbols(symbols)
    rows = await hub.current_rows(symbols)
    return ok_json_response(
        symbols=symbols,
        status=await hub.status_payload(),
        rows=rows,
    )


@app.get("/api/credits")
async def credits(refresh: bool = False) -> JSONResponse:
    if refresh:
        status = await hub.refresh_api_credits()
    else:
        status = await hub.status_payload()
    return ok_json_response(
        status=status,
        note="refresh=true fetches exact daily remaining credits via /api_usage and consumes 1 API credit.",
    )


@app.get("/api/symbol-catalog")
async def symbol_catalog(refresh: bool = False) -> JSONResponse:
    payload = await symbol_catalog_store.get_catalog(refresh=refresh)
    return ok_json_response(**payload)


@app.get("/api/historical/{symbol}")
async def historical(symbol: str, years: int = HISTORICAL_DEFAULT_YEARS, refresh: bool = False) -> JSONResponse:
    payload = await hub.historical_payload(symbol=symbol, years=years, refresh=refresh)
    return ok_json_response(**payload)


@app.get("/api/security-overview/{symbol}")
async def security_overview(
    symbol: str,
    refresh: bool = False,
    include_intraday: bool = True,
    include_market: bool = True,
    include_qqq: bool = True,
) -> JSONResponse:
    payload = await hub.security_overview_payload(
        symbol=symbol,
        refresh=refresh,
        include_intraday=include_intraday,
        include_market=include_market,
        include_qqq=include_qqq,
    )
    return ok_json_response(**payload)


@app.get("/api/security-overview/{symbol}/intraday")
async def security_overview_intraday(symbol: str, refresh: bool = False) -> JSONResponse:
    payload = await hub.security_overview_payload(
        symbol=symbol,
        refresh=refresh,
        include_intraday=True,
        include_market=False,
        include_qqq=False,
    )
    return ok_json_response(
        symbol=payload.get("symbol"),
        technical={
            "vwap_1m": payload.get("technical", {}).get("vwap_1m") if isinstance(payload.get("technical"), dict) else None,
            "vwap_5m": payload.get("technical", {}).get("vwap_5m") if isinstance(payload.get("technical"), dict) else None,
        },
        charts={
            "1min": payload.get("charts", {}).get("1min") if isinstance(payload.get("charts"), dict) else [],
            "5min": payload.get("charts", {}).get("5min") if isinstance(payload.get("charts"), dict) else [],
        },
        source=payload.get("source"),
    )


@app.post("/api/security-overview/{symbol}/clear-cache")
async def clear_security_overview_cache(symbol: str) -> JSONResponse:
    payload = await hub.clear_symbol_overview_cache(symbol=symbol)
    return ok_json_response(**payload)


@app.get("/api/ml/models")
async def ml_models() -> JSONResponse:
    return ok_json_response(models=ML_MODEL_CATALOG)


@app.get("/api/ml/quantile-lstm")
async def quantile_lstm_forecast(
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 80,
    patience: int = 10,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
) -> JSONResponse:
    payload = await _run_quantile_lstm_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
    )
    return ok_json_response(**payload)


@app.get("/api/ml/patchtst")
async def patchtst_forecast(
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 256,
    hidden_size: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 40,
    patience: int = 8,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
) -> JSONResponse:
    payload = await _run_patchtst_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
    )
    return ok_json_response(**payload)


@app.post("/api/ml/quantile-lstm/jobs")
async def start_quantile_lstm_job(req: QuantileLstmJobRequest) -> JSONResponse:
    symbol = normalize_symbols([req.symbol])
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    req.symbol = symbol[0]
    req.months = _normalize_ml_history_months(req.months)

    job_id = ml_job_store.create(kind="quantile_lstm", symbol=req.symbol)
    asyncio.create_task(_run_quantile_lstm_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@app.post("/api/ml/patchtst/jobs")
async def start_patchtst_job(req: QuantileLstmJobRequest) -> JSONResponse:
    symbol = normalize_symbols([req.symbol])
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    req.symbol = symbol[0]
    req.months = _normalize_ml_history_months(req.months)

    job_id = ml_job_store.create(kind="patchtst_quantile", symbol=req.symbol)
    asyncio.create_task(_run_patchtst_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@app.post("/api/ml/compare/jobs")
async def start_ml_compare_job(req: MlComparisonJobRequest) -> JSONResponse:
    symbols = normalize_symbols(req.symbols) if str(req.symbols or "").strip() else []
    if not symbols:
        symbols = ML_COMPARE_DEFAULT_SYMBOLS
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    req.symbols = ",".join(symbols)
    req.months = _normalize_ml_history_months(req.months)

    selected_models = _parse_compare_models(req.models)
    if not selected_models:
        raise HTTPException(status_code=400, detail="At least one valid model is required.")
    req.models = ",".join(selected_models)

    job_id = ml_job_store.create(kind="ml_compare", symbol="MULTI")
    asyncio.create_task(_run_ml_comparison_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@app.get("/api/ml/jobs/{job_id}")
async def ml_job_status(job_id: str) -> JSONResponse:
    payload = ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)


@app.post("/api/ml/jobs/{job_id}/cancel")
async def ml_job_cancel(job_id: str) -> JSONResponse:
    payload = ml_job_store.request_cancel(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)


@app.get("/api/sparkline")
async def sparkline(symbols: str, refresh: bool = False) -> JSONResponse:
    target_symbols = normalize_symbols(symbols)
    if not target_symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    if len(target_symbols) > MAX_BASIC_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.")

    items = await hub.sparkline_payload(target_symbols, refresh=refresh)
    return ok_json_response(
        symbols=target_symbols,
        items=items,
    )


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    queue = hub.register_listener()

    async def event_generator():
        initial_payload = await hub.snapshot_payload()
        yield f"data: {json.dumps(initial_payload)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.unregister_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
