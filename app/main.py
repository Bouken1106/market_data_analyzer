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
API_USAGE_URL = "https://api.twelvedata.com/api_usage"
STOCKS_LIST_URL = "https://api.twelvedata.com/stocks"
TIME_SERIES_URL = "https://api.twelvedata.com/time_series"


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
HISTORICAL_DEFAULT_YEARS = _int_env("HISTORICAL_DEFAULT_YEARS", default=5, minimum=1)
HISTORICAL_MAX_YEARS = _int_env("HISTORICAL_MAX_YEARS", default=10, minimum=1)
HISTORICAL_CACHE_TTL_SEC = _int_env("HISTORICAL_CACHE_TTL_SEC", default=43200, minimum=60)
HISTORICAL_INTERVAL = os.getenv("HISTORICAL_INTERVAL", "1day").strip() or "1day"
HISTORICAL_MAX_POINTS = _int_env("HISTORICAL_MAX_POINTS", default=2000, minimum=100)
SPARKLINE_CACHE_TTL_SEC = _int_env("SPARKLINE_CACHE_TTL_SEC", default=21600, minimum=300)
SPARKLINE_POINTS = _int_env("SPARKLINE_POINTS", default=30, minimum=10)
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
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


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
    years: int = HISTORICAL_DEFAULT_YEARS
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


class MarketDataHub:
    def __init__(self, api_key: str, symbols: list[str], last_price_store: LastPriceStore) -> None:
        self.api_key = api_key
        self.symbols: list[str] = symbols
        self.default_country_key = _normalize_country_key(SYMBOL_CATALOG_COUNTRY)
        self.symbol_country_map = parse_symbol_country_map(SYMBOL_COUNTRY_MAP_RAW)
        self.market_sessions = DEFAULT_MARKET_SESSIONS
        self.prices: dict[str, dict[str, Any]] = {}
        self.last_price_store = last_price_store
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

    async def historical_payload(self, symbol: str, years: int, refresh: bool = False) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        years = max(1, min(years, HISTORICAL_MAX_YEARS))

        cache_key = (normalized, years)
        async with self._historical_lock:
            cached = self._historical_cache.get(cache_key)
            if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), HISTORICAL_CACHE_TTL_SEC):
                payload = dict(cached["payload"])
                payload["source"] = "cache"
                return payload

        end_date = date.today()
        start_date = end_date - timedelta(days=(365 * years) + (years // 4))

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
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        with self._lock:
            self._jobs[job_id] = payload
            self._trim_no_lock()
        return job_id

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return
            item.update(changes)
            item["updated_at"] = datetime.now(timezone.utc).isoformat()

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        self.update(
            job_id,
            status="completed",
            progress=100,
            message="完了しました。",
            result=result,
            error=None,
        )

    def fail(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", message="失敗しました。", error=error, result=None)

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
hub = MarketDataHub(api_key=API_KEY, symbols=DEFAULT_SYMBOLS, last_price_store=last_price_store)
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
ml_job_store = MlJobStore(max_jobs=120)


async def _run_quantile_lstm_pipeline(
    *,
    symbol: str,
    years: int = HISTORICAL_DEFAULT_YEARS,
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
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(2, "ヒストリカルデータを取得しています。")

    effective_years = max(5, years)
    historical_data = await hub.historical_payload(symbol=symbol, years=effective_years, refresh=refresh)
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

    if progress_callback is not None:
        progress_callback(5, "モデル学習を開始します。")

    try:
        model_payload = await asyncio.to_thread(
            run_quantile_lstm_forecast,
            points,
            config_payload,
            progress_callback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Quantile LSTM training failed for %s", symbol, exc_info=exc)
        raise HTTPException(status_code=500, detail="Quantile LSTM training failed.") from exc

    if progress_callback is not None:
        progress_callback(100, "結果を整形しました。")

    return {
        "symbol": historical_data.get("symbol"),
        "years": effective_years,
        "historical_source": historical_data.get("source"),
        **model_payload,
    }


async def _run_patchtst_pipeline(
    *,
    symbol: str,
    years: int = HISTORICAL_DEFAULT_YEARS,
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
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback(2, "ヒストリカルデータを取得しています。")

    effective_years = max(5, years)
    historical_data = await hub.historical_payload(symbol=symbol, years=effective_years, refresh=refresh)
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

    if progress_callback is not None:
        progress_callback(5, "PatchTST学習を開始します。")

    try:
        model_payload = await asyncio.to_thread(
            run_patchtst_forecast,
            points,
            config_payload,
            progress_callback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("PatchTST training failed for %s", symbol, exc_info=exc)
        raise HTTPException(status_code=500, detail="PatchTST training failed.") from exc

    if progress_callback is not None:
        progress_callback(100, "結果を整形しました。")

    return {
        "symbol": historical_data.get("symbol"),
        "years": effective_years,
        "historical_source": historical_data.get("source"),
        **model_payload,
    }


def _progress_callback_for_job(job_id: str):
    def _cb(progress: int, message: str) -> None:
        ml_job_store.update(
            job_id,
            status="running",
            progress=max(0, min(100, int(progress))),
            message=message,
        )

    return _cb


async def _run_quantile_lstm_job(job_id: str, req: QuantileLstmJobRequest) -> None:
    try:
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_quantile_lstm_pipeline(
            symbol=req.symbol,
            years=req.years,
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
        )
        ml_job_store.complete(job_id, result=result)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


async def _run_patchtst_job(job_id: str, req: QuantileLstmJobRequest) -> None:
    try:
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_patchtst_pipeline(
            symbol=req.symbol,
            years=req.years,
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
        )
        ml_job_store.complete(job_id, result=result)
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


@app.get("/api/ml/models")
async def ml_models() -> JSONResponse:
    return ok_json_response(models=ML_MODEL_CATALOG)


@app.get("/api/ml/quantile-lstm")
async def quantile_lstm_forecast(
    symbol: str,
    years: int = HISTORICAL_DEFAULT_YEARS,
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
        years=years,
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
    years: int = HISTORICAL_DEFAULT_YEARS,
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
        years=years,
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

    job_id = ml_job_store.create(kind="quantile_lstm", symbol=req.symbol)
    asyncio.create_task(_run_quantile_lstm_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@app.post("/api/ml/patchtst/jobs")
async def start_patchtst_job(req: QuantileLstmJobRequest) -> JSONResponse:
    symbol = normalize_symbols([req.symbol])
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    req.symbol = symbol[0]

    job_id = ml_job_store.create(kind="patchtst_quantile", symbol=req.symbol)
    asyncio.create_task(_run_patchtst_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@app.get("/api/ml/jobs/{job_id}")
async def ml_job_status(job_id: str) -> JSONResponse:
    payload = ml_job_store.get(job_id)
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
