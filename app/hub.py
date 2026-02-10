"""MarketDataHub – central orchestrator for real-time and historical prices."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import websockets
from fastapi import HTTPException

from .config import (
    API_LIMIT_PER_DAY,
    API_USAGE_URL,
    BETA_MARKET_RECHECK_SEC,
    DAILY_DIFF_MIN_RECHECK_SEC,
    EARLIEST_TIMESTAMP_URL,
    FULL_HISTORY_CHUNK_YEARS,
    FULL_HISTORY_MAX_CHUNKS,
    HISTORICAL_CACHE_TTL_SEC,
    HISTORICAL_DEFAULT_YEARS,
    HISTORICAL_INTERVAL,
    HISTORICAL_MAX_POINTS,
    HISTORICAL_MAX_YEARS,
    LOGGER,
    MARKET_CLOSED_SLEEP_SEC,
    MAX_BASIC_SYMBOLS,
    ML_HISTORY_MAX_MONTHS,
    OVERVIEW_CACHE_TTL_SEC,
    QUOTE_URL,
    REST_PRICE_URL,
    SPARKLINE_CACHE_TTL_SEC,
    SPARKLINE_POINTS,
    SYMBOL_CATALOG_COUNTRY,
    SYMBOL_COUNTRY_MAP_RAW,
    SYMBOL_PATTERN,
    TIME_SERIES_MAX_OUTPUTSIZE,
    TIME_SERIES_URL,
    WS_URL_TEMPLATE,
)
from .market_session import (
    DEFAULT_MARKET_SESSIONS,
    _normalize_country_key,
    infer_country_from_symbol,
    parse_symbol_country_map,
)
from .stores import FullDailyHistoryStore, LastPriceStore
from .utils import (
    _datetime_from_unix,
    fallback_interval_seconds,
    normalize_symbols,
    rest_request_spacing_seconds,
    to_iso8601,
)

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
