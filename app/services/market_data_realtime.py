"""Realtime transport and fallback mixin for ``MarketDataHub``."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import websockets
from fastapi import HTTPException

from ..config import (
    API_USAGE_URL,
    LOGGER,
    MARKET_CLOSED_SLEEP_SEC,
    REST_PRICE_URL,
    WS_URL_TEMPLATE,
)
from ..utils import rest_request_spacing_seconds


class MarketDataRealtimeMixin:
    async def _set_mode(self, mode: str, ws_connected: bool) -> None:
        changed = self.mode != mode or self.ws_connected != ws_connected
        self.mode = mode
        self.ws_connected = ws_connected
        if changed:
            await self.publish({"type": "status", "data": await self.status_payload()})

    async def _websocket_worker(self) -> None:
        if not self._uses_twelvedata():
            while not self._stop_event.is_set():
                await self._set_mode("rest-only", False)
                await asyncio.sleep(1)
            return

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

            ws_url = WS_URL_TEMPLATE.format(api_key=self.twelvedata_api_key)
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
            source_detail={
                "provider": "twelvedata",
                "endpoint": "websocket_quotes_price",
            },
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
        if self.provider == "both":
            td_task = self._poll_one_symbol_twelvedata(client, symbol)
            fmp_task = self._poll_one_symbol_fmp(client, symbol)
            td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
            if isinstance(td_result, Exception):
                LOGGER.warning("REST fallback failed (TD) for %s: %s", symbol, td_result)
            if isinstance(fmp_result, Exception):
                LOGGER.warning("REST fallback failed (FMP) for %s: %s", symbol, fmp_result)
            if td_result is True:
                return
            if fmp_result is True:
                return
            return
        if self.provider == "fmp":
            await self._poll_one_symbol_fmp(client, symbol)
            return
        await self._poll_one_symbol_twelvedata(client, symbol)

    async def _poll_one_symbol_twelvedata(self, client: httpx.AsyncClient, symbol: str) -> bool:
        if not self._uses_twelvedata():
            return False

        try:
            response = await client.get(
                REST_PRICE_URL,
                params={
                    "apikey": self.twelvedata_api_key,
                    "symbol": symbol,
                },
            )
            async with self._credits_lock:
                await self._update_minute_credits_from_response(response)
                await self._consume_daily_credit_estimate(1, source=f"rest:{symbol}")
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("REST fallback failed for %s: %s", symbol, exc)
            return False

        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("REST API error for %s: %s", symbol, payload.get("message"))
            return False

        price = payload.get("price") if isinstance(payload, dict) else None
        if price is None:
            return False

        record = self._build_price_record(symbol=symbol, price=price, source="rest")
        record["source_detail"] = {
            "provider": "twelvedata",
            "endpoint": "price",
        }
        await self._store_and_publish_price(record)
        return True

    async def _poll_one_symbol_fmp(self, client: httpx.AsyncClient, symbol: str) -> bool:
        if not self._uses_fmp():
            return False
        quote = await self._fetch_quote_fmp(client, symbol)
        price = self._pick_float(quote, "price", "close")
        if price is None:
            return False
        record = self._build_price_record(
            symbol=symbol,
            price=price,
            source="rest",
            timestamp=quote.get("timestamp") or quote.get("datetime"),
            source_detail={
                "provider": "fmp",
                "endpoint": "quote",
            },
        )
        await self._store_and_publish_price(record)
        return True

    async def refresh_api_credits(self) -> dict[str, Any]:
        if not self._uses_twelvedata():
            return await self.status_payload()

        timeout = httpx.Timeout(10.0, connect=5.0)
        async with self._credits_lock, httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(API_USAGE_URL, params={"apikey": self.twelvedata_api_key})
            await self._update_minute_credits_from_response(response)
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "error":
                message = payload.get("message", "Failed to fetch API usage.")
                raise HTTPException(status_code=400, detail=message)
            await self._update_daily_credits_from_api_usage(payload)
            return await self.status_payload()
