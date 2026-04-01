"""Implementation helpers for overview, quote, and sparkline queries."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx

from ..config import BETA_MARKET_RECHECK_SEC, FMP_QUOTE_URL, LOGGER, QUOTE_URL, SPARKLINE_POINTS
from ..utils import normalize_symbols
from .market_data_provider_clients import FmpClient, TwelveDataClient
from .market_data_queries_overview_support import OverviewInputs


class MarketDataOverviewOps:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _td_client(self, client: httpx.AsyncClient) -> TwelveDataClient:
        return TwelveDataClient(client, getattr(self.owner, "twelvedata_api_key", ""))

    def _fmp_client(self, client: httpx.AsyncClient) -> FmpClient:
        return FmpClient(client, getattr(self.owner, "fmp_api_key", ""))

    async def fetch_overview_inputs(
        self,
        *,
        client: httpx.AsyncClient,
        request: Any,
        refresh: bool,
    ) -> OverviewInputs:
        quote_task = self.owner._fetch_quote(client, request.symbol)
        day_task = self.owner._fetch_full_daily_series(client, request.symbol, refresh=refresh)
        quote, day_points = await asyncio.gather(quote_task, day_task)

        m1_points: list[dict[str, Any]] = []
        m5_points: list[dict[str, Any]] = []
        if request.include_intraday:
            m1_points, m5_points = await asyncio.gather(
                self.owner._fetch_series(client, request.symbol, "1min", outputsize=390),
                self.owner._fetch_series(client, request.symbol, "5min", outputsize=390),
            )

        market_context: dict[str, Any] | None = None
        if request.include_market:
            market_context = await self.fetch_market_context(
                client,
                refresh=refresh,
                include_qqq=request.include_qqq,
            )

        return OverviewInputs(
            quote=quote,
            day_points=day_points,
            m1_points=m1_points,
            m5_points=m5_points,
            market_context=market_context,
        )

    async def fetch_market_context(
        self,
        client: httpx.AsyncClient,
        refresh: bool = False,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        spy_points = await self.owner._fetch_full_daily_series(
            client,
            "SPY",
            refresh=refresh,
            min_recheck_sec=BETA_MARKET_RECHECK_SEC,
        )
        qqq_points: list[dict[str, Any]] = []
        if include_qqq:
            qqq_points = await self.owner._fetch_full_daily_series(
                client,
                "QQQ",
                refresh=refresh,
                min_recheck_sec=BETA_MARKET_RECHECK_SEC,
            )
        return {
            "spy_points": spy_points[-90:] if len(spy_points) > 90 else spy_points,
            "qqq_points": qqq_points[-90:] if len(qqq_points) > 90 else qqq_points,
        }

    async def fetch_quote(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        if self.owner.provider == "both":
            return await self.fetch_quote_both(client, symbol)
        if self.owner.provider == "fmp":
            return await self.fetch_quote_fmp(client, symbol)
        return await self.fetch_quote_twelvedata(client, symbol)

    async def fetch_quote_twelvedata(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        try:
            api_response = await self._td_client(client).get_quote(QUOTE_URL, symbol=symbol)
            async with self.owner._credits_lock:
                await self.owner._update_minute_credits_from_response(api_response.response)
                await self.owner._consume_daily_credit_estimate(1, source=f"quote:{symbol}")
            payload = api_response.payload
        except Exception as exc:
            LOGGER.warning("Quote fetch failed for %s: %s", symbol, exc)
            return {}
        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Quote API error for %s: %s", symbol, payload.get("message"))
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized = dict(payload)
        normalized["_source_provider"] = "twelvedata"
        normalized["_source_detail"] = self.owner._quote_source_detail("twelvedata")
        return normalized

    async def fetch_quote_both(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        td_task = self.owner._fetch_quote_twelvedata(client, symbol)
        fmp_task = self.owner._fetch_quote_fmp(client, symbol)
        td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
        td_quote = td_result if isinstance(td_result, dict) else {}
        fmp_quote = fmp_result if isinstance(fmp_result, dict) else {}

        if isinstance(td_result, Exception):
            LOGGER.warning("Quote fetch failed (TD) for %s: %s", symbol, td_result)
        if isinstance(fmp_result, Exception):
            LOGGER.warning("Quote fetch failed (FMP) for %s: %s", symbol, fmp_result)

        merged, merged_detail = self.owner._merge_quote_payloads_with_source(
            primary=td_quote,
            primary_name="twelvedata",
            secondary=fmp_quote,
            secondary_name="fmp",
        )
        if merged:
            merged["_source_provider"] = "both"
            merged["_source_detail"] = merged_detail
            return merged
        return td_quote if td_quote else fmp_quote

    async def fetch_quote_fmp(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        try:
            payload = await self._fmp_client(client).get_quote(FMP_QUOTE_URL, symbol=symbol)
        except Exception as exc:
            LOGGER.warning("FMP quote fetch failed for %s: %s", symbol, exc)
            return {}

        row: dict[str, Any] | None = None
        if isinstance(payload, list) and payload:
            first = payload[0]
            row = first if isinstance(first, dict) else None
        elif isinstance(payload, dict):
            if self.owner._is_fmp_error(payload):
                LOGGER.warning("FMP quote API error for %s: %s", symbol, payload.get("Error Message"))
                return {}
            row = payload

        if not isinstance(row, dict):
            return {}

        return {
            "symbol": row.get("symbol") or symbol,
            "name": row.get("name"),
            "exchange": row.get("exchange") or row.get("exchangeShortName"),
            "price": row.get("price"),
            "close": row.get("price") or row.get("close"),
            "previous_close": row.get("previousClose"),
            "open": row.get("open"),
            "high": row.get("dayHigh") or row.get("high"),
            "low": row.get("dayLow") or row.get("low"),
            "volume": row.get("volume"),
            "bid": row.get("bid"),
            "ask": row.get("ask"),
            "timestamp": row.get("timestamp"),
            "datetime": row.get("timestamp"),
            "_source_provider": "fmp",
            "_source_detail": self.owner._quote_source_detail("fmp"),
        }

    async def fetch_sparkline_item(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None:
        points = await self.owner._fetch_series(
            client=client,
            symbol=symbol,
            interval="1day",
            outputsize=max(SPARKLINE_POINTS + 2, 32),
        )
        if not points:
            return None

        values: list[tuple[str, float]] = []
        for item in points:
            dt = str(item.get("t", "")).strip()
            close_value = self.owner._try_parse_float(item.get("c"))
            if not dt or close_value is None:
                continue
            values.append((dt, close_value))

        if len(values) < 2:
            return None

        values.sort(key=lambda item: item[0], reverse=True)
        quote = await self.owner._fetch_quote(client, symbol)
        current_price = self.owner._pick_float(quote, "close", "price")
        reference_close = self.owner._pick_float(quote, "previous_close", "prev_close")
        updated_at = self.owner._best_updated_at(quote, [], [])
        if reference_close is None and len(values) >= 2:
            reference_close = values[1][1]
        change_abs = None
        change_pct = None
        if current_price is not None and reference_close is not None and reference_close > 0:
            change_abs = current_price - reference_close
            change_pct = (change_abs / reference_close) * 100

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
        trend_source = self.owner._series_source_descriptor(points)
        return {
            "symbol": symbol,
            "latest_close": latest_completed_close,
            "latest_close_date": completed[0][0],
            "previous_close": previous_completed_close,
            "previous_close_date": completed[1][0] if len(completed) >= 2 else None,
            "current_price": current_price,
            "reference_close": reference_close,
            "change_abs": change_abs,
            "change_pct": change_pct,
            "updated_at": updated_at,
            "trend_30d": trend_values,
            "trend_from": recent_asc[0][0],
            "trend_to": recent_asc[-1][0],
            "points": len(trend_values),
            "source": trend_source,
        }

    @staticmethod
    def normalize_symbols(symbols: list[str]) -> list[str]:
        return normalize_symbols(symbols)
