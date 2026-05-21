"""Implementation helpers for overview, quote, and sparkline queries."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx

from ..config import LOGGER, settings
from .market_data_sparkline import (
    SparklineValue,
    build_daily_sparkline_payload,
    completed_daily_values,
    daily_close_values,
)
from .market_data_provider_clients import owner_fmp_client, owner_twelvedata_client
from .market_data_queries_overview_support import OverviewInputs


class MarketDataOverviewOps:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

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
            min_recheck_sec=settings.historical.beta_market_recheck_sec,
        )
        qqq_points: list[dict[str, Any]] = []
        if include_qqq:
            qqq_points = await self.owner._fetch_full_daily_series(
                client,
                "QQQ",
                refresh=refresh,
                min_recheck_sec=settings.historical.beta_market_recheck_sec,
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
        provider_symbol = self.owner._format_twelvedata_symbol(symbol)
        try:
            api_response = await owner_twelvedata_client(
                self.owner,
                client,
            ).get_quote(settings.endpoints.quote_url, symbol=provider_symbol)
            async with self.owner._credits_lock:
                await self.owner._update_minute_credits_from_response(api_response.response)
                await self.owner._consume_daily_credit_estimate(1, source=f"quote:{symbol}")
            payload = api_response.payload
        except Exception as exc:
            LOGGER.warning("Quote fetch failed for %s (%s): %s", symbol, provider_symbol, exc)
            return {}
        if isinstance(payload, dict) and payload.get("status") == "error":
            LOGGER.warning("Quote API error for %s (%s): %s", symbol, provider_symbol, payload.get("message"))
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized = dict(payload)
        normalized["symbol"] = symbol
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
            payload = await owner_fmp_client(self.owner, client).get_quote(
                settings.endpoints.fmp_quote_url,
                symbol=symbol,
            )
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

    def _daily_close_values(self, points: list[dict[str, Any]]) -> list[SparklineValue]:
        return daily_close_values(points, parse_close=self.owner._try_parse_float)

    @staticmethod
    def _completed_daily_values(values: list[SparklineValue], *, today_iso: str) -> list[SparklineValue]:
        return completed_daily_values(values, today_iso=today_iso)

    def _build_sparkline_snapshot(
        self,
        *,
        symbol: str,
        points: list[dict[str, Any]],
        completed: list[SparklineValue],
        quote: dict[str, Any],
    ) -> dict[str, Any]:
        current_price = self.owner._pick_float(quote, "close", "price")
        reference_close = self.owner._pick_float(quote, "previous_close", "prev_close")
        updated_at = self.owner._best_updated_at(quote, [], [])
        if reference_close is None and len(completed) >= 2:
            reference_close = completed[1][1]

        trend_source = self.owner._series_source_descriptor(points)
        payload = build_daily_sparkline_payload(
            symbol=symbol,
            completed=completed,
            max_points=settings.overview.sparkline_points,
            current_price=current_price,
            reference_close=reference_close,
            updated_at=updated_at,
            source=trend_source,
        )
        return payload or {}

    async def fetch_sparkline_item(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any] | None:
        points = await self.owner._fetch_series(
            client=client,
            symbol=symbol,
            interval="1day",
            outputsize=max(settings.overview.sparkline_points + 2, 32),
        )
        if not points:
            return None

        values = self._daily_close_values(points)
        if len(values) < 2:
            return None

        quote = await self.owner._fetch_quote(client, symbol)
        completed = self._completed_daily_values(values, today_iso=date.today().isoformat())
        if not completed:
            return None
        return self._build_sparkline_snapshot(
            symbol=symbol,
            points=points,
            completed=completed,
            quote=quote,
        )
