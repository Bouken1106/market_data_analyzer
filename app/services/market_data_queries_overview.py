"""Overview, quote, and sparkline helpers for MarketData query mixins."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    BETA_MARKET_RECHECK_SEC,
    FMP_QUOTE_URL,
    LOGGER,
    MAX_BASIC_SYMBOLS,
    OVERVIEW_CACHE_TTL_SEC,
    QUOTE_URL,
    SPARKLINE_CACHE_TTL_SEC,
    SPARKLINE_POINTS,
    SYMBOL_PATTERN,
)
from ..utils import normalize_symbols


@dataclass(frozen=True)
class OverviewRequest:
    symbol: str
    include_intraday: bool
    include_market: bool
    include_qqq: bool

    @property
    def cache_key(self) -> tuple[str, bool, bool, bool]:
        return (
            self.symbol,
            self.include_intraday,
            self.include_market,
            self.include_qqq,
        )


class MarketDataOverviewMixin:
    async def security_overview_payload(
        self,
        symbol: str,
        refresh: bool = False,
        include_intraday: bool = True,
        include_market: bool = True,
        include_qqq: bool = True,
    ) -> dict[str, Any]:
        request = self._build_overview_request(
            symbol=symbol,
            include_intraday=include_intraday,
            include_market=include_market,
            include_qqq=include_qqq,
        )

        async with self._overview_lock:
            cached = self._overview_cache.get(request.cache_key)
            if cached and not refresh and self._is_cache_fresh(cached.get("cached_epoch"), OVERVIEW_CACHE_TTL_SEC):
                payload = dict(cached["payload"])
                payload["source"] = "cache"
                return payload

        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            quote_task = self._fetch_quote(client, request.symbol)
            day_task = self._fetch_full_daily_series(client, request.symbol, refresh=refresh)
            quote, day_points = await asyncio.gather(quote_task, day_task)

            m1_points: list[dict[str, Any]] = []
            m5_points: list[dict[str, Any]] = []
            if request.include_intraday:
                m1_points, m5_points = await asyncio.gather(
                    self._fetch_series(client, request.symbol, "1min", outputsize=390),
                    self._fetch_series(client, request.symbol, "5min", outputsize=390),
                )

            market_context: dict[str, Any] | None = None
            if request.include_market:
                market_context = await self._fetch_market_context(
                    client,
                    refresh=refresh,
                    include_qqq=request.include_qqq,
                )

        if not day_points:
            raise HTTPException(status_code=404, detail="No overview data found for this symbol.")

        latest_day = day_points[-1]
        price_context = self._build_price_context(
            quote=quote,
            day_points=day_points,
            m1_points=m1_points,
        )

        ma_short = self._moving_average(day_points, window=20)
        ma_mid = self._moving_average(day_points, window=50)
        atr_14 = self._atr(day_points, window=14)
        intraday_vwap_1m = self._intraday_vwap(m1_points)
        intraday_vwap_5m = self._intraday_vwap(m5_points)

        spy_points = market_context.get("spy_points", []) if isinstance(market_context, dict) else []
        qqq_points = market_context.get("qqq_points", []) if isinstance(market_context, dict) else []
        beta_60, corr_60 = self._beta_and_corr_60d(day_points, spy_points) if request.include_market else (None, None)

        spy_latest = spy_points[-1]["c"] if spy_points else None
        spy_prev = spy_points[-2]["c"] if len(spy_points) >= 2 else None
        qqq_latest = qqq_points[-1]["c"] if qqq_points else None
        qqq_prev = qqq_points[-2]["c"] if len(qqq_points) >= 2 else None

        source_detail = self._build_overview_source_detail(
            quote=quote,
            day_points=day_points,
            m1_points=m1_points,
            m5_points=m5_points,
            spy_points=spy_points,
            qqq_points=qqq_points,
            price_context=price_context,
        )
        overview_payload = {
            "symbol": request.symbol,
            "name": self._pick_string(quote, "name", "instrument_name"),
            "exchange": self._pick_string(quote, "exchange"),
            "price": {
                "current": price_context["current_price"],
                "previous_close": price_context["previous_close"],
                "change_abs": price_context["change_abs"],
                "change_pct": price_context["change_pct"],
                "day_open": price_context["day_open"],
                "day_high": price_context["day_high"],
                "day_low": price_context["day_low"],
                "gap_abs": price_context["gap_abs"],
                "gap_pct": price_context["gap_pct"],
                "updated_at": self._best_updated_at(quote, m1_points, day_points),
                "delay_note": self._delay_note(),
            },
            "volume": {
                "today": price_context["day_volume"],
                "avg20": price_context["avg_volume_20"],
                "avg_ratio": price_context["avg_volume_ratio"],
                "turnover": price_context["turnover"],
            },
            "spread": {
                "bid": price_context["bid"],
                "ask": price_context["ask"],
                "spread_abs": price_context["spread_abs"],
                "spread_pct": price_context["spread_pct"],
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
                "nasdaq_proxy": self._build_market_item("QQQ", qqq_latest, qqq_prev) if request.include_qqq else None,
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
            "source": f"{self.provider}-live",
            "source_detail": source_detail,
        }

        async with self._overview_lock:
            self._overview_cache[request.cache_key] = {
                "cached_epoch": time.time(),
                "payload": overview_payload,
            }

        return overview_payload

    @staticmethod
    def _quote_source_detail(provider: str) -> dict[str, str]:
        return {
            "symbol": provider,
            "name": provider,
            "instrument_name": provider,
            "exchange": provider,
            "price": provider,
            "close": provider,
            "previous_close": provider,
            "prev_close": provider,
            "open": provider,
            "high": provider,
            "low": provider,
            "volume": provider,
            "bid": provider,
            "ask": provider,
            "timestamp": provider,
            "datetime": provider,
        }

    def _build_overview_request(
        self,
        *,
        symbol: str,
        include_intraday: bool,
        include_market: bool,
        include_qqq: bool,
    ) -> OverviewRequest:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        return OverviewRequest(
            symbol=normalized,
            include_intraday=bool(include_intraday),
            include_market=bool(include_market),
            include_qqq=bool(include_qqq),
        )

    def _build_price_context(
        self,
        *,
        quote: dict[str, Any],
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_day = day_points[-1]
        previous_day = day_points[-2] if len(day_points) >= 2 else None
        day_series_source = self._series_source_descriptor(day_points)
        quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
        if not isinstance(quote_source_detail, dict):
            quote_source_detail = {}

        current_price = self._pick_float(quote, "close", "price")
        if current_price is None:
            current_price = latest_day["c"]
        previous_close = self._pick_float(quote, "previous_close", "prev_close")
        if previous_close is None and previous_day:
            previous_close = previous_day["c"]

        field_values = {
            "day_open": self._pick_float(quote, "open"),
            "day_high": self._pick_float(quote, "high"),
            "day_low": self._pick_float(quote, "low"),
            "day_volume": self._pick_float(quote, "volume"),
            "bid": self._pick_float(quote, "bid"),
            "ask": self._pick_float(quote, "ask"),
        }
        field_sources = {
            "day_open_source": quote_source_detail.get("open"),
            "day_high_source": quote_source_detail.get("high"),
            "day_low_source": quote_source_detail.get("low"),
            "day_volume_source": quote_source_detail.get("volume"),
            "current_price_source": quote_source_detail.get("close") or quote_source_detail.get("price"),
            "previous_close_source": quote_source_detail.get("previous_close") or quote_source_detail.get("prev_close"),
        }

        self._fill_day_fields_from_intraday(field_values, field_sources, m1_points)
        self._fill_day_fields_from_daily_series(field_values, field_sources, latest_day, day_series_source)

        change_abs, change_pct = self._compute_change_metrics(
            current=current_price,
            previous=previous_close,
        )
        gap_abs, gap_pct = self._compute_change_metrics(
            current=field_values["day_open"],
            previous=previous_close,
        )
        avg_volume_20, avg_volume_ratio, turnover = self._compute_volume_metrics(
            day_points=day_points,
            day_volume=field_values["day_volume"],
            current_price=current_price,
        )
        spread_abs, spread_pct = self._compute_spread_metrics(
            bid=field_values["bid"],
            ask=field_values["ask"],
            current_price=current_price,
        )

        return {
            "current_price": current_price,
            "previous_close": previous_close,
            "day_open": field_values["day_open"],
            "day_high": field_values["day_high"],
            "day_low": field_values["day_low"],
            "day_volume": field_values["day_volume"],
            "bid": field_values["bid"],
            "ask": field_values["ask"],
            "change_abs": change_abs,
            "change_pct": change_pct,
            "gap_abs": gap_abs,
            "gap_pct": gap_pct,
            "avg_volume_20": avg_volume_20,
            "avg_volume_ratio": avg_volume_ratio,
            "turnover": turnover,
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            **field_sources,
        }

    def _fill_day_fields_from_intraday(
        self,
        field_values: dict[str, float | None],
        field_sources: dict[str, str | None],
        m1_points: list[dict[str, Any]],
    ) -> None:
        if not m1_points:
            return
        latest_session = self._extract_latest_session_points(m1_points)
        if not latest_session:
            return

        if field_values["day_open"] is None:
            field_values["day_open"] = latest_session[0]["o"]
        if field_values["day_high"] is None:
            field_values["day_high"] = max((item["h"] for item in latest_session), default=None)
        if field_values["day_low"] is None:
            field_values["day_low"] = min((item["l"] for item in latest_session), default=None)
        if field_values["day_volume"] is None:
            field_values["day_volume"] = sum((item["v"] or 0.0) for item in latest_session)

        if not field_sources["day_open_source"]:
            field_sources["day_open_source"] = "intraday_1min"
        if not field_sources["day_high_source"]:
            field_sources["day_high_source"] = "intraday_1min"
        if not field_sources["day_low_source"]:
            field_sources["day_low_source"] = "intraday_1min"
        if not field_sources["day_volume_source"] and field_values["day_volume"] is not None:
            field_sources["day_volume_source"] = "intraday_1min"

    @staticmethod
    def _fill_day_fields_from_daily_series(
        field_values: dict[str, float | None],
        field_sources: dict[str, str | None],
        latest_day: dict[str, Any],
        day_series_source: str,
    ) -> None:
        fallback_source = f"daily_series({day_series_source})"
        if field_values["day_open"] is None:
            field_values["day_open"] = latest_day["o"]
            field_sources["day_open_source"] = fallback_source
        if field_values["day_high"] is None:
            field_values["day_high"] = latest_day["h"]
            field_sources["day_high_source"] = fallback_source
        if field_values["day_low"] is None:
            field_values["day_low"] = latest_day["l"]
            field_sources["day_low_source"] = fallback_source
        if field_values["day_volume"] is None:
            field_values["day_volume"] = latest_day["v"]
            field_sources["day_volume_source"] = fallback_source
        if not field_sources["current_price_source"]:
            field_sources["current_price_source"] = fallback_source
        if field_sources["previous_close_source"] is None:
            field_sources["previous_close_source"] = fallback_source

    @staticmethod
    def _compute_change_metrics(current: float | None, previous: float | None) -> tuple[float | None, float | None]:
        if current is None or previous is None or previous <= 0:
            return None, None
        change_abs = current - previous
        return change_abs, (change_abs / previous) * 100

    @staticmethod
    def _compute_volume_metrics(
        *,
        day_points: list[dict[str, Any]],
        day_volume: float | None,
        current_price: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        recent_daily_volumes = [p["v"] for p in day_points[-21:-1] if p.get("v") is not None and p["v"] > 0]
        avg_volume_20 = sum(recent_daily_volumes) / len(recent_daily_volumes) if recent_daily_volumes else None
        avg_volume_ratio = (
            (day_volume / avg_volume_20)
            if day_volume is not None and avg_volume_20 is not None and avg_volume_20 > 0
            else None
        )
        turnover = current_price * day_volume if current_price is not None and day_volume is not None else None
        return avg_volume_20, avg_volume_ratio, turnover

    @staticmethod
    def _compute_spread_metrics(
        *,
        bid: float | None,
        ask: float | None,
        current_price: float | None,
    ) -> tuple[float | None, float | None]:
        spread_abs = ask - bid if ask is not None and bid is not None else None
        spread_pct = (
            (spread_abs / current_price) * 100
            if spread_abs is not None and current_price is not None and current_price > 0
            else None
        )
        return spread_abs, spread_pct

    def _build_overview_source_detail(
        self,
        *,
        quote: dict[str, Any],
        day_points: list[dict[str, Any]],
        m1_points: list[dict[str, Any]],
        m5_points: list[dict[str, Any]],
        spy_points: list[dict[str, Any]],
        qqq_points: list[dict[str, Any]],
        price_context: dict[str, Any],
    ) -> dict[str, Any]:
        quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
        if not isinstance(quote_source_detail, dict):
            quote_source_detail = {}
        return {
            "mode": self.provider,
            "components": {
                "quote": quote.get("_source_provider", self.provider) if isinstance(quote, dict) else self.provider,
                "daily_series": self._series_source_descriptor(day_points),
                "intraday_1min": self._series_source_descriptor(m1_points),
                "intraday_5min": self._series_source_descriptor(m5_points),
                "market_spy_daily": self._series_source_descriptor(spy_points),
                "market_qqq_daily": self._series_source_descriptor(qqq_points),
            },
            "fields": {
                "price.current": price_context["current_price_source"] or "unknown",
                "price.previous_close": price_context["previous_close_source"] or "unknown",
                "price.day_open": price_context["day_open_source"] or "unknown",
                "price.day_high": price_context["day_high_source"] or "unknown",
                "price.day_low": price_context["day_low_source"] or "unknown",
                "volume.today": price_context["day_volume_source"] or "unknown",
                "spread.bid": quote_source_detail.get("bid") or "unknown",
                "spread.ask": quote_source_detail.get("ask") or "unknown",
            },
        }

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
        if self.provider == "both":
            return await self._fetch_quote_both(client, symbol)
        if self.provider == "fmp":
            return await self._fetch_quote_fmp(client, symbol)
        return await self._fetch_quote_twelvedata(client, symbol)

    async def _fetch_quote_twelvedata(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        try:
            response = await client.get(
                QUOTE_URL,
                params={
                    "apikey": self.twelvedata_api_key,
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
        if not isinstance(payload, dict):
            return {}
        normalized = dict(payload)
        normalized["_source_provider"] = "twelvedata"
        normalized["_source_detail"] = self._quote_source_detail("twelvedata")
        return normalized

    async def _fetch_quote_both(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        td_task = self._fetch_quote_twelvedata(client, symbol)
        fmp_task = self._fetch_quote_fmp(client, symbol)
        td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
        td_quote = td_result if isinstance(td_result, dict) else {}
        fmp_quote = fmp_result if isinstance(fmp_result, dict) else {}

        if isinstance(td_result, Exception):
            LOGGER.warning("Quote fetch failed (TD) for %s: %s", symbol, td_result)
        if isinstance(fmp_result, Exception):
            LOGGER.warning("Quote fetch failed (FMP) for %s: %s", symbol, fmp_result)

        merged, merged_detail = self._merge_quote_payloads_with_source(
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

    async def _fetch_quote_fmp(self, client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
        try:
            response = await client.get(
                FMP_QUOTE_URL,
                params={
                    "apikey": self.fmp_api_key,
                    "symbol": symbol,
                },
            )
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("FMP quote fetch failed for %s: %s", symbol, exc)
            return {}

        row: dict[str, Any] | None = None
        if isinstance(payload, list) and payload:
            first = payload[0]
            row = first if isinstance(first, dict) else None
        elif isinstance(payload, dict):
            if self._is_fmp_error(payload):
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
            "_source_detail": self._quote_source_detail("fmp"),
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
        points = await self._fetch_series(
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
            close_value = self._try_parse_float(item.get("c"))
            if not dt or close_value is None:
                continue
            values.append((dt, close_value))

        if len(values) < 2:
            return None

        values.sort(key=lambda item: item[0], reverse=True)
        quote = await self._fetch_quote(client, symbol)
        current_price = self._pick_float(quote, "close", "price")
        reference_close = self._pick_float(quote, "previous_close", "prev_close")
        updated_at = self._best_updated_at(quote, [], [])
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
        trend_source = self._series_source_descriptor(points)
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
            "source": f"{self.provider}-live",
            "source_detail": {
                "mode": self.provider,
                "dataset": "sparkline_1day",
                "provider": trend_source,
            },
        }
