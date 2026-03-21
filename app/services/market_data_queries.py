"""Data-query and analytics mixin for ``MarketDataHub``."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    BETA_MARKET_RECHECK_SEC,
    DAILY_DIFF_MIN_RECHECK_SEC,
    EARLIEST_TIMESTAMP_URL,
    FMP_BALANCE_SHEET_URL,
    FMP_CASH_FLOW_URL,
    FMP_DIVIDENDS_URL,
    FMP_DIVIDEND_ADJUSTED_PRICE_URL,
    FMP_HISTORICAL_EOD_URL,
    FMP_INCOME_STATEMENT_URL,
    FMP_KEY_METRICS_TTM_URL,
    FMP_PROFILE_URL,
    FMP_QUOTE_URL,
    FMP_REFERENCE_CACHE_TTL_SEC,
    FMP_RATIOS_TTM_URL,
    FMP_SPLITS_URL,
    FULL_HISTORY_CHUNK_YEARS,
    FULL_HISTORY_MAX_CHUNKS,
    HISTORICAL_CACHE_TTL_SEC,
    HISTORICAL_DEFAULT_YEARS,
    HISTORICAL_INTERVAL,
    HISTORICAL_MAX_POINTS,
    HISTORICAL_MAX_YEARS,
    JQUANTS_API_KEY,
    JQUANTS_DAILY_BARS_URL,
    LOGGER,
    MAX_BASIC_SYMBOLS,
    ML_HISTORY_MAX_MONTHS,
    OVERVIEW_CACHE_TTL_SEC,
    QUOTE_URL,
    SPARKLINE_CACHE_TTL_SEC,
    SPARKLINE_POINTS,
    SYMBOL_PATTERN,
    TIME_SERIES_MAX_OUTPUTSIZE,
    TIME_SERIES_URL,
)
from ..market_session import infer_country_from_symbol
from ..ohlcv import (
    latest_session_points,
    merge_points_by_timestamp as merge_ohlcv_points,
    normalize_ohlcv_point,
)
from ..utils import _datetime_from_unix, normalize_symbols, to_iso8601


class MarketDataQueriesMixin:
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
        requested_years = max(1, int(years))
        fetch_full_history = months is None and requested_years > HISTORICAL_MAX_YEARS
        years = requested_years if fetch_full_history else max(1, min(requested_years, HISTORICAL_MAX_YEARS))
        months = None if months is None else max(1, min(int(months), ML_HISTORY_MAX_MONTHS))

        if months is None:
            cache_key = (normalized, "years:max") if fetch_full_history else (normalized, f"years:{years}")
        else:
            cache_key = (normalized, f"months:{months}")
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
            if fetch_full_history and str(HISTORICAL_INTERVAL).strip().lower() in {"1day", "1d", "day"}:
                points = await self._fetch_full_daily_series(client, symbol=normalized, refresh=refresh)
                source_detail = {
                    "provider": self.provider,
                    "mode": "full_daily_history",
                }
            else:
                if months is None:
                    requested_days = (365 * years) + (years // 4)
                else:
                    requested_days = (31 * months) + 7
                estimated_points = max(200, int(requested_days * 0.8))
                outputsize = min(
                    TIME_SERIES_MAX_OUTPUTSIZE,
                    max(HISTORICAL_MAX_POINTS, estimated_points),
                )
                points, source_detail = await self._fetch_historical_points_with_detail(
                    client,
                    symbol=normalized,
                    interval=HISTORICAL_INTERVAL,
                    outputsize=outputsize,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
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
            "source": f"{self.provider}-live",
            "source_detail": source_detail,
        }

        async with self._historical_lock:
            self._historical_cache[cache_key] = {
                "cached_epoch": time.time(),
                "payload": historical_payload,
            }

        return historical_payload

    async def _fetch_historical_points_with_detail(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._should_use_jquants_for_symbol(symbol, interval):
            points = await self._fetch_series_jquants(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            detail = {
                "mode": "jquants",
                "dataset": "historical_daily",
                "provider": "jquants",
                "points": len(points),
            }
            return points, detail

        if self.provider == "both":
            td_task = self._fetch_series_twelvedata(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            fmp_task = self._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
            td_points, fmp_points = await asyncio.gather(td_task, fmp_task)
            merged = self._merge_points_by_timestamp(fmp_points, td_points)
            detail = {
                "mode": "both",
                "dataset": "historical_daily",
                "merge_policy": "twelvedata_overrides_fmp_on_same_timestamp",
                "providers": {
                    "twelvedata_points": len(td_points),
                    "fmp_points": len(fmp_points),
                    "merged_points": len(merged),
                },
            }
            return merged, detail

        points = await self._fetch_series(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        provider_name = "fmp" if self.provider == "fmp" else "twelvedata"
        detail = {
            "mode": self.provider,
            "dataset": "historical_daily",
            "provider": provider_name,
            "points": len(points),
        }
        return points, detail

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
        day_series_source = self._series_source_descriptor(day_points)

        quote_price = self._pick_float(quote, "close", "price")
        quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
        if not isinstance(quote_source_detail, dict):
            quote_source_detail = {}
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

        day_open_source = quote_source_detail.get("open")
        day_high_source = quote_source_detail.get("high")
        day_low_source = quote_source_detail.get("low")
        day_volume_source = quote_source_detail.get("volume")
        if day_open is None:
            day_open = latest_day["o"]
            day_open_source = f"daily_series({day_series_source})"
        if day_high is None:
            day_high = latest_day["h"]
            day_high_source = f"daily_series({day_series_source})"
        if day_low is None:
            day_low = latest_day["l"]
            day_low_source = f"daily_series({day_series_source})"
        if day_volume is None:
            day_volume = latest_day["v"]
            day_volume_source = f"daily_series({day_series_source})"

        if m1_points:
            latest_session = self._extract_latest_session_points(m1_points)
            if latest_session:
                if not day_open_source:
                    day_open_source = "intraday_1min"
                if not day_high_source:
                    day_high_source = "intraday_1min"
                if not day_low_source:
                    day_low_source = "intraday_1min"
                if not day_volume_source and day_volume is not None:
                    day_volume_source = "intraday_1min"

        current_price_source = quote_source_detail.get("close") or quote_source_detail.get("price")
        if not current_price_source:
            current_price_source = f"daily_series({day_series_source})"
        previous_close_source = quote_source_detail.get("previous_close") or quote_source_detail.get("prev_close")
        if not previous_close_source and previous_close is not None:
            previous_close_source = f"daily_series({day_series_source})"

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
                "delay_note": self._delay_note(),
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
            "source": f"{self.provider}-live",
            "source_detail": {
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
                    "price.current": current_price_source,
                    "price.previous_close": previous_close_source,
                    "price.day_open": day_open_source or "unknown",
                    "price.day_high": day_high_source or "unknown",
                    "price.day_low": day_low_source or "unknown",
                    "volume.today": day_volume_source or "unknown",
                    "spread.bid": quote_source_detail.get("bid") or "unknown",
                    "spread.ask": quote_source_detail.get("ask") or "unknown",
                },
            },
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

    async def fmp_reference_payload(
        self,
        symbol: str,
        refresh: bool = False,
        cache_only: bool = False,
    ) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        if not self.fmp_api_key:
            raise HTTPException(status_code=400, detail="FMP_API_KEY is required for reference data.")

        async with self._fmp_reference_lock:
            cached = self._fmp_reference_cache.get(normalized)
            if cached and not refresh:
                is_fresh = self._is_cache_fresh(cached.get("cached_epoch"), FMP_REFERENCE_CACHE_TTL_SEC)
                if not is_fresh and not cache_only:
                    cached = None
                if cached is not None:
                    payload = dict(cached.get("payload") or {})
                    payload["source"] = "cache-memory" if is_fresh else "cache-memory-stale"
                    payload["cache_ttl_sec"] = FMP_REFERENCE_CACHE_TTL_SEC
                    payload["cache_stale"] = not is_fresh
                    return payload

        if not refresh:
            disk_cached = await self.fmp_reference_store.get(normalized)
            if isinstance(disk_cached, dict):
                cached_at = self._parse_iso_epoch(disk_cached.get("cached_at"))
                is_fresh = cached_at is not None and self._is_cache_fresh(cached_at, FMP_REFERENCE_CACHE_TTL_SEC)
                if is_fresh or cache_only:
                    payload = dict(disk_cached)
                    payload["source"] = "cache-disk" if is_fresh else "cache-disk-stale"
                    payload["cache_ttl_sec"] = FMP_REFERENCE_CACHE_TTL_SEC
                    payload["cache_stale"] = not is_fresh
                    async with self._fmp_reference_lock:
                        self._fmp_reference_cache[normalized] = {
                            "cached_epoch": time.time(),
                            "payload": payload,
                        }
                    return payload

        if cache_only:
            raise HTTPException(status_code=404, detail="No cached FMP reference data found for this symbol.")

        payload = await self._fetch_fmp_reference_live(normalized)
        async with self._fmp_reference_lock:
            self._fmp_reference_cache[normalized] = {
                "cached_epoch": time.time(),
                "payload": payload,
            }
        await self.fmp_reference_store.upsert(normalized, payload)
        return payload

    async def clear_fmp_reference_cache(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        async with self._fmp_reference_lock:
            self._fmp_reference_cache.pop(normalized, None)
        removed_disk = await self.fmp_reference_store.clear(normalized)
        return {
            "symbol": normalized,
            "removed_memory_cache": True,
            "removed_disk_cache": bool(removed_disk),
        }

    async def _fetch_fmp_reference_live(self, symbol: str) -> dict[str, Any]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        two_years_ago = (date.today() - timedelta(days=366 * 2)).isoformat()
        async with httpx.AsyncClient(timeout=timeout) as client:
            profile_task = self._fmp_get_json(
                client,
                FMP_PROFILE_URL,
                params={"symbol": symbol},
            )
            ratios_task = self._fmp_get_json(
                client,
                FMP_RATIOS_TTM_URL,
                params={"symbol": symbol},
            )
            metrics_task = self._fmp_get_json(
                client,
                FMP_KEY_METRICS_TTM_URL,
                params={"symbol": symbol},
            )
            income_task = self._fmp_get_json(
                client,
                FMP_INCOME_STATEMENT_URL,
                params={"symbol": symbol, "limit": 1},
            )
            bs_task = self._fmp_get_json(
                client,
                FMP_BALANCE_SHEET_URL,
                params={"symbol": symbol, "limit": 1},
            )
            cf_task = self._fmp_get_json(
                client,
                FMP_CASH_FLOW_URL,
                params={"symbol": symbol, "limit": 1},
            )
            hist_task = self._fmp_get_json(
                client,
                FMP_DIVIDEND_ADJUSTED_PRICE_URL,
                params={"symbol": symbol, "from": two_years_ago},
            )
            div_task = self._fmp_get_json(
                client,
                FMP_DIVIDENDS_URL,
                params={"symbol": symbol, "from": two_years_ago},
            )
            split_task = self._fmp_get_json(
                client,
                FMP_SPLITS_URL,
                params={"symbol": symbol, "from": two_years_ago},
            )

            (
                profile_raw,
                ratios_raw,
                metrics_raw,
                income_raw,
                bs_raw,
                cf_raw,
                hist_raw,
                div_raw,
                split_raw,
            ) = await asyncio.gather(
                profile_task,
                ratios_task,
                metrics_task,
                income_task,
                bs_task,
                cf_task,
                hist_task,
                div_task,
                split_task,
            )

        profile = self._first_dict(profile_raw)
        ratios = self._first_dict(ratios_raw)
        metrics = self._first_dict(metrics_raw)
        income = self._first_dict(income_raw)
        balance_sheet = self._first_dict(bs_raw)
        cash_flow = self._first_dict(cf_raw)
        historical = self._extract_historical_rows(hist_raw)
        dividends = self._extract_historical_rows(div_raw)
        splits = self._extract_historical_rows(split_raw)

        if not profile and not historical and not income and not balance_sheet and not cash_flow:
            raise HTTPException(status_code=502, detail="Failed to fetch FMP reference data.")

        adjusted_summary = self._build_adjusted_price_summary(historical)
        payload = {
            "symbol": symbol,
            "source": "fmp-live",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cache_ttl_sec": FMP_REFERENCE_CACHE_TTL_SEC,
            "estimated_api_calls_on_refresh": 9,
            "cost_note": "This payload is cached to reduce API credit usage (Free plan: 250 calls/day).",
            "profile": {
                "company_name": profile.get("companyName") or profile.get("company_name"),
                "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "country": profile.get("country"),
                "website": profile.get("website"),
                "ceo": profile.get("ceo"),
                "description": profile.get("description"),
                "market_cap": self._try_parse_float(profile.get("mktCap") or profile.get("marketCap")),
                "beta": self._try_parse_float(profile.get("beta")),
                "employees": profile.get("fullTimeEmployees"),
                "ipo_date": profile.get("ipoDate"),
            },
            "adjusted_prices": adjusted_summary,
            "corporate_actions": {
                "dividends": self._normalize_actions(dividends, action_type="dividend"),
                "splits": self._normalize_actions(splits, action_type="split"),
            },
            "financials": {
                "ratios_ttm": {
                    "pe_ratio_ttm": self._try_parse_float(ratios.get("peRatioTTM")),
                    "pb_ratio_ttm": self._try_parse_float(ratios.get("priceToBookRatioTTM")),
                    "ps_ratio_ttm": self._try_parse_float(ratios.get("priceToSalesRatioTTM")),
                    "roe_ttm": self._try_parse_float(ratios.get("returnOnEquityTTM")),
                    "net_margin_ttm": self._try_parse_float(ratios.get("netProfitMarginTTM")),
                    "current_ratio_ttm": self._try_parse_float(ratios.get("currentRatioTTM")),
                    "debt_to_equity_ttm": self._try_parse_float(ratios.get("debtEquityRatioTTM")),
                },
                "key_metrics_ttm": {
                    "eps_ttm": self._try_parse_float(metrics.get("epsTTM")),
                    "free_cash_flow_per_share_ttm": self._try_parse_float(metrics.get("freeCashFlowPerShareTTM")),
                    "book_value_per_share_ttm": self._try_parse_float(metrics.get("bookValuePerShareTTM")),
                    "dividend_yield_ttm": self._try_parse_float(metrics.get("dividendYieldTTM")),
                },
                "income_statement_latest": {
                    "date": income.get("date"),
                    "revenue": self._try_parse_float(income.get("revenue")),
                    "gross_profit": self._try_parse_float(income.get("grossProfit")),
                    "operating_income": self._try_parse_float(income.get("operatingIncome")),
                    "net_income": self._try_parse_float(income.get("netIncome")),
                    "eps": self._try_parse_float(income.get("eps")),
                },
                "balance_sheet_latest": {
                    "date": balance_sheet.get("date"),
                    "cash_and_short_term_investments": self._try_parse_float(balance_sheet.get("cashAndShortTermInvestments")),
                    "total_assets": self._try_parse_float(balance_sheet.get("totalAssets")),
                    "total_debt": self._try_parse_float(balance_sheet.get("totalDebt")),
                    "total_liabilities": self._try_parse_float(balance_sheet.get("totalLiabilities")),
                    "total_equity": self._try_parse_float(balance_sheet.get("totalStockholdersEquity")),
                },
                "cash_flow_latest": {
                    "date": cash_flow.get("date"),
                    "operating_cash_flow": self._try_parse_float(cash_flow.get("operatingCashFlow")),
                    "capital_expenditure": self._try_parse_float(cash_flow.get("capitalExpenditure")),
                    "free_cash_flow": self._try_parse_float(cash_flow.get("freeCashFlow")),
                    "dividends_paid": self._try_parse_float(cash_flow.get("dividendsPaid")),
                },
            },
        }
        return payload

    async def _fmp_get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        request_params = dict(params or {})
        request_params["apikey"] = self.fmp_api_key
        response = await client.get(url, params=request_params)
        payload = response.json()
        if isinstance(payload, dict):
            message = str(payload.get("Error Message", "")).strip()
            if message:
                raise HTTPException(status_code=400, detail=f"FMP API error: {message}")
        return payload

    @staticmethod
    def _first_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if payload.get("symbol") and any(isinstance(v, (str, int, float, bool, dict, list, type(None))) for v in payload.values()):
                return payload
            data = payload.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return dict(data[0])
            return {}
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return dict(payload[0])
        return {}

    @staticmethod
    def _extract_historical_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            rows = payload.get("historical")
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, dict)]
            rows = payload.get("data")
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, dict)]
            return []
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        return []

    def _build_adjusted_price_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        cleaned = [dict(item) for item in rows if isinstance(item, dict)]
        cleaned.sort(key=lambda item: str(item.get("date", "")))
        latest = cleaned[-1] if cleaned else {}
        close_value = self._try_parse_float(latest.get("close"))
        adj_close_value = self._try_parse_float(latest.get("adjClose") or latest.get("adjustedClose"))
        factor = None
        if close_value and adj_close_value:
            try:
                factor = adj_close_value / close_value if close_value != 0 else None
            except Exception:
                factor = None

        recent: list[dict[str, Any]] = []
        for item in cleaned[-60:]:
            close_item = self._try_parse_float(item.get("close"))
            adj_item = self._try_parse_float(item.get("adjClose") or item.get("adjustedClose"))
            if close_item is None and adj_item is None:
                continue
            recent.append(
                {
                    "date": item.get("date"),
                    "close": close_item,
                    "adj_close": adj_item,
                    "open": self._try_parse_float(item.get("open")),
                    "high": self._try_parse_float(item.get("high")),
                    "low": self._try_parse_float(item.get("low")),
                    "volume": self._try_parse_float(item.get("volume")),
                }
            )

        return {
            "latest_date": latest.get("date"),
            "latest_close": close_value,
            "latest_adj_close": adj_close_value,
            "latest_adjustment_factor": factor,
            "recent_points": recent,
        }

    def _normalize_actions(self, rows: list[dict[str, Any]], action_type: str) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            row = {
                "date": item.get("date"),
                "label": item.get("label"),
            }
            if action_type == "dividend":
                row["dividend"] = self._try_parse_float(item.get("dividend"))
                row["adj_dividend"] = self._try_parse_float(item.get("adjDividend"))
                row["record_date"] = item.get("recordDate")
                row["payment_date"] = item.get("paymentDate")
            else:
                row["numerator"] = self._try_parse_float(item.get("numerator"))
                row["denominator"] = self._try_parse_float(item.get("denominator"))
            cleaned.append(row)
        cleaned.sort(key=lambda item: str(item.get("date", "")), reverse=True)
        return cleaned[:12]

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
        normalized["_source_detail"] = {
            "symbol": "twelvedata",
            "name": "twelvedata",
            "instrument_name": "twelvedata",
            "exchange": "twelvedata",
            "price": "twelvedata",
            "close": "twelvedata",
            "previous_close": "twelvedata",
            "prev_close": "twelvedata",
            "open": "twelvedata",
            "high": "twelvedata",
            "low": "twelvedata",
            "volume": "twelvedata",
            "bid": "twelvedata",
            "ask": "twelvedata",
            "timestamp": "twelvedata",
            "datetime": "twelvedata",
        }
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
            "_source_detail": {
                "symbol": "fmp",
                "name": "fmp",
                "instrument_name": "fmp",
                "exchange": "fmp",
                "price": "fmp",
                "close": "fmp",
                "previous_close": "fmp",
                "prev_close": "fmp",
                "open": "fmp",
                "high": "fmp",
                "low": "fmp",
                "volume": "fmp",
                "bid": "fmp",
                "ask": "fmp",
                "timestamp": "fmp",
                "datetime": "fmp",
            },
        }

    async def _fetch_series(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._should_use_jquants_for_symbol(symbol, interval):
            return await self._fetch_series_jquants(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        if self.provider == "both":
            return await self._fetch_series_both(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        if self.provider == "fmp":
            return await self._fetch_series_fmp(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )
        return await self._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )

    def _should_use_jquants_for_symbol(self, symbol: str, interval: str) -> bool:
        if not JQUANTS_API_KEY:
            return False
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return False
        return infer_country_from_symbol(symbol) == "JAPAN" and self._normalize_jquants_code(symbol) is not None

    @staticmethod
    def _normalize_jquants_code(symbol: str) -> str | None:
        normalized = str(symbol or "").strip().upper()
        if normalized.endswith(".T"):
            normalized = normalized[:-2]
        normalized = normalized.strip()
        if normalized.isdigit() and len(normalized) in {4, 5}:
            return normalized
        return None

    async def _fetch_series_jquants(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        del outputsize
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return []

        code = self._normalize_jquants_code(symbol)
        if not code or not JQUANTS_API_KEY:
            return []

        headers = {"x-api-key": JQUANTS_API_KEY}
        params: dict[str, Any] = {"code": code}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        points: list[dict[str, Any]] = []
        pagination_key: str | None = None
        while True:
            request_params = dict(params)
            if pagination_key:
                request_params["pagination_key"] = pagination_key

            try:
                response = await client.get(JQUANTS_DAILY_BARS_URL, params=request_params, headers=headers)
                payload = response.json()
            except Exception as exc:
                LOGGER.warning("J-Quants daily bars fetch failed for %s: %s", symbol, exc)
                return []

            if response.status_code >= 400:
                LOGGER.warning("J-Quants daily bars API error for %s: %s", symbol, payload)
                return []

            values = None
            if isinstance(payload, dict):
                for key in ("daily_quotes", "quotes", "bars", "dailyBars"):
                    candidate = payload.get(key)
                    if isinstance(candidate, list):
                        values = candidate
                        break
            if not isinstance(values, list):
                return []

            for item in values:
                point = normalize_ohlcv_point(
                    item,
                    timestamp_keys=("Date", "date"),
                    open_keys=("Open", "open", "AdjustmentOpen", "adjustment_open"),
                    high_keys=("High", "high", "AdjustmentHigh", "adjustment_high"),
                    low_keys=("Low", "low", "AdjustmentLow", "adjustment_low"),
                    close_keys=("Close", "close", "AdjustmentClose", "adjustment_close"),
                    volume_keys=("Volume", "volume", "AdjustmentVolume", "adjustment_volume"),
                    source="jquants",
                )
                if point is not None:
                    points.append(point)

            pagination_key = payload.get("pagination_key") if isinstance(payload, dict) else None
            if not pagination_key:
                break

        return sorted(points, key=lambda item: str(item.get("t") or ""))

    async def _fetch_series_twelvedata(
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
                "apikey": self.twelvedata_api_key,
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
            point = normalize_ohlcv_point(
                item,
                timestamp_keys=("datetime",),
                open_keys=("open",),
                high_keys=("high",),
                low_keys=("low",),
                close_keys=("close",),
                volume_keys=("volume",),
                source="twelvedata",
            )
            if point is not None:
                points.append(point)

        return points

    async def _fetch_series_both(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_interval = str(interval or "").strip().lower()
        if normalized_interval not in {"1day", "1d", "day"}:
            return await self._fetch_series_twelvedata(
                client=client,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
            )

        td_task = self._fetch_series_twelvedata(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        fmp_task = self._fetch_series_fmp(
            client=client,
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        td_result, fmp_result = await asyncio.gather(td_task, fmp_task, return_exceptions=True)
        td_points = td_result if isinstance(td_result, list) else []
        fmp_points = fmp_result if isinstance(fmp_result, list) else []

        if isinstance(td_result, Exception):
            LOGGER.warning("Time series fetch failed (TD) for %s %s: %s", symbol, interval, td_result)
        if isinstance(fmp_result, Exception):
            LOGGER.warning("Time series fetch failed (FMP) for %s %s: %s", symbol, interval, fmp_result)

        return self._merge_points_by_timestamp(fmp_points, td_points)

    async def _fetch_series_fmp(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if str(interval or "").strip().lower() not in {"1day", "1d", "day"}:
            return []

        params: dict[str, Any] = {
            "apikey": self.fmp_api_key,
            "symbol": symbol,
        }
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        try:
            response = await client.get(FMP_HISTORICAL_EOD_URL, params=params)
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("FMP time series fetch failed for %s %s: %s", symbol, interval, exc)
            return []

        if self._is_fmp_error(payload):
            LOGGER.warning("FMP time series API error for %s %s: %s", symbol, interval, payload.get("Error Message"))
            return []

        if isinstance(payload, dict):
            values = payload.get("historical") if isinstance(payload.get("historical"), list) else payload.get("data")
        elif isinstance(payload, list):
            values = payload
        else:
            values = None
        if not isinstance(values, list):
            return []

        points: list[dict[str, Any]] = []
        for item in values:
            point = normalize_ohlcv_point(
                item,
                timestamp_keys=("date", "datetime"),
                open_keys=("open",),
                high_keys=("high",),
                low_keys=("low",),
                close_keys=("close",),
                volume_keys=("volume",),
                source="fmp",
            )
            if point is not None:
                points.append(point)

        points.sort(key=lambda item: str(item.get("t", "")))
        if outputsize > 0 and len(points) > outputsize:
            points = points[-outputsize:]
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
            cached_points = await self.full_daily_history_store.get(symbol, copy=False)
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
            point_groups: list[list[dict[str, Any]]] = [cached_points]
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
                    point_groups.append(incremental_points)
                start_cursor = chunk_end + timedelta(days=1)
                chunks += 1

            if start_cursor <= today:
                LOGGER.warning(
                    "Daily cache catch-up truncated for %s: reached chunk limit (%s).",
                    symbol,
                    FULL_HISTORY_MAX_CHUNKS,
                )

            merged_cached = merge_ohlcv_points(*point_groups)
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

        deduped = merge_ohlcv_points(merged)
        await self.full_daily_history_store.upsert(symbol, deduped)
        return deduped

    async def _fetch_earliest_date(self, client: httpx.AsyncClient, symbol: str, interval: str) -> date | None:
        if not self._uses_twelvedata():
            return None
        try:
            response = await client.get(
                EARLIEST_TIMESTAMP_URL,
                params={
                    "apikey": self.twelvedata_api_key,
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
    def _merge_quote_payloads_with_source(
        primary: dict[str, Any],
        primary_name: str,
        secondary: dict[str, Any],
        secondary_name: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not isinstance(primary, dict) and not isinstance(secondary, dict):
            return {}, {}
        out: dict[str, Any] = {}
        detail: dict[str, str] = {}
        keys = {
            "symbol",
            "name",
            "instrument_name",
            "exchange",
            "price",
            "close",
            "previous_close",
            "prev_close",
            "open",
            "high",
            "low",
            "volume",
            "bid",
            "ask",
            "timestamp",
            "datetime",
        }
        for key in keys:
            first = primary.get(key) if isinstance(primary, dict) else None
            second = secondary.get(key) if isinstance(secondary, dict) else None
            if first not in (None, ""):
                out[key] = first
                detail[key] = primary_name
            elif second not in (None, ""):
                out[key] = second
                detail[key] = secondary_name
        return out, detail

    @staticmethod
    def _series_source_descriptor(points: list[dict[str, Any]]) -> str:
        if not points:
            return "none"
        providers: set[str] = set()
        for item in points:
            src = str(item.get("_src", "")).strip().lower()
            if src:
                providers.add(src)
        if not providers:
            return "unknown"
        if len(providers) == 1:
            return next(iter(providers))
        return "mixed"

    @staticmethod
    def _extract_latest_session_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return latest_session_points(points)

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
        return merge_ohlcv_points(base_points, incoming_points)

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
        latest_session = MarketDataQueriesMixin._extract_latest_session_points(points)
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
        symbol_returns = MarketDataQueriesMixin._daily_returns(symbol_points, max_len=60)
        benchmark_returns = MarketDataQueriesMixin._daily_returns(benchmark_points, max_len=60)
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

    async def market_data_lab_quotes_payload(self, symbols: list[str]) -> list[dict[str, Any]]:
        target_symbols = normalize_symbols(symbols, max_items=MAX_BASIC_SYMBOLS)
        if not target_symbols:
            return []

        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            quotes = await asyncio.gather(
                *(self._fetch_quote(client, symbol) for symbol in target_symbols),
                return_exceptions=True,
            )

        items: list[dict[str, Any]] = []
        for symbol, quote in zip(target_symbols, quotes):
            payload = quote if isinstance(quote, dict) else {}
            current_price = self._pick_float(payload, "close", "price")
            previous_close = self._pick_float(payload, "previous_close", "prev_close")
            updated_at = self._best_updated_at(payload, [], [])
            source = str(payload.get("_source_provider") or self.provider or "").strip() or "unknown"

            if current_price is None:
                stored = self.last_price_store.get(symbol) or {}
                stored_price = self._pick_float(stored, "price")
                if stored_price is not None:
                    current_price = stored_price
                    updated_at = str(stored.get("timestamp") or updated_at or "")
                    source = str(stored.get("source") or source or "").strip() or source

            change_abs = None
            change_pct = None
            if current_price is not None and previous_close is not None and previous_close > 0:
                change_abs = current_price - previous_close
                change_pct = (change_abs / previous_close) * 100

            items.append(
                {
                    "symbol": symbol,
                    "name": self._pick_string(payload, "name", "instrument_name"),
                    "exchange": self._pick_string(payload, "exchange"),
                    "price": current_price,
                    "previous_close": previous_close,
                    "change_abs": change_abs,
                    "change_pct": change_pct,
                    "updated_at": updated_at,
                    "source": source,
                }
            )

        return items

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

    async def _update_minute_credits_from_response(self, response: httpx.Response) -> None:
        if not self._uses_twelvedata():
            return
        used_value = self._try_parse_int(response.headers.get("api-credits-used"))
        left_value = self._try_parse_int(response.headers.get("api-credits-left"))
        if used_value is None and left_value is None:
            return

        if used_value is not None:
            self.minute_credits_used = used_value
        if left_value is not None:
            self.minute_credits_left = left_value

    async def _update_daily_credits_from_api_usage(self, payload: dict[str, Any]) -> None:
        if not self._uses_twelvedata():
            return
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
        if not self._uses_twelvedata():
            return
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

    @staticmethod
    def _parse_iso_epoch(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _is_fmp_error(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("status") == "error":
            return True
        message = str(payload.get("Error Message", "")).strip()
        return bool(message)

    def _delay_note(self) -> str:
        if self.provider == "both":
            return "Combined feed: Twelve Data + Financial Modeling Prep."
        if self.provider == "fmp":
            return "Financial Modeling Prep free plan feed."
        return "Twelve Data Basic plan (delayed feed may apply)."
