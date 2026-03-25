"""FMP reference-data helpers for MarketData query mixins."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from ..config import (
    FMP_BALANCE_SHEET_URL,
    FMP_CASH_FLOW_URL,
    FMP_DIVIDENDS_URL,
    FMP_DIVIDEND_ADJUSTED_PRICE_URL,
    FMP_INCOME_STATEMENT_URL,
    FMP_KEY_METRICS_TTM_URL,
    FMP_PROFILE_URL,
    FMP_RATIOS_TTM_URL,
    FMP_REFERENCE_CACHE_TTL_SEC,
    FMP_SPLITS_URL,
    SYMBOL_PATTERN,
)


class MarketDataReferenceMixin:
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
            profile_task = self._fmp_get_json(client, FMP_PROFILE_URL, params={"symbol": symbol})
            ratios_task = self._fmp_get_json(client, FMP_RATIOS_TTM_URL, params={"symbol": symbol})
            metrics_task = self._fmp_get_json(client, FMP_KEY_METRICS_TTM_URL, params={"symbol": symbol})
            income_task = self._fmp_get_json(client, FMP_INCOME_STATEMENT_URL, params={"symbol": symbol, "limit": 1})
            bs_task = self._fmp_get_json(client, FMP_BALANCE_SHEET_URL, params={"symbol": symbol, "limit": 1})
            cf_task = self._fmp_get_json(client, FMP_CASH_FLOW_URL, params={"symbol": symbol, "limit": 1})
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
        return {
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
                    "cash_and_short_term_investments": self._try_parse_float(
                        balance_sheet.get("cashAndShortTermInvestments")
                    ),
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
            if payload.get("symbol") and any(
                isinstance(value, (str, int, float, bool, dict, list, type(None))) for value in payload.values()
            ):
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
