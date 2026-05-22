"""FMP reference-data helpers for MarketData query mixins."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
from .market_data_provider_clients import FmpClient
from .payload_records import first_record, payload_rows
from .ttl_cache import build_cached_response, ttl_cache_lookup_response, ttl_cache_pop, ttl_cache_store


@dataclass(frozen=True)
class FmpReferenceRawPayloads:
    profile: Any
    ratios: Any
    metrics: Any
    income: Any
    balance_sheet: Any
    cash_flow: Any
    historical: Any
    dividends: Any
    splits: Any


@dataclass(frozen=True)
class FmpReferenceData:
    profile: dict[str, Any]
    ratios: dict[str, Any]
    metrics: dict[str, Any]
    income: dict[str, Any]
    balance_sheet: dict[str, Any]
    cash_flow: dict[str, Any]
    historical: list[dict[str, Any]]
    dividends: list[dict[str, Any]]
    splits: list[dict[str, Any]]
    income_history: list[dict[str, Any]] | None = None
    cash_flow_history: list[dict[str, Any]] | None = None


FMP_RATIO_FIELDS = (
    ("pe_ratio_ttm", "peRatioTTM"),
    ("pb_ratio_ttm", "priceToBookRatioTTM"),
    ("ps_ratio_ttm", "priceToSalesRatioTTM"),
    ("roe_ttm", "returnOnEquityTTM"),
    ("net_margin_ttm", "netProfitMarginTTM"),
    ("current_ratio_ttm", "currentRatioTTM"),
    ("debt_to_equity_ttm", "debtEquityRatioTTM"),
)

FMP_KEY_METRIC_FIELDS = (
    ("market_cap_ttm", "marketCapTTM"),
    ("enterprise_value_ttm", "enterpriseValueTTM"),
    ("eps_ttm", "epsTTM"),
    ("net_income_per_share_ttm", "netIncomePerShareTTM"),
    ("revenue_per_share_ttm", "revenuePerShareTTM"),
    ("free_cash_flow_per_share_ttm", "freeCashFlowPerShareTTM"),
    ("book_value_per_share_ttm", "bookValuePerShareTTM"),
    ("dividend_per_share_ttm", "dividendPerShareTTM"),
    ("dividend_yield_ttm", "dividendYieldTTM"),
)

FMP_INCOME_FIELDS = (
    ("revenue", "revenue"),
    ("gross_profit", "grossProfit"),
    ("operating_income", "operatingIncome"),
    ("ebit", "ebit"),
    ("ebitda", "ebitda"),
    ("net_income", "netIncome"),
    ("income_before_tax", "incomeBeforeTax"),
    ("income_tax_expense", "incomeTaxExpense"),
    ("interest_expense", "interestExpense"),
    ("eps", "eps"),
    ("eps_diluted", "epsdiluted"),
    ("weighted_average_shares", "weightedAverageShsOut"),
    ("weighted_average_shares_diluted", "weightedAverageShsOutDil"),
)

FMP_BALANCE_SHEET_FIELDS = (
    ("cash_and_cash_equivalents", "cashAndCashEquivalents"),
    ("short_term_investments", "shortTermInvestments"),
    ("cash_and_short_term_investments", "cashAndShortTermInvestments"),
    ("long_term_investments", "longTermInvestments"),
    ("total_investments", "totalInvestments"),
    ("total_assets", "totalAssets"),
    ("total_debt", "totalDebt"),
    ("net_debt", "netDebt"),
    ("total_liabilities", "totalLiabilities"),
    ("total_equity", "totalStockholdersEquity"),
)

FMP_CASH_FLOW_FIELDS = (
    ("operating_cash_flow", "operatingCashFlow"),
    ("capital_expenditure", "capitalExpenditure"),
    ("free_cash_flow", "freeCashFlow"),
    ("depreciation_and_amortization", "depreciationAndAmortization"),
    ("change_in_working_capital", "changeInWorkingCapital"),
    ("dividends_paid", "dividendsPaid"),
    ("common_stock_repurchased", "commonStockRepurchased"),
    ("repurchases_of_stock", "repurchasesOfStock"),
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

        if not refresh:
            cached_payload = await ttl_cache_lookup_response(
                self._fmp_reference_cache,
                self._fmp_reference_lock,
                normalized,
                ttl_sec=FMP_REFERENCE_CACHE_TTL_SEC,
                copy_fn=dict,
                allow_stale=cache_only,
                source_fresh="cache-memory",
                source_stale="cache-memory-stale",
                include_cache_metadata=True,
            )
            if cached_payload is not None:
                return cached_payload

        if not refresh:
            disk_cached = await self.fmp_reference_store.get(normalized)
            if isinstance(disk_cached, dict):
                cached_at = self._parse_iso_epoch(disk_cached.get("cached_at"))
                is_fresh = cached_at is not None and self._is_cache_fresh(cached_at, FMP_REFERENCE_CACHE_TTL_SEC)
                if is_fresh or cache_only:
                    payload = build_cached_response(
                        disk_cached,
                        source="cache-disk" if is_fresh else "cache-disk-stale",
                        ttl_sec=FMP_REFERENCE_CACHE_TTL_SEC,
                        stale=not is_fresh,
                    )
                    if is_fresh:
                        await ttl_cache_store(
                            self._fmp_reference_cache,
                            self._fmp_reference_lock,
                            normalized,
                            payload,
                        )
                    return payload

        if cache_only:
            raise HTTPException(status_code=404, detail="No cached FMP reference data found for this symbol.")
        if not self.fmp_api_key:
            raise HTTPException(status_code=400, detail="FMP_API_KEY is required for reference data.")

        payload = await self._fetch_fmp_reference_live(normalized)
        await ttl_cache_store(
            self._fmp_reference_cache,
            self._fmp_reference_lock,
            normalized,
            payload,
        )
        await self.fmp_reference_store.upsert(normalized, payload)
        return payload

    async def clear_fmp_reference_cache(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper().strip()
        if not SYMBOL_PATTERN.match(normalized):
            raise HTTPException(status_code=400, detail="Invalid symbol format.")
        await ttl_cache_pop(
            self._fmp_reference_cache,
            self._fmp_reference_lock,
            normalized,
        )
        removed_disk = await self.fmp_reference_store.clear(normalized)
        return {
            "symbol": normalized,
            "removed_memory_cache": True,
            "removed_disk_cache": bool(removed_disk),
        }

    async def _fetch_fmp_reference_live(self, symbol: str) -> dict[str, Any]:
        raw_payloads = await self._fetch_fmp_reference_raw(symbol)
        normalized = self._normalize_fmp_reference_raw(raw_payloads)
        self._validate_fmp_reference_data(normalized)
        return self._build_fmp_reference_payload(symbol=symbol, data=normalized)

    async def _fetch_fmp_reference_raw(self, symbol: str) -> FmpReferenceRawPayloads:
        timeout = httpx.Timeout(40.0, connect=10.0)
        two_years_ago = (date.today() - timedelta(days=366 * 2)).isoformat()
        async with httpx.AsyncClient(timeout=timeout) as client:
            profile_task = self._fmp_get_json(client, FMP_PROFILE_URL, params={"symbol": symbol})
            ratios_task = self._fmp_get_json(client, FMP_RATIOS_TTM_URL, params={"symbol": symbol})
            metrics_task = self._fmp_get_json(client, FMP_KEY_METRICS_TTM_URL, params={"symbol": symbol})
            income_task = self._fmp_get_json(client, FMP_INCOME_STATEMENT_URL, params={"symbol": symbol, "limit": 5})
            bs_task = self._fmp_get_json(client, FMP_BALANCE_SHEET_URL, params={"symbol": symbol, "limit": 1})
            cf_task = self._fmp_get_json(client, FMP_CASH_FLOW_URL, params={"symbol": symbol, "limit": 5})
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
        return FmpReferenceRawPayloads(
            profile=profile_raw,
            ratios=ratios_raw,
            metrics=metrics_raw,
            income=income_raw,
            balance_sheet=bs_raw,
            cash_flow=cf_raw,
            historical=hist_raw,
            dividends=div_raw,
            splits=split_raw,
        )

    def _normalize_fmp_reference_raw(self, raw: FmpReferenceRawPayloads) -> FmpReferenceData:
        return FmpReferenceData(
            profile=self._first_dict(raw.profile),
            ratios=self._first_dict(raw.ratios),
            metrics=self._first_dict(raw.metrics),
            income=self._first_dict(raw.income),
            income_history=payload_rows(raw.income, "data"),
            balance_sheet=self._first_dict(raw.balance_sheet),
            cash_flow=self._first_dict(raw.cash_flow),
            cash_flow_history=payload_rows(raw.cash_flow, "data"),
            historical=self._extract_historical_rows(raw.historical),
            dividends=self._extract_historical_rows(raw.dividends),
            splits=self._extract_historical_rows(raw.splits),
        )

    @staticmethod
    def _validate_fmp_reference_data(data: FmpReferenceData) -> None:
        if not data.profile and not data.historical and not data.income and not data.balance_sheet and not data.cash_flow:
            raise HTTPException(status_code=502, detail="Failed to fetch FMP reference data.")

    def _build_fmp_reference_payload(
        self,
        *,
        symbol: str,
        data: FmpReferenceData,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "source": "fmp-live",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cache_ttl_sec": FMP_REFERENCE_CACHE_TTL_SEC,
            "estimated_api_calls_on_refresh": 9,
            "cost_note": "This payload is cached to reduce API credit usage (Free plan: 250 calls/day).",
            "profile": self._build_profile_payload(data.profile),
            "adjusted_prices": self._build_adjusted_price_summary(data.historical),
            "corporate_actions": self._build_corporate_actions_payload(data),
            "financials": self._build_financials_payload(data),
        }

    def _build_profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "company_name": profile.get("companyName") or profile.get("company_name"),
            "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
            "country": profile.get("country"),
            "website": profile.get("website"),
            "ceo": profile.get("ceo"),
            "description": profile.get("description"),
            "price": self._try_parse_float(profile.get("price")),
            "market_cap": self._try_parse_float(profile.get("mktCap") or profile.get("marketCap")),
            "shares_outstanding": self._try_parse_float(
                profile.get("sharesOutstanding") or profile.get("shares_outstanding")
            ),
            "beta": self._try_parse_float(profile.get("beta")),
            "employees": profile.get("fullTimeEmployees"),
            "ipo_date": profile.get("ipoDate"),
        }

    def _build_corporate_actions_payload(self, data: FmpReferenceData) -> dict[str, Any]:
        return {
            "dividends": self._normalize_actions(data.dividends, action_type="dividend"),
            "splits": self._normalize_actions(data.splits, action_type="split"),
        }

    def _build_financials_payload(self, data: FmpReferenceData) -> dict[str, Any]:
        return {
            "ratios_ttm": self._build_ratios_payload(data.ratios),
            "key_metrics_ttm": self._build_key_metrics_payload(data.metrics),
            "income_statement_latest": self._build_income_statement_payload(data.income),
            "income_statement_history": [
                self._build_income_statement_payload(row)
                for row in (data.income_history or [])[:5]
            ],
            "balance_sheet_latest": self._build_balance_sheet_payload(data.balance_sheet),
            "cash_flow_latest": self._build_cash_flow_payload(data.cash_flow),
            "cash_flow_history": [
                self._build_cash_flow_payload(row)
                for row in (data.cash_flow_history or [])[:5]
            ],
        }

    def _build_ratios_payload(self, ratios: dict[str, Any]) -> dict[str, Any]:
        return self._build_numeric_payload(ratios, FMP_RATIO_FIELDS)

    def _build_key_metrics_payload(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return self._build_numeric_payload(metrics, FMP_KEY_METRIC_FIELDS)

    def _build_income_statement_payload(self, income: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": income.get("date"),
            **self._build_numeric_payload(income, FMP_INCOME_FIELDS),
        }

    def _build_balance_sheet_payload(self, balance_sheet: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": balance_sheet.get("date"),
            **self._build_numeric_payload(balance_sheet, FMP_BALANCE_SHEET_FIELDS),
        }

    def _build_cash_flow_payload(self, cash_flow: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": cash_flow.get("date"),
            **self._build_numeric_payload(cash_flow, FMP_CASH_FLOW_FIELDS),
        }

    def _build_numeric_payload(
        self,
        source: dict[str, Any],
        fields: tuple[tuple[str, str], ...],
    ) -> dict[str, float | None]:
        return {
            output_key: self._try_parse_float(source.get(input_key))
            for output_key, input_key in fields
        }

    async def _fmp_get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        payload = await FmpClient(client, self.fmp_api_key).get_json(url, params=params)
        if isinstance(payload, dict):
            message = str(payload.get("Error Message", "")).strip()
            if message:
                raise HTTPException(status_code=400, detail=f"FMP API error: {message}")
        return payload

    @staticmethod
    def _first_dict(payload: Any) -> dict[str, Any]:
        return first_record(
            payload,
            row_keys=("data",),
            allow_direct_dict=True,
            direct_dict_predicate=MarketDataReferenceMixin._is_fmp_record_payload,
            prefer_direct_dict=True,
        )

    @staticmethod
    def _is_fmp_record_payload(payload: dict[str, Any]) -> bool:
        return bool(
            payload.get("symbol")
            and any(isinstance(value, (str, int, float, bool, dict, list, type(None))) for value in payload.values())
        )

    @staticmethod
    def _extract_historical_rows(payload: Any) -> list[dict[str, Any]]:
        return payload_rows(payload, "historical", "data")

    def _build_adjusted_price_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        cleaned = [dict(item) for item in rows if isinstance(item, dict)]
        cleaned.sort(key=lambda item: str(item.get("date", "")))
        latest = cleaned[-1] if cleaned else {}
        close_value = self._try_parse_float(latest.get("close"))
        adj_close_value = self._try_parse_float(latest.get("adjClose") or latest.get("adjustedClose"))
        factor = self._adjustment_factor(close_value=close_value, adj_close_value=adj_close_value)

        recent: list[dict[str, Any]] = []
        for item in cleaned[-60:]:
            point = self._adjusted_price_point(item)
            if point is None:
                continue
            recent.append(point)

        return {
            "latest_date": latest.get("date"),
            "latest_close": close_value,
            "latest_adj_close": adj_close_value,
            "latest_adjustment_factor": factor,
            "recent_points": recent,
        }

    @staticmethod
    def _adjustment_factor(*, close_value: float | None, adj_close_value: float | None) -> float | None:
        if close_value is None or adj_close_value is None or close_value == 0:
            return None
        return adj_close_value / close_value

    def _adjusted_price_point(self, item: dict[str, Any]) -> dict[str, Any] | None:
        close_item = self._try_parse_float(item.get("close"))
        adj_item = self._try_parse_float(item.get("adjClose") or item.get("adjustedClose"))
        if close_item is None and adj_item is None:
            return None
        return {
            "date": item.get("date"),
            "close": close_item,
            "adj_close": adj_item,
            "open": self._try_parse_float(item.get("open")),
            "high": self._try_parse_float(item.get("high")),
            "low": self._try_parse_float(item.get("low")),
            "volume": self._try_parse_float(item.get("volume")),
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
