"""Free data-source adapters for intrinsic-value calculations.

The adapters in this module normalize public/free-market data into
``FinancialMetrics``.  They do not persist data and they do not expose an API
route; callers can compose them with ``valuation_models.calculate_valuation_report``.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import fields
from datetime import date, timedelta
import io
import os
from typing import Any
import zipfile

import httpx

from ..stooq import fetch_stooq_daily_history
from ..utils import normalize_symbol
from .market_data_math import beta_and_corr
from .market_data_queries_historical_runtime import normalize_jquants_code
from .valuation_models import (
    MARKET_JP,
    MARKET_US,
    SECURITY_REIT,
    FinancialMetrics,
)
from .valuation_numeric import (
    abs_or_none as _abs_or_none,
    div_optional as _div_optional,
    first_dict,
    first_present,
    first_present_text as _first_present_text,
    first_report as _first_report,
    has_value as _has_value,
    parse_float as _parse_float,
    sub_optional as _sub_optional,
    sum_optional as _sum_optional,
)


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_US_10Y_SERIES_ID = "DGS10"
MOF_JGB_YIELD_CURVE_CSV_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
EDINET_DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
EDINET_DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"
JQUANTS_LISTED_INFO_URL = "https://api.jquants.com/v1/listed/info"
JQUANTS_STATEMENTS_URL = "https://api.jquants.com/v1/fins/statements"
JQUANTS_AUTH_REFRESH_URL = "https://api.jquants.com/v1/token/auth_refresh"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class ValuationDataError(RuntimeError):
    """Raised when a requested free data source cannot return usable data."""


class FreeValuationDataClient:
    """Orchestrates public/free source adapters for Japan and US equities."""

    def __init__(
        self,
        *,
        edinet_subscription_key: str | None = None,
        jquants_api_key: str | None = None,
        jquants_id_token: str | None = None,
        jquants_refresh_token: str | None = None,
        fred_api_key: str | None = None,
        fmp_api_key: str | None = None,
        alpha_vantage_api_key: str | None = None,
        sec_user_agent: str | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        self.edinet_subscription_key = _env_default(edinet_subscription_key, "EDINET_API_KEY", "EDINET_SUBSCRIPTION_KEY")
        self.jquants_api_key = _env_default(jquants_api_key, "JQUANTS_API_KEY")
        self.jquants_id_token = _env_default(jquants_id_token, "JQUANTS_ID_TOKEN")
        self.jquants_refresh_token = _env_default(jquants_refresh_token, "JQUANTS_REFRESH_TOKEN")
        self.fred_api_key = _env_default(fred_api_key, "FRED_API_KEY")
        self.fmp_api_key = _env_default(fmp_api_key, "FMP_API_KEY")
        self.alpha_vantage_api_key = _env_default(alpha_vantage_api_key, "ALPHA_VANTAGE_API_KEY")
        self.sec_user_agent = _env_default(sec_user_agent, "SEC_USER_AGENT") or "market-data-analyzer/0.1 contact@example.com"
        self.timeout_sec = timeout_sec

    async def build_us_metrics(self, symbol: str, *, use_fmp_fallback: bool = True) -> FinancialMetrics:
        """Build US metrics from SEC EDGAR, Stooq, FRED, and optional FMP."""

        normalized = normalize_symbol(symbol)
        price_task = asyncio.create_task(self.fetch_stooq_market_context(normalized, benchmark_symbol="SPY.US"))
        rate_task = asyncio.create_task(self.fetch_us_risk_free_rate())
        primary_task = asyncio.create_task(self.fetch_sec_metrics(normalized))

        price_metrics, risk_free_rate = await asyncio.gather(price_task, rate_task)
        try:
            financial_metrics = await primary_task
        except Exception:
            if not use_fmp_fallback or not self.fmp_api_key:
                raise
            financial_metrics = await self.fetch_fmp_metrics(normalized)

        return merge_financial_metrics(
            financial_metrics,
            FinancialMetrics(
                symbol=normalized,
                market=MARKET_US,
                currency=financial_metrics.currency or "USD",
                price=price_metrics.price,
                beta=price_metrics.beta,
                risk_free_rate=risk_free_rate,
                data_sources=price_metrics.data_sources + (f"FRED:{FRED_US_10Y_SERIES_ID}",),
            ),
        )

    async def build_jp_metrics(
        self,
        symbol: str,
        *,
        edinet_doc_id: str | None = None,
    ) -> FinancialMetrics:
        """Build Japan metrics from J-Quants, Stooq, MOF, and optional EDINET."""

        normalized = normalize_symbol(symbol)
        price_task = asyncio.create_task(self.fetch_stooq_market_context(normalized, benchmark_symbol="1306.T"))
        rate_task = asyncio.create_task(self.fetch_jp_risk_free_rate())
        price_metrics, risk_free_rate = await asyncio.gather(price_task, rate_task)
        financial_metrics: FinancialMetrics | None = None
        try:
            financial_metrics = await self.fetch_jquants_metrics(normalized)
        except Exception:
            if not edinet_doc_id:
                raise

        if edinet_doc_id:
            edinet_metrics = await self.fetch_edinet_metrics_by_doc_id(normalized, edinet_doc_id)
            financial_metrics = (
                merge_financial_metrics(financial_metrics, edinet_metrics)
                if financial_metrics is not None
                else edinet_metrics
            )

        if financial_metrics is None:
            raise ValuationDataError("No Japanese financial data source returned usable metrics.")

        combined = merge_financial_metrics(
            financial_metrics,
            FinancialMetrics(
                symbol=normalized,
                market=MARKET_JP,
                currency=financial_metrics.currency or "JPY",
                price=price_metrics.price,
                beta=price_metrics.beta,
                risk_free_rate=risk_free_rate,
                data_sources=price_metrics.data_sources + ("MOF:jgbcm.csv",),
            ),
        )
        return combined

    async def fetch_stooq_market_context(self, symbol: str, *, benchmark_symbol: str) -> FinancialMetrics:
        points_task = asyncio.create_task(fetch_stooq_daily_history(symbol))
        benchmark_task = asyncio.create_task(fetch_stooq_daily_history(benchmark_symbol))
        points, benchmark_points = await asyncio.gather(points_task, benchmark_task)
        price = latest_close(points)
        beta, _corr = beta_and_corr(points, benchmark_points, max_len=252, min_overlap=60)
        market = MARKET_JP if normalize_jquants_code(symbol) else MARKET_US
        currency = "JPY" if market == MARKET_JP else "USD"
        return FinancialMetrics(
            symbol=normalize_symbol(symbol),
            market=market,
            currency=currency,
            price=price,
            beta=beta,
            data_sources=("Stooq:daily_price", f"Stooq:benchmark:{benchmark_symbol}"),
        )

    async def fetch_us_risk_free_rate(self) -> float | None:
        if not self.fred_api_key:
            return None
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.get(
                FRED_OBSERVATIONS_URL,
                params={
                    "series_id": FRED_US_10Y_SERIES_ID,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 30,
                },
            )
            response.raise_for_status()
            return parse_fred_latest_rate(response.json())

    async def fetch_jp_risk_free_rate(self) -> float | None:
        async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True) as client:
            response = await client.get(MOF_JGB_YIELD_CURVE_CSV_URL)
            response.raise_for_status()
            return parse_mof_jgb_10y_csv(response.content)

    async def fetch_sec_metrics(self, symbol: str) -> FinancialMetrics:
        async with httpx.AsyncClient(timeout=self.timeout_sec, headers=self._sec_headers()) as client:
            cik = await fetch_sec_cik_for_ticker(client, symbol)
            facts_response = await client.get(SEC_COMPANY_FACTS_URL.format(cik=cik))
            facts_response.raise_for_status()
            return normalize_sec_company_facts(symbol, cik, facts_response.json())

    async def fetch_jquants_metrics(self, symbol: str) -> FinancialMetrics:
        code = normalize_jquants_code(symbol)
        if not code:
            raise ValuationDataError("J-Quants requires a Japanese numeric stock code.")
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            headers = await self._jquants_headers(client)
            if not headers:
                raise ValuationDataError("JQUANTS_API_KEY, JQUANTS_ID_TOKEN, or JQUANTS_REFRESH_TOKEN is required.")
            info_task = fetch_jquants_pages(client, JQUANTS_LISTED_INFO_URL, headers=headers, params={"code": code})
            statement_task = fetch_jquants_pages(client, JQUANTS_STATEMENTS_URL, headers=headers, params={"code": code})
            info_rows, statement_rows = await asyncio.gather(info_task, statement_task)
        return normalize_jquants_metrics(symbol, info_rows, statement_rows)

    async def find_edinet_documents(
        self,
        symbol: str,
        *,
        lookback_days: int = 90,
        doc_type_codes: tuple[str, ...] = ("120", "140", "160"),
    ) -> list[dict[str, Any]]:
        sec_code = normalize_jquants_code(symbol)
        if not sec_code:
            raise ValuationDataError("EDINET lookup requires a Japanese numeric stock code.")
        if not self.edinet_subscription_key:
            raise ValuationDataError("EDINET_API_KEY or EDINET_SUBSCRIPTION_KEY is required.")

        found: list[dict[str, Any]] = []
        today = date.today()
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            for offset in range(max(1, lookback_days)):
                target_date = today - timedelta(days=offset)
                response = await client.get(
                    EDINET_DOCUMENTS_URL,
                    params={
                        "date": target_date.isoformat(),
                        "type": 2,
                        "Subscription-Key": self.edinet_subscription_key,
                    },
                )
                response.raise_for_status()
                rows = response.json().get("results")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row_sec_code = str(row.get("secCode") or "").strip()
                    row_doc_type = str(row.get("docTypeCode") or "").strip()
                    if row_sec_code.startswith(sec_code[:4]) and row_doc_type in doc_type_codes:
                        found.append(dict(row))
                if found:
                    break
        return found

    async def fetch_edinet_metrics_by_doc_id(self, symbol: str, doc_id: str) -> FinancialMetrics:
        if not self.edinet_subscription_key:
            raise ValuationDataError("EDINET_API_KEY or EDINET_SUBSCRIPTION_KEY is required.")
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.get(
                EDINET_DOCUMENT_URL.format(doc_id=doc_id),
                params={"type": 5, "Subscription-Key": self.edinet_subscription_key},
            )
            response.raise_for_status()
        facts = parse_edinet_xbrl_to_csv_zip(response.content)
        return normalize_edinet_metrics(symbol, facts, doc_id=doc_id)

    async def fetch_fmp_metrics(self, symbol: str) -> FinancialMetrics:
        if not self.fmp_api_key:
            raise ValuationDataError("FMP_API_KEY is required for FMP fallback.")
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            profile, ratios, metrics, income, balance, cash_flow = await asyncio.gather(
                fmp_get_json(client, "profile", self.fmp_api_key, {"symbol": symbol}),
                fmp_get_json(client, "ratios-ttm", self.fmp_api_key, {"symbol": symbol}),
                fmp_get_json(client, "key-metrics-ttm", self.fmp_api_key, {"symbol": symbol}),
                fmp_get_json(client, "income-statement", self.fmp_api_key, {"symbol": symbol, "limit": 5}),
                fmp_get_json(client, "balance-sheet-statement", self.fmp_api_key, {"symbol": symbol, "limit": 1}),
                fmp_get_json(client, "cash-flow-statement", self.fmp_api_key, {"symbol": symbol, "limit": 1}),
            )
        return normalize_fmp_metrics(symbol, profile, ratios, metrics, income, balance, cash_flow)

    async def fetch_alpha_vantage_metrics(self, symbol: str) -> FinancialMetrics:
        if not self.alpha_vantage_api_key:
            raise ValuationDataError("ALPHA_VANTAGE_API_KEY is required.")
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            overview, income, balance, cash_flow = await asyncio.gather(
                alpha_vantage_get_json(client, self.alpha_vantage_api_key, "OVERVIEW", symbol),
                alpha_vantage_get_json(client, self.alpha_vantage_api_key, "INCOME_STATEMENT", symbol),
                alpha_vantage_get_json(client, self.alpha_vantage_api_key, "BALANCE_SHEET", symbol),
                alpha_vantage_get_json(client, self.alpha_vantage_api_key, "CASH_FLOW", symbol),
            )
        return normalize_alpha_vantage_metrics(symbol, overview, income, balance, cash_flow)

    def _sec_headers(self) -> dict[str, str]:
        return {"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    async def _jquants_headers(self, client: httpx.AsyncClient) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.jquants_id_token:
            headers["Authorization"] = f"Bearer {self.jquants_id_token}"
        elif self.jquants_refresh_token:
            response = await client.post(
                JQUANTS_AUTH_REFRESH_URL,
                params={"refreshtoken": self.jquants_refresh_token},
            )
            response.raise_for_status()
            id_token = str(response.json().get("idToken") or "").strip()
            if id_token:
                self.jquants_id_token = id_token
                headers["Authorization"] = f"Bearer {id_token}"
        if self.jquants_api_key:
            headers["x-api-key"] = self.jquants_api_key
        return headers


async def fetch_sec_cik_for_ticker(client: httpx.AsyncClient, symbol: str) -> int:
    ticker_response = await client.get(SEC_COMPANY_TICKERS_URL, headers={"Host": "www.sec.gov"})
    ticker_response.raise_for_status()
    payload = ticker_response.json()
    normalized = normalize_symbol(symbol)
    rows = payload.values() if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if normalize_symbol(row.get("ticker")) == normalized:
            cik = _parse_float(row.get("cik_str"))
            if cik is not None:
                return int(cik)
    raise ValuationDataError(f"SEC CIK not found for ticker {normalized}.")


def normalize_sec_company_facts(symbol: str, cik: int, payload: dict[str, Any]) -> FinancialMetrics:
    facts = payload.get("facts") if isinstance(payload, dict) else {}
    entity_name = payload.get("entityName") if isinstance(payload, dict) else None
    latest_duration = SecFactPicker(facts, annual_only=False)
    latest_annual = SecFactPicker(facts, annual_only=True)
    instant = SecFactPicker(facts, instant_only=True)

    revenue = latest_annual.value(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        unit="USD",
    )
    operating_income = latest_annual.value("OperatingIncomeLoss", unit="USD")
    ebit = operating_income or latest_annual.value(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        unit="USD",
    )
    depreciation = latest_annual.value(
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortizationExpense",
        unit="USD",
    )
    net_income = latest_annual.value("NetIncomeLoss", "ProfitLoss", unit="USD")
    operating_cf = latest_annual.value("NetCashProvidedByUsedInOperatingActivities", unit="USD")
    capex = _abs_or_none(latest_annual.value("PaymentsToAcquirePropertyPlantAndEquipment", unit="USD"))
    dividends_paid = _abs_or_none(
        latest_annual.value("PaymentsOfDividends", "PaymentsOfOrdinaryDividends", unit="USD")
    )
    proceeds = latest_annual.value(
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromBorrowings",
        "ProceedsFromDebtNetOfIssuanceCosts",
        unit="USD",
    )
    repayments = latest_annual.value(
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "PaymentsForRepurchaseOfCommonStock",
        unit="USD",
    )
    net_borrowing = _sub_optional(proceeds, repayments)

    cash = instant.value(
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        unit="USD",
    )
    short_term_investments = instant.value("ShortTermInvestments", unit="USD")
    debt = first_present(
        instant.value("LongTermDebtAndFinanceLeaseObligations", unit="USD"),
        _sum_optional(
            instant.value("ShortTermBorrowings", "ShortTermDebt", "LongTermDebtCurrent", unit="USD"),
            instant.value("LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", unit="USD"),
        ),
    )
    total_assets = instant.value("Assets", unit="USD")
    current_assets = instant.value("AssetsCurrent", unit="USD")
    total_liabilities = instant.value("Liabilities", unit="USD")
    equity = instant.value("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", unit="USD")
    shares = first_present(
        instant.value("EntityCommonStockSharesOutstanding", unit="shares", taxonomy="dei"),
        latest_duration.value("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingDiluted", unit="shares"),
        latest_duration.value("WeightedAverageNumberOfSharesOutstandingBasic", unit="shares"),
    )
    eps = latest_annual.value("EarningsPerShareDiluted", "EarningsPerShareBasic", unit="USD/shares")
    interest_expense = _abs_or_none(latest_annual.value("InterestExpenseNonOperating", "InterestExpense", unit="USD"))
    tax_expense = latest_annual.value("IncomeTaxExpenseBenefit", unit="USD")
    pretax_income = latest_annual.value("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", unit="USD")
    net_income_history = tuple(latest_annual.history("NetIncomeLoss", "ProfitLoss", unit="USD", max_items=5))

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_US,
        currency="USD",
        company_name=str(entity_name or "") or None,
        fiscal_date=latest_annual.latest_end_date(),
        revenue=revenue,
        operating_income=operating_income,
        ebit=ebit,
        depreciation_and_amortization=depreciation,
        ebitda=_sum_optional(ebit, depreciation),
        net_income=net_income,
        eps=eps,
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_sub_optional(operating_cf, capex),
        net_borrowing=net_borrowing,
        cash_and_equivalents=cash,
        short_term_investments=short_term_investments,
        interest_bearing_debt=debt,
        total_liabilities=total_liabilities,
        current_assets=current_assets,
        total_assets=total_assets,
        equity=equity,
        shareholders_equity=equity,
        shares_outstanding=shares,
        dividends_paid=dividends_paid,
        dividend_per_share=_div_optional(dividends_paid, shares),
        interest_expense=interest_expense,
        income_tax_expense=tax_expense,
        income_before_tax=pretax_income,
        net_income_history=net_income_history,
        data_sources=(f"SEC:companyfacts:CIK{cik:010d}",),
        raw={"cik": cik},
    )


class SecFactPicker:
    def __init__(self, facts: Any, *, annual_only: bool = False, instant_only: bool = False) -> None:
        self.facts = facts if isinstance(facts, dict) else {}
        self.annual_only = annual_only
        self.instant_only = instant_only

    def value(self, *concepts: str, unit: str, taxonomy: str = "us-gaap") -> float | None:
        fact = self.fact(*concepts, unit=unit, taxonomy=taxonomy)
        return _parse_float(fact.get("val")) if fact else None

    def fact(self, *concepts: str, unit: str, taxonomy: str = "us-gaap") -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for concept in concepts:
            rows = self._rows(concept, unit=unit, taxonomy=taxonomy)
            candidates.extend(row for row in rows if self._matches_period(row))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (str(row.get("filed") or ""), str(row.get("end") or "")))
        return candidates[-1]

    def history(self, *concepts: str, unit: str, taxonomy: str = "us-gaap", max_items: int = 5) -> list[float]:
        rows: list[dict[str, Any]] = []
        for concept in concepts:
            rows.extend(row for row in self._rows(concept, unit=unit, taxonomy=taxonomy) if self._matches_period(row))
        rows.sort(key=lambda row: (int(row.get("fy") or 0), str(row.get("filed") or "")), reverse=True)
        seen_years: set[int] = set()
        values: list[float] = []
        for row in rows:
            year = int(row.get("fy") or 0)
            if not year or year in seen_years:
                continue
            value = _parse_float(row.get("val"))
            if value is None:
                continue
            seen_years.add(year)
            values.append(value)
            if len(values) >= max_items:
                break
        return values

    def latest_end_date(self) -> str | None:
        rows: list[dict[str, Any]] = []
        taxonomy_rows = self.facts.get("us-gaap") if isinstance(self.facts, dict) else None
        if not isinstance(taxonomy_rows, dict):
            return None
        for concept_payload in taxonomy_rows.values():
            if not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit_rows in units.values():
                if isinstance(unit_rows, list):
                    rows.extend(row for row in unit_rows if isinstance(row, dict) and self._matches_period(row))
        rows.sort(key=lambda row: str(row.get("filed") or ""))
        return str(rows[-1].get("end") or "") if rows else None

    def _rows(self, concept: str, *, unit: str, taxonomy: str) -> list[dict[str, Any]]:
        taxonomy_facts = self.facts.get(taxonomy)
        if not isinstance(taxonomy_facts, dict):
            return []
        concept_payload = taxonomy_facts.get(concept)
        if not isinstance(concept_payload, dict):
            return []
        units = concept_payload.get("units")
        if not isinstance(units, dict):
            return []
        direct_rows = units.get(unit)
        if isinstance(direct_rows, list):
            return [dict(row) for row in direct_rows if isinstance(row, dict)]
        if unit == "USD/shares":
            for candidate_unit in ("USD/shares", "USD-per-shares"):
                candidate = units.get(candidate_unit)
                if isinstance(candidate, list):
                    return [dict(row) for row in candidate if isinstance(row, dict)]
        return []

    def _matches_period(self, row: dict[str, Any]) -> bool:
        if self.instant_only:
            return bool(row.get("end")) and not row.get("start")
        form = str(row.get("form") or "")
        if self.annual_only:
            return form == "10-K" or str(row.get("fp") or "").upper() == "FY"
        return form in {"10-K", "10-Q", "20-F", "40-F"} or not form


async def fetch_jquants_pages(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pagination_key: str | None = None
    while True:
        request_params = dict(params)
        if pagination_key:
            request_params["pagination_key"] = pagination_key
        response = await client.get(url, headers=headers, params=request_params)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("message"):
            raise ValuationDataError(str(payload.get("message")))
        page_rows = extract_jquants_rows(payload)
        rows.extend(page_rows)
        pagination_key = payload.get("pagination_key") if isinstance(payload, dict) else None
        if not pagination_key:
            break
    return rows


def extract_jquants_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("statements", "info", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def normalize_jquants_metrics(symbol: str, info_rows: list[dict[str, Any]], statements: list[dict[str, Any]]) -> FinancialMetrics:
    if not statements:
        raise ValuationDataError("No J-Quants statements were returned.")
    latest = sorted(statements, key=lambda row: (str(row.get("DisclosedDate") or ""), str(row.get("DisclosureNumber") or "")))[-1]
    info = info_rows[0] if info_rows else {}
    issued = _parse_float(latest.get("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"))
    treasury = _parse_float(latest.get("NumberOfTreasuryStockAtTheEndOfFiscalYear"))
    average_shares = _parse_float(latest.get("AverageNumberOfShares"))
    shares = first_present(_sub_optional(issued, treasury), issued, average_shares)
    revenue = _parse_float(latest.get("NetSales"))
    operating_income = _parse_float(latest.get("OperatingProfit"))
    net_income = _parse_float(latest.get("Profit"))
    operating_cf = _parse_float(latest.get("CashFlowsFromOperatingActivities"))
    cash = _parse_float(latest.get("CashAndEquivalents"))
    equity = _parse_float(latest.get("Equity"))
    total_assets = _parse_float(latest.get("TotalAssets"))
    fiscal_date = str(latest.get("CurrentPeriodEndDate") or latest.get("DisclosedDate") or "") or None
    dividend = first_present(
        _parse_float(latest.get("ResultDividendPerShareAnnual")),
        _parse_float(latest.get("ForecastDividendPerShareAnnual")),
        _parse_float(latest.get("DistributionsPerUnit(REIT)")),
    )
    forecast_dividend = first_present(
        _parse_float(latest.get("NextYearForecastDividendPerShareAnnual")),
        _parse_float(latest.get("ForecastDividendPerShareAnnual")),
        _parse_float(latest.get("ForecastDistributionsPerUnit(REIT)")),
    )
    forecast_eps = first_present(
        _parse_float(latest.get("NextYearForecastEarningsPerShare")),
        _parse_float(latest.get("ForecastEarningsPerShare")),
    )
    net_income_history = tuple(
        value
        for value in (
            _parse_float(row.get("Profit"))
            for row in sorted(statements, key=lambda item: str(item.get("CurrentPeriodEndDate") or ""), reverse=True)
            if str(row.get("TypeOfCurrentPeriod") or "").upper() == "FY"
        )
        if value is not None
    )[:5]
    is_reit = bool(_parse_float(latest.get("DistributionsPerUnit(REIT)")) or _parse_float(latest.get("ForecastDistributionsPerUnit(REIT)")))

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_JP,
        currency="JPY",
        company_name=str(info.get("CompanyName") or info.get("CompanyNameEnglish") or "") or None,
        sector=str(info.get("Sector33CodeName") or info.get("Sector17CodeName") or "") or None,
        industry=str(info.get("MarketCodeName") or "") or None,
        security_type=SECURITY_REIT if is_reit else None,
        fiscal_date=fiscal_date,
        revenue=revenue,
        operating_income=operating_income,
        ebit=operating_income,
        net_income=net_income,
        eps=_parse_float(latest.get("EarningsPerShare")),
        forecast_eps=forecast_eps,
        operating_cash_flow=operating_cf,
        cash_and_equivalents=cash,
        total_assets=total_assets,
        equity=equity,
        shareholders_equity=equity,
        shares_outstanding=shares,
        bps=_parse_float(latest.get("BookValuePerShare")),
        dividend_per_share=dividend,
        forecast_dividend_per_share=forecast_dividend,
        payout_ratio=_parse_float(latest.get("ResultPayoutRatioAnnual")),
        net_income_history=net_income_history,
        data_sources=("J-Quants:listed/info", "J-Quants:fins/statements"),
        raw={"jquants_latest_statement": latest, "jquants_listed_info": info},
    )


def parse_edinet_xbrl_to_csv_zip(content: bytes) -> dict[str, float]:
    facts: dict[str, float] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        for name in names:
            if "XBRL_TO_CSV" not in name and "xbrl_to_csv" not in name.lower():
                continue
            raw = archive.read(name)
            text = raw.decode("utf-16", errors="ignore")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            for row in reader:
                key = _first_present_text(row, "要素ID", "Element ID", "element_id")
                value = _first_present_text(row, "値", "Value", "value")
                numeric = _parse_float(value)
                if key and numeric is not None:
                    facts[key] = numeric
    return facts


def normalize_edinet_metrics(symbol: str, facts: dict[str, float], *, doc_id: str) -> FinancialMetrics:
    picker = EdinetFactPicker(facts)
    operating_cf = picker.value("NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromOperatingActivities")
    capex = _abs_or_none(
        picker.value(
            "PurchaseOfPropertyPlantAndEquipment",
            "PaymentsForPurchaseOfPropertyPlantAndEquipment",
            "PurchaseOfIntangibleAssets",
        )
    )
    cash = picker.value("CashAndCashEquivalents", "CashAndCashEquivalentsIFRS", "CashAndDeposits")
    debt = _sum_optional(
        picker.value("BondsAndBorrowingsCurrent", "ShortTermBorrowings", "CurrentPortionOfLongTermBorrowings"),
        picker.value("BondsAndBorrowingsNonCurrent", "LongTermBorrowings", "LongTermDebt"),
    )
    revenue = picker.value("NetSales", "Revenue", "RevenueIFRS")
    operating_income = picker.value("OperatingIncome", "OperatingProfitLoss", "OperatingProfitIFRS")
    net_income = picker.value("ProfitLossAttributableToOwnersOfParent", "NetIncome", "ProfitLoss")
    equity = picker.value("EquityAttributableToOwnersOfParent", "NetAssets", "Equity")
    total_assets = picker.value("Assets", "TotalAssets")
    total_liabilities = picker.value("Liabilities", "TotalLiabilities")
    current_assets = picker.value("CurrentAssets", "AssetsCurrent")
    shares = picker.value("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock", "TotalNumberOfIssuedShares")

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_JP,
        currency="JPY",
        revenue=revenue,
        operating_income=operating_income,
        ebit=operating_income,
        net_income=net_income,
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_sub_optional(operating_cf, capex),
        cash_and_equivalents=cash,
        interest_bearing_debt=debt,
        total_liabilities=total_liabilities,
        current_assets=current_assets,
        total_assets=total_assets,
        equity=equity,
        shareholders_equity=equity,
        shares_outstanding=shares,
        data_sources=(f"EDINET:{doc_id}:csv",),
        raw={"edinet_doc_id": doc_id},
    )


class EdinetFactPicker:
    def __init__(self, facts: dict[str, float]) -> None:
        self.facts = facts

    def value(self, *tokens: str) -> float | None:
        normalized_tokens = tuple(token.lower() for token in tokens)
        for key, value in self.facts.items():
            lower_key = key.lower()
            if any(token in lower_key for token in normalized_tokens):
                return value
        return None


async def fmp_get_json(client: httpx.AsyncClient, endpoint: str, api_key: str, params: dict[str, Any]) -> Any:
    request_params = dict(params)
    request_params["apikey"] = api_key
    response = await client.get(f"{FMP_BASE_URL}/{endpoint}", params=request_params)
    response.raise_for_status()
    return response.json()


def normalize_fmp_metrics(
    symbol: str,
    profile_payload: Any,
    ratios_payload: Any,
    metrics_payload: Any,
    income_payload: Any,
    balance_payload: Any,
    cash_flow_payload: Any,
) -> FinancialMetrics:
    profile = first_dict(profile_payload)
    ratios = first_dict(ratios_payload)
    key_metrics = first_dict(metrics_payload)
    income_rows = [dict(row) for row in income_payload if isinstance(row, dict)] if isinstance(income_payload, list) else []
    income = income_rows[0] if income_rows else first_dict(income_payload)
    balance = first_dict(balance_payload)
    cash_flow = first_dict(cash_flow_payload)
    net_income_history = tuple(
        value
        for value in (_parse_float(row.get("netIncome")) for row in income_rows[:5])
        if value is not None
    )
    capex = _abs_or_none(_parse_float(cash_flow.get("capitalExpenditure")))
    operating_cf = _parse_float(cash_flow.get("operatingCashFlow"))
    cash = first_present(
        _parse_float(balance.get("cashAndCashEquivalents")),
        _parse_float(balance.get("cashAndShortTermInvestments")),
    )
    short_term = _parse_float(balance.get("shortTermInvestments"))

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_US,
        currency=str(income.get("reportedCurrency") or profile.get("currency") or "USD"),
        company_name=str(profile.get("companyName") or "") or None,
        sector=str(profile.get("sector") or "") or None,
        industry=str(profile.get("industry") or "") or None,
        fiscal_date=str(income.get("date") or balance.get("date") or cash_flow.get("date") or "") or None,
        price=_parse_float(profile.get("price")),
        market_cap=_parse_float(profile.get("mktCap") or profile.get("marketCap")),
        revenue=_parse_float(income.get("revenue")),
        operating_income=_parse_float(income.get("operatingIncome")),
        ebit=_parse_float(income.get("ebit")) or _parse_float(income.get("operatingIncome")),
        ebitda=_parse_float(income.get("ebitda")),
        net_income=_parse_float(income.get("netIncome")),
        eps=_parse_float(key_metrics.get("epsTTM")) or _parse_float(income.get("eps")),
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_parse_float(cash_flow.get("freeCashFlow")) or _sub_optional(operating_cf, capex),
        cash_and_equivalents=cash,
        short_term_investments=short_term,
        interest_bearing_debt=_parse_float(balance.get("totalDebt")),
        total_liabilities=_parse_float(balance.get("totalLiabilities")),
        current_assets=_parse_float(balance.get("totalCurrentAssets")),
        total_assets=_parse_float(balance.get("totalAssets")),
        equity=_parse_float(balance.get("totalStockholdersEquity")),
        shareholders_equity=_parse_float(balance.get("totalStockholdersEquity")),
        shares_outstanding=_parse_float(income.get("weightedAverageShsOutDil")) or _parse_float(income.get("weightedAverageShsOut")),
        bps=_parse_float(key_metrics.get("bookValuePerShareTTM")),
        dividends_paid=_abs_or_none(_parse_float(cash_flow.get("dividendsPaid"))),
        dividend_per_share=_parse_float(key_metrics.get("dividendPerShareTTM")),
        roe=_parse_float(ratios.get("returnOnEquityTTM")),
        roa=_parse_float(ratios.get("returnOnAssetsTTM")),
        per=_parse_float(ratios.get("peRatioTTM")),
        pbr=_parse_float(ratios.get("priceToBookRatioTTM")),
        psr=_parse_float(ratios.get("priceToSalesRatioTTM")),
        beta=_parse_float(profile.get("beta")),
        net_income_history=net_income_history,
        data_sources=("FMP:profile", "FMP:income-statement", "FMP:balance-sheet-statement", "FMP:cash-flow-statement"),
        raw={"fmp_profile": profile},
    )


async def alpha_vantage_get_json(
    client: httpx.AsyncClient,
    api_key: str,
    function: str,
    symbol: str,
) -> Any:
    response = await client.get(
        ALPHA_VANTAGE_URL,
        params={"function": function, "symbol": symbol, "apikey": api_key},
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and (payload.get("Error Message") or payload.get("Note")):
        raise ValuationDataError(str(payload.get("Error Message") or payload.get("Note")))
    return payload


def normalize_alpha_vantage_metrics(symbol: str, overview: Any, income: Any, balance: Any, cash_flow: Any) -> FinancialMetrics:
    overview_dict = overview if isinstance(overview, dict) else {}
    annual_income = _first_report(income, "annualReports")
    annual_balance = _first_report(balance, "annualReports")
    annual_cash_flow = _first_report(cash_flow, "annualReports")
    capex = _abs_or_none(_parse_float(annual_cash_flow.get("capitalExpenditures")))
    operating_cf = _parse_float(annual_cash_flow.get("operatingCashflow"))
    dividend_per_share = _parse_float(overview_dict.get("DividendPerShare"))

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_US,
        currency=str(annual_income.get("reportedCurrency") or "USD"),
        company_name=str(overview_dict.get("Name") or "") or None,
        sector=str(overview_dict.get("Sector") or "") or None,
        industry=str(overview_dict.get("Industry") or "") or None,
        fiscal_date=str(annual_income.get("fiscalDateEnding") or "") or None,
        revenue=_parse_float(annual_income.get("totalRevenue")),
        operating_income=_parse_float(annual_income.get("operatingIncome")),
        ebit=_parse_float(annual_income.get("ebit")) or _parse_float(annual_income.get("operatingIncome")),
        net_income=_parse_float(annual_income.get("netIncome")),
        eps=_parse_float(overview_dict.get("EPS")),
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_sub_optional(operating_cf, capex),
        cash_and_equivalents=_parse_float(annual_balance.get("cashAndCashEquivalentsAtCarryingValue")),
        short_term_investments=_parse_float(annual_balance.get("shortTermInvestments")),
        interest_bearing_debt=_sum_optional(
            _parse_float(annual_balance.get("shortTermDebt")),
            _parse_float(annual_balance.get("longTermDebt")),
        ),
        total_liabilities=_parse_float(annual_balance.get("totalLiabilities")),
        current_assets=_parse_float(annual_balance.get("totalCurrentAssets")),
        total_assets=_parse_float(annual_balance.get("totalAssets")),
        equity=_parse_float(annual_balance.get("totalShareholderEquity")),
        shareholders_equity=_parse_float(annual_balance.get("totalShareholderEquity")),
        shares_outstanding=_parse_float(overview_dict.get("SharesOutstanding")),
        dividend_per_share=dividend_per_share,
        per=_parse_float(overview_dict.get("PERatio")),
        pbr=_parse_float(overview_dict.get("PriceToBookRatio")),
        psr=_parse_float(overview_dict.get("PriceToSalesRatioTTM")),
        beta=_parse_float(overview_dict.get("Beta")),
        data_sources=("AlphaVantage:OVERVIEW", "AlphaVantage:INCOME_STATEMENT", "AlphaVantage:BALANCE_SHEET", "AlphaVantage:CASH_FLOW"),
        raw={"alpha_vantage_overview": overview_dict},
    )


def parse_fred_latest_rate(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    for row in observations:
        if not isinstance(row, dict):
            continue
        value = _parse_float(row.get("value"))
        if value is not None:
            return value / 100.0
    return None


def parse_mof_jgb_10y_csv(content: bytes | str) -> float | None:
    if isinstance(content, bytes):
        text = content.decode("cp932", errors="ignore")
    else:
        text = content
    rows = list(csv.reader(io.StringIO(text)))
    header: list[str] | None = None
    ten_year_index: int | None = None
    latest_rate: float | None = None
    for row in rows:
        if not row:
            continue
        if "基準日" in row[0]:
            header = row
            for idx, value in enumerate(row):
                if value.strip() == "10年":
                    ten_year_index = idx
                    break
            continue
        if header is None or ten_year_index is None or len(row) <= ten_year_index:
            continue
        value = _parse_float(row[ten_year_index])
        if value is not None:
            latest_rate = value / 100.0
    return latest_rate


def latest_close(points: list[dict[str, Any]]) -> float | None:
    for point in reversed(points):
        close = _parse_float(point.get("c"))
        if close is not None and close > 0:
            return close
    return None


def merge_financial_metrics(primary: FinancialMetrics, supplement: FinancialMetrics) -> FinancialMetrics:
    """Return ``primary`` with missing fields filled from ``supplement``."""

    values: dict[str, Any] = {}
    for item in fields(FinancialMetrics):
        primary_value = getattr(primary, item.name)
        supplement_value = getattr(supplement, item.name)
        if item.name == "data_sources":
            values[item.name] = tuple(dict.fromkeys(primary.data_sources + supplement.data_sources))
        elif item.name == "raw":
            values[item.name] = {**supplement.raw, **primary.raw}
        elif item.name == "net_income_history":
            values[item.name] = primary_value or supplement_value
        else:
            values[item.name] = primary_value if _has_value(primary_value) else supplement_value
    return FinancialMetrics(**values)


def _env_default(value: str | None, *names: str) -> str:
    if value:
        return str(value).strip()
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            return raw
    return ""
