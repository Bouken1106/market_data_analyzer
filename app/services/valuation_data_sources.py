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

import httpx

from ..stooq import fetch_stooq_daily_history
from ..utils import normalize_symbol
from .market_data_math import beta_and_corr
from .market_data_queries_historical_runtime import normalize_jquants_code
from .payload_records import record_list
from .valuation_edinet_sources import parse_edinet_xbrl_to_csv_zip, normalize_edinet_metrics
from .valuation_errors import ValuationDataError
from .valuation_jquants_sources import (
    extract_jquants_rows,
    fetch_jquants_pages,
    normalize_jquants_metrics,
)
from .valuation_models import (
    MARKET_JP,
    MARKET_US,
    FinancialMetrics,
)
from .valuation_numeric import (
    abs_or_none as _abs_or_none,
    first_dict,
    first_present,
    first_report as _first_report,
    has_value as _has_value,
    parse_float as _parse_float,
    sub_optional as _sub_optional,
    sum_optional as _sum_optional,
)
from .valuation_sec_sources import fetch_sec_cik_for_ticker, normalize_sec_company_facts


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
    income_rows = record_list(income_payload)
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
