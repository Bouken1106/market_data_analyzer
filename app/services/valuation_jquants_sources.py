"""J-Quants adapters for valuation metrics."""

from __future__ import annotations

from typing import Any

import httpx

from ..utils import normalize_symbol
from .valuation_errors import ValuationDataError
from .valuation_models import MARKET_JP, FinancialMetrics
from .valuation_numeric import (
    first_present,
    parse_float as _parse_float,
    sub_optional as _sub_optional,
)
from .valuation_security_rules import SECURITY_REIT


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


def normalize_jquants_metrics(
    symbol: str,
    info_rows: list[dict[str, Any]],
    statements: list[dict[str, Any]],
) -> FinancialMetrics:
    if not statements:
        raise ValuationDataError("No J-Quants statements were returned.")
    latest = sorted(
        statements,
        key=lambda row: (str(row.get("DisclosedDate") or ""), str(row.get("DisclosureNumber") or "")),
    )[-1]
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
    fy_rows = sorted(statements, key=lambda item: str(item.get("CurrentPeriodEndDate") or ""), reverse=True)
    net_income_history = tuple(
        value
        for value in (
            _parse_float(row.get("Profit"))
            for row in fy_rows
            if str(row.get("TypeOfCurrentPeriod") or "").upper() == "FY"
        )
        if value is not None
    )[:5]
    revenue_history = tuple(
        value
        for value in (
            _parse_float(row.get("NetSales"))
            for row in fy_rows
            if str(row.get("TypeOfCurrentPeriod") or "").upper() == "FY"
        )
        if value is not None
    )[:5]
    eps_history = tuple(
        value
        for value in (
            _parse_float(row.get("EarningsPerShare"))
            for row in fy_rows
            if str(row.get("TypeOfCurrentPeriod") or "").upper() == "FY"
        )
        if value is not None
    )[:5]
    is_reit = bool(
        _parse_float(latest.get("DistributionsPerUnit(REIT)"))
        or _parse_float(latest.get("ForecastDistributionsPerUnit(REIT)"))
    )

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
        revenue_history=revenue_history,
        eps_history=eps_history,
        data_sources=("J-Quants:listed/info", "J-Quants:fins/statements"),
        raw={"jquants_latest_statement": latest, "jquants_listed_info": info},
    )
