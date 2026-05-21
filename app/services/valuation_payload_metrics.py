"""FinancialMetrics normalization for valuation UI payloads."""

from __future__ import annotations

from typing import Any

from .valuation_models import MARKET_JP, FinancialMetrics
from .valuation_numeric import (
    dict_at as _dict_at,
    first_positive,
    parse_float as _float,
    path_float as _path_float,
    payload_source,
    positive_abs as _positive_abs,
    positive_div as _positive_div,
    positive_float,
    positive_mul as _positive_mul,
    non_negative_float,
    text_or_none as _text,
)


def financial_metrics_from_payloads(
    symbol: str,
    *,
    market: str,
    overview_payload: dict[str, Any] | None,
    fmp_payload: dict[str, Any] | None,
    risk_free_rate: float,
) -> FinancialMetrics:
    profile = _dict_at(fmp_payload, "profile")
    financials = _dict_at(fmp_payload, "financials")
    ratios = _dict_at(financials, "ratios_ttm")
    key_metrics = _dict_at(financials, "key_metrics_ttm")
    income = _dict_at(financials, "income_statement_latest")
    balance_sheet = _dict_at(financials, "balance_sheet_latest")
    cash_flow = _dict_at(financials, "cash_flow_latest")
    adjusted_prices = _dict_at(fmp_payload, "adjusted_prices")

    overview_price = _path_float(overview_payload, "price", "current")
    profile_price = _float(profile.get("price"))
    fmp_reference_price = first_positive(
        profile_price,
        _float(adjusted_prices.get("latest_close")),
        _float(adjusted_prices.get("latest_adj_close")),
    )
    price = first_positive(
        overview_price,
        fmp_reference_price,
        _float(profile.get("market_price")),
    )
    profile_market_cap = positive_float(profile.get("market_cap"))
    income_eps = positive_float(income.get("eps"))
    eps = first_positive(positive_float(key_metrics.get("eps_ttm")), income_eps)
    shares = first_positive(
        profile.get("shares_outstanding"),
        _positive_div(profile_market_cap, fmp_reference_price),
        income.get("weighted_average_shares_diluted"),
        income.get("weighted_average_shares"),
        _positive_div(income.get("net_income"), income_eps),
        _positive_div(profile_market_cap, price),
    )
    market_cap = first_positive(
        _positive_mul(price, shares),
        key_metrics.get("market_cap_ttm"),
        profile_market_cap,
    )
    dividend_yield = positive_float(key_metrics.get("dividend_yield_ttm"))
    dividend_per_share = first_positive(
        _positive_mul(price, dividend_yield),
        _dividend_per_share_from_actions(fmp_payload),
    )
    beta = first_positive(
        _path_float(overview_payload, "market", "beta_60d_vs_spy"),
        positive_float(profile.get("beta")),
    )
    capex = _positive_abs(cash_flow.get("capital_expenditure"))
    cash_and_equivalents, short_term_investments, long_term_investments = _cash_and_investments(balance_sheet)
    data_sources = tuple(
        item
        for item in (
            f"FMP-reference:{payload_source(fmp_payload)}" if fmp_payload else None,
            f"overview:{payload_source(overview_payload)}" if overview_payload else None,
        )
        if item
    )
    return FinancialMetrics(
        symbol=symbol,
        market=market,
        currency="JPY" if market == MARKET_JP else "USD",
        company_name=_text(profile.get("company_name")),
        sector=_text(profile.get("sector")),
        industry=_text(profile.get("industry")),
        fiscal_date=_text(income.get("date")) or _text(balance_sheet.get("date")) or _text(cash_flow.get("date")),
        price=price,
        shares_outstanding=shares,
        market_cap=market_cap,
        revenue=_float(income.get("revenue")),
        operating_income=_float(income.get("operating_income")),
        ebit=_float(income.get("ebit")) or _float(income.get("operating_income")),
        ebitda=_float(income.get("ebitda")),
        net_income=_float(income.get("net_income")),
        eps=eps,
        operating_cash_flow=_float(cash_flow.get("operating_cash_flow")),
        capital_expenditure=capex,
        free_cash_flow=_float(cash_flow.get("free_cash_flow")),
        cash_and_equivalents=cash_and_equivalents,
        short_term_investments=short_term_investments,
        long_term_investments=long_term_investments,
        interest_bearing_debt=non_negative_float(balance_sheet.get("total_debt")),
        total_liabilities=_float(balance_sheet.get("total_liabilities")),
        total_assets=_float(balance_sheet.get("total_assets")),
        equity=_float(balance_sheet.get("total_equity")),
        shareholders_equity=_float(balance_sheet.get("total_equity")),
        bps=positive_float(key_metrics.get("book_value_per_share_ttm")),
        dividend_per_share=dividend_per_share,
        roe=_float(ratios.get("roe_ttm")),
        per=positive_float(ratios.get("pe_ratio_ttm")),
        pbr=positive_float(ratios.get("pb_ratio_ttm")),
        psr=positive_float(ratios.get("ps_ratio_ttm")),
        ev=positive_float(key_metrics.get("enterprise_value_ttm")),
        interest_expense=_float(income.get("interest_expense")),
        income_tax_expense=_float(income.get("income_tax_expense")),
        income_before_tax=_float(income.get("income_before_tax")),
        beta=beta,
        risk_free_rate=risk_free_rate,
        data_sources=data_sources,
    )


def _dividend_per_share_from_actions(fmp_payload: dict[str, Any] | None) -> float | None:
    actions = _dict_at(fmp_payload, "corporate_actions")
    dividends = actions.get("dividends")
    if not isinstance(dividends, list) or not dividends:
        return None
    total = 0.0
    count = 0
    for item in dividends[:4]:
        if not isinstance(item, dict):
            continue
        amount = positive_float(item.get("adj_dividend")) or positive_float(item.get("dividend"))
        if amount is None:
            continue
        total += amount
        count += 1
    return total if count else None


def _cash_and_investments(balance_sheet: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    cash = non_negative_float(balance_sheet.get("cash_and_cash_equivalents"))
    cash_and_short = non_negative_float(balance_sheet.get("cash_and_short_term_investments"))
    short_term = non_negative_float(balance_sheet.get("short_term_investments"))
    long_term = non_negative_float(balance_sheet.get("long_term_investments"))
    total_investments = non_negative_float(balance_sheet.get("total_investments"))

    if cash is not None:
        if short_term is None and long_term is None and total_investments is not None:
            return cash, total_investments, None
        return cash, short_term, long_term

    if cash_and_short is not None:
        return cash_and_short, None, long_term

    if total_investments is not None:
        return total_investments, None, None

    return None, None, None
