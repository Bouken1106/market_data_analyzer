"""Payload builders for symbol valuation UI/API responses."""

from __future__ import annotations

from statistics import median
from typing import Any

from fastapi import HTTPException

from ..utils import finite_float_or_none, is_valid_symbol, normalize_symbol, utc_now_iso
from .valuation_models import (
    MARKET_JP,
    MARKET_US,
    ComparableMultiples,
    FinancialMetrics,
    ValuationAssumptions,
    calculate_valuation_report,
)


DEFAULT_FAIR_PER = 15.0
DEFAULT_FAIR_PBR = 2.0
DEFAULT_FAIR_PSR = 2.0
DEFAULT_FAIR_EV_SALES = 3.0
DEFAULT_FAIR_EV_EBITDA = 10.0
DEFAULT_FAIR_EV_FCF = 15.0
DEFAULT_FAIR_P_FCF = 15.0
DEFAULT_TARGET_DIVIDEND_YIELD = 0.02
DEFAULT_US_RISK_FREE_RATE = 0.04
DEFAULT_JP_RISK_FREE_RATE = 0.01


async def build_valuation_payload(
    hub: Any,
    symbol: str,
    *,
    refresh: bool = False,
    cache_only: bool = True,
    fair_per: float | None = None,
    fair_pbr: float | None = None,
    fair_psr: float | None = None,
    fair_ev_sales: float | None = None,
    fair_ev_ebitda: float | None = None,
    fair_ev_fcf: float | None = None,
    fair_p_fcf: float | None = None,
    target_dividend_yield: float | None = None,
    risk_free_rate: float | None = None,
    equity_risk_premium: float = 0.055,
    terminal_growth_rate: float = 0.01,
    fcf_growth_rate: float = 0.02,
    forecast_years: int = 5,
) -> dict[str, Any]:
    """Build a UI-friendly valuation payload for one symbol."""

    normalized = normalize_symbol(symbol)
    if not is_valid_symbol(normalized):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    overview_payload, overview_error = await _safe_overview_payload(hub, normalized, refresh=refresh)
    fmp_payload, fmp_error = await _safe_fmp_reference_payload(
        hub,
        normalized,
        refresh=refresh,
        cache_only=cache_only,
    )
    market = _market_for_symbol(normalized)
    resolved_risk_free_rate = _non_negative_or_default(
        risk_free_rate,
        DEFAULT_JP_RISK_FREE_RATE if market == MARKET_JP else DEFAULT_US_RISK_FREE_RATE,
    )
    metrics = _financial_metrics_from_payloads(
        normalized,
        market=market,
        overview_payload=overview_payload,
        fmp_payload=fmp_payload,
        risk_free_rate=resolved_risk_free_rate,
    )
    multiples = ComparableMultiples(
        fair_per=_positive_or_default(fair_per, DEFAULT_FAIR_PER),
        fair_pbr=_positive_or_default(fair_pbr, DEFAULT_FAIR_PBR),
        fair_psr=_positive_or_default(fair_psr, DEFAULT_FAIR_PSR),
        fair_ev_sales=_positive_or_default(fair_ev_sales, DEFAULT_FAIR_EV_SALES),
        fair_ev_ebitda=_positive_or_default(fair_ev_ebitda, DEFAULT_FAIR_EV_EBITDA),
        fair_ev_fcf=_positive_or_default(fair_ev_fcf, DEFAULT_FAIR_EV_FCF),
        fair_p_fcf=_positive_or_default(fair_p_fcf, DEFAULT_FAIR_P_FCF),
        target_dividend_yield=_positive_or_default(target_dividend_yield, DEFAULT_TARGET_DIVIDEND_YIELD),
        source="ui_default_assumptions",
        assumptions={
            "fair_per_default": DEFAULT_FAIR_PER,
            "fair_pbr_default": DEFAULT_FAIR_PBR,
            "fair_psr_default": DEFAULT_FAIR_PSR,
            "fair_ev_sales_default": DEFAULT_FAIR_EV_SALES,
            "fair_ev_ebitda_default": DEFAULT_FAIR_EV_EBITDA,
            "fair_ev_fcf_default": DEFAULT_FAIR_EV_FCF,
            "fair_p_fcf_default": DEFAULT_FAIR_P_FCF,
            "target_dividend_yield_default": DEFAULT_TARGET_DIVIDEND_YIELD,
        },
    )
    assumptions = ValuationAssumptions(
        equity_risk_premium=_rate_or_default(equity_risk_premium, 0.055),
        forecast_years=max(1, min(20, int(forecast_years or 5))),
        fcf_growth_rate=_rate_or_default(fcf_growth_rate, 0.02),
        terminal_growth_rate=_rate_or_default(terminal_growth_rate, 0.01),
        earnings_growth_rate=_rate_or_default(fcf_growth_rate, 0.02),
        dividend_growth_rate=_rate_or_default(terminal_growth_rate, 0.01),
        roe_model_growth_rate=_rate_or_default(terminal_growth_rate, 0.01),
        residual_income_growth_rate=_rate_or_default(terminal_growth_rate, 0.01),
        allow_default_beta=True,
    )
    report = calculate_valuation_report(metrics, multiples=multiples, assumptions=assumptions)
    valuations = _augment_valuations(report.to_dict()["valuations"], metrics.price)
    summary = _valuation_summary(valuations, metrics.price)
    return {
        "symbol": normalized,
        "market": market,
        "currency": metrics.currency,
        "current_price": metrics.price,
        "company_name": metrics.company_name,
        "sector": metrics.sector,
        "industry": metrics.industry,
        "updated_at": utc_now_iso(),
        "source": "valuation-models",
        "input_status": {
            "fundamentals_source": _payload_source(fmp_payload),
            "fundamentals_error": fmp_error,
            "overview_source": _payload_source(overview_payload),
            "overview_error": overview_error,
            "cache_only": cache_only,
            "refresh": refresh,
            "risk_free_rate_source": "request" if risk_free_rate is not None else "static_default",
        },
        "assumptions": {
            "fair_per": multiples.fair_per,
            "fair_pbr": multiples.fair_pbr,
            "fair_psr": multiples.fair_psr,
            "fair_ev_sales": multiples.fair_ev_sales,
            "fair_ev_ebitda": multiples.fair_ev_ebitda,
            "fair_ev_fcf": multiples.fair_ev_fcf,
            "fair_p_fcf": multiples.fair_p_fcf,
            "target_dividend_yield": multiples.target_dividend_yield,
            "risk_free_rate": resolved_risk_free_rate,
            "equity_risk_premium": assumptions.equity_risk_premium,
            "fcf_growth_rate": assumptions.fcf_growth_rate,
            "terminal_growth_rate": assumptions.terminal_growth_rate,
            "forecast_years": assumptions.forecast_years,
        },
        "metrics": report.metrics,
        "summary": summary,
        "valuations": valuations,
    }


async def _safe_overview_payload(hub: Any, symbol: str, *, refresh: bool) -> tuple[dict[str, Any] | None, str | None]:
    fetcher = getattr(hub, "security_overview_payload", None)
    if fetcher is None:
        return None, "security overview service is unavailable"
    try:
        payload = await fetcher(
            symbol=symbol,
            refresh=refresh,
            include_intraday=False,
            include_market=True,
            include_qqq=False,
        )
    except Exception as exc:  # pragma: no cover - defensive around provider failures
        return None, str(exc)
    return payload if isinstance(payload, dict) else None, None


async def _safe_fmp_reference_payload(
    hub: Any,
    symbol: str,
    *,
    refresh: bool,
    cache_only: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if not refresh:
        store_payload = await _fmp_store_payload(hub, symbol)
        if store_payload is not None:
            return store_payload, None

    fetcher = getattr(hub, "fmp_reference_payload", None)
    if fetcher is None:
        if cache_only:
            return None, "No cached FMP reference data found for this symbol."
        return None, "FMP reference service is unavailable"
    if not refresh:
        try:
            payload = await fetcher(symbol, refresh=False, cache_only=True)
        except HTTPException as exc:
            payload = None
            cache_error = str(exc.detail)
        except Exception as exc:  # pragma: no cover - defensive around provider failures
            payload = None
            cache_error = str(exc)
        else:
            cache_error = None
        if isinstance(payload, dict):
            return payload, None
        if cache_only:
            return None, cache_error or "No cached FMP reference data found for this symbol."
    if cache_only:
        return None, "No cached FMP reference data found for this symbol."
    try:
        payload = await fetcher(symbol, refresh=refresh, cache_only=False)
    except HTTPException as exc:
        return None, str(exc.detail)
    except Exception as exc:  # pragma: no cover - defensive around provider failures
        return None, str(exc)
    return payload if isinstance(payload, dict) else None, None


async def _fmp_store_payload(hub: Any, symbol: str) -> dict[str, Any] | None:
    store = getattr(hub, "fmp_reference_store", None)
    getter = getattr(store, "get", None)
    if getter is None:
        return None
    try:
        payload = await getter(symbol)
    except Exception:  # pragma: no cover - cache failures should not block valuation shells
        return None
    return payload if isinstance(payload, dict) else None


def _financial_metrics_from_payloads(
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

    price = first_positive(
        _path_float(overview_payload, "price", "current"),
        _float(profile.get("price")),
        _float(adjusted_prices.get("latest_close")),
        _float(adjusted_prices.get("latest_adj_close")),
    )
    market_cap = _positive_float(profile.get("market_cap"))
    shares = _positive_div(market_cap, price)
    dividend_yield = _positive_float(key_metrics.get("dividend_yield_ttm"))
    dividend_per_share = first_positive(
        _positive_mul(price, dividend_yield),
        _dividend_per_share_from_actions(fmp_payload),
    )
    beta = first_positive(
        _path_float(overview_payload, "market", "beta_60d_vs_spy"),
        _positive_float(profile.get("beta")),
    )
    capex = _positive_abs(cash_flow.get("capital_expenditure"))
    cash_like = _positive_float(balance_sheet.get("cash_and_short_term_investments"))
    data_sources = tuple(
        item
        for item in (
            f"FMP-reference:{_payload_source(fmp_payload)}" if fmp_payload else None,
            f"overview:{_payload_source(overview_payload)}" if overview_payload else None,
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
        net_income=_float(income.get("net_income")),
        eps=first_positive(_float(income.get("eps")), _positive_float(key_metrics.get("eps_ttm"))),
        operating_cash_flow=_float(cash_flow.get("operating_cash_flow")),
        capital_expenditure=capex,
        free_cash_flow=_float(cash_flow.get("free_cash_flow")),
        cash_and_equivalents=cash_like,
        interest_bearing_debt=_non_negative_float(balance_sheet.get("total_debt")),
        total_liabilities=_float(balance_sheet.get("total_liabilities")),
        total_assets=_float(balance_sheet.get("total_assets")),
        equity=_float(balance_sheet.get("total_equity")),
        shareholders_equity=_float(balance_sheet.get("total_equity")),
        bps=_positive_float(key_metrics.get("book_value_per_share_ttm")),
        dividend_per_share=dividend_per_share,
        roe=_float(ratios.get("roe_ttm")),
        per=_positive_float(ratios.get("pe_ratio_ttm")),
        pbr=_positive_float(ratios.get("pb_ratio_ttm")),
        psr=_positive_float(ratios.get("ps_ratio_ttm")),
        beta=beta,
        risk_free_rate=risk_free_rate,
        data_sources=data_sources,
    )


def _augment_valuations(valuations: list[dict[str, Any]], current_price: float | None) -> list[dict[str, Any]]:
    price = _positive_float(current_price)
    enriched: list[dict[str, Any]] = []
    for item in valuations:
        row = dict(item)
        theoretical = _positive_float(row.get("theoretical_price"))
        row["upside_pct"] = ((theoretical / price) - 1.0) * 100.0 if theoretical is not None and price else None
        enriched.append(row)
    return enriched


def _valuation_summary(valuations: list[dict[str, Any]], current_price: float | None) -> dict[str, Any]:
    prices = [
        value
        for value in (_positive_float(item.get("theoretical_price")) for item in valuations)
        if value is not None
    ]
    calculated = len(prices)
    price = _positive_float(current_price)
    if not prices:
        return {
            "calculated_count": 0,
            "method_count": len(valuations),
            "median_price": None,
            "median_upside_pct": None,
            "min_price": None,
            "max_price": None,
        }
    median_price = float(median(prices))
    return {
        "calculated_count": calculated,
        "method_count": len(valuations),
        "median_price": median_price,
        "median_upside_pct": ((median_price / price) - 1.0) * 100.0 if price else None,
        "min_price": min(prices),
        "max_price": max(prices),
    }


def _market_for_symbol(symbol: str) -> str:
    return MARKET_JP if symbol.endswith(".T") else MARKET_US


def _dict_at(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def _path_float(payload: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _float(value)


def _float(value: Any) -> float | None:
    return finite_float_or_none(value)


def _positive_float(value: Any) -> float | None:
    return finite_float_or_none(value, minimum=0.0, strict_minimum=True)


def _non_negative_float(value: Any) -> float | None:
    return finite_float_or_none(value, minimum=0.0)


def _positive_abs(value: Any) -> float | None:
    numeric = _float(value)
    if numeric is None:
        return None
    return abs(numeric) if numeric != 0 else None


def _positive_or_default(value: float | None, default: float) -> float:
    return _positive_float(value) or default


def _non_negative_or_default(value: float | None, default: float) -> float:
    numeric = _non_negative_float(value)
    return default if numeric is None else numeric


def _rate_or_default(value: Any, default: float) -> float:
    numeric = _float(value)
    if numeric is None:
        return default
    return max(-0.5, min(0.5, numeric))


def _positive_div(top: float | None, bottom: float | None) -> float | None:
    if top is None or bottom is None or bottom <= 0:
        return None
    value = top / bottom
    return value if value > 0 else None


def _positive_mul(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    value = left * right
    return value if value > 0 else None


def first_positive(*values: float | None) -> float | None:
    for value in values:
        if value is not None and value > 0:
            return value
    return None


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
        amount = _positive_float(item.get("adj_dividend")) or _positive_float(item.get("dividend"))
        if amount is None:
            continue
        total += amount
        count += 1
    return total if count else None


def _text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _payload_source(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    return str(payload.get("source") or payload.get("_cache_source") or "payload").strip() or "payload"
