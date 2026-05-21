"""Request-input normalization for valuation payload builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .valuation_models import MARKET_JP, MARKET_US, ComparableMultiples, ValuationAssumptions
from .valuation_numeric import non_negative_float, parse_float, positive_float

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
DEFAULT_EQUITY_RISK_PREMIUM = 0.055
DEFAULT_TERMINAL_GROWTH_RATE = 0.01
DEFAULT_FCF_GROWTH_RATE = 0.02
DEFAULT_FORECAST_YEARS = 5


@dataclass(frozen=True)
class ValuationPayloadOptions:
    fair_per: float | None = None
    fair_pbr: float | None = None
    fair_psr: float | None = None
    fair_ev_sales: float | None = None
    fair_ev_ebitda: float | None = None
    fair_ev_fcf: float | None = None
    fair_p_fcf: float | None = None
    target_dividend_yield: float | None = None
    risk_free_rate: float | None = None
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE
    fcf_growth_rate: float = DEFAULT_FCF_GROWTH_RATE
    forecast_years: int = DEFAULT_FORECAST_YEARS


def market_for_symbol(symbol: str) -> str:
    return MARKET_JP if symbol.endswith(".T") else MARKET_US


def resolve_risk_free_rate(market: str, requested_rate: Any) -> float:
    default = DEFAULT_JP_RISK_FREE_RATE if market == MARKET_JP else DEFAULT_US_RISK_FREE_RATE
    numeric = non_negative_float(requested_rate)
    return default if numeric is None else numeric


def build_comparable_multiples(options: ValuationPayloadOptions) -> ComparableMultiples:
    return ComparableMultiples(
        fair_per=_positive_or_default(options.fair_per, DEFAULT_FAIR_PER),
        fair_pbr=_positive_or_default(options.fair_pbr, DEFAULT_FAIR_PBR),
        fair_psr=_positive_or_default(options.fair_psr, DEFAULT_FAIR_PSR),
        fair_ev_sales=_positive_or_default(options.fair_ev_sales, DEFAULT_FAIR_EV_SALES),
        fair_ev_ebitda=_positive_or_default(options.fair_ev_ebitda, DEFAULT_FAIR_EV_EBITDA),
        fair_ev_fcf=_positive_or_default(options.fair_ev_fcf, DEFAULT_FAIR_EV_FCF),
        fair_p_fcf=_positive_or_default(options.fair_p_fcf, DEFAULT_FAIR_P_FCF),
        target_dividend_yield=_positive_or_default(options.target_dividend_yield, DEFAULT_TARGET_DIVIDEND_YIELD),
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


def build_valuation_assumptions(options: ValuationPayloadOptions) -> ValuationAssumptions:
    return ValuationAssumptions(
        equity_risk_premium=_rate_or_default(options.equity_risk_premium, DEFAULT_EQUITY_RISK_PREMIUM),
        forecast_years=max(1, min(20, int(options.forecast_years or DEFAULT_FORECAST_YEARS))),
        fcf_growth_rate=_rate_or_default(options.fcf_growth_rate, DEFAULT_FCF_GROWTH_RATE),
        terminal_growth_rate=_rate_or_default(options.terminal_growth_rate, DEFAULT_TERMINAL_GROWTH_RATE),
        earnings_growth_rate=_rate_or_default(options.fcf_growth_rate, DEFAULT_FCF_GROWTH_RATE),
        dividend_growth_rate=_rate_or_default(options.terminal_growth_rate, DEFAULT_TERMINAL_GROWTH_RATE),
        roe_model_growth_rate=_rate_or_default(options.terminal_growth_rate, DEFAULT_TERMINAL_GROWTH_RATE),
        residual_income_growth_rate=_rate_or_default(options.terminal_growth_rate, DEFAULT_TERMINAL_GROWTH_RATE),
        allow_default_beta=True,
    )


def valuation_assumptions_payload(
    *,
    multiples: ComparableMultiples,
    assumptions: ValuationAssumptions,
    risk_free_rate: float,
) -> dict[str, Any]:
    return {
        "fair_per": multiples.fair_per,
        "fair_pbr": multiples.fair_pbr,
        "fair_psr": multiples.fair_psr,
        "fair_ev_sales": multiples.fair_ev_sales,
        "fair_ev_ebitda": multiples.fair_ev_ebitda,
        "fair_ev_fcf": multiples.fair_ev_fcf,
        "fair_p_fcf": multiples.fair_p_fcf,
        "target_dividend_yield": multiples.target_dividend_yield,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": assumptions.equity_risk_premium,
        "fcf_growth_rate": assumptions.fcf_growth_rate,
        "terminal_growth_rate": assumptions.terminal_growth_rate,
        "forecast_years": assumptions.forecast_years,
    }


def _positive_or_default(value: Any, default: float) -> float:
    return positive_float(value) or default


def _rate_or_default(value: Any, default: float) -> float:
    numeric = parse_float(value)
    if numeric is None:
        return default
    return max(-0.5, min(0.5, numeric))
