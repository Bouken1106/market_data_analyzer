"""Intrinsic-value calculation models for Japanese and US equities.

This module is intentionally UI/API agnostic.  It accepts normalized market and
financial inputs, calculates derived metrics, and returns one result per
valuation method with either a theoretical price or a concrete reason why the
method cannot be calculated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Any

from .valuation_numeric import (
    abs_div_optional as _rate,
    add_optional as _add,
    clean_finite_dict as _clean_dict,
    div_optional as _div,
    median_positive as _median_positive,
    mul_optional as _mul,
    non_negative_float as _non_negative,
    parse_float as _finite,
    positive_float as _positive,
    sub_optional as _sub,
)
from .valuation_security_rules import (
    SECURITY_BANK,
    SECURITY_INSURANCE,
    SECURITY_OPERATING,
    SECURITY_REIT,
    infer_security_type,
    method_blocked_for_security_type,
)


MARKET_JP = "JP"
MARKET_US = "US"

VALUATION_ROLE_STANDARD = "standard"
VALUATION_ROLE_SUPPORTING = "supporting"
VALUATION_ROLE_DOWNSIDE = "downside_reference"


@dataclass(frozen=True)
class ComparableMultiples:
    """Fair multiples from peer median, sector median, or own historical median."""

    fair_per: float | None = None
    fair_pbr: float | None = None
    fair_psr: float | None = None
    fair_ev_sales: float | None = None
    fair_ev_ebitda: float | None = None
    fair_ev_fcf: float | None = None
    fair_p_fcf: float | None = None
    fair_p_ffo: float | None = None
    target_dividend_yield: float | None = None
    adjusted_fair_per: float | None = None
    adjusted_fair_ev_ebitda: float | None = None
    adjusted_fair_ev_sales: float | None = None
    quality_score: float | None = None
    ev_sales_quality_score: float | None = None
    source: str | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValuationAssumptions:
    """Default assumptions used only when a method needs an explicit assumption."""

    equity_risk_premium: float = 0.047
    tax_rate: float = 0.30
    default_credit_spread: float = 0.02
    forecast_years: int = 5
    earnings_growth_rate: float = 0.02
    fcf_growth_rate: float = 0.02
    dividend_growth_rate: float = 0.01
    terminal_growth_rate: float = 0.01
    roe_model_growth_rate: float = 0.01
    residual_income_growth_rate: float = 0.01
    allow_default_beta: bool = False
    default_beta: float = 1.0
    peer_quality_adjustment_k: float = 0.20
    min_peer_count: int = 3
    min_projection_growth_rate: float = -0.10
    max_projection_growth_rate: float = 0.25
    min_reinvestment_rate: float = -0.25
    max_reinvestment_rate: float = 0.85
    reverse_dcf_min_growth_rate: float = -0.20
    reverse_dcf_max_growth_rate: float = 1.00


@dataclass(frozen=True)
class FinancialMetrics:
    """Normalized financial snapshot.

    Monetary statement fields are company totals in the reporting currency.
    ``capital_expenditure`` is a positive cash outflow.  Parsers should convert
    provider-specific negative capex conventions before constructing this model.
    Rates are decimals, e.g. 0.04 for 4%.
    """

    symbol: str
    market: str
    currency: str | None = None
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    security_type: str | None = None
    fiscal_date: str | None = None

    price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None

    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    depreciation_and_amortization: float | None = None
    net_income: float | None = None
    normalized_net_income: float | None = None
    eps: float | None = None
    forecast_eps: float | None = None

    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    net_borrowing: float | None = None
    working_capital_change: float | None = None

    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    long_term_investments: float | None = None
    interest_bearing_debt: float | None = None
    total_liabilities: float | None = None
    current_assets: float | None = None
    total_assets: float | None = None
    equity: float | None = None
    shareholders_equity: float | None = None
    adjusted_net_assets: float | None = None
    nav: float | None = None
    nav_per_share: float | None = None

    bps: float | None = None
    dividend_per_share: float | None = None
    forecast_dividend_per_share: float | None = None
    dividends_paid: float | None = None
    share_repurchases: float | None = None
    payout_ratio: float | None = None

    interest_expense: float | None = None
    income_tax_expense: float | None = None
    income_before_tax: float | None = None

    roe: float | None = None
    roa: float | None = None
    roic: float | None = None
    per: float | None = None
    pbr: float | None = None
    psr: float | None = None
    ev: float | None = None
    ev_ebitda: float | None = None
    ev_sales: float | None = None
    ev_fcf: float | None = None
    beta: float | None = None
    risk_free_rate: float | None = None
    cost_of_equity: float | None = None
    debt_cost: float | None = None
    wacc: float | None = None
    ffo: float | None = None
    ffo_per_share: float | None = None

    net_income_history: tuple[float, ...] = ()
    revenue_history: tuple[float, ...] = ()
    eps_history: tuple[float, ...] = ()
    free_cash_flow_history: tuple[float, ...] = ()
    data_sources: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValuationResult:
    method_name: str
    theoretical_price: float | None
    used_data: dict[str, Any] = field(default_factory=dict)
    data_sources: tuple[str, ...] = ()
    assumptions: dict[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None
    valuation_role: str = VALUATION_ROLE_SUPPORTING
    is_standard_candidate: bool = False
    calculation_date: str | None = None
    fiscal_date: str | None = None

    @property
    def is_calculated(self) -> bool:
        return self.theoretical_price is not None and self.unavailable_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "theoretical_price": self.theoretical_price,
            "used_data": self.used_data,
            "data_sources": list(self.data_sources),
            "assumptions": self.assumptions,
            "unavailable_reason": self.unavailable_reason,
            "is_calculated": self.is_calculated,
            "valuation_role": self.valuation_role,
            "is_standard_candidate": self.is_standard_candidate,
            "calculation_date": self.calculation_date,
            "fiscal_date": self.fiscal_date,
        }


@dataclass(frozen=True)
class ValuationReport:
    symbol: str
    market: str
    currency: str | None
    metrics: dict[str, Any]
    valuations: tuple[ValuationResult, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "currency": self.currency,
            "metrics": self.metrics,
            "valuations": [item.to_dict() for item in self.valuations],
            "diagnostics": self.diagnostics,
        }


def calculate_valuation_report(
    metrics: FinancialMetrics,
    *,
    multiples: ComparableMultiples | None = None,
    assumptions: ValuationAssumptions | None = None,
    peers: list[FinancialMetrics] | None = None,
) -> ValuationReport:
    """Calculate all supported valuation methods for a normalized snapshot."""

    resolved_assumptions = assumptions or ValuationAssumptions()
    resolved_multiples = multiples or ComparableMultiples()
    if peers is not None:
        resolved_multiples = build_quality_adjusted_multiples_from_metrics(
            metrics,
            peers,
            base_multiples=resolved_multiples,
            assumptions=resolved_assumptions,
        )
    derived = DerivedValuationMetrics(metrics, resolved_assumptions)
    calculator = ValuationCalculator(
        metrics=metrics,
        derived=derived,
        multiples=resolved_multiples,
        assumptions=resolved_assumptions,
    )
    valuations = tuple(calculator.calculate_all())
    diagnostics = {
        "standard_valuation": standard_theoretical_price(valuations),
        "reverse_dcf": calculator.reverse_dcf(),
    }
    return ValuationReport(
        symbol=metrics.symbol,
        market=metrics.market,
        currency=metrics.currency,
        metrics=derived.summary(),
        valuations=valuations,
        diagnostics=diagnostics,
    )


class DerivedValuationMetrics:
    def __init__(self, metrics: FinancialMetrics, assumptions: ValuationAssumptions) -> None:
        self.metrics = metrics
        self.assumptions = assumptions

    @property
    def security_type(self) -> str:
        return infer_security_type(
            explicit_type=self.metrics.security_type,
            sector=self.metrics.sector,
            industry=self.metrics.industry,
            company_name=self.metrics.company_name,
        )

    @property
    def shares(self) -> float | None:
        return _positive(self.metrics.shares_outstanding)

    @property
    def price(self) -> float | None:
        return _positive(self.metrics.price)

    @property
    def equity(self) -> float | None:
        return _positive(self.metrics.shareholders_equity) or _positive(self.metrics.equity)

    @property
    def cash_like(self) -> float | None:
        cash = _non_negative(self.metrics.cash_and_equivalents)
        short_term = _non_negative(self.metrics.short_term_investments) or 0.0
        if cash is None:
            return None
        return cash + short_term

    @property
    def debt(self) -> float | None:
        return _non_negative(self.metrics.interest_bearing_debt)

    @property
    def market_cap(self) -> float | None:
        return _positive(self.metrics.market_cap) or _mul(self.price, self.shares)

    @property
    def ebit(self) -> float | None:
        return _finite(self.metrics.ebit) if self.metrics.ebit is not None else _finite(self.metrics.operating_income)

    @property
    def ebitda(self) -> float | None:
        explicit = _finite(self.metrics.ebitda)
        if explicit is not None:
            return explicit
        return _add(self.ebit, _non_negative(self.metrics.depreciation_and_amortization))

    @property
    def free_cash_flow(self) -> float | None:
        explicit = _finite(self.metrics.free_cash_flow)
        if explicit is not None:
            return explicit
        return _sub(self.metrics.operating_cash_flow, self.metrics.capital_expenditure)

    @property
    def net_debt(self) -> float | None:
        return _sub(self.debt, self.cash_like)

    @property
    def nopat(self) -> float | None:
        return _mul(self.ebit, 1.0 - self.tax_rate)

    @property
    def invested_capital(self) -> float | None:
        return _sub(_add(self.equity, self.debt), self.cash_like)

    @property
    def reinvestment_amount(self) -> float | None:
        capex = _finite(self.metrics.capital_expenditure)
        depreciation = _non_negative(self.metrics.depreciation_and_amortization)
        working_capital_change = _finite(self.metrics.working_capital_change)
        if capex is None:
            return None
        return capex - (depreciation or 0.0) + (working_capital_change or 0.0)

    @property
    def reinvestment_rate(self) -> float | None:
        rate = _div(self.reinvestment_amount, self.nopat)
        if rate is None:
            return None
        if rate <= 0:
            return None
        return _clamp(
            rate,
            self.assumptions.min_reinvestment_rate,
            self.assumptions.max_reinvestment_rate,
        )

    @property
    def sustainable_growth_rate(self) -> float | None:
        roic = self.roic
        reinvestment_rate = self.reinvestment_rate
        if roic is None or reinvestment_rate is None:
            return None
        if roic <= 0 or reinvestment_rate <= 0:
            return None
        return _bounded_growth(roic * reinvestment_rate, self.assumptions)

    @property
    def fcff(self) -> float | None:
        nopat = self.nopat
        reinvestment = self.reinvestment_amount
        if nopat is None or reinvestment is None:
            return None
        return nopat - reinvestment

    @property
    def eps(self) -> float | None:
        explicit = _finite(self.metrics.eps)
        if explicit is not None:
            return explicit
        return _div(self.metrics.net_income, self.shares)

    @property
    def normalized_eps(self) -> float | None:
        normalized_income = _finite(self.metrics.normalized_net_income)
        if normalized_income is None and self.metrics.net_income_history:
            valid = [_finite(item) for item in self.metrics.net_income_history]
            cleaned = [item for item in valid if item is not None]
            if cleaned:
                normalized_income = sum(cleaned) / len(cleaned)
        return _div(normalized_income, self.shares)

    @property
    def bps(self) -> float | None:
        return _finite(self.metrics.bps) or _div(self.equity, self.shares)

    @property
    def revenue_growth_3y(self) -> float | None:
        return _cagr_latest_first(self.metrics.revenue_history, max_periods=3)

    @property
    def revenue_growth_5y(self) -> float | None:
        return _cagr_latest_first(self.metrics.revenue_history, max_periods=5)

    @property
    def eps_growth_3y(self) -> float | None:
        return _cagr_latest_first(self.metrics.eps_history, max_periods=3)

    @property
    def eps_growth_5y(self) -> float | None:
        return _cagr_latest_first(self.metrics.eps_history, max_periods=5)

    @property
    def fcf_growth_3y(self) -> float | None:
        return _cagr_latest_first(self.metrics.free_cash_flow_history, max_periods=3)

    @property
    def operating_margin(self) -> float | None:
        return _div(self.metrics.operating_income, self.metrics.revenue)

    @property
    def gross_margin(self) -> float | None:
        return _div(self.metrics.gross_profit, self.metrics.revenue)

    @property
    def ebitda_margin(self) -> float | None:
        return _div(self.ebitda, self.metrics.revenue)

    @property
    def fcf_margin(self) -> float | None:
        return _div(self.free_cash_flow, self.metrics.revenue)

    @property
    def revenue_per_share(self) -> float | None:
        return _div(self.metrics.revenue, self.shares)

    @property
    def fcf_per_share(self) -> float | None:
        return _div(self.free_cash_flow, self.shares)

    @property
    def ffo_per_share(self) -> float | None:
        return _finite(self.metrics.ffo_per_share) or _div(self.metrics.ffo, self.shares)

    @property
    def nav_per_share(self) -> float | None:
        explicit = _finite(self.metrics.nav_per_share)
        if explicit is not None:
            return explicit
        nav = _positive(self.metrics.nav) or _positive(self.metrics.adjusted_net_assets)
        return _div(nav, self.shares)

    @property
    def dividend_per_share(self) -> float | None:
        explicit = _non_negative(self.metrics.dividend_per_share)
        if explicit is not None:
            return explicit
        dividends_paid = _finite(self.metrics.dividends_paid)
        if dividends_paid is None:
            return None
        return _div(abs(dividends_paid), self.shares)

    @property
    def forecast_dividend_per_share(self) -> float | None:
        return _positive(self.metrics.forecast_dividend_per_share) or self.dividend_per_share

    @property
    def shareholder_payout_ratio(self) -> float | None:
        net_income = _positive(self.metrics.net_income)
        distributions = _add(_non_negative(self.metrics.dividends_paid) or 0.0, _non_negative(self.metrics.share_repurchases) or 0.0)
        payout = _div(distributions, net_income) if distributions else None
        if payout is None:
            payout = _normalize_ratio(self.metrics.payout_ratio)
        if payout is None:
            payout = _div(self.dividend_per_share, self.eps)
        growth = self.sustainable_growth_rate
        roe = self.roe
        if growth is not None and roe is not None and roe > 0:
            fundamental_payout = 1.0 - growth / roe
            if payout is None:
                payout = fundamental_payout
            elif payout < 0.25:
                payout = max(payout, fundamental_payout)
        if payout is None:
            return None
        return _clamp(payout, 0.0, 1.0)

    @property
    def tax_rate(self) -> float:
        explicit = _rate(self.metrics.income_tax_expense, self.metrics.income_before_tax)
        if explicit is not None:
            return _clamp(explicit, 0.0, 0.55)
        return _clamp(self.assumptions.tax_rate, 0.0, 0.55)

    @property
    def beta(self) -> float | None:
        beta = _finite(self.metrics.beta)
        if beta is not None:
            return beta
        if self.assumptions.allow_default_beta:
            return self.assumptions.default_beta
        return None

    @property
    def cost_of_equity(self) -> float | None:
        explicit = _positive(self.metrics.cost_of_equity)
        if explicit is not None:
            return explicit
        rf = _finite(self.metrics.risk_free_rate)
        beta = self.beta
        if rf is None or beta is None:
            return None
        return rf + beta * self.assumptions.equity_risk_premium

    @property
    def debt_cost(self) -> float | None:
        explicit = _positive(self.metrics.debt_cost)
        if explicit is not None:
            return explicit
        expense = _finite(self.metrics.interest_expense)
        debt = self.debt
        if expense is not None and debt is not None and debt > 0:
            return abs(expense) / debt
        rf = _finite(self.metrics.risk_free_rate)
        if rf is None:
            return None
        return rf + self.assumptions.default_credit_spread

    @property
    def wacc(self) -> float | None:
        explicit = _positive(self.metrics.wacc)
        if explicit is not None:
            return explicit

        cost_of_equity = self.cost_of_equity
        if cost_of_equity is None:
            return None

        market_cap = self.market_cap
        debt = self.debt or 0.0
        if market_cap is None or market_cap <= 0:
            return None
        if debt <= 0:
            return cost_of_equity

        debt_cost = self.debt_cost
        if debt_cost is None:
            return None
        capital = market_cap + debt
        return cost_of_equity * (market_cap / capital) + debt_cost * (1.0 - self.tax_rate) * (debt / capital)

    @property
    def ev(self) -> float | None:
        explicit = _positive(self.metrics.ev)
        if explicit is not None:
            return explicit
        market_cap = self.market_cap
        debt = self.debt or 0.0
        cash = self.cash_like or 0.0
        if market_cap is None:
            return None
        return market_cap + debt - cash

    @property
    def roe(self) -> float | None:
        explicit = _finite(self.metrics.roe)
        if explicit is not None:
            return explicit
        return _div(self.metrics.net_income, self.equity)

    @property
    def roa(self) -> float | None:
        explicit = _finite(self.metrics.roa)
        if explicit is not None:
            return explicit
        return _div(self.metrics.net_income, self.metrics.total_assets)

    @property
    def roic(self) -> float | None:
        explicit = _finite(self.metrics.roic)
        if explicit is not None:
            return explicit
        return _div(self.nopat, self.invested_capital)

    def blended_growth_rate(self, *, include_eps: bool = True) -> float:
        candidates = [
            self.revenue_growth_3y,
            self.revenue_growth_5y,
            self.sustainable_growth_rate,
        ]
        if include_eps:
            forecast_eps_growth = _div(_sub(self.metrics.forecast_eps, self.eps), self.eps)
            candidates.extend([forecast_eps_growth, self.eps_growth_3y, self.eps_growth_5y])
        valid = [_bounded_growth(item, self.assumptions) for item in candidates if item is not None]
        if not valid:
            return _bounded_growth(self.assumptions.earnings_growth_rate, self.assumptions)
        return _bounded_growth(_median(valid), self.assumptions)

    def summary(self) -> dict[str, Any]:
        fcf = self.free_cash_flow
        ev = self.ev
        return {
            "price": self.price,
            "shares_outstanding": self.shares,
            "market_cap": self.market_cap,
            "revenue": _finite(self.metrics.revenue),
            "gross_profit": _finite(self.metrics.gross_profit),
            "operating_income": _finite(self.metrics.operating_income),
            "ebit": self.ebit,
            "ebitda": self.ebitda,
            "net_income": _finite(self.metrics.net_income),
            "eps": self.eps,
            "operating_cash_flow": _finite(self.metrics.operating_cash_flow),
            "capital_expenditure": _finite(self.metrics.capital_expenditure),
            "free_cash_flow": fcf,
            "cash_and_equivalents": _finite(self.metrics.cash_and_equivalents),
            "short_term_investments": _finite(self.metrics.short_term_investments),
            "long_term_investments": _finite(self.metrics.long_term_investments),
            "interest_bearing_debt": self.debt,
            "net_debt": self.net_debt,
            "net_assets": _finite(self.metrics.equity),
            "shareholders_equity": self.equity,
            "bps": self.bps,
            "dividend_per_share": self.dividend_per_share,
            "dividends_paid": _finite(self.metrics.dividends_paid),
            "share_repurchases": _finite(self.metrics.share_repurchases),
            "payout_ratio": self.shareholder_payout_ratio,
            "nopat": self.nopat,
            "invested_capital": self.invested_capital,
            "reinvestment_amount": self.reinvestment_amount,
            "reinvestment_rate": self.reinvestment_rate,
            "sustainable_growth_rate": self.sustainable_growth_rate,
            "revenue_growth_3y": self.revenue_growth_3y,
            "revenue_growth_5y": self.revenue_growth_5y,
            "eps_growth_3y": self.eps_growth_3y,
            "eps_growth_5y": self.eps_growth_5y,
            "fcf_growth_3y": self.fcf_growth_3y,
            "roe": self.roe,
            "roa": self.roa,
            "roic": self.roic,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "ebitda_margin": self.ebitda_margin,
            "fcf_margin": self.fcf_margin,
            "per": _finite(self.metrics.per) or _div(self.price, self.eps),
            "pbr": _finite(self.metrics.pbr) or _div(self.price, self.bps),
            "psr": _finite(self.metrics.psr) or _div(self.market_cap, self.metrics.revenue),
            "ev": ev,
            "ev_ebitda": _finite(self.metrics.ev_ebitda) or _div(ev, self.ebitda),
            "ev_sales": _finite(self.metrics.ev_sales) or _div(ev, self.metrics.revenue),
            "ev_fcf": _finite(self.metrics.ev_fcf) or _div(ev, fcf),
            "beta": self.beta,
            "risk_free_rate": _finite(self.metrics.risk_free_rate),
            "cost_of_equity": self.cost_of_equity,
            "debt_cost": self.debt_cost,
            "wacc": self.wacc,
            "security_type": self.security_type,
            "fiscal_date": self.metrics.fiscal_date,
            "data_sources": list(self.metrics.data_sources),
        }


class ValuationCalculator:
    def __init__(
        self,
        *,
        metrics: FinancialMetrics,
        derived: DerivedValuationMetrics,
        multiples: ComparableMultiples,
        assumptions: ValuationAssumptions,
    ) -> None:
        self.metrics = metrics
        self.derived = derived
        self.multiples = multiples
        self.assumptions = assumptions

    def calculate_all(self) -> list[ValuationResult]:
        results = [
            self.growth_adjusted_per(),
            self.growth_adjusted_ev_ebitda(),
            self.growth_adjusted_ev_sales(),
            self.justified_per(),
            self.justified_pbr(),
            self.roic_reinvestment_dcf(),
            self.interest_adjusted_per(),
            self.interest_adjusted_ev_ebitda(),
            self.actual_per(),
            self.forecast_per(),
            self.normalized_per(),
            self.pbr(),
            self.roe_linked_pbr(),
            self.psr(),
            self.ev_sales(),
            self.ev_ebitda(),
            self.ev_fcf(),
            self.p_fcf(),
            self.dividend_yield(),
            self.gordon_growth(),
            self.dcf(),
            self.fcfe(),
            self.residual_income(),
            self.ncav(),
            self.net_cash(),
        ]
        if self.derived.security_type == SECURITY_REIT:
            results.extend([self.nav(), self.ffo_multiple()])
        if self.derived.security_type == SECURITY_INSURANCE:
            results.append(self.adjusted_net_asset())
        return results

    def growth_adjusted_per(self) -> ValuationResult:
        adjusted_multiple = self._neutral_adjusted_multiple(
            adjusted=self.multiples.adjusted_fair_per,
            base=self.multiples.fair_per,
        )
        quality_source = self._quality_adjustment_source(self.multiples.adjusted_fair_per)
        return self._per_share_multiple_method(
            "成長・収益性補正PER法",
            base_value=self.derived.eps,
            multiple=adjusted_multiple,
            base_name="eps",
            multiple_name="adjusted_fair_per",
            missing_base_reason="actual EPS is missing or non-positive",
            missing_multiple_reason="quality-adjusted fair PER is missing",
            blocked_group="per",
            assumptions={
                "quality_score": self._neutral_quality_score(self.multiples.quality_score),
                "quality_adjustment_source": quality_source,
                "peer_quality_adjustment_k": self.assumptions.peer_quality_adjustment_k,
            },
            valuation_role=self._quality_adjusted_role(quality_source),
        )

    def growth_adjusted_ev_ebitda(self) -> ValuationResult:
        adjusted_multiple = self._neutral_adjusted_multiple(
            adjusted=self.multiples.adjusted_fair_ev_ebitda,
            base=self.multiples.fair_ev_ebitda,
        )
        quality_source = self._quality_adjustment_source(self.multiples.adjusted_fair_ev_ebitda)
        return self._enterprise_multiple_method(
            "成長・収益性補正EV/EBITDA法",
            base_value=self.derived.ebitda,
            multiple=adjusted_multiple,
            base_name="ebitda",
            multiple_name="adjusted_fair_ev_ebitda",
            missing_base_reason="EBITDA is missing or non-positive",
            missing_multiple_reason="quality-adjusted fair EV/EBITDA is missing",
            blocked_group="ev_ebitda",
            assumptions={
                "quality_score": self._neutral_quality_score(self.multiples.quality_score),
                "quality_adjustment_source": quality_source,
                "peer_quality_adjustment_k": self.assumptions.peer_quality_adjustment_k,
            },
            valuation_role=self._quality_adjusted_role(quality_source),
        )

    def growth_adjusted_ev_sales(self) -> ValuationResult:
        adjusted_multiple = self._neutral_adjusted_multiple(
            adjusted=self.multiples.adjusted_fair_ev_sales,
            base=self.multiples.fair_ev_sales,
        )
        return self._enterprise_multiple_method(
            "成長・収益性補正EV/Sales法",
            base_value=self.metrics.revenue,
            multiple=adjusted_multiple,
            base_name="revenue",
            multiple_name="adjusted_fair_ev_sales",
            missing_base_reason="revenue is missing or non-positive",
            missing_multiple_reason="quality-adjusted fair EV/Sales is missing",
            blocked_group="ev",
            assumptions={
                "quality_score": self._neutral_quality_score(self.multiples.ev_sales_quality_score),
                "quality_adjustment_source": self._quality_adjustment_source(self.multiples.adjusted_fair_ev_sales),
                "peer_quality_adjustment_k": self.assumptions.peer_quality_adjustment_k,
            },
            valuation_role=VALUATION_ROLE_SUPPORTING,
        )

    def justified_per(self) -> ValuationResult:
        method = "Justified PER法"
        eps = _positive(self.derived.eps)
        payout = self.derived.shareholder_payout_ratio
        cost_of_equity = self.derived.cost_of_equity
        growth = self.derived.blended_growth_rate(include_eps=True)
        if self._blocked_for_security_type("per"):
            return self._unavailable(method, "sector rule excludes this method")
        if eps is None:
            return self._unavailable(method, "EPS is missing or non-positive")
        if payout is None or payout <= 0:
            return self._unavailable(method, "shareholder payout ratio is missing or non-positive", {"eps": eps})
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"eps": eps, "payout_ratio": payout})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than growth rate",
                {"cost_of_equity": cost_of_equity, "growth_rate": growth},
            )
        justified_per = payout * (1.0 + growth) / (cost_of_equity - growth)
        if justified_per <= 0:
            return self._unavailable(method, "justified PER is non-positive", {"justified_per": justified_per})
        return self._priced(
            method,
            eps * justified_per,
            {
                "eps": eps,
                "shareholder_payout_ratio": payout,
                "growth_rate": growth,
                "cost_of_equity": cost_of_equity,
                "justified_per": justified_per,
            },
            {"growth_rate": growth},
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def justified_pbr(self) -> ValuationResult:
        method = "Justified PBR法"
        bps = _positive(self.derived.bps)
        roe = self.derived.roe
        cost_of_equity = self.derived.cost_of_equity
        growth = self.derived.blended_growth_rate(include_eps=True)
        if bps is None:
            return self._unavailable(method, "BPS is missing or non-positive")
        if roe is None:
            return self._unavailable(method, "ROE is missing", {"bps": bps})
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"bps": bps, "roe": roe})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than growth rate",
                {"cost_of_equity": cost_of_equity, "growth_rate": growth},
            )
        theoretical_pbr = (roe - growth) / (cost_of_equity - growth)
        if theoretical_pbr <= 0:
            return self._unavailable(method, "theoretical PBR is non-positive", {"theoretical_pbr": theoretical_pbr})
        return self._priced(
            method,
            bps * theoretical_pbr,
            {
                "bps": bps,
                "roe": roe,
                "growth_rate": growth,
                "cost_of_equity": cost_of_equity,
                "theoretical_pbr": theoretical_pbr,
            },
            {"growth_rate": growth},
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def roic_reinvestment_dcf(self) -> ValuationResult:
        method = "ROIC・再投資率DCF法"
        if self._blocked_for_security_type("fcf"):
            return self._unavailable(method, "sector rule excludes this method")
        nopat = self.derived.nopat
        reinvestment = self.derived.reinvestment_amount
        reinvestment_rate = self.derived.reinvestment_rate
        roic = self.derived.roic
        fcff = self.derived.fcff
        wacc = self.derived.wacc
        terminal_growth = self.assumptions.terminal_growth_rate
        if nopat is None or nopat <= 0:
            return self._unavailable(method, "NOPAT is missing or non-positive")
        if reinvestment is None or reinvestment_rate is None:
            return self._unavailable(method, "reinvestment amount or reinvestment rate is missing", {"nopat": nopat})
        if roic is None:
            return self._unavailable(method, "ROIC is missing", {"nopat": nopat})
        if fcff is None or fcff <= 0:
            return self._unavailable(method, "FCFF is missing or non-positive", {"fcff": fcff, "nopat": nopat})
        if wacc is None:
            return self._unavailable(method, "WACC is missing", {"fcff": fcff})
        if wacc <= terminal_growth:
            return self._unavailable(
                method,
                "WACC is not greater than terminal growth rate",
                {"wacc": wacc, "terminal_growth_rate": terminal_growth},
            )
        growth = _bounded_growth(roic * reinvestment_rate, self.assumptions)
        enterprise_value = self._discounted_cash_flow_enterprise_value(
            base_cash_flow=fcff,
            discount_rate=wacc,
            cash_flow_growth=growth,
            terminal_growth=terminal_growth,
        )
        return self._enterprise_value_method(
            method,
            enterprise_value,
            {
                "nopat": nopat,
                "roic": roic,
                "reinvestment_amount": reinvestment,
                "reinvestment_rate": reinvestment_rate,
                "fcff": fcff,
                "wacc": wacc,
            },
            {"fcff_growth_rate": growth, "terminal_growth_rate": terminal_growth},
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def interest_adjusted_per(self) -> ValuationResult:
        method = "金利補正PER法"
        eps = _positive(self.derived.eps)
        cost_of_equity = self.derived.cost_of_equity
        growth = self.derived.blended_growth_rate(include_eps=True)
        payout = self.derived.shareholder_payout_ratio
        if self._blocked_for_security_type("per"):
            return self._unavailable(method, "sector rule excludes this method")
        if eps is None:
            return self._unavailable(method, "EPS is missing or non-positive")
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"eps": eps})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than growth rate",
                {"cost_of_equity": cost_of_equity, "growth_rate": growth},
            )
        numerator = (payout if payout and payout > 0 else 1.0) * (1.0 + growth)
        theoretical_per = numerator / (cost_of_equity - growth)
        return self._priced(
            method,
            eps * theoretical_per,
            {
                "eps": eps,
                "growth_rate": growth,
                "cost_of_equity": cost_of_equity,
                "shareholder_payout_ratio": payout,
                "theoretical_per": theoretical_per,
            },
            {"growth_rate": growth},
            valuation_role=VALUATION_ROLE_SUPPORTING,
        )

    def interest_adjusted_ev_ebitda(self) -> ValuationResult:
        method = "金利補正EV/EBITDA法"
        if self._blocked_for_security_type("ev_ebitda"):
            return self._unavailable(method, "sector rule excludes this method")
        ebitda = _positive(self.derived.ebitda)
        fcff = _positive(self.derived.fcff) or _positive(self.derived.free_cash_flow)
        wacc = self.derived.wacc
        growth = self.derived.blended_growth_rate(include_eps=False)
        if ebitda is None:
            return self._unavailable(method, "EBITDA is missing or non-positive")
        if fcff is None:
            return self._unavailable(method, "FCFF is missing or non-positive", {"ebitda": ebitda})
        if wacc is None:
            return self._unavailable(method, "WACC is missing", {"ebitda": ebitda, "fcff": fcff})
        if wacc <= growth:
            return self._unavailable(
                method,
                "WACC is not greater than growth rate",
                {"wacc": wacc, "growth_rate": growth},
            )
        conversion = fcff / ebitda
        theoretical_multiple = conversion * (1.0 + growth) / (wacc - growth)
        if theoretical_multiple <= 0:
            return self._unavailable(method, "theoretical EV/EBITDA is non-positive", {"theoretical_multiple": theoretical_multiple})
        return self._enterprise_value_method(
            method,
            ebitda * theoretical_multiple,
            {
                "ebitda": ebitda,
                "fcff": fcff,
                "fcff_conversion_rate": conversion,
                "growth_rate": growth,
                "wacc": wacc,
                "theoretical_ev_ebitda": theoretical_multiple,
            },
            {"growth_rate": growth},
            valuation_role=VALUATION_ROLE_SUPPORTING,
        )

    def actual_per(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "実績PER法",
            base_value=self.derived.eps,
            multiple=self.multiples.fair_per,
            base_name="eps",
            multiple_name="fair_per",
            missing_base_reason="actual EPS is missing or non-positive",
            missing_multiple_reason="fair PER is missing",
            blocked_group="per",
        )

    def forecast_per(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "予想PER法",
            base_value=self.metrics.forecast_eps,
            multiple=self.multiples.fair_per,
            base_name="forecast_eps",
            multiple_name="fair_per",
            missing_base_reason="forecast EPS is missing or non-positive",
            missing_multiple_reason="fair PER is missing",
            blocked_group="per",
        )

    def normalized_per(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "正規化PER法",
            base_value=self.derived.normalized_eps,
            multiple=self.multiples.fair_per,
            base_name="normalized_eps",
            multiple_name="fair_per",
            missing_base_reason="normalized EPS is missing or non-positive",
            missing_multiple_reason="fair PER is missing",
            blocked_group="per",
        )

    def pbr(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "PBR法",
            base_value=self.derived.bps,
            multiple=self.multiples.fair_pbr,
            base_name="bps",
            multiple_name="fair_pbr",
            missing_base_reason="BPS is missing or non-positive",
            missing_multiple_reason="fair PBR is missing",
        )

    def roe_linked_pbr(self) -> ValuationResult:
        method = "ROE連動PBR法"
        bps = _positive(self.derived.bps)
        roe = self.derived.roe
        cost_of_equity = self.derived.cost_of_equity
        growth = self.assumptions.roe_model_growth_rate
        if bps is None:
            return self._unavailable(method, "BPS is missing or non-positive")
        if roe is None:
            return self._unavailable(method, "ROE is missing", {"bps": bps})
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"bps": bps, "roe": roe})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than growth rate",
                {"cost_of_equity": cost_of_equity, "growth_rate": growth},
            )
        theoretical_pbr = (roe - growth) / (cost_of_equity - growth)
        if theoretical_pbr <= 0:
            return self._unavailable(method, "theoretical PBR is non-positive", {"theoretical_pbr": theoretical_pbr})
        return self._priced(
            method,
            bps * theoretical_pbr,
            {"bps": bps, "roe": roe, "cost_of_equity": cost_of_equity, "theoretical_pbr": theoretical_pbr},
            {"growth_rate": growth},
        )

    def psr(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "PSR法",
            base_value=self.derived.revenue_per_share,
            multiple=self.multiples.fair_psr,
            base_name="revenue_per_share",
            multiple_name="fair_psr",
            missing_base_reason="revenue per share is missing or non-positive",
            missing_multiple_reason="fair PSR is missing",
            blocked_group="sales",
        )

    def ev_sales(self) -> ValuationResult:
        return self._enterprise_multiple_method(
            "EV/Sales法",
            base_value=self.metrics.revenue,
            multiple=self.multiples.fair_ev_sales,
            base_name="revenue",
            multiple_name="fair_ev_sales",
            missing_base_reason="revenue is missing or non-positive",
            missing_multiple_reason="fair EV/Sales is missing",
            blocked_group="ev",
        )

    def ev_ebitda(self) -> ValuationResult:
        return self._enterprise_multiple_method(
            "EV/EBITDA法",
            base_value=self.derived.ebitda,
            multiple=self.multiples.fair_ev_ebitda,
            base_name="ebitda",
            multiple_name="fair_ev_ebitda",
            missing_base_reason="EBITDA is missing or non-positive",
            missing_multiple_reason="fair EV/EBITDA is missing",
            blocked_group="ev_ebitda",
        )

    def ev_fcf(self) -> ValuationResult:
        return self._enterprise_multiple_method(
            "EV/FCF法",
            base_value=self.derived.free_cash_flow,
            multiple=self.multiples.fair_ev_fcf,
            base_name="free_cash_flow",
            multiple_name="fair_ev_fcf",
            missing_base_reason="FCF is missing or non-positive",
            missing_multiple_reason="fair EV/FCF is missing",
            blocked_group="fcf",
        )

    def p_fcf(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "P/FCF法",
            base_value=self.derived.fcf_per_share,
            multiple=self.multiples.fair_p_fcf,
            base_name="fcf_per_share",
            multiple_name="fair_p_fcf",
            missing_base_reason="FCF per share is missing or non-positive",
            missing_multiple_reason="fair P/FCF is missing",
            blocked_group="fcf",
        )

    def dividend_yield(self) -> ValuationResult:
        method = "配当利回り法"
        dividend = _positive(self.derived.dividend_per_share)
        target_yield = _positive(self.multiples.target_dividend_yield)
        if dividend is None:
            return self._unavailable(method, "dividend per share is missing or non-positive")
        if target_yield is None:
            return self._unavailable(method, "target dividend yield is missing", {"dividend_per_share": dividend})
        return self._priced(
            method,
            dividend / target_yield,
            {"dividend_per_share": dividend, "target_dividend_yield": target_yield},
        )

    def gordon_growth(self) -> ValuationResult:
        method = "ゴードン成長モデル"
        forecast_dividend = _positive(self.metrics.forecast_dividend_per_share)
        current_dividend = _positive(self.derived.dividend_per_share)
        cost_of_equity = self.derived.cost_of_equity
        growth = self.assumptions.dividend_growth_rate
        if forecast_dividend is not None:
            next_dividend = forecast_dividend
            used_data = {"next_dividend": next_dividend, "dividend_source": "forecast"}
        elif current_dividend is not None:
            next_dividend = current_dividend * (1.0 + growth)
            used_data = {"current_dividend": current_dividend, "next_dividend": next_dividend}
        else:
            return self._unavailable(method, "next dividend is missing or non-positive")
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", used_data)
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than dividend growth rate",
                {"cost_of_equity": cost_of_equity, "dividend_growth_rate": growth},
            )
        return self._priced(
            method,
            next_dividend / (cost_of_equity - growth),
            {**used_data, "cost_of_equity": cost_of_equity},
            {"dividend_growth_rate": growth},
        )

    def dcf(self) -> ValuationResult:
        method = "簡易DCF法"
        if self._blocked_for_security_type("fcf"):
            return self._unavailable(method, "sector rule excludes this method")
        fcf = self.derived.free_cash_flow
        wacc = self.derived.wacc
        terminal_growth = self.assumptions.terminal_growth_rate
        if fcf is None or fcf <= 0:
            return self._unavailable(method, "FCF is missing or non-positive")
        if wacc is None:
            return self._unavailable(method, "WACC is missing", {"free_cash_flow": fcf})
        if wacc <= terminal_growth:
            return self._unavailable(
                method,
                "WACC is not greater than terminal growth rate",
                {"wacc": wacc, "terminal_growth_rate": terminal_growth},
            )
        enterprise_value = self._discounted_cash_flow_enterprise_value(
            base_cash_flow=fcf,
            discount_rate=wacc,
            cash_flow_growth=self.assumptions.fcf_growth_rate,
            terminal_growth=terminal_growth,
        )
        return self._enterprise_value_method(
            method,
            enterprise_value,
            {"free_cash_flow": fcf, "wacc": wacc},
            {"fcf_growth_rate": self.assumptions.fcf_growth_rate, "terminal_growth_rate": terminal_growth},
        )

    def fcfe(self) -> ValuationResult:
        method = "FCFE法"
        if self._blocked_for_security_type("fcf"):
            return self._unavailable(method, "sector rule excludes this method")
        operating_cf = _finite(self.metrics.operating_cash_flow)
        capex = _finite(self.metrics.capital_expenditure)
        net_borrowing = _finite(self.metrics.net_borrowing)
        cost_of_equity = self.derived.cost_of_equity
        terminal_growth = self.assumptions.terminal_growth_rate
        if operating_cf is None or capex is None or net_borrowing is None:
            return self._unavailable(method, "operating CF, capex, or net borrowing is missing")
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing")
        if cost_of_equity <= terminal_growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than terminal growth rate",
                {"cost_of_equity": cost_of_equity, "terminal_growth_rate": terminal_growth},
            )
        fcfe = operating_cf - capex + net_borrowing
        if fcfe <= 0:
            return self._unavailable(method, "FCFE is non-positive", {"fcfe": fcfe})
        equity_value = self._discounted_cash_flow_enterprise_value(
            base_cash_flow=fcfe,
            discount_rate=cost_of_equity,
            cash_flow_growth=self.assumptions.fcf_growth_rate,
            terminal_growth=terminal_growth,
        )
        shares = self.derived.shares
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing", {"fcfe": fcfe})
        return self._priced(
            method,
            equity_value / shares,
            {"fcfe": fcfe, "cost_of_equity": cost_of_equity, "shares_outstanding": shares},
            {"fcfe_growth_rate": self.assumptions.fcf_growth_rate, "terminal_growth_rate": terminal_growth},
        )

    def residual_income(self) -> ValuationResult:
        method = "残余利益モデル"
        equity = self.derived.equity
        net_income = _finite(self.metrics.net_income)
        cost_of_equity = self.derived.cost_of_equity
        shares = self.derived.shares
        growth = self.assumptions.residual_income_growth_rate
        if equity is None or shares is None:
            return self._unavailable(method, "equity or shares outstanding is missing")
        if net_income is None:
            return self._unavailable(method, "net income is missing", {"equity": equity})
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"equity": equity})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than residual income growth rate",
                {"cost_of_equity": cost_of_equity, "growth_rate": growth},
            )
        residual_income = net_income - cost_of_equity * equity

        years = self._forecast_years()
        pv = self._present_value_with_terminal(
            base_value=residual_income,
            discount_rate=cost_of_equity,
            growth_rate=growth,
            terminal_growth=growth,
            years=years,
        )
        return self._priced(
            method,
            (equity + pv) / shares,
            {
                "equity": equity,
                "net_income": net_income,
                "residual_income": residual_income,
                "cost_of_equity": cost_of_equity,
                "shares_outstanding": shares,
            },
            {"residual_income_growth_rate": growth, "forecast_years": years},
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def ncav(self) -> ValuationResult:
        method = "NCAV法"
        current_assets = _finite(self.metrics.current_assets)
        total_liabilities = _finite(self.metrics.total_liabilities)
        shares = self.derived.shares
        if current_assets is None or total_liabilities is None:
            return self._unavailable(method, "current assets or total liabilities is missing")
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing")
        ncav = current_assets - total_liabilities
        if ncav <= 0:
            return self._unavailable(method, "NCAV is non-positive", {"ncav": ncav})
        return self._priced(
            method,
            ncav / shares,
            {"current_assets": current_assets, "total_liabilities": total_liabilities, "ncav": ncav},
            valuation_role=VALUATION_ROLE_DOWNSIDE,
        )

    def net_cash(self) -> ValuationResult:
        method = "ネットキャッシュ価値"
        cash_like = self.derived.cash_like
        debt = self.derived.debt
        shares = self.derived.shares
        if cash_like is None or debt is None:
            return self._unavailable(method, "cash, short-term investments, or interest-bearing debt is missing")
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing")
        net_cash = cash_like - debt
        if net_cash <= 0:
            return self._unavailable(method, "net cash is non-positive", {"net_cash": net_cash})
        return self._priced(
            method,
            net_cash / shares,
            {"cash_and_short_term_investments": cash_like, "interest_bearing_debt": debt, "net_cash": net_cash},
            valuation_role=VALUATION_ROLE_DOWNSIDE,
        )

    def nav(self) -> ValuationResult:
        method = "NAV法"
        nav_per_share = _positive(self.derived.nav_per_share)
        if nav_per_share is None:
            return self._unavailable(method, "NAV per share is missing or non-positive")
        return self._priced(method, nav_per_share, {"nav_per_share": nav_per_share}, valuation_role=VALUATION_ROLE_STANDARD)

    def ffo_multiple(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "FFO倍率法",
            base_value=self.derived.ffo_per_share,
            multiple=self.multiples.fair_p_ffo,
            base_name="ffo_per_share",
            multiple_name="fair_p_ffo",
            missing_base_reason="FFO per share is missing or non-positive",
            missing_multiple_reason="fair P/FFO is missing",
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def adjusted_net_asset(self) -> ValuationResult:
        method = "修正純資産法"
        adjusted_net_assets = _positive(self.metrics.adjusted_net_assets)
        shares = self.derived.shares
        if adjusted_net_assets is None:
            return self._unavailable(method, "adjusted net assets is missing or non-positive")
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing")
        return self._priced(
            method,
            adjusted_net_assets / shares,
            {"adjusted_net_assets": adjusted_net_assets},
            valuation_role=VALUATION_ROLE_STANDARD,
        )

    def reverse_dcf(self) -> dict[str, Any]:
        if self._blocked_for_security_type("fcf"):
            return {"is_calculated": False, "unavailable_reason": "sector rule excludes this diagnostic"}

        current_ev = _positive(self.derived.ev)
        base_fcff = _positive(self.derived.fcff) or _positive(self.derived.free_cash_flow)
        wacc = self.derived.wacc
        terminal_growth = self.assumptions.terminal_growth_rate
        if current_ev is None:
            return {"is_calculated": False, "unavailable_reason": "current EV is missing or non-positive"}
        if base_fcff is None:
            return {"is_calculated": False, "unavailable_reason": "FCFF is missing or non-positive", "current_ev": current_ev}
        if wacc is None:
            return {"is_calculated": False, "unavailable_reason": "WACC is missing", "current_ev": current_ev, "fcff": base_fcff}
        if wacc <= terminal_growth:
            return {
                "is_calculated": False,
                "unavailable_reason": "WACC is not greater than terminal growth rate",
                "current_ev": current_ev,
                "fcff": base_fcff,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth,
            }

        lower = self.assumptions.reverse_dcf_min_growth_rate
        upper = self.assumptions.reverse_dcf_max_growth_rate
        if lower >= upper:
            return {
                "is_calculated": False,
                "unavailable_reason": "reverse DCF growth search bounds are invalid",
                "current_ev": current_ev,
                "fcff": base_fcff,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth,
                "growth_search_low": lower,
                "growth_search_high": upper,
            }

        def enterprise_value_at(growth: float) -> float:
            return self._discounted_cash_flow_enterprise_value(
                base_cash_flow=base_fcff,
                discount_rate=wacc,
                cash_flow_growth=growth,
                terminal_growth=terminal_growth,
            )

        low_value = enterprise_value_at(lower)
        high_value = enterprise_value_at(upper)
        if current_ev < low_value or current_ev > high_value:
            return {
                "is_calculated": False,
                "unavailable_reason": "current EV is outside the reverse DCF growth search range",
                "current_ev": current_ev,
                "fcff": base_fcff,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth,
                "growth_search_low": lower,
                "growth_search_high": upper,
                "enterprise_value_at_low": low_value,
                "enterprise_value_at_high": high_value,
                "past_revenue_growth_3y": self.derived.revenue_growth_3y,
                "past_fcf_growth_3y": self.derived.fcf_growth_3y,
                "sustainable_growth_rate": self.derived.sustainable_growth_rate,
            }

        left = lower
        right = upper
        for _ in range(80):
            midpoint = (left + right) / 2.0
            value = enterprise_value_at(midpoint)
            if value < current_ev:
                left = midpoint
            else:
                right = midpoint
        implied_growth = (left + right) / 2.0
        return _clean_dict(
            {
                "is_calculated": True,
                "current_price_implied_fcff_growth_rate": implied_growth,
                "current_ev": current_ev,
                "fcff": base_fcff,
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth,
                "forecast_years": self._forecast_years(),
                "growth_search_low": lower,
                "growth_search_high": upper,
                "past_revenue_growth_3y": self.derived.revenue_growth_3y,
                "past_fcf_growth_3y": self.derived.fcf_growth_3y,
                "sustainable_growth_rate": self.derived.sustainable_growth_rate,
                "peer_average_growth_rate": self.multiples.assumptions.get("peer_revenue_growth_median"),
            }
        )

    def _per_share_multiple_method(
        self,
        method: str,
        *,
        base_value: float | None,
        multiple: float | None,
        base_name: str,
        multiple_name: str,
        missing_base_reason: str,
        missing_multiple_reason: str,
        blocked_group: str | None = None,
        assumptions: dict[str, Any] | None = None,
        valuation_role: str = VALUATION_ROLE_SUPPORTING,
    ) -> ValuationResult:
        if blocked_group and self._blocked_for_security_type(blocked_group):
            return self._unavailable(method, "sector rule excludes this method")
        base = _positive(base_value)
        if base is None:
            return self._unavailable(method, missing_base_reason, {base_name: base_value})
        fair_multiple = _positive(multiple)
        if fair_multiple is None:
            return self._unavailable(method, missing_multiple_reason, {base_name: base})
        return self._priced(
            method,
            base * fair_multiple,
            {base_name: base, multiple_name: fair_multiple},
            assumptions,
            valuation_role=valuation_role,
        )

    def _enterprise_multiple_method(
        self,
        method: str,
        *,
        base_value: float | None,
        multiple: float | None,
        base_name: str,
        multiple_name: str,
        missing_base_reason: str,
        missing_multiple_reason: str,
        blocked_group: str,
        assumptions: dict[str, Any] | None = None,
        valuation_role: str = VALUATION_ROLE_SUPPORTING,
    ) -> ValuationResult:
        if self._blocked_for_security_type(blocked_group):
            return self._unavailable(method, "sector rule excludes this method")
        base = _positive(base_value)
        if base is None:
            return self._unavailable(method, missing_base_reason, {base_name: base_value})
        fair_multiple = _positive(multiple)
        if fair_multiple is None:
            return self._unavailable(method, missing_multiple_reason, {base_name: base})
        return self._enterprise_value_method(
            method,
            base * fair_multiple,
            {base_name: base, multiple_name: fair_multiple},
            assumptions,
            valuation_role=valuation_role,
        )

    def _enterprise_value_method(
        self,
        method: str,
        enterprise_value: float | None,
        used_data: dict[str, Any],
        assumptions: dict[str, Any] | None = None,
        valuation_role: str = VALUATION_ROLE_SUPPORTING,
    ) -> ValuationResult:
        ev = _positive(enterprise_value)
        debt = self.derived.debt
        cash = self.derived.cash_like
        shares = self.derived.shares
        if ev is None:
            return self._unavailable(method, "enterprise value is missing or non-positive", used_data)
        if debt is None or cash is None:
            return self._unavailable(method, "cash or interest-bearing debt is missing", used_data)
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing", used_data)
        equity_value = ev - debt + cash
        if equity_value <= 0:
            return self._unavailable(method, "equity value is non-positive", {**used_data, "equity_value": equity_value})
        return self._priced(
            method,
            equity_value / shares,
            {**used_data, "enterprise_value": ev, "interest_bearing_debt": debt, "cash_like": cash},
            assumptions,
            valuation_role=valuation_role,
        )

    def _discounted_cash_flow_enterprise_value(
        self,
        *,
        base_cash_flow: float,
        discount_rate: float,
        cash_flow_growth: float,
        terminal_growth: float,
    ) -> float:
        return self._present_value_with_terminal(
            base_value=base_cash_flow,
            discount_rate=discount_rate,
            growth_rate=cash_flow_growth,
            terminal_growth=terminal_growth,
            years=self._forecast_years(),
        )

    @staticmethod
    def _present_value_with_terminal(
        *,
        base_value: float,
        discount_rate: float,
        growth_rate: float,
        terminal_growth: float,
        years: int,
    ) -> float:
        pv = 0.0
        future_value = base_value
        for year in range(1, years + 1):
            future_value *= 1.0 + growth_rate
            pv += future_value / ((1.0 + discount_rate) ** year)
        terminal_value = future_value * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
        return pv + terminal_value / ((1.0 + discount_rate) ** years)

    def _forecast_years(self) -> int:
        return max(1, min(20, int(self.assumptions.forecast_years or 1)))

    def _blocked_for_security_type(self, method_group: str) -> bool:
        return method_blocked_for_security_type(self.derived.security_type, method_group)

    def _neutral_adjusted_multiple(self, *, adjusted: float | None, base: float | None) -> float | None:
        if self.multiples.assumptions.get("quality_adjustment_unavailable_reason"):
            return adjusted
        return _positive(adjusted) or _positive(base)

    def _neutral_quality_score(self, score: float | None) -> float | None:
        if self.multiples.assumptions.get("quality_adjustment_unavailable_reason"):
            return score
        return _finite(score) if score is not None else 0.0

    def _quality_adjustment_source(self, adjusted: float | None) -> str | None:
        if self.multiples.assumptions.get("quality_adjustment_unavailable_reason"):
            return "unavailable"
        if _positive(adjusted) is not None:
            return self.multiples.source or "quality_adjusted_peer_median"
        return "neutral_fallback_no_peer_quality"

    def _quality_adjusted_role(self, quality_source: str | None) -> str:
        if quality_source == "neutral_fallback_no_peer_quality":
            return VALUATION_ROLE_SUPPORTING
        return VALUATION_ROLE_STANDARD

    def _priced(
        self,
        method: str,
        price: float | None,
        used_data: dict[str, Any],
        assumptions: dict[str, Any] | None = None,
        valuation_role: str = VALUATION_ROLE_SUPPORTING,
    ) -> ValuationResult:
        normalized_price = _positive(price)
        if normalized_price is None:
            return self._unavailable(method, "calculated price is missing or non-positive", used_data, assumptions)
        if self._is_obvious_outlier(normalized_price):
            return self._unavailable(method, "calculated price was rejected as an obvious outlier", used_data, assumptions)
        return ValuationResult(
            method_name=method,
            theoretical_price=normalized_price,
            used_data=_clean_dict(used_data),
            data_sources=self.metrics.data_sources,
            assumptions=self._method_assumptions(assumptions),
            valuation_role=valuation_role,
            is_standard_candidate=valuation_role == VALUATION_ROLE_STANDARD,
            calculation_date=date.today().isoformat(),
            fiscal_date=self.metrics.fiscal_date,
        )

    def _unavailable(
        self,
        method: str,
        reason: str,
        used_data: dict[str, Any] | None = None,
        assumptions: dict[str, Any] | None = None,
        valuation_role: str = VALUATION_ROLE_SUPPORTING,
    ) -> ValuationResult:
        return ValuationResult(
            method_name=method,
            theoretical_price=None,
            used_data=_clean_dict(used_data or {}),
            data_sources=self.metrics.data_sources,
            assumptions=self._method_assumptions(assumptions),
            unavailable_reason=reason,
            valuation_role=valuation_role,
            is_standard_candidate=False,
            calculation_date=date.today().isoformat(),
            fiscal_date=self.metrics.fiscal_date,
        )

    def _method_assumptions(self, method_assumptions: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "equity_risk_premium": self.assumptions.equity_risk_premium,
            "tax_rate": self.derived.tax_rate,
        }
        if self.multiples.source:
            payload["multiple_source"] = self.multiples.source
        if self.multiples.assumptions:
            payload["multiple_assumptions"] = dict(self.multiples.assumptions)
        if method_assumptions:
            payload.update(method_assumptions)
        return _clean_dict(payload)

    def _is_obvious_outlier(self, theoretical_price: float) -> bool:
        if not math.isfinite(theoretical_price) or theoretical_price <= 0:
            return True
        if theoretical_price > 1e12:
            return True
        current_price = self.derived.price
        if current_price is None:
            return False
        return theoretical_price > current_price * 1000.0


def build_comparable_multiples_from_metrics(
    peers: list[FinancialMetrics],
    *,
    source: str = "peer_median",
) -> ComparableMultiples:
    """Build fair multiples from already-normalized peer snapshots."""

    assumptions = {"peer_count": len(peers)}
    derived_peers = [DerivedValuationMetrics(item, ValuationAssumptions()) for item in peers]
    return ComparableMultiples(
        fair_per=_median_positive([_div(peer.price, peer.eps) for peer in derived_peers if peer.eps and peer.eps > 0]),
        fair_pbr=_median_positive([_div(peer.price, peer.bps) for peer in derived_peers if peer.bps and peer.bps > 0]),
        fair_psr=_median_positive(
            [_div(peer.market_cap, peer.metrics.revenue) for peer in derived_peers if peer.metrics.revenue]
        ),
        fair_ev_sales=_median_positive(
            [_div(peer.ev, peer.metrics.revenue) for peer in derived_peers if peer.metrics.revenue]
        ),
        fair_ev_ebitda=_median_positive(
            [_div(peer.ev, peer.ebitda) for peer in derived_peers if peer.ebitda and peer.ebitda > 0]
        ),
        fair_ev_fcf=_median_positive(
            [_div(peer.ev, peer.free_cash_flow) for peer in derived_peers if peer.free_cash_flow and peer.free_cash_flow > 0]
        ),
        fair_p_fcf=_median_positive(
            [_div(peer.price, peer.fcf_per_share) for peer in derived_peers if peer.fcf_per_share and peer.fcf_per_share > 0]
        ),
        fair_p_ffo=_median_positive(
            [_div(peer.price, peer.ffo_per_share) for peer in derived_peers if peer.ffo_per_share and peer.ffo_per_share > 0]
        ),
        target_dividend_yield=_median_positive(
            [
                _div(peer.dividend_per_share, peer.price)
                for peer in derived_peers
                if peer.dividend_per_share and peer.price
            ]
        ),
        source=source,
        assumptions=assumptions,
    )


def build_quality_adjusted_multiples_from_metrics(
    target: FinancialMetrics,
    peers: list[FinancialMetrics],
    *,
    base_multiples: ComparableMultiples | None = None,
    assumptions: ValuationAssumptions | None = None,
    source: str = "quality_adjusted_peer_median",
) -> ComparableMultiples:
    """Build peer median multiples and growth/profitability-adjusted variants."""

    resolved_assumptions = assumptions or ValuationAssumptions()
    peer_multiples = build_comparable_multiples_from_metrics(peers, source="peer_median")
    base = base_multiples or ComparableMultiples()
    fair_per = base.fair_per or peer_multiples.fair_per
    fair_pbr = base.fair_pbr or peer_multiples.fair_pbr
    fair_psr = base.fair_psr or peer_multiples.fair_psr
    fair_ev_sales = base.fair_ev_sales or peer_multiples.fair_ev_sales
    fair_ev_ebitda = base.fair_ev_ebitda or peer_multiples.fair_ev_ebitda
    fair_ev_fcf = base.fair_ev_fcf or peer_multiples.fair_ev_fcf
    fair_p_fcf = base.fair_p_fcf or peer_multiples.fair_p_fcf
    fair_p_ffo = base.fair_p_ffo or peer_multiples.fair_p_ffo
    target_dividend_yield = base.target_dividend_yield or peer_multiples.target_dividend_yield

    payload = {
        **peer_multiples.assumptions,
        **base.assumptions,
        "peer_count": len(peers),
        "min_peer_count": resolved_assumptions.min_peer_count,
    }
    if len(peers) < resolved_assumptions.min_peer_count:
        payload["quality_adjustment_unavailable_reason"] = "peer count is below minimum"
        return ComparableMultiples(
            fair_per=fair_per,
            fair_pbr=fair_pbr,
            fair_psr=fair_psr,
            fair_ev_sales=fair_ev_sales,
            fair_ev_ebitda=fair_ev_ebitda,
            fair_ev_fcf=fair_ev_fcf,
            fair_p_fcf=fair_p_fcf,
            fair_p_ffo=fair_p_ffo,
            target_dividend_yield=target_dividend_yield,
            source=base.source or source,
            assumptions=payload,
        )

    target_derived = DerivedValuationMetrics(target, resolved_assumptions)
    peer_derived = [DerivedValuationMetrics(item, resolved_assumptions) for item in peers]
    quality_score, quality_components = _quality_score(target_derived, peer_derived)
    ev_sales_score, ev_sales_components = _ev_sales_quality_score(target_derived, peer_derived)
    payload["quality_score_components"] = quality_components
    payload["ev_sales_quality_score_components"] = ev_sales_components
    payload["peer_revenue_growth_median"] = _median_present(
        [_peer_revenue_growth(item) for item in peer_derived]
    )

    k = resolved_assumptions.peer_quality_adjustment_k
    return ComparableMultiples(
        fair_per=fair_per,
        fair_pbr=fair_pbr,
        fair_psr=fair_psr,
        fair_ev_sales=fair_ev_sales,
        fair_ev_ebitda=fair_ev_ebitda,
        fair_ev_fcf=fair_ev_fcf,
        fair_p_fcf=fair_p_fcf,
        fair_p_ffo=fair_p_ffo,
        target_dividend_yield=target_dividend_yield,
        adjusted_fair_per=_adjusted_multiple(fair_per, quality_score, k),
        adjusted_fair_ev_ebitda=_adjusted_multiple(fair_ev_ebitda, quality_score, k),
        adjusted_fair_ev_sales=_adjusted_multiple(fair_ev_sales, ev_sales_score, k),
        quality_score=quality_score,
        ev_sales_quality_score=ev_sales_score,
        source=base.source or source,
        assumptions=payload,
    )


def standard_theoretical_price(valuations: tuple[ValuationResult, ...]) -> dict[str, Any]:
    """Return the standard theoretical price as the median of valid standard candidates."""

    standard_items = [item for item in valuations if item.is_calculated and item.is_standard_candidate]
    prices = [item.theoretical_price for item in standard_items if item.theoretical_price is not None]
    filtered = _filter_price_outliers(prices)
    if not filtered:
        return {
            "standard_theoretical_price": None,
            "candidate_count": 0,
            "method_names": [],
        }
    method_names = [
        item.method_name
        for item in standard_items
        if item.theoretical_price is not None and item.theoretical_price in filtered
    ]
    return {
        "standard_theoretical_price": _median(filtered),
        "candidate_count": len(filtered),
        "method_names": method_names,
        "used_standard_candidates": True,
    }


def _quality_score(
    target: DerivedValuationMetrics,
    peers: list[DerivedValuationMetrics],
) -> tuple[float, dict[str, float]]:
    components = {
        "revenue_growth": (_peer_revenue_growth(target), [_peer_revenue_growth(item) for item in peers], 0.25),
        "roic": (target.roic, [item.roic for item in peers], 0.20),
        "roe": (target.roe, [item.roe for item in peers], 0.15),
        "operating_margin": (target.operating_margin, [item.operating_margin for item in peers], 0.15),
        "fcf_margin": (target.fcf_margin, [item.fcf_margin for item in peers], 0.15),
        "financial_leverage": (_net_debt_level(target), [_net_debt_level(item) for item in peers], -0.10),
    }
    return _weighted_z_score(components)


def _ev_sales_quality_score(
    target: DerivedValuationMetrics,
    peers: list[DerivedValuationMetrics],
) -> tuple[float, dict[str, float]]:
    components = {
        "revenue_growth": (_peer_revenue_growth(target), [_peer_revenue_growth(item) for item in peers], 0.30),
        "gross_margin": (target.gross_margin, [item.gross_margin for item in peers], 0.15),
        "operating_margin": (target.operating_margin, [item.operating_margin for item in peers], 0.15),
        "ebitda_margin": (target.ebitda_margin, [item.ebitda_margin for item in peers], 0.15),
        "fcf_margin": (target.fcf_margin, [item.fcf_margin for item in peers], 0.10),
        "roic": (target.roic, [item.roic for item in peers], 0.10),
        "net_debt_level": (_net_debt_level(target), [_net_debt_level(item) for item in peers], -0.05),
    }
    return _weighted_z_score(components)


def _weighted_z_score(components: dict[str, tuple[float | None, list[float | None], float]]) -> tuple[float, dict[str, float]]:
    weighted = 0.0
    details: dict[str, float] = {}
    used_weight = 0.0
    for name, (target_value, peer_values, weight) in components.items():
        z_score = _z_score(target_value, peer_values)
        if z_score is None:
            continue
        details[name] = z_score
        weighted += weight * z_score
        used_weight += abs(weight)
    if used_weight <= 0:
        return 0.0, details
    return _clamp(weighted, -3.0, 3.0), details


def _z_score(target_value: float | None, peer_values: list[float | None]) -> float | None:
    target = _finite(target_value)
    values = [item for item in (_finite(value) for value in peer_values) if item is not None]
    if target is None or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    std_dev = math.sqrt(variance)
    if std_dev <= 0:
        return 0.0
    return _clamp((target - mean) / std_dev, -3.0, 3.0)


def _adjusted_multiple(base_multiple: float | None, quality_score: float | None, k: float) -> float | None:
    base = _positive(base_multiple)
    score = _finite(quality_score)
    if base is None or score is None:
        return None
    return base * math.exp(_clamp(k, 0.0, 1.0) * score)


def _peer_revenue_growth(derived: DerivedValuationMetrics) -> float | None:
    return derived.revenue_growth_3y if derived.revenue_growth_3y is not None else derived.revenue_growth_5y


def _net_debt_level(derived: DerivedValuationMetrics) -> float | None:
    net_debt = derived.net_debt
    if net_debt is None:
        return None
    ebitda = _positive(derived.ebitda)
    if ebitda is not None:
        return net_debt / ebitda
    equity = _positive(derived.equity)
    if equity is not None:
        return net_debt / equity
    revenue = _positive(derived.metrics.revenue)
    return net_debt / revenue if revenue is not None else None


def _filter_price_outliers(prices: list[float | None]) -> list[float]:
    values = sorted(value for value in (_positive(price) for price in prices) if value is not None)
    if len(values) < 3:
        return values
    center = _median(values)
    return [value for value in values if center * 0.10 <= value <= center * 10.0]


def _cagr_latest_first(history: tuple[float, ...], *, max_periods: int) -> float | None:
    values = [item for item in (_positive(value) for value in history) if item is not None]
    if len(values) < 2:
        return None
    periods = min(max(1, max_periods), len(values) - 1)
    latest = values[0]
    older = values[periods]
    if older <= 0:
        return None
    return (latest / older) ** (1.0 / periods) - 1.0


def _bounded_growth(value: float | None, assumptions: ValuationAssumptions) -> float:
    parsed = _finite(value)
    if parsed is None:
        return _clamp(assumptions.earnings_growth_rate, assumptions.min_projection_growth_rate, assumptions.max_projection_growth_rate)
    return _clamp(parsed, assumptions.min_projection_growth_rate, assumptions.max_projection_growth_rate)


def _normalize_ratio(value: Any) -> float | None:
    numeric = _finite(value)
    if numeric is None:
        return None
    if numeric > 1.0 and numeric <= 100.0:
        numeric = numeric / 100.0
    return numeric


def _median(values: list[float]) -> float:
    cleaned = sorted(values)
    midpoint = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[midpoint]
    return (cleaned[midpoint - 1] + cleaned[midpoint]) / 2.0


def _median_present(values: list[float | None]) -> float | None:
    cleaned = sorted(value for value in values if value is not None)
    return _median(cleaned) if cleaned else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
