"""Intrinsic-value calculation models for Japanese and US equities.

This module is intentionally UI/API agnostic.  It accepts normalized market and
financial inputs, calculates derived metrics, and returns one result per
valuation method with either a theoretical price or a concrete reason why the
method cannot be calculated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


MARKET_JP = "JP"
MARKET_US = "US"

SECURITY_OPERATING = "operating"
SECURITY_BANK = "bank"
SECURITY_INSURANCE = "insurance"
SECURITY_REIT = "reit"


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
    source: str | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValuationAssumptions:
    """Default assumptions used only when a method needs an explicit assumption."""

    equity_risk_premium: float = 0.055
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

    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
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
        }


@dataclass(frozen=True)
class ValuationReport:
    symbol: str
    market: str
    currency: str | None
    metrics: dict[str, Any]
    valuations: tuple[ValuationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "currency": self.currency,
            "metrics": self.metrics,
            "valuations": [item.to_dict() for item in self.valuations],
        }


def calculate_valuation_report(
    metrics: FinancialMetrics,
    *,
    multiples: ComparableMultiples | None = None,
    assumptions: ValuationAssumptions | None = None,
) -> ValuationReport:
    """Calculate all supported valuation methods for a normalized snapshot."""

    resolved_assumptions = assumptions or ValuationAssumptions()
    resolved_multiples = multiples or ComparableMultiples()
    derived = DerivedValuationMetrics(metrics, resolved_assumptions)
    calculator = ValuationCalculator(
        metrics=metrics,
        derived=derived,
        multiples=resolved_multiples,
        assumptions=resolved_assumptions,
    )
    valuations = tuple(calculator.calculate_all())
    return ValuationReport(
        symbol=metrics.symbol,
        market=metrics.market,
        currency=metrics.currency,
        metrics=derived.summary(),
        valuations=valuations,
    )


class DerivedValuationMetrics:
    def __init__(self, metrics: FinancialMetrics, assumptions: ValuationAssumptions) -> None:
        self.metrics = metrics
        self.assumptions = assumptions

    @property
    def security_type(self) -> str:
        explicit = _norm_text(self.metrics.security_type)
        if explicit in {SECURITY_BANK, SECURITY_INSURANCE, SECURITY_REIT, SECURITY_OPERATING}:
            return explicit

        text = " ".join(
            item
            for item in (
                _norm_text(self.metrics.sector),
                _norm_text(self.metrics.industry),
                _norm_text(self.metrics.company_name),
            )
            if item
        )
        if any(token in text for token in ("reit", "不動産投資信託", "投資法人")):
            return SECURITY_REIT
        if any(token in text for token in ("bank", "banks", "銀行")):
            return SECURITY_BANK
        if any(token in text for token in ("insurance", "保険")):
            return SECURITY_INSURANCE
        return SECURITY_OPERATING

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
        nopat = _mul(self.ebit, 1.0 - self.tax_rate)
        invested_capital = _sub(_add(self.equity, self.debt), self.cash_like)
        return _div(nopat, invested_capital)

    def summary(self) -> dict[str, Any]:
        fcf = self.free_cash_flow
        ev = self.ev
        return {
            "price": self.price,
            "shares_outstanding": self.shares,
            "market_cap": self.market_cap,
            "revenue": _finite(self.metrics.revenue),
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
            "interest_bearing_debt": self.debt,
            "net_assets": _finite(self.metrics.equity),
            "shareholders_equity": self.equity,
            "bps": self.bps,
            "dividend_per_share": self.dividend_per_share,
            "payout_ratio": _finite(self.metrics.payout_ratio) or _div(self.dividend_per_share, self.eps),
            "roe": self.roe,
            "roa": self.roa,
            "roic": self.roic,
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
        dividend = _positive(self.derived.forecast_dividend_per_share)
        cost_of_equity = self.derived.cost_of_equity
        growth = self.assumptions.dividend_growth_rate
        if dividend is None:
            return self._unavailable(method, "next dividend is missing or non-positive")
        if cost_of_equity is None:
            return self._unavailable(method, "cost of equity is missing", {"next_dividend": dividend})
        if cost_of_equity <= growth:
            return self._unavailable(
                method,
                "cost of equity is not greater than dividend growth rate",
                {"cost_of_equity": cost_of_equity, "dividend_growth_rate": growth},
            )
        next_dividend = dividend * (1.0 + growth)
        return self._priced(
            method,
            next_dividend / (cost_of_equity - growth),
            {"next_dividend": next_dividend, "cost_of_equity": cost_of_equity},
            {"dividend_growth_rate": growth},
        )

    def dcf(self) -> ValuationResult:
        method = "簡易DCF法"
        if self._blocked_for_sector("fcf"):
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
        if self._blocked_for_sector("fcf"):
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
        if residual_income <= 0:
            return self._unavailable(method, "residual income is non-positive", {"residual_income": residual_income})

        pv = 0.0
        future_residual = residual_income
        years = self._forecast_years()
        for year in range(1, years + 1):
            future_residual *= 1.0 + growth
            pv += future_residual / ((1.0 + cost_of_equity) ** year)
        terminal = future_residual * (1.0 + growth) / (cost_of_equity - growth)
        pv += terminal / ((1.0 + cost_of_equity) ** years)
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
        )

    def nav(self) -> ValuationResult:
        method = "NAV法"
        nav_per_share = _positive(self.derived.nav_per_share)
        if nav_per_share is None:
            return self._unavailable(method, "NAV per share is missing or non-positive")
        return self._priced(method, nav_per_share, {"nav_per_share": nav_per_share})

    def ffo_multiple(self) -> ValuationResult:
        return self._per_share_multiple_method(
            "FFO倍率法",
            base_value=self.derived.ffo_per_share,
            multiple=self.multiples.fair_p_ffo,
            base_name="ffo_per_share",
            multiple_name="fair_p_ffo",
            missing_base_reason="FFO per share is missing or non-positive",
            missing_multiple_reason="fair P/FFO is missing",
        )

    def adjusted_net_asset(self) -> ValuationResult:
        method = "修正純資産法"
        adjusted_net_assets = _positive(self.metrics.adjusted_net_assets)
        shares = self.derived.shares
        if adjusted_net_assets is None:
            return self._unavailable(method, "adjusted net assets is missing or non-positive")
        if shares is None:
            return self._unavailable(method, "shares outstanding is missing")
        return self._priced(method, adjusted_net_assets / shares, {"adjusted_net_assets": adjusted_net_assets})

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
    ) -> ValuationResult:
        if blocked_group and self._blocked_for_sector(blocked_group):
            return self._unavailable(method, "sector rule excludes this method")
        base = _positive(base_value)
        if base is None:
            return self._unavailable(method, missing_base_reason, {base_name: base_value})
        fair_multiple = _positive(multiple)
        if fair_multiple is None:
            return self._unavailable(method, missing_multiple_reason, {base_name: base})
        return self._priced(method, base * fair_multiple, {base_name: base, multiple_name: fair_multiple})

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
    ) -> ValuationResult:
        if self._blocked_for_sector(blocked_group):
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
        )

    def _enterprise_value_method(
        self,
        method: str,
        enterprise_value: float | None,
        used_data: dict[str, Any],
        assumptions: dict[str, Any] | None = None,
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
        )

    def _discounted_cash_flow_enterprise_value(
        self,
        *,
        base_cash_flow: float,
        discount_rate: float,
        cash_flow_growth: float,
        terminal_growth: float,
    ) -> float:
        years = self._forecast_years()
        pv = 0.0
        future_cash_flow = base_cash_flow
        for year in range(1, years + 1):
            future_cash_flow *= 1.0 + cash_flow_growth
            pv += future_cash_flow / ((1.0 + discount_rate) ** year)
        terminal_value = future_cash_flow * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
        return pv + terminal_value / ((1.0 + discount_rate) ** years)

    def _forecast_years(self) -> int:
        return max(1, min(20, int(self.assumptions.forecast_years or 1)))

    def _blocked_for_sector(self, method_group: str) -> bool:
        security_type = self.derived.security_type
        if security_type == SECURITY_BANK:
            return method_group in {"sales", "ev", "ev_ebitda", "fcf"}
        if security_type == SECURITY_INSURANCE:
            return method_group in {"sales", "ev", "ev_ebitda", "fcf"}
        if security_type == SECURITY_REIT:
            return method_group in {"per", "sales", "ev", "ev_ebitda", "fcf"}
        return False

    def _priced(
        self,
        method: str,
        price: float | None,
        used_data: dict[str, Any],
        assumptions: dict[str, Any] | None = None,
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
        )

    def _unavailable(
        self,
        method: str,
        reason: str,
        used_data: dict[str, Any] | None = None,
        assumptions: dict[str, Any] | None = None,
    ) -> ValuationResult:
        return ValuationResult(
            method_name=method,
            theoretical_price=None,
            used_data=_clean_dict(used_data or {}),
            data_sources=self.metrics.data_sources,
            assumptions=self._method_assumptions(assumptions),
            unavailable_reason=reason,
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


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _positive(value: Any) -> float | None:
    numeric = _finite(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _non_negative(value: Any) -> float | None:
    numeric = _finite(value)
    if numeric is None or numeric < 0:
        return None
    return numeric


def _div(numerator: Any, denominator: Any) -> float | None:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


def _mul(left: Any, right: Any) -> float | None:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return None
    return left_value * right_value


def _add(left: Any, right: Any) -> float | None:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None and right_value is None:
        return None
    return (left_value or 0.0) + (right_value or 0.0)


def _sub(left: Any, right: Any) -> float | None:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _rate(numerator: Any, denominator: Any) -> float | None:
    value = _div(numerator, denominator)
    if value is None:
        return None
    return abs(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if value is None:
            continue
        cleaned[key] = value
    return cleaned


def _median_positive(values: list[float | None]) -> float | None:
    cleaned = sorted(value for value in (_positive(item) for item in values) if value is not None)
    if not cleaned:
        return None
    midpoint = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[midpoint]
    return (cleaned[midpoint - 1] + cleaned[midpoint]) / 2.0
