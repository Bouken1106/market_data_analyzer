"""Payload builders for symbol valuation UI/API responses."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..utils import exception_detail_text, is_valid_symbol, normalize_symbol, utc_now_iso
from .valuation_models import calculate_valuation_report
from .valuation_payload_inputs import (
    DEFAULT_EQUITY_RISK_PREMIUM,
    DEFAULT_FCF_GROWTH_RATE,
    DEFAULT_FORECAST_YEARS,
    DEFAULT_MARKET_ASSUMPTION_DATE,
    DEFAULT_PEER_QUALITY_ADJUSTMENT_K,
    DEFAULT_TERMINAL_GROWTH_RATE,
    ValuationPayloadOptions,
    build_comparable_multiples,
    build_valuation_assumptions,
    market_for_symbol,
    resolve_risk_free_rate,
    valuation_assumptions_payload,
)
from .valuation_payload_metrics import financial_metrics_from_payloads, payload_source
from .valuation_payload_summary import valuation_summary, valuations_with_upside


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
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE,
    fcf_growth_rate: float = DEFAULT_FCF_GROWTH_RATE,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
    peer_quality_adjustment_k: float = DEFAULT_PEER_QUALITY_ADJUSTMENT_K,
) -> dict[str, Any]:
    """Build a UI-friendly valuation payload for one symbol."""

    normalized = normalize_symbol(symbol)
    if not is_valid_symbol(normalized):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    options = ValuationPayloadOptions(
        fair_per=fair_per,
        fair_pbr=fair_pbr,
        fair_psr=fair_psr,
        fair_ev_sales=fair_ev_sales,
        fair_ev_ebitda=fair_ev_ebitda,
        fair_ev_fcf=fair_ev_fcf,
        fair_p_fcf=fair_p_fcf,
        target_dividend_yield=target_dividend_yield,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        terminal_growth_rate=terminal_growth_rate,
        fcf_growth_rate=fcf_growth_rate,
        forecast_years=forecast_years,
        peer_quality_adjustment_k=peer_quality_adjustment_k,
    )
    overview_payload, overview_error = await _safe_overview_payload(hub, normalized, refresh=refresh)
    fmp_payload, fmp_error = await _safe_fmp_reference_payload(
        hub,
        normalized,
        refresh=refresh,
        cache_only=cache_only,
    )
    market = market_for_symbol(normalized)
    resolved_risk_free_rate = resolve_risk_free_rate(market, options.risk_free_rate)
    metrics = financial_metrics_from_payloads(
        normalized,
        market=market,
        overview_payload=overview_payload,
        fmp_payload=fmp_payload,
        risk_free_rate=resolved_risk_free_rate,
    )
    multiples = build_comparable_multiples(options)
    assumptions = build_valuation_assumptions(options)
    report = calculate_valuation_report(metrics, multiples=multiples, assumptions=assumptions)
    valuations = valuations_with_upside(report.to_dict()["valuations"], metrics.price)
    summary = valuation_summary(valuations, metrics.price)
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
            "fundamentals_source": payload_source(fmp_payload),
            "fundamentals_error": fmp_error,
            "overview_source": payload_source(overview_payload),
            "overview_error": overview_error,
            "cache_only": cache_only,
            "refresh": refresh,
            "risk_free_rate_source": "request" if risk_free_rate is not None else f"market_default:{DEFAULT_MARKET_ASSUMPTION_DATE}",
        },
        "assumptions": valuation_assumptions_payload(
            multiples=multiples,
            assumptions=assumptions,
            risk_free_rate=resolved_risk_free_rate,
        ),
        "metrics": report.metrics,
        "diagnostics": report.to_dict()["diagnostics"],
        "summary": summary,
        "valuations": valuations,
    }


async def _safe_overview_payload(hub: Any, symbol: str, *, refresh: bool) -> tuple[dict[str, Any] | None, str | None]:
    fetcher = getattr(hub, "security_overview_payload", None)
    if fetcher is None:
        return None, "security overview service is unavailable"
    return await _safe_dict_payload(
        fetcher,
        symbol=symbol,
        refresh=refresh,
        include_intraday=False,
        include_market=True,
        include_qqq=False,
    )


async def _safe_fmp_reference_payload(
    hub: Any,
    symbol: str,
    *,
    refresh: bool,
    cache_only: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    fetcher = getattr(hub, "fmp_reference_payload", None)
    if fetcher is not None:
        if not refresh:
            payload, cache_error = await _safe_dict_payload(fetcher, symbol, refresh=False, cache_only=cache_only)
            if payload is not None:
                return payload, None
            if cache_only:
                return None, cache_error or "No cached FMP reference data found for this symbol."
        if cache_only:
            return None, "No cached FMP reference data found for this symbol."
        return await _safe_dict_payload(fetcher, symbol, refresh=refresh, cache_only=False)

    if not refresh:
        store_payload = await _fmp_store_payload(hub, symbol)
        if store_payload is not None:
            payload = dict(store_payload)
            payload["source"] = "cache-store-stale"
            payload["cache_stale"] = True
            return payload, None
    if cache_only:
        return None, "No cached FMP reference data found for this symbol."
    return None, "FMP reference service is unavailable"


async def _safe_dict_payload(fetcher: Any, *args: Any, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = await fetcher(*args, **kwargs)
    except HTTPException as exc:
        return None, exception_detail_text(exc)
    except Exception as exc:  # pragma: no cover - defensive around provider failures
        return None, exception_detail_text(exc)
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
