"""Saved portfolio normalization and risk analysis helpers."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..ohlcv import close_values_by_date
from ..utils import date_or_none, exception_detail_text, percent_of
from .portfolio_common import apply_market_value_weights, price_map_from_rows
from .portfolio_holdings import (
    MAX_HOLDINGS_PER_REGION,
    REGION_CURRENCIES,
    REGION_LABELS,
    normalize_region,
    normalize_region_holdings,
    resolve_region_holdings,
)

DEFAULT_ANALYSIS_LOOKBACK_DAYS = 252
MIN_ANALYSIS_LOOKBACK_DAYS = 63
MAX_ANALYSIS_LOOKBACK_DAYS = 756
MIN_RETURN_OBSERVATIONS = 60
MAX_HISTORICAL_CLOSE_AGE_DAYS = 14


@dataclass
class _HistoricalSeriesBundle:
    last_close_by_symbol: dict[str, float]
    last_close_date_by_symbol: dict[str, str]
    series_by_symbol: dict[str, pd.Series]
    analyzed_symbols: set[str]


@dataclass
class _PricedHoldingsBundle:
    rows: list[dict[str, Any]]
    total_market_value: float
    priced_holdings_count: int
    analyzed_market_value: float
    stale_historical_symbols: list[str]


def resolve_lookback_days(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_ANALYSIS_LOOKBACK_DAYS
    return max(MIN_ANALYSIS_LOOKBACK_DAYS, min(parsed, MAX_ANALYSIS_LOOKBACK_DAYS))


def _historical_close_is_stale(last_close_date: str | None, *, today: date | None = None) -> bool:
    close_date = date_or_none(last_close_date)
    if close_date is None:
        return True
    current_date = today or datetime.now(timezone.utc).date()
    return (current_date - close_date).days > MAX_HISTORICAL_CLOSE_AGE_DAYS


def _build_current_price_map(rows: list[Any]) -> dict[str, float]:
    return {
        symbol: price
        for symbol, price in price_map_from_rows(rows).items()
        if price is not None
    }


def _close_series_from_points(points: Any) -> pd.Series | None:
    if not isinstance(points, list):
        return None

    values = close_values_by_date(points, close_keys=("c",))
    if not values:
        return None

    series = pd.Series(values, dtype="float64")
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _empty_risk_payload(*, lookback_days: int | None, note: str | None) -> dict[str, Any]:
    return {
        "lookback_days": lookback_days,
        "observation_count": 0,
        "analysis_start": None,
        "analysis_end": None,
        "daily_volatility_pct": None,
        "annualized_volatility_pct": None,
        "value_at_risk_95_pct": None,
        "value_at_risk_95_amount": None,
        "expected_shortfall_95_pct": None,
        "expected_shortfall_95_amount": None,
        "max_drawdown_pct": None,
        "note": note,
    }


def _empty_region_summary() -> dict[str, Any]:
    return {
        "holdings_count": 0,
        "priced_holdings_count": 0,
        "analyzed_holdings_count": 0,
        "market_value": None,
        "risk_coverage_pct": None,
        "top_holding_symbol": None,
        "top_holding_weight_pct": None,
        "effective_holdings": None,
    }


def _region_metadata(region: str) -> dict[str, str]:
    normalized_region = normalize_region(region)
    return {
        "region": normalized_region,
        "label": REGION_LABELS[normalized_region],
        "currency": REGION_CURRENCIES[normalized_region],
    }


def _empty_region_payload(region: str) -> dict[str, Any]:
    return {
        **_region_metadata(region),
        "holdings": [],
        "summary": _empty_region_summary(),
        "risk": _empty_risk_payload(lookback_days=None, note="No holdings saved."),
        "warnings": [],
    }


def _top_holding_summary(holdings: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    priced = [item for item in holdings if isinstance(item.get("weight"), (int, float))]
    if not priced:
        return None, None
    top = max(priced, key=lambda item: float(item["weight"]))
    return str(top["symbol"]), float(top["weight"])


def _effective_holdings(holdings: list[dict[str, Any]]) -> float | None:
    weights = np.array(
        [
            float(item["weight"]) / 100.0
            for item in holdings
            if isinstance(item.get("weight"), (int, float)) and float(item["weight"]) > 0
        ],
        dtype=float,
    )
    if weights.size == 0:
        return None
    hhi = float(np.sum(weights ** 2))
    if hhi <= 0:
        return None
    return 1.0 / hhi


def _build_risk_payload(
    *,
    lookback_days: int,
    market_value: float | None,
    portfolio_values: pd.Series | None,
) -> dict[str, Any]:
    base_payload = _empty_risk_payload(lookback_days=lookback_days, note=None)
    if portfolio_values is None or len(portfolio_values) < 2:
        base_payload["note"] = "Risk metrics need more historical data."
        return base_payload

    returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < MIN_RETURN_OBSERVATIONS:
        base_payload["observation_count"] = int(len(returns))
        base_payload["analysis_start"] = portfolio_values.index[0].date().isoformat()
        base_payload["analysis_end"] = portfolio_values.index[-1].date().isoformat()
        base_payload["note"] = f"At least {MIN_RETURN_OBSERVATIONS} daily return observations are required."
        return base_payload

    returns_values = returns.to_numpy(dtype=float)
    daily_volatility = float(np.std(returns_values, ddof=1))
    var_threshold = float(np.quantile(returns_values, 0.05))
    tail = returns_values[returns_values <= var_threshold]
    drawdown = (portfolio_values / portfolio_values.cummax()) - 1.0

    var_pct = max(0.0, -var_threshold * 100.0)
    expected_shortfall_pct = max(0.0, -float(np.mean(tail)) * 100.0) if tail.size else None
    max_drawdown_pct = max(0.0, -float(drawdown.min()) * 100.0)

    base_payload.update(
        {
            "observation_count": int(len(returns)),
            "analysis_start": portfolio_values.index[0].date().isoformat(),
            "analysis_end": portfolio_values.index[-1].date().isoformat(),
            "daily_volatility_pct": daily_volatility * 100.0,
            "annualized_volatility_pct": daily_volatility * math.sqrt(252.0) * 100.0,
            "value_at_risk_95_pct": var_pct,
            "value_at_risk_95_amount": (market_value * (var_pct / 100.0)) if market_value is not None else None,
            "expected_shortfall_95_pct": expected_shortfall_pct,
            "expected_shortfall_95_amount": (
                market_value * (expected_shortfall_pct / 100.0)
                if market_value is not None and expected_shortfall_pct is not None
                else None
            ),
            "max_drawdown_pct": max_drawdown_pct,
        }
    )
    return base_payload


def _portfolio_value_series(
    series_by_symbol: dict[str, pd.Series],
    quantities_by_symbol: dict[str, float],
    *,
    lookback_days: int,
) -> pd.Series | None:
    if not series_by_symbol:
        return None

    price_frame = pd.concat(series_by_symbol, axis=1).sort_index().ffill().dropna(how="any")
    if price_frame.empty:
        return None
    price_frame = price_frame.tail(lookback_days + 1)
    quantity_series = pd.Series(quantities_by_symbol, dtype="float64")
    value_frame = price_frame.mul(quantity_series, axis=1)
    portfolio_values = value_frame.sum(axis=1)
    return portfolio_values if not portfolio_values.empty else None


def _historical_months_for_lookback(lookback_days: int) -> int:
    return max(6, int(math.ceil(lookback_days / 21.0)) + 3)


async def _fetch_region_historical_results(
    hub: Any,
    *,
    symbols: list[str],
    lookback_days: int,
) -> list[Any]:
    historical_months = _historical_months_for_lookback(lookback_days)
    historical_years = max(1, int(math.ceil(historical_months / 12.0)))
    return await asyncio.gather(
        *[
            hub.historical_payload(
                symbol=symbol,
                years=historical_years,
                months=historical_months,
                refresh=False,
            )
            for symbol in symbols
        ],
        return_exceptions=True,
    )


def _build_historical_series_bundle(
    *,
    symbols: list[str],
    historical_results: list[Any],
    warnings: list[str],
) -> _HistoricalSeriesBundle:
    bundle = _HistoricalSeriesBundle(
        last_close_by_symbol={},
        last_close_date_by_symbol={},
        series_by_symbol={},
        analyzed_symbols=set(),
    )
    for symbol, result in zip(symbols, historical_results):
        if isinstance(result, Exception):
            warnings.append(f"{symbol}: historical data unavailable ({exception_detail_text(result)}).")
            continue
        points = result.get("points") if isinstance(result, dict) else None
        series = _close_series_from_points(points)
        if series is None:
            warnings.append(f"{symbol}: no usable close-price history returned.")
            continue
        last_close = float(series.iloc[-1])
        bundle.last_close_by_symbol[symbol] = last_close
        bundle.last_close_date_by_symbol[symbol] = series.index[-1].date().isoformat()
        bundle.series_by_symbol[symbol] = series
        bundle.analyzed_symbols.add(symbol)
    return bundle


async def _current_price_map_from_task(current_rows_task: asyncio.Task, warnings: list[str]) -> dict[str, float]:
    try:
        current_rows = await current_rows_task
    except Exception as exc:
        warnings.append(f"Latest prices unavailable ({exc}). Falling back to historical close where possible.")
        return {}
    return _build_current_price_map(current_rows)


def _last_price_fields(
    *,
    current_price: float | None,
    latest_close: float | None,
    last_close_date: str | None,
) -> tuple[float | None, str]:
    historical_close_stale = (
        current_price is None
        and latest_close is not None
        and _historical_close_is_stale(last_close_date)
    )
    if current_price is not None:
        return current_price, "market"
    if historical_close_stale:
        return None, "stale_historical_close"
    if latest_close is not None:
        return latest_close, "historical_close"
    return None, "unavailable"


def _build_priced_holdings_bundle(
    *,
    holdings: list[dict[str, float]],
    current_price_by_symbol: dict[str, float],
    historical: _HistoricalSeriesBundle,
) -> _PricedHoldingsBundle:
    rows: list[dict[str, Any]] = []
    total_market_value = 0.0
    priced_holdings_count = 0
    analyzed_market_value = 0.0
    stale_historical_symbols: list[str] = []

    for holding in holdings:
        symbol = holding["symbol"]
        quantity = float(holding["quantity"])
        current_price = current_price_by_symbol.get(symbol)
        latest_close = historical.last_close_by_symbol.get(symbol)
        last_close_date = historical.last_close_date_by_symbol.get(symbol)
        last_price, last_price_source = _last_price_fields(
            current_price=current_price,
            latest_close=latest_close,
            last_close_date=last_close_date,
        )
        market_value = (last_price * quantity) if last_price is not None else None
        if market_value is not None:
            priced_holdings_count += 1
            total_market_value += market_value
            if symbol in historical.analyzed_symbols:
                analyzed_market_value += market_value
        elif last_price_source == "stale_historical_close":
            stale_historical_symbols.append(symbol)

        rows.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "last_price": last_price,
                "last_price_source": last_price_source,
                "last_close_date": last_close_date,
                "market_value": market_value,
                "weight": None,
                "risk_included": symbol in historical.analyzed_symbols,
            }
        )

    apply_market_value_weights(rows, total_market_value)
    rows.sort(
        key=lambda item: (
            item.get("market_value") is None,
            -(float(item["market_value"]) if isinstance(item.get("market_value"), (int, float)) else 0.0),
            str(item["symbol"]),
        )
    )
    return _PricedHoldingsBundle(
        rows=rows,
        total_market_value=total_market_value,
        priced_holdings_count=priced_holdings_count,
        analyzed_market_value=analyzed_market_value,
        stale_historical_symbols=stale_historical_symbols,
    )


def _append_stale_historical_warnings(
    *,
    symbols: list[str],
    last_close_date_by_symbol: dict[str, str],
    warnings: list[str],
) -> None:
    for symbol in symbols:
        last_close_date = last_close_date_by_symbol.get(symbol)
        if last_close_date:
            warnings.append(
                f"{symbol}: latest available close is stale ({last_close_date}). Current market value is omitted."
            )
        else:
            warnings.append(
                f"{symbol}: latest available close is stale. Current market value is omitted."
            )


def _append_risk_coverage_note(
    *,
    risk_payload: dict[str, Any],
    analyzed_symbols: set[str],
    analyzed_market_value: float,
    total_market_value: float,
    market_value_or_none: float | None,
) -> None:
    if not analyzed_symbols or market_value_or_none is None or analyzed_market_value >= total_market_value:
        return
    coverage_pct = percent_of(analyzed_market_value, total_market_value)
    if coverage_pct is None:
        return
    existing_note = str(risk_payload.get("note") or "").strip()
    coverage_note = f"Risk metrics cover {coverage_pct:.1f}% of current market value."
    risk_payload["note"] = f"{existing_note} {coverage_note}".strip()


def _build_region_summary(
    *,
    holdings_rows: list[dict[str, Any]],
    priced_holdings_count: int,
    analyzed_symbols: set[str],
    total_market_value: float,
    analyzed_market_value: float,
) -> dict[str, Any]:
    top_holding_symbol, top_holding_weight = _top_holding_summary(holdings_rows)
    return {
        "holdings_count": len(holdings_rows),
        "priced_holdings_count": priced_holdings_count,
        "analyzed_holdings_count": len(analyzed_symbols),
        "market_value": total_market_value if total_market_value > 0 else None,
        "risk_coverage_pct": percent_of(analyzed_market_value, total_market_value),
        "top_holding_symbol": top_holding_symbol,
        "top_holding_weight_pct": top_holding_weight,
        "effective_holdings": _effective_holdings(holdings_rows),
    }


async def analyze_region_portfolio(
    hub: Any,
    *,
    region: str,
    holdings: list[dict[str, float]],
    lookback_days: int,
) -> dict[str, Any]:
    normalized_region = normalize_region(region)
    if not holdings:
        return _empty_region_payload(normalized_region)

    symbols = [item["symbol"] for item in holdings]
    quantities_by_symbol = {item["symbol"]: float(item["quantity"]) for item in holdings}
    warnings: list[str] = []

    current_rows_task = asyncio.create_task(hub.current_rows(symbols))
    historical_results = await _fetch_region_historical_results(
        hub,
        symbols=symbols,
        lookback_days=lookback_days,
    )
    historical = _build_historical_series_bundle(
        symbols=symbols,
        historical_results=historical_results,
        warnings=warnings,
    )
    current_price_by_symbol = await _current_price_map_from_task(current_rows_task, warnings)
    priced = _build_priced_holdings_bundle(
        holdings=holdings,
        current_price_by_symbol=current_price_by_symbol,
        historical=historical,
    )

    portfolio_values = _portfolio_value_series(
        {symbol: historical.series_by_symbol[symbol] for symbol in symbols if symbol in historical.series_by_symbol},
        {symbol: quantities_by_symbol[symbol] for symbol in symbols if symbol in historical.series_by_symbol},
        lookback_days=lookback_days,
    )
    market_value_or_none = priced.total_market_value if priced.total_market_value > 0 else None
    risk_market_value = priced.analyzed_market_value if priced.analyzed_market_value > 0 else market_value_or_none
    risk_payload = _build_risk_payload(
        lookback_days=lookback_days,
        market_value=risk_market_value,
        portfolio_values=portfolio_values,
    )
    _append_stale_historical_warnings(
        symbols=priced.stale_historical_symbols,
        last_close_date_by_symbol=historical.last_close_date_by_symbol,
        warnings=warnings,
    )
    _append_risk_coverage_note(
        risk_payload=risk_payload,
        analyzed_symbols=historical.analyzed_symbols,
        analyzed_market_value=priced.analyzed_market_value,
        total_market_value=priced.total_market_value,
        market_value_or_none=market_value_or_none,
    )

    return {
        **_region_metadata(normalized_region),
        "holdings": priced.rows,
        "summary": _build_region_summary(
            holdings_rows=priced.rows,
            priced_holdings_count=priced.priced_holdings_count,
            analyzed_symbols=historical.analyzed_symbols,
            total_market_value=priced.total_market_value,
            analyzed_market_value=priced.analyzed_market_value,
        ),
        "risk": risk_payload,
        "warnings": warnings,
    }


async def analyze_saved_portfolio(
    hub: Any,
    *,
    jp_holdings: Any,
    us_holdings: Any,
    lookback_days: Any = DEFAULT_ANALYSIS_LOOKBACK_DAYS,
) -> dict[str, Any]:
    resolved_lookback_days = resolve_lookback_days(lookback_days)
    normalized_jp_holdings = normalize_region_holdings(jp_holdings, region="jp")
    normalized_us_holdings = normalize_region_holdings(us_holdings, region="us")
    jp_payload, us_payload = await asyncio.gather(
        analyze_region_portfolio(
            hub,
            region="jp",
            holdings=normalized_jp_holdings,
            lookback_days=resolved_lookback_days,
        ),
        analyze_region_portfolio(
            hub,
            region="us",
            holdings=normalized_us_holdings,
            lookback_days=resolved_lookback_days,
        ),
    )
    return {
        "lookback_days": resolved_lookback_days,
        "portfolio": {
            "jp_holdings": normalized_jp_holdings,
            "us_holdings": normalized_us_holdings,
        },
        "regions": {
            "jp": jp_payload,
            "us": us_payload,
        },
    }
