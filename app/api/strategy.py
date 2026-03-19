"""Strategy backtest and allocation routes."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import StrategyEvaluationRequest
from ..services.paper_portfolio import paper_portfolio_payload, to_valid_price
from ..strategy_engine import (
    buy_and_hold_backtest,
    build_price_matrix,
    compute_returns,
    estimate_window_stats,
    run_backtest,
    target_weights,
)
from ..utils import normalize_symbols, ok_json_response
from .deps import HubDep, PaperPortfolioStoreDep

router = APIRouter()


async def _fetch_strategy_points(
    hub: Any,
    symbols: list[str],
    months: int,
    refresh: bool,
) -> dict[str, list[dict[str, Any]]]:
    tasks = [
        hub.historical_payload(symbol=symbol, months=months, refresh=refresh)
        for symbol in symbols
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for idx, item in enumerate(responses):
        symbol = symbols[idx]
        if isinstance(item, Exception):
            continue
        points = item.get("points") if isinstance(item, dict) else None
        if isinstance(points, list) and points:
            out[symbol] = points
    return out


def _normalize_strategy_method(raw: str) -> str:
    method = str(raw or "").strip().lower()
    aliases = {
        "equal": "equal_weight",
        "equal_weight": "equal_weight",
        "inverse_volatility": "inverse_volatility",
        "inverse_vol": "inverse_volatility",
        "min_variance": "min_variance",
        "minimum_variance": "min_variance",
    }
    normalized = aliases.get(method)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: equal_weight, inverse_volatility, min_variance.",
        )
    return normalized


def _normalize_rebalance_frequency(raw: str) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
        "quarterly": "quarterly",
    }
    normalized = aliases.get(value)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="rebalance_frequency must be daily, weekly, monthly, or quarterly.",
        )
    return normalized


def _build_trade_proposals(
    target_allocations: list[dict[str, Any]],
    current_positions: list[dict[str, Any]],
    latest_prices: dict[str, float],
    portfolio_equity: float,
    min_trade_value: float,
) -> dict[str, Any]:
    current_qty: dict[str, float] = {}
    for position in current_positions:
        if not isinstance(position, dict):
            continue
        symbol = str(position.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        quantity = position.get("quantity")
        try:
            parsed_quantity = float(quantity)
        except (TypeError, ValueError):
            parsed_quantity = 0.0
        if not math.isfinite(parsed_quantity):
            parsed_quantity = 0.0
        current_qty[symbol] = parsed_quantity

    proposals: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    safe_equity = max(0.0, float(portfolio_equity))
    threshold = max(0.0, float(min_trade_value))
    for item in target_allocations:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        target_weight = item.get("target_weight")
        try:
            weight = float(target_weight)
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight):
            weight = 0.0
        weight = max(0.0, min(1.0, weight))

        price = latest_prices.get(symbol)
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) <= 0:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "missing_market_price",
                }
            )
            continue

        target_value = safe_equity * weight
        current_shares = current_qty.get(symbol, 0.0)
        current_value = current_shares * float(price)
        delta_value = target_value - current_value
        delta_shares = delta_value / float(price)
        side = "hold"
        if delta_shares > 1e-9:
            side = "buy"
        elif delta_shares < -1e-9:
            side = "sell"
        order_value = abs(delta_shares) * float(price)
        if order_value < threshold:
            side = "hold"

        proposals.append(
            {
                "symbol": symbol,
                "side": side,
                "target_weight": weight,
                "target_weight_pct": weight * 100.0,
                "target_value": target_value,
                "current_shares": current_shares,
                "current_value": current_value,
                "delta_shares": delta_shares,
                "quantity": abs(delta_shares),
                "delta_value": delta_value,
                "order_value": order_value,
                "price": float(price),
            }
        )
    return {"trades": proposals, "skipped": skipped}


@router.post("/api/strategy/evaluate")
async def strategy_evaluate(
    req: StrategyEvaluationRequest,
    hub: HubDep,
    paper_portfolio_store: PaperPortfolioStoreDep,
) -> JSONResponse:
    symbols = normalize_symbols(req.symbols)
    if len(symbols) < 2:
        raise HTTPException(status_code=400, detail="At least two symbols are required.")

    method = _normalize_strategy_method(req.method)
    rebalance_frequency = _normalize_rebalance_frequency(req.rebalance_frequency)
    months = max(3, min(int(req.months), 60))
    lookback_days = max(20, min(int(req.lookback_days), 756))
    max_weight = float(req.max_weight)
    if not math.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise HTTPException(status_code=400, detail="max_weight must be in (0, 1].")
    if max_weight * len(symbols) < 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"max_weight={max_weight} is too small for {len(symbols)} symbols.",
        )

    initial_capital = float(req.initial_capital)
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise HTTPException(status_code=400, detail="initial_capital must be greater than 0.")

    commission_bps = float(req.commission_bps)
    slippage_bps = float(req.slippage_bps)
    if not math.isfinite(commission_bps) or commission_bps < 0:
        raise HTTPException(status_code=400, detail="commission_bps must be >= 0.")
    if not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise HTTPException(status_code=400, detail="slippage_bps must be >= 0.")
    transaction_cost_rate = (commission_bps + slippage_bps) / 10_000.0
    rebalance_threshold_pct = max(0.0, float(req.rebalance_threshold_pct))
    min_trade_value = max(0.0, float(req.min_trade_value))

    normalized_benchmark = normalize_symbols([req.benchmark_symbol])
    benchmark_symbol = normalized_benchmark[0] if normalized_benchmark else "SPY"
    fetch_symbols = list(symbols)
    if benchmark_symbol not in fetch_symbols:
        fetch_symbols.append(benchmark_symbol)

    points_by_symbol = await _fetch_strategy_points(hub, fetch_symbols, months=months, refresh=bool(req.refresh))
    asset_points = {symbol: points_by_symbol.get(symbol, []) for symbol in symbols}
    price_dates, prices, aligned_symbols = build_price_matrix(asset_points)
    if not price_dates or prices.shape[0] < (lookback_days + 30):
        raise HTTPException(
            status_code=400,
            detail="Not enough aligned historical data to evaluate the strategy.",
        )
    returns = compute_returns(prices)
    return_dates = price_dates[1:]

    if returns.shape[0] < (lookback_days + 10):
        raise HTTPException(
            status_code=400,
            detail="Not enough return observations for the requested lookback_days.",
        )

    backtest = run_backtest(
        symbols=aligned_symbols,
        return_dates=return_dates,
        returns=returns,
        method=method,
        lookback_days=lookback_days,
        rebalance_frequency=rebalance_frequency,
        rebalance_threshold_pct=rebalance_threshold_pct,
        max_weight=max_weight,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
    )

    latest_window = returns[-lookback_days:, :]
    weights = target_weights(method=method, returns_window=latest_window, max_weight=max_weight)
    plan = estimate_window_stats(aligned_symbols, latest_window, weights)

    benchmark_metrics: dict[str, Any] = {
        "symbol": benchmark_symbol,
        "total_return_pct": None,
        "cagr_pct": None,
        "volatility_pct": None,
        "sharpe": None,
        "max_drawdown_pct": None,
        "win_rate_pct": None,
    }
    benchmark_points = points_by_symbol.get(benchmark_symbol, [])
    if benchmark_points:
        bench_dates, bench_prices, _ = build_price_matrix({benchmark_symbol: benchmark_points})
        if len(bench_dates) >= 2:
            bench_map = {
                bench_dates[i]: float(bench_prices[i, 0])
                for i in range(len(bench_dates))
            }
            aligned_bench_prices = [bench_map.get(d) for d in price_dates]
            if all(isinstance(item, (int, float)) and float(item) > 0 for item in aligned_bench_prices):
                aligned_arr = np.asarray(aligned_bench_prices, dtype=np.float64).reshape(-1, 1)
                bench_returns = compute_returns(aligned_arr)
                bench_backtest = buy_and_hold_backtest(
                    return_dates=return_dates,
                    returns=bench_returns.reshape(-1),
                    initial_capital=initial_capital,
                )
                benchmark_metrics = {"symbol": benchmark_symbol, **bench_backtest.get("metrics", {})}

    portfolio_state = await paper_portfolio_payload(hub, paper_portfolio_store)
    latest_rows = await hub.current_rows(aligned_symbols)
    latest_prices = {
        str(item.get("symbol") or "").upper().strip(): float(to_valid_price(item.get("price")) or 0.0)
        for item in latest_rows
        if isinstance(item, dict) and to_valid_price(item.get("price")) is not None
    }
    trade_proposals = _build_trade_proposals(
        target_allocations=plan.get("symbols", []),
        current_positions=portfolio_state.get("positions", []),
        latest_prices=latest_prices,
        portfolio_equity=float(portfolio_state.get("equity") or initial_capital),
        min_trade_value=min_trade_value,
    )

    return ok_json_response(
        settings={
            "symbols": aligned_symbols,
            "method": method,
            "months": months,
            "lookback_days": lookback_days,
            "rebalance_frequency": rebalance_frequency,
            "rebalance_threshold_pct": rebalance_threshold_pct,
            "max_weight": max_weight,
            "initial_capital": initial_capital,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "benchmark_symbol": benchmark_symbol,
        },
        allocation_plan=plan,
        backtest=backtest,
        benchmark=benchmark_metrics,
        trade_proposals=trade_proposals,
        data_summary={
            "from": price_dates[0],
            "to": price_dates[-1],
            "price_points": len(price_dates),
            "return_points": len(return_dates),
        },
    )
