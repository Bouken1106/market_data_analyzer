"""Shared business helpers for paper portfolio routes and related services."""

from __future__ import annotations

import math
from typing import Any

from fastapi import HTTPException

_PRICE_UNAVAILABLE_DETAIL = "Current market price is unavailable. Set price manually."


def to_valid_price(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def to_finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


async def resolve_trade_price(hub: Any, symbol: str, explicit_price: float | None) -> tuple[float, str]:
    if explicit_price is not None:
        parsed = to_valid_price(explicit_price)
        if parsed is None:
            raise HTTPException(status_code=400, detail="price must be greater than 0.")
        return parsed, "manual"

    rows = await hub.current_rows([symbol])
    if not rows:
        raise HTTPException(status_code=400, detail=_PRICE_UNAVAILABLE_DETAIL)

    latest = rows[0] if isinstance(rows[0], dict) else {}
    parsed = to_valid_price(latest.get("price"))
    if parsed is None:
        raise HTTPException(status_code=400, detail=_PRICE_UNAVAILABLE_DETAIL)
    return parsed, "market"


async def paper_portfolio_payload(hub: Any, paper_portfolio_store: Any) -> dict[str, Any]:
    state = await paper_portfolio_store.get_state()
    positions_raw = state.get("positions") if isinstance(state, dict) else {}
    if not isinstance(positions_raw, dict):
        positions_raw = {}

    symbols = sorted(str(symbol).upper().strip() for symbol in positions_raw.keys())
    rows = await hub.current_rows(symbols) if symbols else []
    price_map: dict[str, float | None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        price_map[symbol] = to_valid_price(row.get("price"))

    positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    has_market_value = False

    for symbol in symbols:
        item = positions_raw.get(symbol, {})
        quantity = to_finite_number(item.get("quantity")) or 0.0
        avg_cost = to_valid_price(item.get("avg_cost")) or 0.0
        if abs(quantity) <= 1e-12:
            continue

        cost_basis = abs(quantity) * avg_cost
        total_cost_basis += cost_basis
        last_price = price_map.get(symbol)
        market_value = None
        unrealized_pnl = None
        unrealized_pnl_pct = None
        if last_price is not None:
            has_market_value = True
            market_value = quantity * last_price
            total_market_value += market_value if isinstance(market_value, (int, float)) else 0.0
            if quantity > 0:
                unrealized_pnl = (last_price - avg_cost) * quantity
            else:
                unrealized_pnl = (avg_cost - last_price) * abs(quantity)
            if cost_basis > 0:
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100

        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "cost_basis": cost_basis,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "weight": None,
            }
        )

    if total_market_value > 0:
        for item in positions:
            market_value = item.get("market_value")
            if isinstance(market_value, (int, float)):
                item["weight"] = (float(market_value) / total_market_value) * 100

    cash = to_valid_price(state.get("cash")) or 0.0
    initial_cash = to_valid_price(state.get("initial_cash")) or cash
    equity = cash + total_market_value
    unrealized_total = total_market_value - total_cost_basis if has_market_value else None
    total_return_pct = ((equity - initial_cash) / initial_cash * 100) if initial_cash > 0 else None

    trades = state.get("trades") if isinstance(state.get("trades"), list) else []
    recent_trades = []
    for item in reversed(trades[-50:]):
        if isinstance(item, dict):
            recent_trades.append(dict(item))

    return {
        "initial_cash": initial_cash,
        "cash": cash,
        "market_value": total_market_value,
        "equity": equity,
        "cost_basis": total_cost_basis,
        "unrealized_pnl": unrealized_total,
        "total_return_pct": total_return_pct,
        "positions": positions,
        "recent_trades": recent_trades,
        "trade_count": len(trades),
        "updated_at": state.get("updated_at"),
    }
