"""Shared business helpers for paper portfolio routes and related services."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..utils import finite_float_or_none
from .portfolio_common import apply_market_value_weights, positive_price_or_none, price_map_from_rows

_PRICE_UNAVAILABLE_DETAIL = "Current market price is unavailable. Set price manually."


def _as_position_symbols(positions_raw: dict[str, Any]) -> list[str]:
    return sorted(str(symbol).upper().strip() for symbol in positions_raw.keys())


def _build_price_map(rows: list[Any]) -> dict[str, float | None]:
    return price_map_from_rows(rows, include_missing=True)


def _position_pnl_fields(quantity: float, avg_cost: float, last_price: float | None) -> dict[str, float | None]:
    if last_price is None:
        return {"market_value": None, "unrealized_pnl": None, "unrealized_pnl_pct": None}

    cost_basis = abs(quantity) * avg_cost
    market_value = quantity * last_price
    if quantity > 0:
        unrealized_pnl = (last_price - avg_cost) * quantity
    else:
        unrealized_pnl = (avg_cost - last_price) * abs(quantity)
    unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else None
    return {
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
    }


def to_valid_price(value: Any) -> float | None:
    return positive_price_or_none(value)


def to_finite_number(value: Any) -> float | None:
    return finite_float_or_none(value)


def _normalize_positions_raw(state: Any) -> dict[str, Any]:
    positions_raw = state.get("positions") if isinstance(state, dict) else {}
    if not isinstance(positions_raw, dict):
        return {}
    return positions_raw


def _build_position_rows(
    positions_raw: dict[str, Any],
    price_map: dict[str, float | None],
) -> tuple[list[dict[str, Any]], float, float, bool]:
    positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    has_market_value = False

    for symbol in _as_position_symbols(positions_raw):
        item = positions_raw.get(symbol, {})
        quantity = to_finite_number(item.get("quantity")) or 0.0
        avg_cost = to_valid_price(item.get("avg_cost")) or 0.0
        if abs(quantity) <= 1e-12:
            continue

        cost_basis = abs(quantity) * avg_cost
        total_cost_basis += cost_basis
        last_price = price_map.get(symbol)
        pnl_fields = _position_pnl_fields(quantity, avg_cost, last_price)
        market_value = pnl_fields["market_value"]
        if market_value is not None:
            has_market_value = True
            total_market_value += market_value

        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "cost_basis": cost_basis,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pnl": pnl_fields["unrealized_pnl"],
                "unrealized_pnl_pct": pnl_fields["unrealized_pnl_pct"],
                "weight": None,
            }
        )

    return positions, total_market_value, total_cost_basis, has_market_value


def _apply_position_weights(positions: list[dict[str, Any]], total_market_value: float) -> None:
    apply_market_value_weights(positions, total_market_value)


def _build_recent_trades(state: Any) -> tuple[list[dict[str, Any]], int]:
    trades = state.get("trades") if isinstance(state, dict) and isinstance(state.get("trades"), list) else []
    recent_trades = [dict(item) for item in reversed(trades[-50:]) if isinstance(item, dict)]
    return recent_trades, len(trades)


def _portfolio_summary(
    *,
    state: Any,
    total_market_value: float,
    total_cost_basis: float,
    has_market_value: bool,
) -> dict[str, float | None]:
    cash = to_valid_price(state.get("cash")) or 0.0
    initial_cash = to_valid_price(state.get("initial_cash")) or cash
    equity = cash + total_market_value
    unrealized_total = total_market_value - total_cost_basis if has_market_value else None
    total_return_pct = ((equity - initial_cash) / initial_cash * 100) if initial_cash > 0 else None
    return {
        "initial_cash": initial_cash,
        "cash": cash,
        "market_value": total_market_value,
        "equity": equity,
        "cost_basis": total_cost_basis,
        "unrealized_pnl": unrealized_total,
        "total_return_pct": total_return_pct,
    }


async def resolve_trade_price(hub: Any, symbol: str, explicit_price: float | None) -> tuple[float, str]:
    if explicit_price is not None:
        parsed = to_valid_price(explicit_price)
        if parsed is None:
            raise HTTPException(status_code=400, detail="price must be greater than 0.")
        return parsed, "manual"

    async def _overview_fallback() -> tuple[float | None, str | None]:
        fallback_loader = getattr(hub, "security_overview_payload", None)
        if not callable(fallback_loader):
            return None, None
        try:
            overview = await fallback_loader(
                symbol=symbol,
                refresh=False,
                include_intraday=False,
                include_market=False,
                include_qqq=False,
            )
        except Exception:
            return None, None
        overview_price = (
            overview.get("price", {}).get("current")
            if isinstance(overview, dict) and isinstance(overview.get("price"), dict)
            else None
        )
        parsed_price = to_valid_price(overview_price)
        if parsed_price is None:
            return None, None
        return parsed_price, "daily"

    rows = await hub.current_rows([symbol])
    if not rows:
        parsed, source = await _overview_fallback()
        if parsed is not None and source is not None:
            return parsed, source
        raise HTTPException(status_code=400, detail=_PRICE_UNAVAILABLE_DETAIL)

    latest = rows[0] if isinstance(rows[0], dict) else {}
    parsed = to_valid_price(latest.get("price"))
    if parsed is None:
        parsed, source = await _overview_fallback()
        if parsed is not None and source is not None:
            return parsed, source
        raise HTTPException(status_code=400, detail=_PRICE_UNAVAILABLE_DETAIL)
    return parsed, "market"


async def paper_portfolio_payload(hub: Any, paper_portfolio_store: Any) -> dict[str, Any]:
    state = await paper_portfolio_store.get_state()
    positions_raw = _normalize_positions_raw(state)

    symbols = _as_position_symbols(positions_raw)
    rows = await hub.current_rows(symbols) if symbols else []
    price_map = _build_price_map(rows)
    positions, total_market_value, total_cost_basis, has_market_value = _build_position_rows(
        positions_raw,
        price_map,
    )
    _apply_position_weights(positions, total_market_value)
    recent_trades, trade_count = _build_recent_trades(state)

    return {
        **_portfolio_summary(
            state=state,
            total_market_value=total_market_value,
            total_cost_basis=total_cost_basis,
            has_market_value=has_market_value,
        ),
        "positions": positions,
        "recent_trades": recent_trades,
        "trade_count": trade_count,
        "updated_at": state.get("updated_at"),
    }
