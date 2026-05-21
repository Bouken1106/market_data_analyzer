"""Pure trade application rules for the paper portfolio store."""

from __future__ import annotations

from typing import Callable

from ..utils import finite_float_or_none, is_valid_symbol, normalize_symbol

SUPPORTED_TRADE_SIDES = frozenset({"buy", "sell", "short", "cover"})
TradeApplier = Callable[..., tuple[float, float | None]]


def validate_trade_request(symbol: str, side: str, quantity: float, price: float) -> tuple[str, str, float, float]:
    normalized_symbol = normalize_symbol(symbol)
    normalized_side = str(side or "").lower().strip()
    qty = finite_float_or_none(quantity, minimum=0.0, strict_minimum=True)
    px = finite_float_or_none(price, minimum=0.0, strict_minimum=True)

    if not is_valid_symbol(normalized_symbol):
        raise ValueError("Invalid symbol format.")
    if normalized_side not in SUPPORTED_TRADE_SIDES:
        raise ValueError("side must be buy, sell, short, or cover.")
    if qty is None:
        raise ValueError("quantity must be greater than 0.")
    if px is None:
        raise ValueError("price must be greater than 0.")
    return normalized_symbol, normalized_side, qty, px


def position_snapshot(positions: dict[str, dict[str, float]], symbol: str) -> tuple[float, float]:
    current = positions.get(symbol, {"quantity": 0.0, "avg_cost": 0.0})
    return float(current.get("quantity") or 0.0), float(current.get("avg_cost") or 0.0)


def store_position(
    positions: dict[str, dict[str, float]],
    symbol: str,
    *,
    quantity: float,
    avg_cost: float,
) -> None:
    positions[symbol] = {
        "quantity": quantity,
        "avg_cost": avg_cost,
    }


def apply_buy_trade(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    quantity: float,
    price: float,
    current_qty: float,
    current_avg_cost: float,
) -> tuple[float, float | None]:
    if current_qty < -1e-9:
        raise ValueError("Cannot buy while short position exists. Use cover.")
    total_cost = quantity * price
    if cash + 1e-9 < total_cost:
        raise ValueError("Insufficient cash balance.")
    new_qty = current_qty + quantity
    if new_qty <= 0:
        raise ValueError("Invalid resulting quantity.")
    new_avg_cost = ((current_qty * current_avg_cost) + total_cost) / new_qty
    store_position(positions, symbol, quantity=new_qty, avg_cost=new_avg_cost)
    return cash - total_cost, None


def apply_sell_trade(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    quantity: float,
    price: float,
    current_qty: float,
    current_avg_cost: float,
) -> tuple[float, float | None]:
    if current_qty < -1e-9:
        raise ValueError("Cannot sell while short position exists. Use cover or short.")
    if current_qty + 1e-9 < quantity:
        raise ValueError("Sell quantity exceeds current position.")
    proceeds = quantity * price
    realized_pnl = (price - current_avg_cost) * quantity
    remaining_qty = current_qty - quantity
    if remaining_qty <= 1e-9:
        positions.pop(symbol, None)
    else:
        store_position(positions, symbol, quantity=remaining_qty, avg_cost=current_avg_cost)
    return cash + proceeds, realized_pnl


def apply_short_trade(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    quantity: float,
    price: float,
    current_qty: float,
    current_avg_cost: float,
) -> tuple[float, float | None]:
    if current_qty > 1e-9:
        raise ValueError("Cannot short while long position exists. Use sell.")
    proceeds = quantity * price
    short_size = abs(current_qty)
    new_short_size = short_size + quantity
    if new_short_size <= 0:
        raise ValueError("Invalid resulting short quantity.")
    new_avg_cost = ((short_size * current_avg_cost) + (quantity * price)) / new_short_size
    store_position(positions, symbol, quantity=-new_short_size, avg_cost=new_avg_cost)
    return cash + proceeds, None


def apply_cover_trade(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    quantity: float,
    price: float,
    current_qty: float,
    current_avg_cost: float,
) -> tuple[float, float | None]:
    if current_qty >= -1e-9:
        raise ValueError("No short position to cover.")
    short_size = abs(current_qty)
    if short_size + 1e-9 < quantity:
        raise ValueError("Cover quantity exceeds current short position.")
    total_cost = quantity * price
    if cash + 1e-9 < total_cost:
        raise ValueError("Insufficient cash balance.")
    realized_pnl = (current_avg_cost - price) * quantity
    remaining_short = short_size - quantity
    if remaining_short <= 1e-9:
        positions.pop(symbol, None)
    else:
        store_position(positions, symbol, quantity=-remaining_short, avg_cost=current_avg_cost)
    return cash - total_cost, realized_pnl


_TRADE_APPLIERS: dict[str, TradeApplier] = {
    "buy": apply_buy_trade,
    "sell": apply_sell_trade,
    "short": apply_short_trade,
    "cover": apply_cover_trade,
}


def apply_trade_to_portfolio_state(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    side: str,
    quantity: float,
    price: float,
) -> tuple[float, float | None]:
    current_qty, current_avg_cost = position_snapshot(positions, symbol)
    apply_trade = _TRADE_APPLIERS.get(side)
    if apply_trade is None:
        raise ValueError("side must be buy, sell, short, or cover.")
    return apply_trade(
        cash=cash,
        positions=positions,
        symbol=symbol,
        quantity=quantity,
        price=price,
        current_qty=current_qty,
        current_avg_cost=current_avg_cost,
    )
