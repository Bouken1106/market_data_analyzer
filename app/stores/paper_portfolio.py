"""Persistent store for paper-trading portfolio state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, utc_now_iso, write_json_file

_SUPPORTED_TRADE_SIDES = frozenset({"buy", "sell", "short", "cover"})
_MAX_STORED_TRADES = 1000


def _validate_trade_request(symbol: str, side: str, quantity: float, price: float) -> tuple[str, str, float, float]:
    normalized_symbol = normalize_symbol(symbol)
    normalized_side = str(side or "").lower().strip()
    qty = float(quantity)
    px = float(price)

    if not is_valid_symbol(normalized_symbol):
        raise ValueError("Invalid symbol format.")
    if normalized_side not in _SUPPORTED_TRADE_SIDES:
        raise ValueError("side must be buy, sell, short, or cover.")
    if qty <= 0:
        raise ValueError("quantity must be greater than 0.")
    if px <= 0:
        raise ValueError("price must be greater than 0.")
    return normalized_symbol, normalized_side, qty, px


def _position_snapshot(positions: dict[str, dict[str, float]], symbol: str) -> tuple[float, float]:
    current = positions.get(symbol, {"quantity": 0.0, "avg_cost": 0.0})
    return float(current.get("quantity") or 0.0), float(current.get("avg_cost") or 0.0)


def _store_position(
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


def _apply_buy_trade(
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
    _store_position(positions, symbol, quantity=new_qty, avg_cost=new_avg_cost)
    return cash - total_cost, None


def _apply_sell_trade(
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
        _store_position(positions, symbol, quantity=remaining_qty, avg_cost=current_avg_cost)
    return cash + proceeds, realized_pnl


def _apply_short_trade(
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
    _store_position(positions, symbol, quantity=-new_short_size, avg_cost=new_avg_cost)
    return cash + proceeds, None


def _apply_cover_trade(
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
        _store_position(positions, symbol, quantity=-remaining_short, avg_cost=current_avg_cost)
    return cash - total_cost, realized_pnl


def _apply_trade_to_portfolio_state(
    *,
    cash: float,
    positions: dict[str, dict[str, float]],
    symbol: str,
    side: str,
    quantity: float,
    price: float,
) -> tuple[float, float | None]:
    current_qty, current_avg_cost = _position_snapshot(positions, symbol)
    if side == "buy":
        return _apply_buy_trade(
            cash=cash,
            positions=positions,
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_qty=current_qty,
            current_avg_cost=current_avg_cost,
        )
    if side == "sell":
        return _apply_sell_trade(
            cash=cash,
            positions=positions,
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_qty=current_qty,
            current_avg_cost=current_avg_cost,
        )
    if side == "short":
        return _apply_short_trade(
            cash=cash,
            positions=positions,
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_qty=current_qty,
            current_avg_cost=current_avg_cost,
        )
    return _apply_cover_trade(
        cash=cash,
        positions=positions,
        symbol=symbol,
        quantity=quantity,
        price=price,
        current_qty=current_qty,
        current_avg_cost=current_avg_cost,
    )


class PaperPortfolioStore:
    def __init__(self, cache_path: Path, default_initial_cash: float = 1_000_000.0) -> None:
        self.cache_path = cache_path
        self.default_initial_cash = float(default_initial_cash)
        self._lock = asyncio.Lock()
        self._state = self._load_from_disk()

    def _empty_state(self, initial_cash: float | None = None) -> dict[str, Any]:
        base_cash = self.default_initial_cash if initial_cash is None else float(initial_cash)
        return {
            "initial_cash": base_cash,
            "cash": base_cash,
            "positions": {},
            "trades": [],
            "updated_at": utc_now_iso(),
        }

    def _snapshot_state_no_lock(self) -> dict[str, Any]:
        return {
            "initial_cash": float(self._state["initial_cash"]),
            "cash": float(self._state["cash"]),
            "positions": {
                symbol: {
                    "quantity": float(item["quantity"]),
                    "avg_cost": float(item["avg_cost"]),
                }
                for symbol, item in self._state["positions"].items()
            },
            "trades": [dict(item) for item in self._state["trades"]],
            "updated_at": str(self._state["updated_at"]),
        }

    def _load_from_disk(self) -> dict[str, Any]:
        payload = read_json_file(self.cache_path)
        if payload is None:
            return self._empty_state()

        if not isinstance(payload, dict):
            return self._empty_state()

        initial_cash = self._to_positive_float(payload.get("initial_cash"), fallback=self.default_initial_cash)
        cash = self._to_non_negative_float(payload.get("cash"), fallback=initial_cash)
        positions = self._normalize_positions(payload.get("positions"))
        trades = self._normalize_trades(payload.get("trades"))
        updated_at = str(payload.get("updated_at") or utc_now_iso())
        return {
            "initial_cash": initial_cash,
            "cash": cash,
            "positions": positions,
            "trades": trades,
            "updated_at": updated_at,
        }

    @staticmethod
    def _to_positive_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if parsed <= 0:
            return float(fallback)
        return parsed

    @staticmethod
    def _to_non_negative_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if parsed < 0:
            return float(fallback)
        return parsed

    def _normalize_positions(self, raw: Any) -> dict[str, dict[str, float]]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, float]] = {}
        for symbol_raw, item in raw.items():
            symbol = normalize_symbol(symbol_raw)
            if not is_valid_symbol(symbol):
                continue
            if not isinstance(item, dict):
                continue
            quantity = self._to_non_negative_or_negative_float(item.get("quantity"))
            avg_cost = self._to_positive_float(item.get("avg_cost"), fallback=0.0)
            if quantity is None or abs(quantity) <= 1e-12:
                continue
            if avg_cost <= 0:
                continue
            out[symbol] = {
                "quantity": quantity,
                "avg_cost": avg_cost,
            }
        return out

    def _normalize_trades(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            side = str(item.get("side") or "").lower().strip()
            if not is_valid_symbol(symbol):
                continue
            if side not in _SUPPORTED_TRADE_SIDES:
                continue
            qty = self._to_positive_float(item.get("quantity"), fallback=-1)
            price = self._to_positive_float(item.get("price"), fallback=-1)
            if qty <= 0 or price <= 0:
                continue
            out.append(
                {
                    "timestamp": str(item.get("timestamp") or utc_now_iso()),
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "price": price,
                    "realized_pnl": self._to_non_negative_or_negative_float(item.get("realized_pnl")),
                    "cash_after": self._to_non_negative_float(item.get("cash_after"), fallback=0.0),
                }
            )
        return out[-_MAX_STORED_TRADES:]

    @staticmethod
    def _to_non_negative_or_negative_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed != parsed:  # NaN guard
            return None
        return parsed

    def _write_no_lock(self) -> None:
        payload = self._snapshot_state_no_lock()
        payload["trades"] = payload["trades"][-_MAX_STORED_TRADES:]
        try:
            write_json_file(self.cache_path, payload)
        except Exception as exc:
            LOGGER.warning("Failed to write paper portfolio cache: %s", exc)

    async def get_state(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_state_no_lock()

    async def apply_trade(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        normalized_symbol, normalized_side, qty, px = _validate_trade_request(symbol, side, quantity, price)

        async with self._lock:
            cash = float(self._state["cash"])
            positions = self._state["positions"]
            cash, realized_pnl = _apply_trade_to_portfolio_state(
                cash=cash,
                positions=positions,
                symbol=normalized_symbol,
                side=normalized_side,
                quantity=qty,
                price=px,
            )

            timestamp = utc_now_iso()
            trade = {
                "timestamp": timestamp,
                "symbol": normalized_symbol,
                "side": normalized_side,
                "quantity": qty,
                "price": px,
                "realized_pnl": realized_pnl,
                "cash_after": cash,
            }
            trades = self._state["trades"]
            trades.append(trade)
            if len(trades) > _MAX_STORED_TRADES:
                del trades[:-_MAX_STORED_TRADES]

            self._state["cash"] = cash
            self._state["updated_at"] = timestamp
            self._write_no_lock()
            return dict(trade)

    async def reset(self, initial_cash: float | None = None) -> dict[str, Any]:
        async with self._lock:
            next_initial_cash = (
                self._to_positive_float(initial_cash, fallback=self.default_initial_cash)
                if initial_cash is not None
                else self.default_initial_cash
            )
            self._state = self._empty_state(next_initial_cash)
            self._write_no_lock()
            return self._snapshot_state_no_lock()
