"""Persistent store for paper-trading portfolio state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, utc_now_iso, write_json_file


class PaperPortfolioStore:
    def __init__(self, cache_path: Path, default_initial_cash: float = 1_000_000.0) -> None:
        self.cache_path = cache_path
        self.default_initial_cash = float(default_initial_cash)
        self._lock = asyncio.Lock()
        self._state = self._load_from_disk()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "initial_cash": self.default_initial_cash,
            "cash": self.default_initial_cash,
            "positions": {},
            "trades": [],
            "updated_at": utc_now_iso(),
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
            if side not in {"buy", "sell", "short", "cover"}:
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
        return out[-1000:]

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
        payload = {
            "initial_cash": self._state["initial_cash"],
            "cash": self._state["cash"],
            "positions": self._state["positions"],
            "trades": self._state["trades"][-1000:],
            "updated_at": self._state["updated_at"],
        }
        try:
            write_json_file(self.cache_path, payload)
        except Exception as exc:
            LOGGER.warning("Failed to write paper portfolio cache: %s", exc)

    async def get_state(self) -> dict[str, Any]:
        async with self._lock:
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

    async def apply_trade(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        normalized_symbol = normalize_symbol(symbol)
        normalized_side = str(side or "").lower().strip()
        qty = float(quantity)
        px = float(price)

        if not is_valid_symbol(normalized_symbol):
            raise ValueError("Invalid symbol format.")
        if normalized_side not in {"buy", "sell", "short", "cover"}:
            raise ValueError("side must be buy, sell, short, or cover.")
        if qty <= 0:
            raise ValueError("quantity must be greater than 0.")
        if px <= 0:
            raise ValueError("price must be greater than 0.")

        async with self._lock:
            cash = float(self._state["cash"])
            positions = self._state["positions"]
            current = positions.get(normalized_symbol, {"quantity": 0.0, "avg_cost": 0.0})
            current_qty = float(current.get("quantity") or 0.0)
            current_avg_cost = float(current.get("avg_cost") or 0.0)

            realized_pnl: float | None = None
            if normalized_side == "buy":
                if current_qty < -1e-9:
                    raise ValueError("Cannot buy while short position exists. Use cover.")
                total_cost = qty * px
                if cash + 1e-9 < total_cost:
                    raise ValueError("Insufficient cash balance.")
                new_qty = current_qty + qty
                if new_qty <= 0:
                    raise ValueError("Invalid resulting quantity.")
                new_avg_cost = ((current_qty * current_avg_cost) + total_cost) / new_qty
                cash -= total_cost
                positions[normalized_symbol] = {
                    "quantity": new_qty,
                    "avg_cost": new_avg_cost,
                }
            elif normalized_side == "sell":
                if current_qty < -1e-9:
                    raise ValueError("Cannot sell while short position exists. Use cover or short.")
                if current_qty + 1e-9 < qty:
                    raise ValueError("Sell quantity exceeds current position.")
                proceeds = qty * px
                realized_pnl = (px - current_avg_cost) * qty
                remaining_qty = current_qty - qty
                cash += proceeds
                if remaining_qty <= 1e-9:
                    positions.pop(normalized_symbol, None)
                else:
                    positions[normalized_symbol] = {
                        "quantity": remaining_qty,
                        "avg_cost": current_avg_cost,
                    }
            elif normalized_side == "short":
                if current_qty > 1e-9:
                    raise ValueError("Cannot short while long position exists. Use sell.")
                proceeds = qty * px
                short_size = abs(current_qty)
                new_short_size = short_size + qty
                if new_short_size <= 0:
                    raise ValueError("Invalid resulting short quantity.")
                new_avg_cost = ((short_size * current_avg_cost) + (qty * px)) / new_short_size
                cash += proceeds
                positions[normalized_symbol] = {
                    "quantity": -new_short_size,
                    "avg_cost": new_avg_cost,
                }
            else:  # cover
                if current_qty >= -1e-9:
                    raise ValueError("No short position to cover.")
                short_size = abs(current_qty)
                if short_size + 1e-9 < qty:
                    raise ValueError("Cover quantity exceeds current short position.")
                total_cost = qty * px
                if cash + 1e-9 < total_cost:
                    raise ValueError("Insufficient cash balance.")
                realized_pnl = (current_avg_cost - px) * qty
                remaining_short = short_size - qty
                cash -= total_cost
                if remaining_short <= 1e-9:
                    positions.pop(normalized_symbol, None)
                else:
                    positions[normalized_symbol] = {
                        "quantity": -remaining_short,
                        "avg_cost": current_avg_cost,
                    }

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
            if len(trades) > 1000:
                del trades[:-1000]

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
            self._state = {
                "initial_cash": next_initial_cash,
                "cash": next_initial_cash,
                "positions": {},
                "trades": [],
                "updated_at": utc_now_iso(),
            }
            self._write_no_lock()
            return {
                "initial_cash": float(self._state["initial_cash"]),
                "cash": float(self._state["cash"]),
                "positions": {},
                "trades": [],
                "updated_at": str(self._state["updated_at"]),
            }
