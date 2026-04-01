"""Persistent store for paper-trading portfolio state."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import is_valid_symbol, normalize_symbol, read_json_file, utc_now_iso, write_json_file
from .paper_portfolio_engine import (
    SUPPORTED_TRADE_SIDES,
    apply_trade_to_portfolio_state,
    validate_trade_request,
)

_MAX_STORED_TRADES = 1000


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
            if side not in SUPPORTED_TRADE_SIDES:
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
        normalized_symbol, normalized_side, qty, px = validate_trade_request(symbol, side, quantity, price)

        async with self._lock:
            cash = float(self._state["cash"])
            positions = self._state["positions"]
            cash, realized_pnl = apply_trade_to_portfolio_state(
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
