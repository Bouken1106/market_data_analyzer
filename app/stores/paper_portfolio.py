"""Persistent store for paper-trading portfolio state."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import LOGGER, SYMBOL_PATTERN


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
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_from_disk(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return self._empty_state()
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()

        if not isinstance(payload, dict):
            return self._empty_state()

        initial_cash = self._to_positive_float(payload.get("initial_cash"), fallback=self.default_initial_cash)
        cash = self._to_non_negative_float(payload.get("cash"), fallback=initial_cash)
        positions = self._normalize_positions(payload.get("positions"))
        trades = self._normalize_trades(payload.get("trades"))
        updated_at = str(payload.get("updated_at") or datetime.now(timezone.utc).isoformat())
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
            symbol = str(symbol_raw or "").upper().strip()
            if not symbol or not SYMBOL_PATTERN.match(symbol):
                continue
            if not isinstance(item, dict):
                continue
            quantity = self._to_non_negative_float(item.get("quantity"), fallback=0.0)
            avg_cost = self._to_non_negative_float(item.get("avg_cost"), fallback=0.0)
            if quantity <= 0:
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
            symbol = str(item.get("symbol") or "").upper().strip()
            side = str(item.get("side") or "").lower().strip()
            if not symbol or not SYMBOL_PATTERN.match(symbol):
                continue
            if side not in {"buy", "sell"}:
                continue
            qty = self._to_positive_float(item.get("quantity"), fallback=-1)
            price = self._to_positive_float(item.get("price"), fallback=-1)
            if qty <= 0 or price <= 0:
                continue
            out.append(
                {
                    "timestamp": str(item.get("timestamp") or datetime.now(timezone.utc).isoformat()),
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
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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
        normalized_symbol = str(symbol or "").upper().strip()
        normalized_side = str(side or "").lower().strip()
        qty = float(quantity)
        px = float(price)

        if not normalized_symbol or not SYMBOL_PATTERN.match(normalized_symbol):
            raise ValueError("Invalid symbol format.")
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell.")
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
            else:
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

            timestamp = datetime.now(timezone.utc).isoformat()
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
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_no_lock()
            return {
                "initial_cash": float(self._state["initial_cash"]),
                "cash": float(self._state["cash"]),
                "positions": {},
                "trades": [],
                "updated_at": str(self._state["updated_at"]),
            }
