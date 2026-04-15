from __future__ import annotations

import asyncio
import json
import unittest

from app.api.portfolio import paper_portfolio_reset, paper_trade
from fastapi import HTTPException
from app.models import PaperPortfolioResetRequest, PaperTradeRequest
from app.stores.paper_portfolio_engine import apply_trade_to_portfolio_state, validate_trade_request
from app.utils import utc_now_iso


class _FakeHub:
    def __init__(self) -> None:
        self.full_daily_history_store = object()
        self.rows_by_symbol: dict[str, dict[str, object]] = {
            "AAPL": {"symbol": "AAPL", "price": 125.0},
            "MSFT": {"symbol": "MSFT", "price": None},
            "7203.T": {"symbol": "7203.T", "price": None},
        }
        self.current_rows_calls: list[list[str]] = []
        self.overview_calls: list[dict[str, object]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def current_rows(self, symbols: list[str]):
        self.current_rows_calls.append(list(symbols))
        return [dict(self.rows_by_symbol[symbol]) for symbol in symbols if symbol in self.rows_by_symbol]

    async def security_overview_payload(
        self,
        *,
        symbol: str,
        refresh: bool = False,
        include_intraday: bool = True,
        include_market: bool = True,
        include_qqq: bool = True,
    ) -> dict[str, object]:
        self.overview_calls.append(
            {
                "symbol": symbol,
                "refresh": refresh,
                "include_intraday": include_intraday,
                "include_market": include_market,
                "include_qqq": include_qqq,
            }
        )
        if symbol == "7203.T":
            return {
                "symbol": symbol,
                "price": {"current": 500.0},
            }
        return {"symbol": symbol, "price": {"current": None}}


class _FakePaperPortfolioStore:
    def __init__(self, initial_cash: float = 1_000.0) -> None:
        self.default_initial_cash = initial_cash
        self.state = {
            "initial_cash": initial_cash,
            "cash": initial_cash,
            "positions": {},
            "trades": [],
            "updated_at": utc_now_iso(),
        }

    async def get_state(self):
        return {
            "initial_cash": self.state["initial_cash"],
            "cash": self.state["cash"],
            "positions": {symbol: dict(item) for symbol, item in self.state["positions"].items()},
            "trades": [dict(item) for item in self.state["trades"]],
            "updated_at": self.state["updated_at"],
        }

    async def apply_trade(self, symbol: str, side: str, quantity: float, price: float):
        normalized_symbol, normalized_side, qty, px = validate_trade_request(symbol, side, quantity, price)
        cash, realized_pnl = apply_trade_to_portfolio_state(
            cash=float(self.state["cash"]),
            positions=self.state["positions"],
            symbol=normalized_symbol,
            side=normalized_side,
            quantity=qty,
            price=px,
        )
        trade = {
            "timestamp": utc_now_iso(),
            "symbol": normalized_symbol,
            "side": normalized_side,
            "quantity": qty,
            "price": px,
            "realized_pnl": realized_pnl,
            "cash_after": cash,
        }
        self.state["cash"] = cash
        self.state["trades"].append(trade)
        self.state["updated_at"] = trade["timestamp"]
        return dict(trade)

    async def reset(self, initial_cash: float | None = None):
        base_cash = self.default_initial_cash if initial_cash is None else float(initial_cash)
        self.state = {
            "initial_cash": base_cash,
            "cash": base_cash,
            "positions": {},
            "trades": [],
            "updated_at": utc_now_iso(),
        }
        return await self.get_state()


class PaperPortfolioApiTest(unittest.TestCase):
    def _build_dependencies(self) -> tuple[_FakeHub, _FakePaperPortfolioStore]:
        return _FakeHub(), _FakePaperPortfolioStore(initial_cash=1_000.0)

    def test_trade_endpoint_uses_market_price_and_returns_updated_payload(self) -> None:
        hub, store = self._build_dependencies()

        response = asyncio.run(
            paper_trade(
                PaperTradeRequest(symbol="aapl", side="BUY", quantity=2),
                hub=hub,
                paper_portfolio_store=store,
            )
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["trade"]["symbol"], "AAPL")
        self.assertEqual(payload["trade"]["execution_source"], "market")
        self.assertEqual(payload["cash"], 750.0)
        self.assertEqual(payload["market_value"], 250.0)
        self.assertEqual(payload["equity"], 1_000.0)
        self.assertEqual(payload["trade_count"], 1)
        self.assertEqual(len(payload["positions"]), 1)
        self.assertEqual(payload["positions"][0]["symbol"], "AAPL")
        self.assertEqual(payload["positions"][0]["weight"], 100.0)
        self.assertEqual(hub.current_rows_calls[0], ["AAPL"])
        self.assertEqual(hub.current_rows_calls[1], ["AAPL"])

    def test_trade_endpoint_rejects_missing_market_price(self) -> None:
        hub, store = self._build_dependencies()

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                paper_trade(
                    PaperTradeRequest(symbol="MSFT", side="buy", quantity=1),
                    hub=hub,
                    paper_portfolio_store=store,
                )
            )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(exc.exception.detail, "Current market price is unavailable. Set price manually.")

    def test_trade_endpoint_falls_back_to_daily_overview_price(self) -> None:
        hub, store = self._build_dependencies()

        response = asyncio.run(
            paper_trade(
                PaperTradeRequest(symbol="7203.T", side="buy", quantity=1),
                hub=hub,
                paper_portfolio_store=store,
            )
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["trade"]["symbol"], "7203.T")
        self.assertEqual(payload["trade"]["execution_source"], "daily")
        self.assertEqual(payload["cash"], 500.0)
        self.assertEqual(hub.current_rows_calls[0], ["7203.T"])
        self.assertEqual(
            hub.overview_calls[-1],
            {
                "symbol": "7203.T",
                "refresh": False,
                "include_intraday": False,
                "include_market": False,
                "include_qqq": False,
            },
        )

    def test_reset_endpoint_clears_positions_and_uses_explicit_initial_cash(self) -> None:
        hub, store = self._build_dependencies()

        trade_response = asyncio.run(
            paper_trade(
                PaperTradeRequest(symbol="AAPL", side="buy", quantity=1, price=100.0),
                hub=hub,
                paper_portfolio_store=store,
            )
        )
        reset_response = asyncio.run(
            paper_portfolio_reset(
                PaperPortfolioResetRequest(initial_cash=5_000.0),
                hub=hub,
                paper_portfolio_store=store,
            )
        )

        self.assertEqual(trade_response.status_code, 200)
        self.assertEqual(reset_response.status_code, 200)
        payload = json.loads(reset_response.body)
        self.assertEqual(payload["initial_cash"], 5_000.0)
        self.assertEqual(payload["cash"], 5_000.0)
        self.assertEqual(payload["positions"], [])
        self.assertEqual(payload["recent_trades"], [])
        self.assertEqual(payload["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()
