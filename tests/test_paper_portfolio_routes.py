from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.application import create_app
from app.bootstrap import AppServices
from app.stores.paper_portfolio_engine import apply_trade_to_portfolio_state, validate_trade_request
from app.utils import utc_now_iso


class _FakeHub:
    def __init__(self) -> None:
        self.full_daily_history_store = object()
        self.rows_by_symbol: dict[str, dict[str, object]] = {
            "AAPL": {"symbol": "AAPL", "price": 125.0},
            "MSFT": {"symbol": "MSFT", "price": None},
        }
        self.current_rows_calls: list[list[str]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def current_rows(self, symbols: list[str]) -> list[dict[str, object]]:
        self.current_rows_calls.append(list(symbols))
        return [dict(self.rows_by_symbol[symbol]) for symbol in symbols if symbol in self.rows_by_symbol]


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

    async def get_state(self) -> dict[str, object]:
        return {
            "initial_cash": self.state["initial_cash"],
            "cash": self.state["cash"],
            "positions": {symbol: dict(item) for symbol, item in self.state["positions"].items()},
            "trades": [dict(item) for item in self.state["trades"]],
            "updated_at": self.state["updated_at"],
        }

    async def apply_trade(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, object]:
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

    async def reset(self, initial_cash: float | None = None) -> dict[str, object]:
        base_cash = self.default_initial_cash if initial_cash is None else float(initial_cash)
        self.state = {
            "initial_cash": base_cash,
            "cash": base_cash,
            "positions": {},
            "trades": [],
            "updated_at": utc_now_iso(),
        }
        return await self.get_state()


class PaperPortfolioRoutesTest(unittest.TestCase):
    def _build_app(self) -> tuple[_FakeHub, _FakePaperPortfolioStore, object]:
        hub = _FakeHub()
        store = _FakePaperPortfolioStore()
        app = create_app(
            AppServices(
                hub=hub,
                symbol_catalog_store=object(),
                paper_portfolio_store=store,
                ui_state_store=object(),
            )
        )
        return hub, store, app

    def test_portfolio_route_returns_aggregated_payload(self) -> None:
        hub, store, app = self._build_app()
        store.state["cash"] = 750.0
        store.state["positions"] = {
            "AAPL": {"quantity": 2.0, "avg_cost": 100.0},
        }

        with TestClient(app) as client:
            response = client.get("/api/portfolio")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["cash"], 750.0)
        self.assertEqual(payload["market_value"], 250.0)
        self.assertEqual(payload["equity"], 1_000.0)
        self.assertEqual(payload["positions"][0]["symbol"], "AAPL")
        self.assertEqual(payload["positions"][0]["weight"], 100.0)
        self.assertEqual(hub.current_rows_calls[-1], ["AAPL"])

    def test_trade_route_uses_market_price_and_returns_updated_state(self) -> None:
        hub, _store, app = self._build_app()

        with TestClient(app) as client:
            response = client.post(
                "/api/portfolio/trades",
                json={"symbol": "aapl", "side": "BUY", "quantity": 2},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["trade"]["symbol"], "AAPL")
        self.assertEqual(payload["trade"]["execution_source"], "market")
        self.assertEqual(payload["cash"], 750.0)
        self.assertEqual(payload["market_value"], 250.0)
        self.assertEqual(payload["equity"], 1_000.0)
        self.assertEqual(payload["trade_count"], 1)
        self.assertEqual(hub.current_rows_calls[0], ["AAPL"])
        self.assertEqual(hub.current_rows_calls[1], ["AAPL"])

    def test_trade_route_rejects_missing_market_price_and_reset_route_clears_state(self) -> None:
        _hub, store, app = self._build_app()

        with TestClient(app) as client:
            trade_response = client.post(
                "/api/portfolio/trades",
                json={"symbol": "MSFT", "side": "buy", "quantity": 1},
            )
            buy_response = client.post(
                "/api/portfolio/trades",
                json={"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.0},
            )
            reset_response = client.post("/api/portfolio/reset", json={"initial_cash": 5_000.0})

        self.assertEqual(trade_response.status_code, 400)
        self.assertEqual(trade_response.json()["detail"], "Current market price is unavailable. Set price manually.")

        self.assertEqual(buy_response.status_code, 200)
        self.assertEqual(reset_response.status_code, 200)
        payload = reset_response.json()
        self.assertEqual(payload["initial_cash"], 5_000.0)
        self.assertEqual(payload["cash"], 5_000.0)
        self.assertEqual(payload["positions"], [])
        self.assertEqual(payload["recent_trades"], [])
        self.assertEqual(payload["trade_count"], 0)
        self.assertEqual(store.state["positions"], {})
        self.assertEqual(store.state["trades"], [])


if __name__ == "__main__":
    unittest.main()
