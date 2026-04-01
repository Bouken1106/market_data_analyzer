from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.stores.paper_portfolio import PaperPortfolioStore


class PaperPortfolioStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _cache_path(self) -> Path:
        return Path(self._tmpdir.name) / "paper_portfolio.json"

    def test_load_from_disk_normalizes_symbols_positions_and_trades(self) -> None:
        cache_path = self._cache_path()
        cache_path.write_text(
            json.dumps(
                {
                    "initial_cash": -1,
                    "cash": -10,
                    "positions": {
                        " aapl ": {"quantity": "2", "avg_cost": "10"},
                        "NVDA": {"quantity": "-3", "avg_cost": 50},
                        "bad symbol!": {"quantity": 1, "avg_cost": 10},
                        "MSFT": {"quantity": 0, "avg_cost": 20},
                        "TSLA": {"quantity": "nan", "avg_cost": 100},
                    },
                    "trades": [
                        {"symbol": " aapl ", "side": "BUY", "quantity": "1", "price": "10", "cash_after": "990"},
                        {"symbol": "bad symbol!", "side": "buy", "quantity": 1, "price": 10},
                        {"symbol": "MSFT", "side": "hold", "quantity": 1, "price": 10},
                        {"symbol": "NVDA", "side": "short", "quantity": 0, "price": 10},
                    ],
                }
            ),
            encoding="utf-8",
        )

        store = PaperPortfolioStore(cache_path=cache_path, default_initial_cash=1_000.0)
        state = asyncio.run(store.get_state())

        self.assertEqual(state["initial_cash"], 1_000.0)
        self.assertEqual(state["cash"], 1_000.0)
        self.assertEqual(sorted(state["positions"].keys()), ["AAPL", "NVDA"])
        self.assertEqual(state["positions"]["AAPL"]["quantity"], 2.0)
        self.assertEqual(state["positions"]["NVDA"]["quantity"], -3.0)
        self.assertEqual(len(state["trades"]), 1)
        self.assertEqual(state["trades"][0]["symbol"], "AAPL")
        self.assertEqual(state["trades"][0]["side"], "buy")

    def test_apply_trade_and_reset_persist_state_to_disk(self) -> None:
        cache_path = self._cache_path()
        store = PaperPortfolioStore(cache_path=cache_path, default_initial_cash=1_000.0)

        buy_trade = asyncio.run(store.apply_trade(symbol="aapl", side="buy", quantity=2, price=100))
        sell_trade = asyncio.run(store.apply_trade(symbol="AAPL", side="sell", quantity=1, price=110))
        state_after_sell = asyncio.run(store.get_state())
        reset_state = asyncio.run(store.reset(initial_cash=500.0))
        persisted = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(buy_trade["cash_after"], 800.0)
        self.assertEqual(sell_trade["realized_pnl"], 10.0)
        self.assertEqual(state_after_sell["cash"], 910.0)
        self.assertEqual(state_after_sell["positions"]["AAPL"]["quantity"], 1.0)
        self.assertEqual(reset_state["initial_cash"], 500.0)
        self.assertEqual(reset_state["cash"], 500.0)
        self.assertEqual(persisted["positions"], {})
        self.assertEqual(persisted["trades"], [])


if __name__ == "__main__":
    unittest.main()
