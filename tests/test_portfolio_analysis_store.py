from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.stores.portfolio_analysis import PortfolioAnalysisStore


class PortfolioAnalysisStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _cache_path(self) -> Path:
        return Path(self._tmpdir.name) / "saved_portfolios.json"

    def test_save_update_reload_and_delete_portfolio(self) -> None:
        store = PortfolioAnalysisStore(cache_path=self._cache_path())

        created = asyncio.run(
            store.save_portfolio(
                portfolio_id=None,
                name=" Core Allocation ",
                jp_holdings=[{"symbol": "7203", "quantity": 100}, {"symbol": "7203.T", "quantity": 50}],
                us_holdings=[{"symbol": "aapl", "quantity": 10}],
            )
        )
        updated = asyncio.run(
            store.save_portfolio(
                portfolio_id=created["portfolio_id"],
                name="Core Allocation v2",
                jp_holdings=[{"symbol": "6758", "quantity": 30}],
                us_holdings=[{"symbol": "AAPL", "quantity": 10}, {"symbol": "MSFT", "quantity": 8}],
            )
        )

        reloaded = PortfolioAnalysisStore(cache_path=self._cache_path())
        portfolios = asyncio.run(reloaded.list_portfolios())
        deleted = asyncio.run(reloaded.delete_portfolio(created["portfolio_id"]))
        persisted = json.loads(self._cache_path().read_text(encoding="utf-8"))

        self.assertEqual(created["name"], "Core Allocation")
        self.assertEqual(created["jp_holdings"][0]["symbol"], "7203.T")
        self.assertEqual(created["jp_holdings"][0]["quantity"], 150.0)
        self.assertEqual(updated["name"], "Core Allocation v2")
        self.assertEqual(len(portfolios), 1)
        self.assertEqual(portfolios[0]["jp_holdings"][0]["symbol"], "6758.T")
        self.assertEqual(len(portfolios[0]["us_holdings"]), 2)
        self.assertTrue(deleted)
        self.assertEqual(persisted["portfolios"], [])

    def test_save_reload_and_clear_draft(self) -> None:
        store = PortfolioAnalysisStore(cache_path=self._cache_path())

        saved_draft = asyncio.run(
            store.save_draft(
                portfolio_id="draft-portfolio",
                name=" Draft   Portfolio ",
                lookback_days=126,
                jp_rows=[{"symbol": "NTT", "quantity": "100"}, {"symbol": "", "quantity": ""}],
                us_rows=[{"symbol": "Apple", "quantity": "5"}],
            )
        )

        reloaded = PortfolioAnalysisStore(cache_path=self._cache_path())
        loaded_draft = asyncio.run(reloaded.get_draft())
        asyncio.run(reloaded.clear_draft())
        cleared_draft = asyncio.run(reloaded.get_draft())
        persisted = json.loads(self._cache_path().read_text(encoding="utf-8"))

        self.assertEqual(saved_draft["name"], "Draft Portfolio")
        self.assertEqual(saved_draft["lookback_days"], 126)
        self.assertEqual(saved_draft["jp_rows"][0]["symbol"], "NTT")
        self.assertEqual(saved_draft["us_rows"][0]["symbol"], "Apple")
        self.assertEqual(loaded_draft["portfolio_id"], "draft-portfolio")
        self.assertEqual(loaded_draft["lookback_days"], 126)
        self.assertIsNone(cleared_draft)
        self.assertIsNone(persisted["draft"])


if __name__ == "__main__":
    unittest.main()
