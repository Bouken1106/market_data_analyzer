from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.application import create_app
from app.bootstrap import AppServices
from app.stores.portfolio_analysis import PortfolioAnalysisStore


class _FakeHub:
    def __init__(self) -> None:
        self.price_map = {
            "7203.T": 3_100.0,
            "9432.T": 152.0,
            "6758.T": 12_800.0,
            "AAPL": 190.0,
            "MSFT": 412.0,
        }

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def current_rows(self, symbols: list[str]) -> list[dict[str, object]]:
        return [
            {"symbol": symbol, "price": self.price_map[symbol]}
            for symbol in symbols
            if symbol in self.price_map
        ]

    async def historical_payload(
        self,
        symbol: str,
        years: int = 1,
        months: int | None = None,
        refresh: bool = False,
        **kwargs,
    ) -> dict[str, object]:
        del years, months, refresh, kwargs
        base = self.price_map.get(symbol, 100.0)
        points: list[dict[str, object]] = []
        start = date(2025, 1, 1)
        for offset in range(190):
            close = base * (1.0 + (offset * 0.0008) + (((offset % 7) - 3) * 0.0025))
            points.append(
                {
                    "t": (start + timedelta(days=offset)).isoformat(),
                    "c": round(close, 4),
                }
            )
        return {"symbol": symbol, "points": points}


class _FakeSymbolCatalogStore:
    async def get_catalog(
        self,
        refresh: bool = False,
        cache_only: bool = False,
        *,
        country: str | None = None,
    ) -> dict[str, object]:
        del refresh, cache_only
        normalized_country = str(country or "").strip().lower()
        if normalized_country == "japan":
            return {
                "symbols": [
                    {"symbol": "9432.T", "name": "Nippon Telegraph and Telephone Corporation", "exchange": "JPX"},
                    {"symbol": "7203.T", "name": "Toyota Motor Corporation", "exchange": "JPX"},
                    {"symbol": "4419.T", "name": "Finatext Holdings Ltd.", "exchange": "JPX"},
                    {"symbol": "3850.T", "name": "NTT DATA INTRAMART CORPORATION", "exchange": "JPX"},
                ]
            }
        return {
            "symbols": [
                {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
            ]
        }


class _StaleHistoricalHub:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def current_rows(self, symbols: list[str]) -> list[dict[str, object]]:
        del symbols
        return []

    async def historical_payload(
        self,
        symbol: str,
        years: int = 1,
        months: int | None = None,
        refresh: bool = False,
        **kwargs,
    ) -> dict[str, object]:
        del years, months, refresh, kwargs
        if symbol != "4419.T":
            return {"symbol": symbol, "points": []}
        points: list[dict[str, object]] = []
        start = date(2025, 7, 1)
        for offset in range(150):
            current = start + timedelta(days=offset)
            close = 1000.0 + (offset * 0.3)
            points.append({"t": current.isoformat(), "c": round(close, 4)})
        return {"symbol": symbol, "points": points}


class PortfolioAnalysisApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = PortfolioAnalysisStore(cache_path=Path(self._tmpdir.name) / "saved_portfolios.json")
        self.app = create_app(
            AppServices(
                hub=_FakeHub(),
                symbol_catalog_store=_FakeSymbolCatalogStore(),
                paper_portfolio_store=object(),
                ui_state_store=object(),
                portfolio_analysis_store=self.store,
            )
        )

    def test_save_list_analyze_and_delete_portfolio(self) -> None:
        with TestClient(self.app) as client:
            save_response = client.post(
                "/api/portfolio-analysis/portfolios",
                json={
                    "name": "Income + Growth",
                    "jp_holdings": [{"symbol": "NTT", "quantity": 100}],
                    "us_holdings": [{"symbol": "AAPL", "quantity": 8}, {"symbol": "MSFT", "quantity": 5}],
                },
            )
            portfolio_id = save_response.json()["portfolio"]["portfolio_id"]
            list_response = client.get("/api/portfolio-analysis/portfolios")
            analyze_response = client.post(
                "/api/portfolio-analysis/analyze",
                json={
                    "jp_holdings": [{"symbol": "NTT", "quantity": 100}],
                    "us_holdings": [{"symbol": "AAPL", "quantity": 8}, {"symbol": "MSFT", "quantity": 5}],
                    "lookback_days": 126,
                },
            )
            delete_response = client.delete(f"/api/portfolio-analysis/portfolios/{portfolio_id}")

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(save_response.json()["portfolio"]["jp_holdings"][0]["symbol"], "9432.T")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["portfolios"]), 1)
        self.assertEqual(analyze_response.status_code, 200)
        analysis_payload = analyze_response.json()
        self.assertEqual(analysis_payload["portfolio"]["jp_holdings"][0]["symbol"], "9432.T")
        self.assertEqual(analysis_payload["regions"]["jp"]["summary"]["holdings_count"], 1)
        self.assertEqual(analysis_payload["regions"]["us"]["summary"]["holdings_count"], 2)
        self.assertIsNotNone(analysis_payload["regions"]["jp"]["risk"]["annualized_volatility_pct"])
        self.assertIsNotNone(analysis_payload["regions"]["us"]["risk"]["value_at_risk_95_amount"])
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["portfolios"], [])

    def test_portfolio_draft_roundtrip(self) -> None:
        with TestClient(self.app) as client:
            initial_response = client.get("/api/portfolio-analysis/draft")
            save_response = client.post(
                "/api/portfolio-analysis/draft",
                json={
                    "portfolio_id": "draft-one",
                    "name": " Draft Alpha ",
                    "lookback_days": 126,
                    "jp_rows": [{"symbol": "NTT", "quantity": "100"}],
                    "us_rows": [{"symbol": "AAPL", "quantity": "5"}],
                },
            )
            reload_response = client.get("/api/portfolio-analysis/draft")

        self.assertEqual(initial_response.status_code, 200)
        self.assertIsNone(initial_response.json()["draft"])
        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(save_response.json()["draft"]["portfolio_id"], "draft-one")
        self.assertEqual(save_response.json()["draft"]["name"], "Draft Alpha")
        self.assertEqual(save_response.json()["draft"]["jp_rows"][0]["symbol"], "NTT")
        self.assertEqual(reload_response.status_code, 200)
        self.assertEqual(reload_response.json()["draft"]["lookback_days"], 126)

    def test_analyze_endpoint_rejects_invalid_symbol(self) -> None:
        with TestClient(self.app) as client:
            response = client.post(
                "/api/portfolio-analysis/analyze",
                json={
                    "jp_holdings": [{"symbol": "bad symbol!", "quantity": 100}],
                    "us_holdings": [],
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid symbol format", response.json()["detail"])

    def test_analyze_endpoint_omits_stale_historical_close_from_last_price(self) -> None:
        app = create_app(
            AppServices(
                hub=_StaleHistoricalHub(),
                symbol_catalog_store=_FakeSymbolCatalogStore(),
                paper_portfolio_store=object(),
                ui_state_store=object(),
                portfolio_analysis_store=self.store,
            )
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/portfolio-analysis/analyze",
                json={
                    "jp_holdings": [{"symbol": "4419", "quantity": 100}],
                    "us_holdings": [],
                    "lookback_days": 126,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        holding = payload["regions"]["jp"]["holdings"][0]
        self.assertEqual(holding["symbol"], "4419.T")
        self.assertIsNone(holding["last_price"])
        self.assertEqual(holding["last_price_source"], "stale_historical_close")
        self.assertIsNone(holding["market_value"])
        self.assertTrue(payload["regions"]["jp"]["warnings"])
        self.assertIn("stale", payload["regions"]["jp"]["warnings"][0].lower())


if __name__ == "__main__":
    unittest.main()
