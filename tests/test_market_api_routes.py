from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.market_stream import build_market_stream_response
from app.application import create_app
from app.bootstrap import AppServices


class _FakeHub:
    def __init__(self) -> None:
        self.provider = "both"
        self.full_daily_history_store = object()
        self.symbols = ["AAPL", "MSFT"]
        self.listeners: list[asyncio.Queue] = []
        self.set_symbols_calls: list[list[str]] = []
        self.current_rows_calls: list[list[str]] = []
        self.historical_calls: list[dict[str, object]] = []
        self.overview_calls: list[dict[str, object]] = []
        self.sparkline_calls: list[dict[str, object]] = []
        self.refresh_api_credits_calls = 0
        self.rows_by_symbol = {
            "AAPL": {"symbol": "AAPL", "price": 123.45, "updated_at": "2026-04-03T00:00:00Z"},
            "MSFT": {"symbol": "MSFT", "price": 234.56, "updated_at": "2026-04-03T00:01:00Z"},
        }

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def snapshot_payload(self) -> dict[str, object]:
        return {
            "rows": [self.rows_by_symbol["AAPL"], self.rows_by_symbol["MSFT"]],
            "status": {"mode": "rest-fallback"},
        }

    async def set_symbols(self, symbols: list[str]) -> None:
        self.set_symbols_calls.append(list(symbols))
        self.symbols = list(symbols)

    async def current_rows(self, symbols: list[str]) -> list[dict[str, object]]:
        self.current_rows_calls.append(list(symbols))
        return [dict(self.rows_by_symbol[symbol]) for symbol in symbols if symbol in self.rows_by_symbol]

    async def status_payload(self) -> dict[str, object]:
        return {"mode": "rest-fallback", "symbols": list(self.symbols)}

    async def refresh_api_credits(self) -> dict[str, object]:
        self.refresh_api_credits_calls += 1
        return {"daily_credits_left": 321, "source": "api_usage"}

    async def historical_payload(
        self,
        symbol: str,
        years: int = 5,
        refresh: bool = False,
        **kwargs,
    ) -> dict[str, object]:
        self.historical_calls.append(
            {
                "symbol": symbol,
                "years": years,
                "refresh": refresh,
                **kwargs,
            }
        )
        return {
            "symbol": symbol,
            "points": [
                {"t": "2026-04-01", "o": 120.0, "c": 121.0},
                {"t": "2026-04-02", "o": 121.0, "c": 123.45},
            ],
        }

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
        return {
            "symbol": symbol,
            "price": {"current": 123.45},
            "market": {"spy": {"symbol": "SPY"}} if include_market else {},
        }

    async def sparkline_payload(self, symbols: list[str], *, refresh: bool = False) -> list[dict[str, object]]:
        self.sparkline_calls.append({"symbols": list(symbols), "refresh": refresh})
        return [
            {
                "symbol": symbol,
                "points": [{"t": "2026-04-01", "c": 1.0}, {"t": "2026-04-02", "c": 2.0}],
                "latest_close": float(index + 1) * 100.0,
            }
            for index, symbol in enumerate(symbols)
        ]

    def register_listener(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.listeners.append(queue)
        return queue

    def unregister_listener(self, queue: asyncio.Queue) -> None:
        if queue in self.listeners:
            self.listeners.remove(queue)


class _FakeSymbolCatalogStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get_catalog(self, *, refresh: bool = False, cache_only: bool = False) -> dict[str, object]:
        self.calls.append({"refresh": refresh, "cache_only": cache_only})
        return {
            "items": [{"symbol": "AAPL", "name": "Apple Inc."}],
            "count": 1,
            "source": "cache",
        }


class _FakeUiStateStore:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def get_watchlist_commentary(self) -> dict[str, object] | None:
        return self.payload

    def set_watchlist_commentary(self, payload: dict[str, object]) -> None:
        self.payload = dict(payload)


class _FakePaperPortfolioStore:
    async def get_state(self) -> dict[str, object]:
        return {
            "initial_cash": 1000.0,
            "cash": 1000.0,
            "positions": {},
            "trades": [],
            "updated_at": "2026-04-03T00:00:00Z",
        }


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return True


class MarketApiRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hub = _FakeHub()
        self.symbol_catalog_store = _FakeSymbolCatalogStore()
        self.ui_state_store = _FakeUiStateStore()
        self.app = create_app(
            AppServices(
                hub=self.hub,
                symbol_catalog_store=self.symbol_catalog_store,
                paper_portfolio_store=_FakePaperPortfolioStore(),
                ui_state_store=self.ui_state_store,
            )
        )

    def test_snapshot_symbols_credits_and_catalog_routes(self) -> None:
        with TestClient(self.app) as client:
            snapshot_response = client.get("/api/snapshot")
            symbols_response = client.post("/api/symbols", json={"symbols": " aapl, msft "})
            credits_response = client.get("/api/credits")
            refreshed_credits_response = client.get("/api/credits?refresh=true")
            catalog_response = client.get("/api/symbol-catalog?refresh=true&cache_only=true")

        self.assertEqual(snapshot_response.status_code, 200)
        self.assertEqual(len(snapshot_response.json()["rows"]), 2)
        self.assertEqual(snapshot_response.json()["status"]["mode"], "rest-fallback")

        self.assertEqual(symbols_response.status_code, 200)
        self.assertEqual(symbols_response.json()["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(self.hub.set_symbols_calls[-1], ["AAPL", "MSFT"])
        self.assertEqual(self.hub.current_rows_calls[-1], ["AAPL", "MSFT"])

        self.assertEqual(credits_response.status_code, 200)
        self.assertIn("api_usage", credits_response.json()["note"])
        self.assertEqual(refreshed_credits_response.status_code, 200)
        self.assertEqual(refreshed_credits_response.json()["status"]["daily_credits_left"], 321)
        self.assertEqual(self.hub.refresh_api_credits_calls, 1)

        self.assertEqual(catalog_response.status_code, 200)
        self.assertEqual(catalog_response.json()["count"], 1)
        self.assertEqual(
            self.symbol_catalog_store.calls[-1],
            {"refresh": True, "cache_only": True},
        )

    def test_historical_overview_sparkline_watchlist_and_stream_routes(self) -> None:
        commentary_payload = {
            "symbols": ["AAPL", "MSFT"],
            "current_date": "2026-04-03",
            "generated_at": "2026-04-03T00:00:00Z",
            "model": "test-model",
            "comment": "AAPL: strong.\nMSFT: steady.",
            "metrics": [],
            "prompt": "prompt",
        }

        with patch(
            "app.api.market.build_watchlist_commentary_payload",
            new=AsyncMock(return_value=commentary_payload),
        ) as build_payload_mock:
            with TestClient(self.app) as client:
                historical_response = client.get("/api/historical/AAPL?years=2&refresh=true")
                overview_response = client.get(
                    "/api/security-overview/AAPL?refresh=true&include_intraday=false&include_market=false&include_qqq=false"
                )
                sparkline_response = client.get("/api/sparkline?symbols=AAPL,MSFT&refresh=true")
                commentary_response = client.get("/api/watchlist-commentary?symbols=AAPL,MSFT&refresh=true")
                portfolio_response = client.get("/api/portfolio")

        self.assertEqual(historical_response.status_code, 200)
        self.assertEqual(historical_response.json()["symbol"], "AAPL")
        self.assertEqual(
            self.hub.historical_calls[-1],
            {"symbol": "AAPL", "years": 2, "refresh": True},
        )

        self.assertEqual(overview_response.status_code, 200)
        self.assertEqual(overview_response.json()["price"]["current"], 123.45)
        self.assertEqual(
            self.hub.overview_calls[-1],
            {
                "symbol": "AAPL",
                "refresh": True,
                "include_intraday": False,
                "include_market": False,
                "include_qqq": False,
            },
        )

        self.assertEqual(sparkline_response.status_code, 200)
        self.assertEqual(sparkline_response.json()["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(
            self.hub.sparkline_calls[-1],
            {"symbols": ["AAPL", "MSFT"], "refresh": True},
        )

        self.assertEqual(commentary_response.status_code, 200)
        self.assertEqual(commentary_response.json()["comment"], commentary_payload["comment"])
        self.assertEqual(self.ui_state_store.payload, commentary_payload)
        build_payload_mock.assert_awaited_once_with(self.hub, ["AAPL", "MSFT"], refresh=True)

        self.assertEqual(portfolio_response.status_code, 200)
        self.assertEqual(portfolio_response.json()["equity"], 1000.0)

    def test_market_stream_response_emits_initial_snapshot_and_unregisters_listener(self) -> None:
        async def run_test() -> None:
            request = _FakeRequest()
            response = build_market_stream_response(request, self.hub)
            self.assertEqual(response.media_type, "text/event-stream")
            first_chunk = await anext(response.body_iterator)
            self.assertTrue(first_chunk.startswith("data: "))
            self.assertIn('"status": {"mode": "rest-fallback"}', first_chunk)

            with self.assertRaises(StopAsyncIteration):
                await anext(response.body_iterator)

        asyncio.run(run_test())
        self.assertEqual(self.hub.listeners, [])


if __name__ == "__main__":
    unittest.main()
