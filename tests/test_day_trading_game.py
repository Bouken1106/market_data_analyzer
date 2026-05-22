from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from app.application import create_app
from app.bootstrap import AppServices
from app.services.day_trading_game import build_day_trading_session


class _FakeHub:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class DayTradingGameServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_build_session_uses_yfinance_close_as_execution_price(self) -> None:
        index = pd.date_range(
            "2026-04-01 09:30",
            periods=12,
            freq="15min",
            tz="America/New_York",
        )
        frame = pd.DataFrame(
            {
                "Open": [100 + step for step in range(12)],
                "High": [101 + step for step in range(12)],
                "Low": [99 + step for step in range(12)],
                "Close": [100.5 + step for step in range(12)],
                "Volume": [1000 + step for step in range(12)],
            },
            index=index,
        )

        async def fake_fetch(symbol: str) -> pd.DataFrame:
            self.assertEqual(symbol, "AAPL")
            return frame

        payload = await build_day_trading_session(
            market="us",
            symbol="AAPL",
            rng=random.Random(1),
            fetch_history=fake_fetch,
        )

        self.assertEqual(payload["market"], "us")
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["date"], "2026-04-01")
        self.assertEqual(payload["candle_count"], 12)
        self.assertEqual(payload["candles"][0]["execution_price"], 100.5)
        self.assertEqual(payload["candles"][0]["execution_price_method"], "close")

    async def test_build_session_falls_back_to_iqr_midpoint_when_close_is_missing(self) -> None:
        index = pd.date_range(
            "2026-04-02 09:30",
            periods=12,
            freq="15min",
            tz="America/New_York",
        )
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 12,
                "High": [104.0] * 12,
                "Low": [96.0] * 12,
                "Q1": [98.0] * 12,
                "Q3": [102.0] * 12,
            },
            index=index,
        )

        async def fake_fetch(_: str) -> pd.DataFrame:
            return frame

        payload = await build_day_trading_session(
            market="us",
            symbol="AAPL",
            rng=random.Random(1),
            fetch_history=fake_fetch,
        )

        self.assertEqual(payload["candles"][0]["execution_price"], 100.0)
        self.assertEqual(payload["candles"][0]["execution_price_method"], "iqr_midpoint")


class DayTradingGameApiTest(unittest.TestCase):
    def test_day_trading_game_session_route_returns_payload(self) -> None:
        app = create_app(
            AppServices(
                hub=_FakeHub(),
                symbol_catalog_store=object(),
                paper_portfolio_store=object(),
                ui_state_store=object(),
            )
        )
        payload = {
            "game_id": "test",
            "market": "jp",
            "market_label": "Japan",
            "symbol": "7203.T",
            "date": "2026-04-01",
            "timezone": "Asia/Tokyo",
            "currency": "JPY",
            "currency_symbol": "¥",
            "currency_digits": 0,
            "interval": "15m",
            "period": "60d",
            "source": "yfinance",
            "execution_price_rule": "close",
            "candle_count": 1,
            "session_start": "09:00",
            "session_end": "09:00",
            "candles": [],
        }

        with patch(
            "app.api.day_trading_game.build_day_trading_session",
            new=AsyncMock(return_value=payload),
        ) as build_mock:
            with TestClient(app) as client:
                response = client.get("/api/day-trading-game/session?market=jp")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["symbol"], "7203.T")
        build_mock.assert_awaited_once_with(market="jp", symbol=None)


if __name__ == "__main__":
    unittest.main()
