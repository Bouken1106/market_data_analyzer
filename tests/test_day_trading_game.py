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


def _intraday_index(
    dates: tuple[str, ...],
    *,
    periods_per_day: int = 12,
    timezone: str = "America/New_York",
) -> pd.DatetimeIndex:
    indexes = [
        pd.date_range(
            f"{date_key} 09:30",
            periods=periods_per_day,
            freq="15min",
            tz=timezone,
        )
        for date_key in dates
    ]
    combined = indexes[0]
    for extra in indexes[1:]:
        combined = combined.append(extra)
    return combined


class _LastChoiceRandom:
    def shuffle(self, items) -> None:
        return None

    def choice(self, items):
        return items[-1]


class DayTradingGameServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_build_session_uses_yfinance_close_as_execution_price(self) -> None:
        dates = ("2026-04-01", "2026-04-02", "2026-04-03")
        index = _intraday_index(dates)
        steps = range(len(index))
        frame = pd.DataFrame(
            {
                "Open": [100 + step for step in steps],
                "High": [101 + step for step in steps],
                "Low": [99 + step for step in steps],
                "Close": [100.5 + step for step in steps],
                "Volume": [1000 + step for step in steps],
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
        self.assertEqual(payload["symbol_name"], "Apple")
        self.assertEqual(payload["symbol_label"], "Apple (AAPL)")
        self.assertEqual(payload["date"], "2026-04-01")
        self.assertEqual(payload["start_date"], "2026-04-01")
        self.assertEqual(payload["end_date"], "2026-04-03")
        self.assertEqual(payload["date_range"], "2026-04-01 to 2026-04-03")
        self.assertEqual(payload["session_dates"], list(dates))
        self.assertEqual(payload["session_day_count"], 3)
        self.assertEqual(payload["candle_count"], 36)
        self.assertEqual(payload["mode"], "intraday")
        self.assertEqual(payload["moving_averages"][0]["label"], "MA5")
        self.assertEqual(payload["moving_averages"][1]["label"], "MA20")
        self.assertEqual(payload["candles"][0]["execution_price"], 100.5)
        self.assertEqual(payload["candles"][0]["execution_price_method"], "close")
        self.assertEqual(payload["candles"][12]["date"], "2026-04-02")
        self.assertEqual(payload["candles"][-1]["date"], "2026-04-03")
        self.assertIsNone(payload["candles"][0]["moving_averages"]["short"])
        self.assertEqual(payload["candles"][4]["moving_averages"]["short"], 102.5)

    async def test_build_session_falls_back_to_iqr_midpoint_when_close_is_missing(self) -> None:
        index = _intraday_index(("2026-04-02", "2026-04-03", "2026-04-06"))
        frame = pd.DataFrame(
            {
                "Open": [100.0] * len(index),
                "High": [104.0] * len(index),
                "Low": [96.0] * len(index),
                "Q1": [98.0] * len(index),
                "Q3": [102.0] * len(index),
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

    async def test_build_session_adds_japanese_symbol_label(self) -> None:
        dates = ("2026-04-01", "2026-04-02", "2026-04-03")
        index = _intraday_index(dates, timezone="Asia/Tokyo")
        frame = pd.DataFrame(
            {
                "Open": [100.0] * len(index),
                "High": [101.0] * len(index),
                "Low": [99.0] * len(index),
                "Close": [100.5] * len(index),
            },
            index=index,
        )

        async def fake_fetch(symbol: str) -> pd.DataFrame:
            self.assertEqual(symbol, "7974.T")
            return frame

        payload = await build_day_trading_session(
            market="jp",
            symbol="7974.T",
            rng=random.Random(1),
            fetch_history=fake_fetch,
        )

        self.assertEqual(payload["symbol"], "7974.T")
        self.assertEqual(payload["symbol_name"], "Nintendo")
        self.assertEqual(payload["symbol_label"], "Nintendo (7974.T)")

    async def test_daily_session_uses_prior_history_for_moving_averages(self) -> None:
        index = pd.date_range("2026-01-02", periods=55, freq="B", tz="America/New_York")
        steps = range(len(index))
        frame = pd.DataFrame(
            {
                "Open": [100 + step for step in steps],
                "High": [101 + step for step in steps],
                "Low": [99 + step for step in steps],
                "Close": [100.5 + step for step in steps],
            },
            index=index,
        )

        async def fake_fetch(symbol: str, *, interval: str, period: str) -> pd.DataFrame:
            self.assertEqual(symbol, "AAPL")
            self.assertEqual(interval, "1d")
            self.assertEqual(period, "2y")
            return frame

        payload = await build_day_trading_session(
            market="us",
            mode="daily",
            symbol="AAPL",
            rng=_LastChoiceRandom(),
            fetch_history=fake_fetch,
        )

        self.assertEqual(payload["mode"], "daily")
        self.assertEqual(payload["interval"], "1d")
        self.assertEqual(payload["period"], "2y")
        self.assertEqual(payload["step_label"], "Next Day")
        self.assertEqual(payload["session_day_count"], 30)
        self.assertEqual(payload["candle_count"], 30)
        self.assertEqual(payload["moving_averages"][0]["label"], "MA5")
        self.assertEqual(payload["moving_averages"][1]["label"], "MA25")
        self.assertEqual(payload["start_date"], index[-30].date().isoformat())
        self.assertEqual(payload["end_date"], index[-1].date().isoformat())
        self.assertEqual(payload["candles"][0]["moving_averages"]["short"], 123.5)
        self.assertEqual(payload["candles"][0]["moving_averages"]["mid"], 113.5)


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
            "symbol_name": "Toyota Motor",
            "symbol_label": "Toyota Motor (7203.T)",
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
        build_mock.assert_awaited_once_with(market="jp", mode="intraday", symbol=None)


if __name__ == "__main__":
    unittest.main()
