import asyncio
import unittest
from datetime import date, timedelta

from fastapi import HTTPException

from app.leadlag.data_adapter import HubHistoricalLeadLagAdapter
from app.leadlag.schemas import LeadLagConfig
from app.leadlag.service import LeadLagService


def _business_dates(start: str, count: int) -> list[str]:
    current = date.fromisoformat(start)
    values: list[str] = []
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _build_points(dates, opens, closes):
    return [
        {"t": day, "o": float(open_price), "c": float(close_price)}
        for day, open_price, close_price in zip(dates, opens, closes)
    ]


class _FakeHub:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    async def historical_payload(self, symbol: str, years: int = 30, refresh: bool = False, **kwargs):
        self.calls.append(
            {
                "symbol": symbol,
                "years": years,
                "refresh": refresh,
                **kwargs,
            }
        )
        return {"symbol": symbol, "points": self.payloads[symbol]}


class LeadLagServiceTest(unittest.TestCase):
    def test_analyze_returns_latest_signal_and_strategy_summary(self) -> None:
        dates = _business_dates("2024-01-01", 35)
        payloads = {
            "USA1": _build_points(dates, [100 + i * 1.0 for i in range(len(dates))], [100 + i * 1.1 for i in range(len(dates))]),
            "USA2": _build_points(dates, [50 + i * 0.4 for i in range(len(dates))], [50 + i * 0.35 for i in range(len(dates))]),
            "JP1.T": _build_points(dates, [100 + i * 0.8 for i in range(len(dates))], [100 + i * 1.2 for i in range(len(dates))]),
            "JP2.T": _build_points(dates, [90 + i * 0.25 for i in range(len(dates))], [90.1 + i * 0.3 for i in range(len(dates))]),
            "JP3.T": _build_points(dates, [80 + i * 0.18 for i in range(len(dates))], [80.2 + i * 0.22 for i in range(len(dates))]),
            "JP4.T": _build_points(dates, [70 + i * 0.12 for i in range(len(dates))], [69.9 + i * 0.15 for i in range(len(dates))]),
        }
        hub = _FakeHub(payloads)
        service = LeadLagService(hub)
        config = LeadLagConfig(
            us_symbols=("USA1", "USA2"),
            jp_symbols=("JP1.T", "JP2.T", "JP3.T", "JP4.T"),
            rolling_window_days=3,
            lambda_reg=0.9,
            n_components=2,
            quantile_q=0.25,
            cfull_start="2024-01-01",
            cfull_end="2024-01-04",
            cyclical_symbols=frozenset({"USA1", "JP1.T", "JP3.T"}),
            defensive_symbols=frozenset({"USA2", "JP2.T", "JP4.T"}),
            refresh=False,
            include_backtest=True,
            include_transfer_matrix=True,
            history_years=10,
        )

        result = asyncio.run(service.analyze(config))

        self.assertIn("latest_signal", result)
        self.assertIn("strategy", result)
        self.assertTrue(result["latest_signal"]["predicted_rows"])
        self.assertIsNotNone(result["strategy"]["summary"]["signal_days"])
        self.assertIn("recent_1m_summary", result["strategy"])
        self.assertLess(result["strategy"]["recent_1m_summary"]["signal_days"], result["strategy"]["summary"]["signal_days"])
        self.assertTrue(hub.calls)
        self.assertTrue(all(call.get("source_preference") == "stooq" for call in hub.calls))
        self.assertTrue(all(call.get("allow_api_fallback") is False for call in hub.calls))

    def test_adapter_exposes_specific_fetch_failure_reason(self) -> None:
        class _FailingHub:
            async def historical_payload(self, symbol: str, years: int = 30, refresh: bool = False, **kwargs):
                del years, refresh, kwargs
                if symbol == "JP2.T":
                    raise HTTPException(status_code=404, detail="Stooq daily CSV fetch failed for JP2.T. All connection attempts failed")
                return {
                    "symbol": symbol,
                    "points": _build_points(
                        [
                            "2024-01-01",
                            "2024-01-02",
                            "2024-01-03",
                        ],
                        [100, 101, 102],
                        [101, 102, 103],
                    ),
                }

        adapter = HubHistoricalLeadLagAdapter(_FailingHub(), history_years=10)
        batch = asyncio.run(adapter.fetch_points(("USA1", "JP1.T", "JP2.T")))
        excluded = batch.failures
        self.assertEqual(
            excluded["JP2.T"],
            "Stooq daily CSV fetch failed for JP2.T. All connection attempts failed",
        )


if __name__ == "__main__":
    unittest.main()
