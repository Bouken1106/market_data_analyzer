import asyncio
import unittest

from app.leadlag.schemas import LeadLagConfig
from app.leadlag.service import LeadLagService


def _build_points(dates, opens, closes):
    return [
        {"t": day, "o": float(open_price), "c": float(close_price)}
        for day, open_price, close_price in zip(dates, opens, closes)
    ]


class _FakeHub:
    def __init__(self, payloads):
        self.payloads = payloads

    async def historical_payload(self, symbol: str, years: int = 30, refresh: bool = False):
        return {"symbol": symbol, "points": self.payloads[symbol]}


class LeadLagServiceTest(unittest.TestCase):
    def test_analyze_returns_latest_signal_and_strategy_summary(self) -> None:
        dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
        ]
        payloads = {
            "USA1": _build_points(dates, [100, 100, 101, 102, 103, 104, 105, 106, 107], [100, 101, 102, 103, 104, 105, 106, 107, 108]),
            "USA2": _build_points(dates, [50, 50, 50.5, 51, 51.3, 51.5, 51.7, 52, 52.3], [50, 50.5, 51, 51.3, 51.5, 51.7, 52, 52.3, 52.6]),
            "JP1.T": _build_points(dates, [100, 100, 101, 102, 103, 104, 105, 106, 107], [100, 101.5, 102.0, 103.2, 104.3, 105.1, 106.4, 107.3, 108.6]),
            "JP2.T": _build_points(dates, [90, 90.2, 90.8, 91.1, 91.7, 92.4, 92.8, 93.2, 93.5], [90.1, 90.6, 91.0, 91.5, 92.1, 92.7, 93.0, 93.4, 93.9]),
            "JP3.T": _build_points(dates, [80, 80.4, 80.8, 81.2, 81.6, 82.0, 82.4, 82.8, 83.2], [80.2, 80.7, 81.0, 81.4, 81.9, 82.2, 82.6, 83.0, 83.4]),
            "JP4.T": _build_points(dates, [70, 69.8, 70.2, 70.6, 71.0, 71.4, 71.8, 72.1, 72.5], [69.9, 70.1, 70.5, 70.9, 71.2, 71.6, 72.0, 72.4, 72.7]),
        }
        service = LeadLagService(_FakeHub(payloads))
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


if __name__ == "__main__":
    unittest.main()
