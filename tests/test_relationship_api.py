import unittest

from fastapi.testclient import TestClient

from app.application import create_app
from app.bootstrap import AppServices


def _build_points(base: float, step: float, count: int = 30) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    close = base
    for index in range(count):
        close += step
        points.append(
            {
                "t": f"2024-01-{index + 1:02d}",
                "o": close,
                "h": close,
                "l": close,
                "c": close,
                "v": 1000.0 + index,
            }
        )
    return points


class _FakeHub:
    def __init__(self) -> None:
        self.full_daily_history_store = object()
        self.data = {
            "AAPL": _build_points(100.0, 1.0),
            "MSFT": _build_points(200.0, 2.0),
            "XOM": _build_points(150.0, -0.5),
        }

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def historical_payload(self, symbol: str, months: int = 12, refresh: bool = False, **kwargs):
        del months, refresh, kwargs
        points = self.data.get(symbol)
        if points is None:
            raise ValueError("missing symbol")
        return {"symbol": symbol, "points": points}


class _FakeMlJobStore:
    def get(self, job_id: str):
        del job_id
        return None

    def request_cancel(self, job_id: str):
        del job_id
        return None


class RelationshipApiTest(unittest.TestCase):
    def test_relationship_api_returns_summary_and_pairs(self) -> None:
        services = AppServices(
            hub=_FakeHub(),
            symbol_catalog_store=object(),
            paper_portfolio_store=object(),
            ui_state_store=object(),
            stock_ml_page_store=object(),
            ml_job_store=_FakeMlJobStore(),
        )
        app = create_app(services)

        with TestClient(app) as client:
            response = client.get("/api/relationships?symbols=AAPL,MSFT,XOM&window_days=20&top_pairs=5")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["requested_symbols"], ["AAPL", "MSFT", "XOM"])
        self.assertEqual(payload["analyzed_symbols"], ["AAPL", "MSFT", "XOM"])
        self.assertEqual(len(payload["pair_candidates"]), 3)
        self.assertEqual(payload["pair_candidates"][0]["left"], "AAPL")
        self.assertEqual(payload["pair_candidates"][0]["right"], "MSFT")
        self.assertIn("summary", payload)
        self.assertEqual(payload["skipped_symbols"], [])


if __name__ == "__main__":
    unittest.main()
