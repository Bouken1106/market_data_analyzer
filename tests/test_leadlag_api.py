import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from app.api.deps import init_routes
from app.routes import router as app_router


def _build_points(dates, opens, closes):
    return [
        {"t": day, "o": float(open_price), "c": float(close_price)}
        for day, open_price, close_price in zip(dates, opens, closes)
    ]


class _FakeHub:
    def __init__(self, payloads):
        self.payloads = payloads

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def historical_payload(self, symbol: str, years: int = 30, refresh: bool = False):
        return {"symbol": symbol, "points": self.payloads[symbol]}


class LeadLagApiTest(unittest.TestCase):
    def test_leadlag_page_and_api(self) -> None:
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

        app = FastAPI()
        init_routes(
            app,
            hub=_FakeHub(payloads),
            symbol_catalog_store=object(),
            paper_portfolio_store=object(),
            ui_state_store=object(),
        )
        static_dir = Path(__file__).resolve().parents[1] / "app" / "static"
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        app.include_router(app_router)
        with TestClient(app) as client:
            page_response = client.get("/leadlag-lab")
            defaults_response = client.get("/api/leadlag/config")
            analyze_response = client.post(
                "/api/leadlag/analyze",
                json={
                    "us_symbols": "USA1,USA2",
                    "jp_symbols": "JP1.T,JP2.T,JP3.T,JP4.T",
                    "cyclical_symbols": "USA1,JP1.T,JP3.T",
                    "defensive_symbols": "USA2,JP2.T,JP4.T",
                    "rolling_window_days": 3,
                    "lambda_reg": 0.9,
                    "n_components": 2,
                    "quantile_q": 0.25,
                    "cfull_start": "2024-01-01",
                    "cfull_end": "2024-01-04",
                },
            )

        self.assertEqual(page_response.status_code, 200)
        self.assertEqual(defaults_response.status_code, 200)
        self.assertEqual(analyze_response.status_code, 200)
        self.assertIn("history_years", defaults_response.json()["defaults"])
        self.assertTrue(analyze_response.json()["ok"])


if __name__ == "__main__":
    unittest.main()
