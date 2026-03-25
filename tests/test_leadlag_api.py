import unittest
from pathlib import Path
from datetime import date, timedelta

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


def _business_dates(start: str, count: int) -> list[str]:
    current = date.fromisoformat(start)
    values: list[str] = []
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return values


class _FakeHub:
    def __init__(self, payloads):
        self.payloads = payloads

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def historical_payload(self, symbol: str, years: int = 30, refresh: bool = False, **kwargs):
        return {"symbol": symbol, "points": self.payloads[symbol]}


class LeadLagApiTest(unittest.TestCase):
    def test_leadlag_page_and_api(self) -> None:
        dates = _business_dates("2024-01-01", 35)
        payloads = {
            "USA1": _build_points(dates, [100 + i * 1.0 for i in range(len(dates))], [100 + i * 1.1 for i in range(len(dates))]),
            "USA2": _build_points(dates, [50 + i * 0.4 for i in range(len(dates))], [50 + i * 0.35 for i in range(len(dates))]),
            "JP1.T": _build_points(dates, [100 + i * 0.8 for i in range(len(dates))], [100 + i * 1.2 for i in range(len(dates))]),
            "JP2.T": _build_points(dates, [90 + i * 0.25 for i in range(len(dates))], [90.1 + i * 0.3 for i in range(len(dates))]),
            "JP3.T": _build_points(dates, [80 + i * 0.18 for i in range(len(dates))], [80.2 + i * 0.22 for i in range(len(dates))]),
            "JP4.T": _build_points(dates, [70 + i * 0.12 for i in range(len(dates))], [69.9 + i * 0.15 for i in range(len(dates))]),
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
        self.assertIn("直近 L 営業日を 1 つの窓として", page_response.text)
        self.assertIn("Strategy Period", page_response.text)
        self.assertIn("history_years", defaults_response.json()["defaults"])
        self.assertEqual(
            defaults_response.json()["defaults"]["universe"]["us"],
            defaults_response.json()["defaults"]["us_symbols"],
        )
        self.assertEqual(
            defaults_response.json()["defaults"]["universe"]["jp"],
            defaults_response.json()["defaults"]["jp_symbols"],
        )
        self.assertIn("daily_rows", analyze_response.json()["strategy"])
        self.assertTrue(analyze_response.json()["ok"])


if __name__ == "__main__":
    unittest.main()
