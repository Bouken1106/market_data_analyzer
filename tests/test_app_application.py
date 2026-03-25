import unittest

from fastapi.testclient import TestClient

from app.application import create_app
from app.bootstrap import AppServices


class _FakeHub:
    def __init__(self) -> None:
        self.full_daily_history_store = object()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _FakeMlJobStore:
    def get(self, job_id: str):
        del job_id
        return None

    def request_cancel(self, job_id: str):
        del job_id
        return None


class ApplicationRoutesTest(unittest.TestCase):
    def test_create_app_registers_static_pages_and_ml_routes(self) -> None:
        services = AppServices(
            hub=_FakeHub(),
            symbol_catalog_store=object(),
            paper_portfolio_store=object(),
            ui_state_store=object(),
            stock_ml_page_store=object(),
            ml_job_store=_FakeMlJobStore(),
        )

        app = create_app(services)
        route_paths = {route.path for route in app.router.routes}
        expected_paths = {
            "/",
            "/market-data-lab",
            "/ml-lab",
            "/strategy-lab",
            "/compare-lab",
            "/leadlag-lab",
            "/historical/{symbol}",
            "/api/ml/models",
            "/api/strategy/evaluate",
            "/api/ml/jobs/{job_id}",
        }
        self.assertTrue(expected_paths.issubset(route_paths))

        with TestClient(app) as client:
            self.assertEqual(client.get("/market-data-lab").status_code, 200)
            self.assertEqual(client.get("/ml-lab").status_code, 200)
            self.assertEqual(client.get("/strategy-lab").status_code, 200)
            self.assertEqual(client.get("/compare-lab").status_code, 200)
            self.assertEqual(client.get("/leadlag-lab").status_code, 200)
            self.assertEqual(client.get("/historical/AAPL").status_code, 200)
            self.assertEqual(client.get("/api/ml/models").status_code, 200)

            ml_job_response = client.get("/api/ml/jobs/missing-job")
            self.assertEqual(ml_job_response.status_code, 404)
            self.assertEqual(ml_job_response.json()["detail"], "Job not found.")


if __name__ == "__main__":
    unittest.main()
