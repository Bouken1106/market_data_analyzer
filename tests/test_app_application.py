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


class ApplicationRoutesTest(unittest.TestCase):
    def test_create_app_registers_remaining_static_pages_and_api_routes(self) -> None:
        services = AppServices(
            hub=_FakeHub(),
            symbol_catalog_store=object(),
            paper_portfolio_store=object(),
            ui_state_store=object(),
        )

        app = create_app(services)
        route_paths = {route.path for route in app.router.routes}
        expected_paths = {
            "/",
            "/relationship-lab",
            "/leadlag-lab",
            "/historical/{symbol}",
            "/api/relationships",
        }
        self.assertTrue(expected_paths.issubset(route_paths))
        self.assertNotIn("/market-data-lab", route_paths)
        self.assertNotIn("/ml-lab", route_paths)
        self.assertNotIn("/strategy-lab", route_paths)
        self.assertNotIn("/compare-lab", route_paths)

        with TestClient(app) as client:
            self.assertEqual(client.get("/relationship-lab").status_code, 200)
            self.assertEqual(client.get("/leadlag-lab").status_code, 200)
            self.assertEqual(client.get("/historical/AAPL").status_code, 200)
            self.assertEqual(client.get("/market-data-lab").status_code, 404)
            self.assertEqual(client.get("/ml-lab").status_code, 404)
            self.assertEqual(client.get("/strategy-lab").status_code, 404)
            self.assertEqual(client.get("/compare-lab").status_code, 404)
            self.assertEqual(client.get("/api/ml/models").status_code, 404)
            self.assertEqual(client.post("/api/strategy/evaluate").status_code, 404)


if __name__ == "__main__":
    unittest.main()
