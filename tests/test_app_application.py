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
    def test_create_app_registers_static_pages_and_api_routes(self) -> None:
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

        with TestClient(app) as client:
            relationship_response = client.get("/relationship-lab")
            leadlag_response = client.get("/leadlag-lab")
            historical_response = client.get("/historical/AAPL")
            static_js_response = client.get("/static/app.terminal.js")

        self.assertEqual(relationship_response.status_code, 200)
        self.assertEqual(leadlag_response.status_code, 200)
        self.assertEqual(historical_response.status_code, 200)
        self.assertEqual(static_js_response.status_code, 200)
        self.assertEqual(
            relationship_response.headers.get("Cache-Control"),
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.assertEqual(
            historical_response.headers.get("Cache-Control"),
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.assertEqual(
            static_js_response.headers.get("Cache-Control"),
            "no-store, no-cache, must-revalidate, max-age=0",
        )


if __name__ == "__main__":
    unittest.main()
