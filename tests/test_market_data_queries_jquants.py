import asyncio
import json
import unittest

import httpx

from app.services.market_data_queries import MarketDataQueriesMixin


class _DummyQueries(MarketDataQueriesMixin):
    def __init__(self) -> None:
        self.provider = "twelvedata"


class MarketDataQueriesJQuantsTest(unittest.TestCase):
    def test_normalize_jquants_code_accepts_tse_suffix(self) -> None:
        self.assertEqual(_DummyQueries._normalize_jquants_code("1617.T"), "1617")
        self.assertEqual(_DummyQueries._normalize_jquants_code("86970"), "86970")
        self.assertIsNone(_DummyQueries._normalize_jquants_code("AAPL"))

    def test_fetch_series_jquants_normalizes_daily_quotes_and_pagination(self) -> None:
        async def run_test() -> None:
            responses = [
                {
                    "daily_quotes": [
                        {
                            "Date": "2024-01-04",
                            "Open": 100.0,
                            "High": 101.0,
                            "Low": 99.0,
                            "Close": 100.5,
                            "Volume": 1000.0,
                        }
                    ],
                    "pagination_key": "next-page",
                },
                {
                    "daily_quotes": [
                        {
                            "Date": "2024-01-05",
                            "Open": 101.0,
                            "High": 102.0,
                            "Low": 100.0,
                            "Close": 101.5,
                            "Volume": 1100.0,
                        }
                    ]
                },
            ]
            seen_queries: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                seen_queries.append(str(request.url))
                self.assertEqual(request.headers.get("x-api-key"), "test-jquants-key")
                return httpx.Response(
                    status_code=200,
                    content=json.dumps(responses[len(seen_queries) - 1]),
                    headers={"content-type": "application/json"},
                    request=request,
                )

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)
            queries = _DummyQueries()

            from app.services import market_data_queries as module

            original_key = module.JQUANTS_API_KEY
            module.JQUANTS_API_KEY = "test-jquants-key"
            try:
                points = await queries._fetch_series_jquants(
                    client=client,
                    symbol="1617.T",
                    interval="1day",
                    outputsize=500,
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                )
            finally:
                module.JQUANTS_API_KEY = original_key
                await client.aclose()

            self.assertEqual(len(points), 2)
            self.assertEqual(points[0]["t"], "2024-01-04")
            self.assertEqual(points[0]["o"], 100.0)
            self.assertEqual(points[1]["c"], 101.5)
            self.assertEqual(points[1]["_src"], "jquants")
            self.assertIn("code=1617", seen_queries[0])
            self.assertIn("from=2024-01-01", seen_queries[0])
            self.assertIn("to=2024-01-31", seen_queries[0])
            self.assertIn("pagination_key=next-page", seen_queries[1])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
