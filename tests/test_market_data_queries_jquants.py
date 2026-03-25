import asyncio
import json
import unittest

import httpx
from fastapi import HTTPException

from app.services.market_data_queries import MarketDataQueriesMixin


class _DummyQueries(MarketDataQueriesMixin):
    def __init__(self) -> None:
        self.provider = "twelvedata"
        self._historical_cache = {}
        self._historical_lock = asyncio.Lock()


class _MemoryDailyHistoryStore:
    def __init__(self) -> None:
        self.cleared: list[str] = []
        self.upserts: dict[str, list[dict[str, object]]] = {}

    async def clear(self, symbol: str | None = None) -> int:
        if symbol:
            self.cleared.append(symbol)
        return 0

    async def get(self, symbol: str, *, copy: bool = True) -> list[dict[str, object]]:
        points = self.upserts.get(symbol, [])
        return [dict(item) for item in points] if copy else points

    async def upsert(self, symbol: str, points: list[dict[str, object]]) -> None:
        self.upserts[symbol] = [dict(item) for item in points]

    async def last_updated_epoch(self, symbol: str) -> float | None:
        return None


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

    def test_fetch_series_jquants_retries_with_subscription_window(self) -> None:
        async def run_test() -> None:
            responses = [
                httpx.Response(
                    status_code=400,
                    content=json.dumps(
                        {
                            "message": (
                                "Your subscription covers the following dates: "
                                "2023-12-29 ~ 2025-12-29. If you want more data, please check other plans."
                            )
                        }
                    ),
                    headers={"content-type": "application/json"},
                ),
                httpx.Response(
                    status_code=200,
                    content=json.dumps(
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
                            ]
                        }
                    ),
                    headers={"content-type": "application/json"},
                ),
            ]
            seen_queries: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                seen_queries.append(str(request.url))
                response = responses[len(seen_queries) - 1]
                response.request = request
                return response

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)
            queries = _DummyQueries()

            from app.services import market_data_queries as module

            original_key = module.JQUANTS_API_KEY
            original_interval = module.JQUANTS_MIN_REQUEST_INTERVAL_SEC
            original_backoff = module.JQUANTS_RATE_LIMIT_BACKOFF_SEC
            module.JQUANTS_API_KEY = "test-jquants-key"
            module.JQUANTS_MIN_REQUEST_INTERVAL_SEC = 0.0
            module.JQUANTS_RATE_LIMIT_BACKOFF_SEC = 0.0
            try:
                points = await queries._fetch_series_jquants(
                    client=client,
                    symbol="1617.T",
                    interval="1day",
                    outputsize=500,
                    start_date="2020-01-01",
                    end_date="2026-03-23",
                )
            finally:
                module.JQUANTS_API_KEY = original_key
                module.JQUANTS_MIN_REQUEST_INTERVAL_SEC = original_interval
                module.JQUANTS_RATE_LIMIT_BACKOFF_SEC = original_backoff
                await client.aclose()

            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["t"], "2024-01-04")
            self.assertIn("from=2020-01-01", seen_queries[0])
            self.assertIn("to=2026-03-23", seen_queries[0])
            self.assertIn("from=2023-12-29", seen_queries[1])
            self.assertIn("to=2025-12-29", seen_queries[1])

        asyncio.run(run_test())

    def test_fetch_full_daily_series_uses_jquants_fallback_without_earliest_lookup(self) -> None:
        class _JQuantsOnlyQueries(_DummyQueries):
            def __init__(self) -> None:
                super().__init__()
                self.full_daily_history_store = _MemoryDailyHistoryStore()

            def _should_use_jquants_for_symbol(self, symbol: str, interval: str) -> bool:
                return True

            async def _fetch_series(
                self,
                client: httpx.AsyncClient,
                symbol: str,
                interval: str,
                outputsize: int,
                start_date: str | None = None,
                end_date: str | None = None,
            ) -> list[dict[str, object]]:
                del client, interval, outputsize, start_date, end_date
                return [{"t": "2024-01-04", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000.0}]

            async def _fetch_earliest_date(self, client: httpx.AsyncClient, symbol: str, interval: str):
                raise AssertionError("earliest timestamp lookup should be skipped for J-Quants symbols")

        async def run_test() -> None:
            queries = _JQuantsOnlyQueries()
            points = await queries._fetch_full_daily_series(client=None, symbol="1617.T", refresh=True)
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["t"], "2024-01-04")
            self.assertEqual(queries.full_daily_history_store.upserts["1617.T"][0]["c"], 100.5)

        asyncio.run(run_test())

    def test_historical_points_fall_back_to_fmp_when_twelvedata_returns_no_daily_series(self) -> None:
        class _TwelvedataFallbackQueries(_DummyQueries):
            def __init__(self) -> None:
                super().__init__()
                self.fmp_api_key = "has-fmp"

            async def _fetch_series(
                self,
                client: httpx.AsyncClient,
                symbol: str,
                interval: str,
                outputsize: int,
                start_date: str | None = None,
                end_date: str | None = None,
            ) -> list[dict[str, object]]:
                del client, symbol, interval, outputsize, start_date, end_date
                return []

            async def _fetch_series_fmp(
                self,
                client: httpx.AsyncClient,
                symbol: str,
                interval: str,
                outputsize: int,
                start_date: str | None = None,
                end_date: str | None = None,
            ) -> list[dict[str, object]]:
                del client, symbol, interval, outputsize, start_date, end_date
                return [{"t": "2024-01-04", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000.0}]

        async def run_test() -> None:
            queries = _TwelvedataFallbackQueries()
            points, detail = await queries._fetch_historical_points_with_detail(
                client=None,
                symbol="XLK",
                interval="1day",
                outputsize=500,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            self.assertEqual(len(points), 1)
            self.assertEqual(detail["provider"], "fmp")
            self.assertEqual(detail["mode"], "twelvedata_with_fmp_fallback")

        asyncio.run(run_test())

    def test_historical_payload_can_use_stooq_without_api_fallback(self) -> None:
        class _StooqOnlyQueries(_DummyQueries):
            def __init__(self) -> None:
                super().__init__()
                self.full_daily_history_store = _MemoryDailyHistoryStore()

            async def _fetch_historical_points_with_detail(self, *args, **kwargs):
                raise AssertionError("API historical fetch should not be used when source_preference=stooq")

        async def run_test() -> None:
            queries = _StooqOnlyQueries()

            from app.services import market_data_queries as module

            original_fetch = module.fetch_stooq_daily_history

            async def fake_fetch(symbol: str, *, client=None, timeout_sec: float = 25.0):
                del client, timeout_sec
                self.assertEqual(symbol, "XLB")
                return [
                    {"t": "2024-01-02", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000.0, "_src": "stooq"},
                    {"t": "2024-01-03", "o": 101.0, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1100.0, "_src": "stooq"},
                ]

            module.fetch_stooq_daily_history = fake_fetch
            try:
                payload = await queries.historical_payload(
                    symbol="XLB",
                    years=5,
                    refresh=True,
                    source_preference="stooq",
                    allow_api_fallback=False,
                )
            finally:
                module.fetch_stooq_daily_history = original_fetch

            self.assertEqual(payload["symbol"], "XLB")
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["source_detail"]["provider"], "stooq")
            self.assertEqual(payload["source_detail"]["mode"], "stooq_live")
            self.assertEqual(queries.full_daily_history_store.upserts["XLB"][0]["_src"], "stooq")

        asyncio.run(run_test())

    def test_historical_payload_reports_stooq_empty_response_detail(self) -> None:
        class _StooqEmptyQueries(_DummyQueries):
            def __init__(self) -> None:
                super().__init__()
                self.full_daily_history_store = _MemoryDailyHistoryStore()

            async def _fetch_historical_points_with_detail(self, *args, **kwargs):
                raise AssertionError("API historical fetch should not run for Stooq-only mode")

        async def run_test() -> None:
            queries = _StooqEmptyQueries()

            from app.services import market_data_queries as module

            original_fetch = module.fetch_stooq_daily_history

            async def fake_fetch(symbol: str, *, client=None, timeout_sec: float = 25.0):
                del symbol, client, timeout_sec
                return []

            module.fetch_stooq_daily_history = fake_fetch
            try:
                with self.assertRaises(HTTPException) as ctx:
                    await queries.historical_payload(
                        symbol="XLB",
                        years=5,
                        refresh=True,
                        source_preference="stooq",
                        allow_api_fallback=False,
                    )
            finally:
                module.fetch_stooq_daily_history = original_fetch

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertIn("Stooq daily CSV", str(ctx.exception.detail))
            self.assertIn("requested date range", str(ctx.exception.detail))

        asyncio.run(run_test())

    def test_historical_payload_reports_stooq_fetch_failure_detail(self) -> None:
        class _StooqFailureQueries(_DummyQueries):
            def __init__(self) -> None:
                super().__init__()
                self.full_daily_history_store = _MemoryDailyHistoryStore()

            async def _fetch_historical_points_with_detail(self, *args, **kwargs):
                raise AssertionError("API historical fetch should not run for Stooq-only mode")

        async def run_test() -> None:
            queries = _StooqFailureQueries()

            from app.services import market_data_queries as module

            original_fetch = module.fetch_stooq_daily_history

            async def fake_fetch(symbol: str, *, client=None, timeout_sec: float = 25.0):
                del symbol, client, timeout_sec
                raise httpx.ConnectError("All connection attempts failed")

            module.fetch_stooq_daily_history = fake_fetch
            try:
                with self.assertRaises(HTTPException) as ctx:
                    await queries.historical_payload(
                        symbol="XLB",
                        years=5,
                        refresh=True,
                        source_preference="stooq",
                        allow_api_fallback=False,
                    )
            finally:
                module.fetch_stooq_daily_history = original_fetch

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertIn("Stooq daily CSV fetch failed", str(ctx.exception.detail))
            self.assertIn("All connection attempts failed", str(ctx.exception.detail))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
