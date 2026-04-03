from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException

from app.services.lmstudio_client import LmStudioChatResult
from app.services.watchlist_commentary_service import WatchlistCommentaryService


class _FakeHub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def sparkline_payload(self, symbols: list[str], *, refresh: bool = False) -> list[dict[str, object]]:
        self.calls.append({"symbols": list(symbols), "refresh": refresh})
        return [
            {
                "symbol": "AAPL",
                "latest_close": 110.0,
                "previous_close": 100.0,
                "trend_30d": [100.0, 102.0, 107.0, 110.0],
            },
            {
                "symbol": "MSFT",
                "latest_close": 205.0,
                "previous_close": 200.0,
                "trend_30d": [180.0, 190.0, 198.0, 205.0],
            },
        ]


class _FakeClient:
    def __init__(self, results: list[LmStudioChatResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_format: dict[str, object] | None,
    ) -> LmStudioChatResult:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if not self.results:
            raise AssertionError("Unexpected extra chat call")
        return self.results.pop(0)


class WatchlistCommentaryServiceTest(unittest.TestCase):
    def test_build_payload_returns_metrics_and_commentary(self) -> None:
        service = WatchlistCommentaryService()
        service.client = _FakeClient(
            [
                LmStudioChatResult(
                    content='{"picks":[{"symbol":"AAPL","comment":"強い値動き。"},{"symbol":"MSFT","comment":"堅調。"}]}',
                    status_code=200,
                    error_detail=None,
                    model="demo-model",
                )
            ]
        )
        hub = _FakeHub()

        payload = asyncio.run(service.build_payload(hub, ["AAPL", "MSFT"], refresh=True))

        self.assertEqual(payload["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["comment"], "AAPL: 強い値動き。\nMSFT: 堅調。")
        self.assertEqual(len(payload["metrics"]), 2)
        self.assertEqual(payload["metrics"][0]["symbol"], "AAPL")
        self.assertEqual(hub.calls, [{"symbols": ["AAPL", "MSFT"], "refresh": True}])

    def test_request_commentary_falls_back_to_plain_text_when_repair_fails(self) -> None:
        service = WatchlistCommentaryService()
        fake_client = _FakeClient(
            [
                LmStudioChatResult(content="", status_code=400, error_detail="schema error", model="demo-model"),
                LmStudioChatResult(
                    content="AAPL strong move\nMSFT stable trend",
                    status_code=200,
                    error_detail=None,
                    model="fallback-model",
                ),
                LmStudioChatResult(content="", status_code=500, error_detail="repair failed", model="repair-model"),
            ]
        )
        service.client = fake_client

        commentary, model = asyncio.run(service._request_commentary("prompt", ["AAPL", "MSFT"]))

        self.assertEqual(model, "fallback-model")
        self.assertEqual(commentary, "AAPL: AAPL strong move\nMSFT: MSFT stable trend")
        self.assertEqual(fake_client.calls[0]["response_format"]["type"], "json_schema")
        self.assertIsNone(fake_client.calls[1]["response_format"])
        self.assertEqual(fake_client.calls[2]["response_format"]["type"], "json_schema")

    def test_request_commentary_raises_502_on_hard_lmstudio_error(self) -> None:
        service = WatchlistCommentaryService()
        service.client = _FakeClient(
            [
                LmStudioChatResult(
                    content="",
                    status_code=500,
                    error_detail="server exploded",
                    model="demo-model",
                )
            ]
        )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(service._request_commentary("prompt", ["AAPL", "MSFT"]))

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("server exploded", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
