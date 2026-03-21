"""Thin adapter that reuses the existing historical data fetch pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HistoricalPointBatch:
    """Historical OHLC payloads collected from the shared hub."""

    points_by_symbol: dict[str, list[dict[str, Any]]]
    point_counts: dict[str, int]
    failures: dict[str, str]


class HubHistoricalLeadLagAdapter:
    """Fetch historical series through the existing MarketDataHub interface."""

    def __init__(self, hub: Any, *, history_years: int) -> None:
        self.hub = hub
        self.history_years = max(5, int(history_years))

    async def fetch_points(
        self,
        symbols: tuple[str, ...],
        *,
        refresh: bool = False,
    ) -> HistoricalPointBatch:
        tasks = [
            self.hub.historical_payload(symbol=symbol, years=self.history_years, refresh=refresh)
            for symbol in symbols
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        points_by_symbol: dict[str, list[dict[str, Any]]] = {}
        point_counts: dict[str, int] = {}
        failures: dict[str, str] = {}

        for index, response in enumerate(responses):
            symbol = symbols[index]
            if isinstance(response, Exception):
                failures[symbol] = str(response)
                continue

            points = response.get("points") if isinstance(response, dict) else None
            if not isinstance(points, list) or not points:
                failures[symbol] = "No historical points returned."
                continue

            safe_points = [dict(item) for item in points if isinstance(item, dict)]
            if not safe_points:
                failures[symbol] = "No valid historical points returned."
                continue

            points_by_symbol[symbol] = safe_points
            point_counts[symbol] = len(safe_points)

        return HistoricalPointBatch(
            points_by_symbol=points_by_symbol,
            point_counts=point_counts,
            failures=failures,
        )
