"""Service package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["MarketDataQueriesMixin", "MarketDataRealtimeMixin", "MarketDataStateMixin"]


def __getattr__(name: str) -> Any:
    if name == "MarketDataQueriesMixin":
        from .market_data_queries import MarketDataQueriesMixin

        return MarketDataQueriesMixin
    if name == "MarketDataRealtimeMixin":
        from .market_data_realtime import MarketDataRealtimeMixin

        return MarketDataRealtimeMixin
    if name == "MarketDataStateMixin":
        from .market_data_state import MarketDataStateMixin

        return MarketDataStateMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
