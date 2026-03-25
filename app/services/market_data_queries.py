"""Data-query and analytics mixin for ``MarketDataHub``."""

from __future__ import annotations

# Keep these module-level names available for callers/tests that monkeypatch them.
from ..config import JQUANTS_API_KEY, JQUANTS_MIN_REQUEST_INTERVAL_SEC, JQUANTS_RATE_LIMIT_BACKOFF_SEC
from ..stooq import fetch_stooq_daily_history
from .market_data_queries_common import MarketDataQueryCommonMixin
from .market_data_queries_historical import MarketDataHistoricalMixin
from .market_data_queries_overview import MarketDataOverviewMixin
from .market_data_queries_reference import MarketDataReferenceMixin


class MarketDataQueriesMixin(
    MarketDataReferenceMixin,
    MarketDataOverviewMixin,
    MarketDataHistoricalMixin,
    MarketDataQueryCommonMixin,
):
    """Combined query mixin composed from focused query helpers."""
