"""Shared compatibility facade for market-data query helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from ..market_session import DEFAULT_COUNTRY_KEY, DEFAULT_MARKET_SESSIONS
from ..ohlcv import latest_session_points, merge_points_by_timestamp as merge_ohlcv_points
from .market_data_market_hours import (
    is_country_market_open,
    is_symbol_market_open,
    open_symbols,
    resolve_symbol_country_key,
)
from .market_data_math import atr, beta_and_corr, daily_returns, intraday_vwap, moving_average
from .market_data_payload_utils import (
    best_updated_at,
    build_market_item,
    delay_note,
    is_fmp_error,
    merge_quote_payloads_with_source,
    parse_timestamp,
    pick_float,
    pick_string,
    series_source_descriptor,
)


class MarketDataQueryCommonMixin:
    def _format_twelvedata_symbol(self, symbol: str) -> str:
        normalized = str(symbol or "").strip().upper()
        if not normalized or ":" in normalized:
            return normalized
        country_key = self._resolve_symbol_country_key(normalized)
        if country_key == "JAPAN":
            code = normalized[:-2] if normalized.endswith(".T") else normalized
            if code.isdigit() and len(code) in {4, 5}:
                return f"{code}:JPX"
        return normalized

    @staticmethod
    def _pick_float(payload: dict[str, Any], *keys: str) -> float | None:
        return pick_float(payload, *keys)

    @staticmethod
    def _pick_string(payload: dict[str, Any], *keys: str) -> str | None:
        return pick_string(payload, *keys)

    @staticmethod
    def _merge_quote_payloads_with_source(
        primary: dict[str, Any],
        primary_name: str,
        secondary: dict[str, Any],
        secondary_name: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return merge_quote_payloads_with_source(primary, primary_name, secondary, secondary_name)

    @staticmethod
    def _series_source_descriptor(points: list[dict[str, Any]]) -> str:
        return series_source_descriptor(points)

    @staticmethod
    def _extract_latest_session_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return latest_session_points(points)

    @staticmethod
    def _point_date(point: dict[str, Any]) -> date | None:
        raw_t = str(point.get("t", "")).strip()
        if not raw_t:
            return None
        date_text = raw_t.split(" ")[0]
        try:
            return date.fromisoformat(date_text)
        except ValueError:
            return None

    @staticmethod
    def _merge_points_by_timestamp(
        base_points: list[dict[str, Any]],
        incoming_points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return merge_ohlcv_points(base_points, incoming_points)

    @staticmethod
    def _moving_average(points: list[dict[str, Any]], window: int) -> float | None:
        return moving_average(points, window)

    @staticmethod
    def _atr(points: list[dict[str, Any]], window: int = 14) -> float | None:
        return atr(points, window)

    @staticmethod
    def _intraday_vwap(points: list[dict[str, Any]]) -> float | None:
        return intraday_vwap(points)

    @staticmethod
    def _daily_returns(points: list[dict[str, Any]], max_len: int) -> dict[str, float]:
        return daily_returns(points, max_len)

    @staticmethod
    def _beta_and_corr_60d(
        symbol_points: list[dict[str, Any]],
        benchmark_points: list[dict[str, Any]],
    ) -> tuple[float | None, float | None]:
        return beta_and_corr(symbol_points, benchmark_points, max_len=60, min_overlap=20)

    @staticmethod
    def _parse_timestamp(raw: Any) -> str | None:
        return parse_timestamp(raw)

    def _best_updated_at(
        self,
        quote_payload: dict[str, Any],
        intraday_points: list[dict[str, Any]],
        day_points: list[dict[str, Any]],
    ) -> str | None:
        return best_updated_at(quote_payload, intraday_points, day_points)

    @staticmethod
    def _build_market_item(symbol: str, latest: float | None, previous: float | None) -> dict[str, Any]:
        return build_market_item(symbol, latest, previous)

    async def _update_minute_credits_from_response(self, response) -> None:
        if not self._uses_twelvedata():
            return
        used_value = self._try_parse_int(response.headers.get("api-credits-used"))
        left_value = self._try_parse_int(response.headers.get("api-credits-left"))
        if used_value is None and left_value is None:
            return

        if used_value is not None:
            self.minute_credits_used = used_value
        if left_value is not None:
            self.minute_credits_left = left_value

    async def _update_daily_credits_from_api_usage(self, payload: dict[str, Any]) -> None:
        if not self._uses_twelvedata():
            return
        if not isinstance(payload, dict):
            return
        daily_usage = self._try_parse_int(payload.get("daily_usage"))
        plan_daily_limit = self._try_parse_int(payload.get("plan_daily_limit"))
        if daily_usage is None and plan_daily_limit is None:
            return

        if plan_daily_limit is not None:
            self.daily_credits_limit = plan_daily_limit
        if daily_usage is not None:
            self.daily_credits_used = max(0, daily_usage)
        if self.daily_credits_limit is not None and self.daily_credits_used is not None:
            self.daily_credits_left = max(0, self.daily_credits_limit - self.daily_credits_used)

        self.daily_credits_source = "api_usage"
        self.daily_credits_is_estimated = False
        self.daily_credits_updated_at = datetime.now(timezone.utc).isoformat()
        await self.publish({"type": "status", "data": await self.status_payload()})

    async def _consume_daily_credit_estimate(self, amount: int, source: str) -> None:
        if not self._uses_twelvedata():
            return
        if amount <= 0:
            return
        if self.daily_credits_limit is None or self.daily_credits_used is None:
            return

        self.daily_credits_used = max(0, self.daily_credits_used + amount)
        self.daily_credits_left = max(0, self.daily_credits_limit - self.daily_credits_used)
        self.daily_credits_source = source
        self.daily_credits_is_estimated = True
        self.daily_credits_updated_at = datetime.now(timezone.utc).isoformat()

        await self.publish({"type": "status", "data": await self.status_payload()})

    def _resolve_symbol_country_key(self, symbol: str) -> str:
        return resolve_symbol_country_key(
            symbol,
            symbol_country_map=getattr(self, "symbol_country_map", {}),
            default_country_key=getattr(self, "default_country_key", DEFAULT_COUNTRY_KEY),
        )

    def _is_country_market_open(self, country_key: str, now_utc: datetime) -> bool:
        return is_country_market_open(
            country_key,
            market_sessions=getattr(self, "market_sessions", DEFAULT_MARKET_SESSIONS),
            now_utc=now_utc,
        )

    def _is_symbol_market_open(self, symbol: str, now_utc: datetime | None = None) -> bool:
        return is_symbol_market_open(
            symbol,
            symbol_country_map=getattr(self, "symbol_country_map", {}),
            default_country_key=getattr(self, "default_country_key", DEFAULT_COUNTRY_KEY),
            market_sessions=getattr(self, "market_sessions", DEFAULT_MARKET_SESSIONS),
            now_utc=now_utc,
        )

    def _open_symbols(self, symbols: list[str]) -> list[str]:
        return open_symbols(
            symbols,
            symbol_country_map=getattr(self, "symbol_country_map", {}),
            default_country_key=getattr(self, "default_country_key", DEFAULT_COUNTRY_KEY),
            market_sessions=getattr(self, "market_sessions", DEFAULT_MARKET_SESSIONS),
        )

    @staticmethod
    def _try_parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_parse_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_iso_epoch(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _is_fmp_error(payload: Any) -> bool:
        return is_fmp_error(payload)

    def _delay_note(self) -> str:
        return delay_note(self.provider)
