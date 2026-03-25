"""Shared helpers for MarketData query mixins."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

from ..market_session import infer_country_from_symbol
from ..ohlcv import latest_session_points, merge_points_by_timestamp as merge_ohlcv_points
from ..utils import _datetime_from_unix


class MarketDataQueryCommonMixin:
    @staticmethod
    def _pick_float(payload: dict[str, Any], *keys: str) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                return num
        return None

    @staticmethod
    def _pick_string(payload: dict[str, Any], *keys: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = str(payload.get(key, "")).strip()
            if value:
                return value
        return None

    @staticmethod
    def _merge_quote_payloads_with_source(
        primary: dict[str, Any],
        primary_name: str,
        secondary: dict[str, Any],
        secondary_name: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not isinstance(primary, dict) and not isinstance(secondary, dict):
            return {}, {}
        out: dict[str, Any] = {}
        detail: dict[str, str] = {}
        keys = {
            "symbol",
            "name",
            "instrument_name",
            "exchange",
            "price",
            "close",
            "previous_close",
            "prev_close",
            "open",
            "high",
            "low",
            "volume",
            "bid",
            "ask",
            "timestamp",
            "datetime",
        }
        for key in keys:
            first = primary.get(key) if isinstance(primary, dict) else None
            second = secondary.get(key) if isinstance(secondary, dict) else None
            if first not in (None, ""):
                out[key] = first
                detail[key] = primary_name
            elif second not in (None, ""):
                out[key] = second
                detail[key] = secondary_name
        return out, detail

    @staticmethod
    def _series_source_descriptor(points: list[dict[str, Any]]) -> str:
        if not points:
            return "none"
        providers: set[str] = set()
        for item in points:
            src = str(item.get("_src", "")).strip().lower()
            if src:
                providers.add(src)
        if not providers:
            return "unknown"
        if len(providers) == 1:
            return next(iter(providers))
        return "mixed"

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
        closes = [item["c"] for item in points if isinstance(item.get("c"), (int, float))]
        if len(closes) < window or window <= 0:
            return None
        sample = closes[-window:]
        return sum(sample) / window

    @staticmethod
    def _atr(points: list[dict[str, Any]], window: int = 14) -> float | None:
        if len(points) < window + 1:
            return None
        trs: list[float] = []
        prev_close = points[0]["c"]
        for item in points[1:]:
            high = item.get("h")
            low = item.get("l")
            close = item.get("c")
            if not isinstance(high, (int, float)) or not isinstance(low, (int, float)) or not isinstance(close, (int, float)):
                prev_close = close if isinstance(close, (int, float)) else prev_close
                continue
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            if tr >= 0:
                trs.append(tr)
            prev_close = close
        if len(trs) < window:
            return None
        sample = trs[-window:]
        return sum(sample) / window

    @staticmethod
    def _intraday_vwap(points: list[dict[str, Any]]) -> float | None:
        if not points:
            return None
        latest_session = latest_session_points(points)
        if not latest_session:
            return None
        pv_sum = 0.0
        v_sum = 0.0
        for item in latest_session:
            close = item.get("c")
            volume = item.get("v")
            if not isinstance(close, (int, float)) or not isinstance(volume, (int, float)) or volume <= 0:
                continue
            pv_sum += close * volume
            v_sum += volume
        if v_sum <= 0:
            return None
        return pv_sum / v_sum

    @staticmethod
    def _daily_returns(points: list[dict[str, Any]], max_len: int) -> dict[str, float]:
        closes: list[tuple[str, float]] = []
        for item in points:
            raw_t = str(item.get("t", "")).strip()
            close = item.get("c")
            if not raw_t or not isinstance(close, (int, float)) or close <= 0:
                continue
            closes.append((raw_t.split(" ")[0], close))
        if len(closes) < 2:
            return {}
        target = closes[-(max_len + 1):]
        out: dict[str, float] = {}
        for idx in range(1, len(target)):
            date_key, close_value = target[idx]
            prev_close = target[idx - 1][1]
            if prev_close <= 0:
                continue
            out[date_key] = (close_value / prev_close) - 1
        return out

    @staticmethod
    def _beta_and_corr_60d(
        symbol_points: list[dict[str, Any]],
        benchmark_points: list[dict[str, Any]],
    ) -> tuple[float | None, float | None]:
        symbol_returns = MarketDataQueryCommonMixin._daily_returns(symbol_points, max_len=60)
        benchmark_returns = MarketDataQueryCommonMixin._daily_returns(benchmark_points, max_len=60)
        common_dates = sorted(set(symbol_returns.keys()) & set(benchmark_returns.keys()))
        if len(common_dates) < 20:
            return None, None

        x = [benchmark_returns[d] for d in common_dates]
        y = [symbol_returns[d] for d in common_dates]
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        cov = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y)) / max(1, len(x) - 1)
        var_x = sum((xv - mean_x) ** 2 for xv in x) / max(1, len(x) - 1)
        var_y = sum((yv - mean_y) ** 2 for yv in y) / max(1, len(y) - 1)
        if var_x <= 0 or var_y <= 0:
            return None, None
        beta = cov / var_x
        corr = cov / math.sqrt(var_x * var_y)
        return beta, corr

    @staticmethod
    def _parse_timestamp(raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            parsed = _datetime_from_unix(raw)
            return parsed.isoformat() if parsed is not None else None
        text = str(raw).strip()
        if not text:
            return None
        if text.isdigit():
            parsed = _datetime_from_unix(text)
            return parsed.isoformat() if parsed is not None else None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _best_updated_at(
        self,
        quote_payload: dict[str, Any],
        intraday_points: list[dict[str, Any]],
        day_points: list[dict[str, Any]],
    ) -> str | None:
        candidates = [
            self._parse_timestamp(quote_payload.get("timestamp")) if isinstance(quote_payload, dict) else None,
            self._parse_timestamp(quote_payload.get("datetime")) if isinstance(quote_payload, dict) else None,
            self._parse_timestamp(intraday_points[-1]["t"]) if intraday_points else None,
            self._parse_timestamp(day_points[-1]["t"]) if day_points else None,
        ]
        for item in candidates:
            if item:
                return item
        return None

    @staticmethod
    def _build_market_item(symbol: str, latest: float | None, previous: float | None) -> dict[str, Any]:
        change_abs = None
        change_pct = None
        if latest is not None and previous is not None and previous > 0:
            change_abs = latest - previous
            change_pct = (change_abs / previous) * 100
        return {
            "symbol": symbol,
            "price": latest,
            "change_abs": change_abs,
            "change_pct": change_pct,
        }

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
        normalized_symbol = symbol.upper().strip()
        mapped_country = self.symbol_country_map.get(normalized_symbol)
        if mapped_country:
            return mapped_country
        inferred_country = infer_country_from_symbol(normalized_symbol)
        if inferred_country:
            return inferred_country
        return self.default_country_key

    def _is_country_market_open(self, country_key: str, now_utc: datetime) -> bool:
        session = self.market_sessions.get(country_key)
        if session is None:
            return True

        local_now = now_utc.astimezone(session.tz)
        if local_now.weekday() not in session.weekdays:
            return False
        current_minutes = (local_now.hour * 60) + local_now.minute

        if session.open_minutes <= session.close_minutes:
            return session.open_minutes <= current_minutes < session.close_minutes
        return current_minutes >= session.open_minutes or current_minutes < session.close_minutes

    def _is_symbol_market_open(self, symbol: str, now_utc: datetime | None = None) -> bool:
        utc_now = now_utc or datetime.now(timezone.utc)
        country_key = self._resolve_symbol_country_key(symbol)
        return self._is_country_market_open(country_key, utc_now)

    def _open_symbols(self, symbols: list[str]) -> list[str]:
        now_utc = datetime.now(timezone.utc)
        return [symbol for symbol in symbols if self._is_symbol_market_open(symbol, now_utc=now_utc)]

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
        if not isinstance(payload, dict):
            return False
        if payload.get("status") == "error":
            return True
        message = str(payload.get("Error Message", "")).strip()
        return bool(message)

    def _delay_note(self) -> str:
        if self.provider == "both":
            return "Combined feed: Twelve Data + Financial Modeling Prep."
        if self.provider == "fmp":
            return "Financial Modeling Prep free plan feed."
        return "Twelve Data Basic plan (delayed feed may apply)."
