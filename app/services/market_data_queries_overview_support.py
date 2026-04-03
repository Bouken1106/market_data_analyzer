"""Pure helpers for overview and sparkline payload assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException

from ..config import SYMBOL_PATTERN

PickFloat = Callable[[dict[str, Any], str], float | None]
BuildMarketItem = Callable[[str, float | None, float | None], dict[str, Any]]
BetaAndCorr = Callable[[list[dict[str, Any]], list[dict[str, Any]]], tuple[float | None, float | None]]
LatestSessionPoints = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
SeriesSourceDescriptor = Callable[[list[dict[str, Any]]], str]
PickString = Callable[..., str | None]
BestUpdatedAt = Callable[[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]], str | None]
DelayNote = Callable[[], str]

_SUPPORT_STATUS = {
    "order_book": "not_supported_on_current_data_source",
    "corporate_events": "not_supported_on_current_data_source",
    "earnings_calendar": "not_supported_on_current_data_source",
    "news_headlines": "not_supported_on_current_data_source",
    "sector_etf": "not_supported_on_current_data_source",
}


@dataclass(frozen=True)
class OverviewRequest:
    symbol: str
    include_intraday: bool
    include_market: bool
    include_qqq: bool

    @property
    def cache_key(self) -> tuple[str, bool, bool, bool]:
        return (
            self.symbol,
            self.include_intraday,
            self.include_market,
            self.include_qqq,
        )


@dataclass(frozen=True)
class OverviewInputs:
    quote: dict[str, Any]
    day_points: list[dict[str, Any]]
    m1_points: list[dict[str, Any]]
    m5_points: list[dict[str, Any]]
    market_context: dict[str, Any] | None


@dataclass(frozen=True)
class OverviewMarketSection:
    spy_points: list[dict[str, Any]]
    qqq_points: list[dict[str, Any]]
    payload: dict[str, Any]


def build_overview_request(
    *,
    symbol: str,
    include_intraday: bool,
    include_market: bool,
    include_qqq: bool,
) -> OverviewRequest:
    normalized = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(normalized):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")
    return OverviewRequest(
        symbol=normalized,
        include_intraday=bool(include_intraday),
        include_market=bool(include_market),
        include_qqq=bool(include_qqq),
    )


def support_status_payload() -> dict[str, str]:
    return dict(_SUPPORT_STATUS)


def build_overview_source_detail(
    *,
    quote: dict[str, Any],
    day_points: list[dict[str, Any]],
    m1_points: list[dict[str, Any]],
    m5_points: list[dict[str, Any]],
    spy_points: list[dict[str, Any]],
    qqq_points: list[dict[str, Any]],
    price_context: dict[str, Any],
    series_source_descriptor: SeriesSourceDescriptor,
) -> dict[str, Any]:
    quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
    if not isinstance(quote_source_detail, dict):
        quote_source_detail = {}
    return {
        "quote_provider": quote.get("_source_provider") if isinstance(quote, dict) else None,
        "chart_sources": {
            "1min": series_source_descriptor(m1_points),
            "5min": series_source_descriptor(m5_points),
            "1day": series_source_descriptor(day_points),
            "SPY": series_source_descriptor(spy_points),
            "QQQ": series_source_descriptor(qqq_points),
        },
        "fields": {
            "price.current": price_context["current_price_source"] or "unknown",
            "price.previous_close": price_context["previous_close_source"] or "unknown",
            "price.day_open": price_context["day_open_source"] or "unknown",
            "price.day_high": price_context["day_high_source"] or "unknown",
            "price.day_low": price_context["day_low_source"] or "unknown",
            "volume.today": price_context["day_volume_source"] or "unknown",
            "spread.bid": quote_source_detail.get("bid") or "unknown",
            "spread.ask": quote_source_detail.get("ask") or "unknown",
        },
    }


def build_overview_payload(
    *,
    request: OverviewRequest,
    inputs: OverviewInputs,
    provider: str,
    price_context: dict[str, Any],
    technical: dict[str, float | None],
    market_payload: dict[str, Any],
    source_detail: dict[str, Any],
    pick_string: PickString,
    best_updated_at: BestUpdatedAt,
    delay_note: DelayNote,
) -> dict[str, Any]:
    return {
        "symbol": request.symbol,
        "name": pick_string(inputs.quote, "name", "instrument_name"),
        "exchange": pick_string(inputs.quote, "exchange"),
        "price": {
            "current": price_context["current_price"],
            "previous_close": price_context["previous_close"],
            "change_abs": price_context["change_abs"],
            "change_pct": price_context["change_pct"],
            "day_open": price_context["day_open"],
            "day_high": price_context["day_high"],
            "day_low": price_context["day_low"],
            "gap_abs": price_context["gap_abs"],
            "gap_pct": price_context["gap_pct"],
            "updated_at": best_updated_at(inputs.quote, inputs.m1_points, inputs.day_points),
            "delay_note": delay_note(),
        },
        "volume": {
            "today": price_context["day_volume"],
            "avg20": price_context["avg_volume_20"],
            "avg_ratio": price_context["avg_volume_ratio"],
            "turnover": price_context["turnover"],
        },
        "spread": {
            "bid": price_context["bid"],
            "ask": price_context["ask"],
            "spread_abs": price_context["spread_abs"],
            "spread_pct": price_context["spread_pct"],
        },
        "technical": technical,
        "market": market_payload,
        "charts": {
            "1min": inputs.m1_points,
            "5min": inputs.m5_points,
            "1day": inputs.day_points,
        },
        "support_status": {
            **support_status_payload(),
        },
        "source": f"{provider}-live",
        "source_detail": source_detail,
    }


def build_price_context(
    *,
    quote: dict[str, Any],
    day_points: list[dict[str, Any]],
    m1_points: list[dict[str, Any]],
    pick_float: Callable[..., float | None],
    series_source_descriptor: SeriesSourceDescriptor,
    extract_latest_session_points: LatestSessionPoints,
) -> dict[str, Any]:
    latest_day = day_points[-1]
    previous_day = day_points[-2] if len(day_points) >= 2 else None
    day_series_source = series_source_descriptor(day_points)
    quote_source_detail = quote.get("_source_detail") if isinstance(quote, dict) else {}
    if not isinstance(quote_source_detail, dict):
        quote_source_detail = {}

    current_price = pick_float(quote, "close", "price")
    if current_price is None:
        current_price = latest_day["c"]
    previous_close = pick_float(quote, "previous_close", "prev_close")
    if previous_close is None and previous_day:
        previous_close = previous_day["c"]

    field_values = {
        "day_open": pick_float(quote, "open"),
        "day_high": pick_float(quote, "high"),
        "day_low": pick_float(quote, "low"),
        "day_volume": pick_float(quote, "volume"),
        "bid": pick_float(quote, "bid"),
        "ask": pick_float(quote, "ask"),
    }
    field_sources = {
        "day_open_source": quote_source_detail.get("open"),
        "day_high_source": quote_source_detail.get("high"),
        "day_low_source": quote_source_detail.get("low"),
        "day_volume_source": quote_source_detail.get("volume"),
        "current_price_source": quote_source_detail.get("close") or quote_source_detail.get("price"),
        "previous_close_source": quote_source_detail.get("previous_close") or quote_source_detail.get("prev_close"),
    }

    fill_day_fields_from_intraday(
        field_values=field_values,
        field_sources=field_sources,
        m1_points=m1_points,
        extract_latest_session_points=extract_latest_session_points,
    )
    fill_day_fields_from_daily_series(
        field_values=field_values,
        field_sources=field_sources,
        latest_day=latest_day,
        day_series_source=day_series_source,
    )

    change_abs, change_pct = compute_change_metrics(current=current_price, previous=previous_close)
    gap_abs, gap_pct = compute_change_metrics(current=field_values["day_open"], previous=previous_close)
    avg_volume_20, avg_volume_ratio, turnover = compute_volume_metrics(
        day_points=day_points,
        day_volume=field_values["day_volume"],
        current_price=current_price,
    )
    spread_abs, spread_pct = compute_spread_metrics(
        bid=field_values["bid"],
        ask=field_values["ask"],
        current_price=current_price,
    )

    return {
        "current_price": current_price,
        "previous_close": previous_close,
        "day_open": field_values["day_open"],
        "day_high": field_values["day_high"],
        "day_low": field_values["day_low"],
        "day_volume": field_values["day_volume"],
        "bid": field_values["bid"],
        "ask": field_values["ask"],
        "change_abs": change_abs,
        "change_pct": change_pct,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "avg_volume_20": avg_volume_20,
        "avg_volume_ratio": avg_volume_ratio,
        "turnover": turnover,
        "spread_abs": spread_abs,
        "spread_pct": spread_pct,
        **field_sources,
    }


def fill_day_fields_from_intraday(
    *,
    field_values: dict[str, float | None],
    field_sources: dict[str, str | None],
    m1_points: list[dict[str, Any]],
    extract_latest_session_points: LatestSessionPoints,
) -> None:
    if not m1_points:
        return
    latest_session = extract_latest_session_points(m1_points)
    if not latest_session:
        return

    if field_values["day_open"] is None:
        field_values["day_open"] = latest_session[0]["o"]
    if field_values["day_high"] is None:
        field_values["day_high"] = max((item["h"] for item in latest_session), default=None)
    if field_values["day_low"] is None:
        field_values["day_low"] = min((item["l"] for item in latest_session), default=None)
    if field_values["day_volume"] is None:
        field_values["day_volume"] = sum((item["v"] or 0.0) for item in latest_session)

    if not field_sources["day_open_source"]:
        field_sources["day_open_source"] = "intraday_1min"
    if not field_sources["day_high_source"]:
        field_sources["day_high_source"] = "intraday_1min"
    if not field_sources["day_low_source"]:
        field_sources["day_low_source"] = "intraday_1min"
    if not field_sources["day_volume_source"] and field_values["day_volume"] is not None:
        field_sources["day_volume_source"] = "intraday_1min"


def fill_day_fields_from_daily_series(
    *,
    field_values: dict[str, float | None],
    field_sources: dict[str, str | None],
    latest_day: dict[str, Any],
    day_series_source: str,
) -> None:
    fallback_source = f"daily_series({day_series_source})"
    if field_values["day_open"] is None:
        field_values["day_open"] = latest_day["o"]
        field_sources["day_open_source"] = fallback_source
    if field_values["day_high"] is None:
        field_values["day_high"] = latest_day["h"]
        field_sources["day_high_source"] = fallback_source
    if field_values["day_low"] is None:
        field_values["day_low"] = latest_day["l"]
        field_sources["day_low_source"] = fallback_source
    if field_values["day_volume"] is None:
        field_values["day_volume"] = latest_day["v"]
        field_sources["day_volume_source"] = fallback_source
    if not field_sources["current_price_source"]:
        field_sources["current_price_source"] = fallback_source
    if field_sources["previous_close_source"] is None:
        field_sources["previous_close_source"] = fallback_source


def compute_change_metrics(*, current: float | None, previous: float | None) -> tuple[float | None, float | None]:
    if current is None or previous is None or previous <= 0:
        return None, None
    change_abs = current - previous
    return change_abs, (change_abs / previous) * 100


def compute_volume_metrics(
    *,
    day_points: list[dict[str, Any]],
    day_volume: float | None,
    current_price: float | None,
) -> tuple[float | None, float | None, float | None]:
    recent_daily_volumes = [p["v"] for p in day_points[-21:-1] if p.get("v") is not None and p["v"] > 0]
    avg_volume_20 = sum(recent_daily_volumes) / len(recent_daily_volumes) if recent_daily_volumes else None
    avg_volume_ratio = (
        (day_volume / avg_volume_20)
        if day_volume is not None and avg_volume_20 is not None and avg_volume_20 > 0
        else None
    )
    turnover = current_price * day_volume if current_price is not None and day_volume is not None else None
    return avg_volume_20, avg_volume_ratio, turnover


def compute_spread_metrics(
    *,
    bid: float | None,
    ask: float | None,
    current_price: float | None,
) -> tuple[float | None, float | None]:
    spread_abs = ask - bid if ask is not None and bid is not None else None
    spread_pct = (
        (spread_abs / current_price) * 100
        if spread_abs is not None and current_price is not None and current_price > 0
        else None
    )
    return spread_abs, spread_pct


def build_market_section(
    *,
    day_points: list[dict[str, Any]],
    market_context: dict[str, Any] | None,
    include_market: bool,
    include_qqq: bool,
    beta_and_corr_60d: BetaAndCorr,
    build_market_item: BuildMarketItem,
) -> OverviewMarketSection:
    spy_points = market_context.get("spy_points", []) if isinstance(market_context, dict) else []
    qqq_points = market_context.get("qqq_points", []) if isinstance(market_context, dict) else []
    beta_60, corr_60 = beta_and_corr_60d(day_points, spy_points) if include_market else (None, None)
    spy_latest = spy_points[-1]["c"] if spy_points else None
    spy_prev = spy_points[-2]["c"] if len(spy_points) >= 2 else None
    qqq_latest = qqq_points[-1]["c"] if qqq_points else None
    qqq_prev = qqq_points[-2]["c"] if len(qqq_points) >= 2 else None
    return OverviewMarketSection(
        spy_points=spy_points,
        qqq_points=qqq_points,
        payload={
            "sp500_proxy": build_market_item("SPY", spy_latest, spy_prev),
            "nasdaq_proxy": build_market_item("QQQ", qqq_latest, qqq_prev) if include_qqq else None,
            "beta_60d_vs_spy": beta_60,
            "corr_60d_vs_spy": corr_60,
        },
    )
