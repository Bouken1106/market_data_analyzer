"""Provider historical-series payload normalization helpers."""

from __future__ import annotations

from typing import Any

from ..ohlcv import normalize_ohlcv_point
from .payload_records import payload_rows


def normalize_twelvedata_series_payload(payload: Any) -> list[dict[str, Any]]:
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []
    return _normalize_series_rows(
        values,
        timestamp_keys=("datetime",),
        source="twelvedata",
    )


def normalize_fmp_historical_payload(payload: Any, *, outputsize: int) -> list[dict[str, Any]]:
    points = _normalize_series_rows(
        payload_rows(payload, "historical", "data"),
        timestamp_keys=("date", "datetime"),
        source="fmp",
    )
    points.sort(key=lambda item: str(item.get("t", "")))
    if outputsize > 0 and len(points) > outputsize:
        return points[-outputsize:]
    return points


def _normalize_series_rows(
    rows: list[dict[str, Any]],
    *,
    timestamp_keys: tuple[str, ...],
    source: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in rows:
        point = normalize_ohlcv_point(
            item,
            timestamp_keys=timestamp_keys,
            open_keys=("open",),
            high_keys=("high",),
            low_keys=("low",),
            close_keys=("close",),
            volume_keys=("volume",),
            source=source,
        )
        if point is not None:
            points.append(point)
    return points
