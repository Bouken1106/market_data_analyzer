"""Helpers for normalizing and merging OHLCV point collections."""

from __future__ import annotations

from typing import Any, Iterable


def _pick_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _pick_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        try:
            return float(payload.get(key))
        except (TypeError, ValueError):
            continue
    return None


def normalize_ohlcv_point(
    payload: dict[str, Any],
    *,
    timestamp_keys: tuple[str, ...] = ("t", "datetime", "date"),
    open_keys: tuple[str, ...] = ("o", "open"),
    high_keys: tuple[str, ...] = ("h", "high"),
    low_keys: tuple[str, ...] = ("l", "low"),
    close_keys: tuple[str, ...] = ("c", "close"),
    volume_keys: tuple[str, ...] = ("v", "volume"),
    source: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    timestamp = _pick_text(payload, timestamp_keys)
    close_value = _pick_float(payload, close_keys)
    if not timestamp or close_value is None or close_value <= 0:
        return None

    open_value = _pick_float(payload, open_keys)
    high_value = _pick_float(payload, high_keys)
    low_value = _pick_float(payload, low_keys)
    volume_value = _pick_float(payload, volume_keys)

    if open_value is None or open_value <= 0:
        open_value = close_value
    if high_value is None or high_value <= 0:
        high_value = max(open_value, close_value)
    if low_value is None or low_value <= 0:
        low_value = min(open_value, close_value)

    point = {
        "t": timestamp,
        "o": open_value,
        "h": max(high_value, open_value, close_value),
        "l": min(low_value, open_value, close_value),
        "c": close_value,
        "v": volume_value,
    }
    resolved_source = str(source or payload.get("_src") or "").strip()
    if resolved_source:
        point["_src"] = resolved_source
    return point


def normalize_ohlcv_points(
    points: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in points:
        point = normalize_ohlcv_point(item, **kwargs)
        if point is not None:
            normalized.append(point)
    return normalized


def merge_points_by_timestamp(*point_groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for points in point_groups:
        for item in points:
            if not isinstance(item, dict):
                continue
            timestamp = str(item.get("t") or "").strip()
            if not timestamp:
                continue
            merged[timestamp] = item
    if not merged:
        return []
    return [merged[key] for key in sorted(merged)]


def latest_session_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []
    latest_date = str(points[-1].get("t") or "").split(" ")[0]
    if not latest_date:
        return points

    start_idx = len(points) - 1
    while start_idx > 0:
        prior_date = str(points[start_idx - 1].get("t") or "").split(" ")[0]
        if prior_date != latest_date:
            break
        start_idx -= 1
    return points[start_idx:]
