"""Shared numeric and payload helpers for valuation services."""

from __future__ import annotations

import math
from typing import Any


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", ".", "None", "null", "NaN"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def positive_float(value: Any) -> float | None:
    numeric = parse_float(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def non_negative_float(value: Any) -> float | None:
    numeric = parse_float(value)
    if numeric is None or numeric < 0:
        return None
    return numeric


def abs_or_none(value: Any) -> float | None:
    parsed = parse_float(value)
    return abs(parsed) if parsed is not None else None


def positive_abs(value: Any) -> float | None:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return abs(numeric) if numeric != 0 else None


def sum_optional(*values: Any) -> float | None:
    parsed = [parse_float(value) for value in values]
    cleaned = [value for value in parsed if value is not None]
    if not cleaned:
        return None
    return sum(cleaned)


def sub_optional(left: Any, right: Any) -> float | None:
    left_value = parse_float(left)
    right_value = parse_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def div_optional(left: Any, right: Any) -> float | None:
    left_value = parse_float(left)
    right_value = parse_float(right)
    if left_value is None or right_value in (None, 0):
        return None
    return left_value / right_value


def positive_div(left: Any, right: Any) -> float | None:
    value = div_optional(left, right)
    return value if value is not None and value > 0 else None


def mul_optional(left: Any, right: Any) -> float | None:
    left_value = parse_float(left)
    right_value = parse_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value * right_value


def positive_mul(left: Any, right: Any) -> float | None:
    value = mul_optional(left, right)
    return value if value is not None and value > 0 else None


def first_positive(*values: Any) -> float | None:
    for value in values:
        numeric = positive_float(value)
        if numeric is not None:
            return numeric
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if has_value(value):
            return value
    return None


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, tuple):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    return True


def dict_at(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def path_float(payload: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return parse_float(value)


def first_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return dict(item)
        return {}
    if isinstance(payload, dict):
        rows = payload.get("data")
        if isinstance(rows, list):
            return first_dict(rows)
        return dict(payload)
    return {}


def first_report(payload: Any, key: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        rows = payload.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return dict(rows[0])
    return {}


def first_present_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


def text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def payload_source(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    return str(payload.get("source") or payload.get("_cache_source") or "payload").strip() or "payload"
