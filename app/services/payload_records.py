"""Helpers for extracting dictionary records from provider payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def record_list(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def payload_row_list(payload: Any, *row_keys: str) -> list[dict[str, Any]] | None:
    """Return rows when the payload shape explicitly contains a row list.

    Unlike ``payload_rows``, this preserves the difference between an empty
    provider row list and an unrecognized provider payload.
    """

    if isinstance(payload, list):
        return record_list(payload)
    if not isinstance(payload, dict):
        return None
    for key in row_keys:
        rows = payload.get(key)
        if isinstance(rows, list):
            return record_list(rows)
    return None


def first_record(
    payload: Any,
    *,
    row_keys: Iterable[str] = ("data",),
    allow_direct_dict: bool = True,
    direct_dict_predicate: Callable[[dict[str, Any]], bool] | None = None,
    prefer_direct_dict: bool = False,
) -> dict[str, Any]:
    if isinstance(payload, list):
        rows = record_list(payload)
        return rows[0] if rows else {}
    if not isinstance(payload, dict):
        return {}

    if prefer_direct_dict and _can_use_direct_dict(payload, allow_direct_dict, direct_dict_predicate):
        return dict(payload)

    for key in row_keys:
        rows = record_list(payload.get(key))
        if rows:
            return rows[0]

    if _can_use_direct_dict(payload, allow_direct_dict, direct_dict_predicate):
        return dict(payload)
    return {}


def payload_rows(payload: Any, *row_keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return record_list(payload)
    if not isinstance(payload, dict):
        return []
    for key in row_keys:
        rows = record_list(payload.get(key))
        if rows:
            return rows
    return []


def _can_use_direct_dict(
    payload: dict[str, Any],
    allow_direct_dict: bool,
    predicate: Callable[[dict[str, Any]], bool] | None,
) -> bool:
    if not allow_direct_dict:
        return False
    return predicate(payload) if predicate is not None else True
