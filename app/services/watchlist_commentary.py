"""Watchlist commentary generation for the market API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ..config import (
    LMSTUDIO_API_KEY,
    LMSTUDIO_CHAT_COMPLETIONS_URL,
    LMSTUDIO_MODEL,
    LMSTUDIO_TIMEOUT_SEC,
)
from ..utils import utc_now_iso
from .watchlist_commentary_support import (
    WATCHLIST_RESPONSE_FORMAT,
    build_base_messages,
    build_repair_messages,
    build_watchlist_prompt,
    chat_lmstudio,
    commentary_from_json,
    compute_watch_metrics,
    fallback_commentary,
    metrics_payload,
)


async def _request_lmstudio_commentary(prompt: str, valid_symbols: list[str]) -> tuple[str, str]:
    base_messages = build_base_messages(prompt)
    raw_commentary, status_code, error_detail, used_model = await chat_lmstudio(
        api_url=LMSTUDIO_CHAT_COMPLETIONS_URL,
        api_key=LMSTUDIO_API_KEY,
        model=LMSTUDIO_MODEL,
        timeout_sec=LMSTUDIO_TIMEOUT_SEC,
        messages=base_messages,
        max_tokens=320,
        response_format=WATCHLIST_RESPONSE_FORMAT,
    )
    if status_code >= 400:
        if status_code in {400, 404, 422}:
            raw_commentary, status_code, error_detail, used_model = await chat_lmstudio(
                api_url=LMSTUDIO_CHAT_COMPLETIONS_URL,
                api_key=LMSTUDIO_API_KEY,
                model=LMSTUDIO_MODEL,
                timeout_sec=LMSTUDIO_TIMEOUT_SEC,
                messages=base_messages,
                max_tokens=320,
                response_format=None,
            )
        if status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"LM Studio error: {error_detail or f'HTTP {status_code}'}",
            )

    if not raw_commentary:
        raise HTTPException(status_code=502, detail="LM Studio returned an empty commentary.")

    commentary = commentary_from_json(raw_commentary, valid_symbols)
    if commentary:
        return commentary, used_model

    repair_messages = build_repair_messages(raw_commentary, valid_symbols)
    repaired_commentary, repair_status, repair_error, repair_model = await chat_lmstudio(
        api_url=LMSTUDIO_CHAT_COMPLETIONS_URL,
        api_key=LMSTUDIO_API_KEY,
        model=LMSTUDIO_MODEL,
        timeout_sec=LMSTUDIO_TIMEOUT_SEC,
        messages=repair_messages,
        max_tokens=220,
        response_format=WATCHLIST_RESPONSE_FORMAT,
    )
    if repair_status >= 400 and repair_status in {400, 404, 422}:
        repaired_commentary, repair_status, repair_error, repair_model = await chat_lmstudio(
            api_url=LMSTUDIO_CHAT_COMPLETIONS_URL,
            api_key=LMSTUDIO_API_KEY,
            model=LMSTUDIO_MODEL,
            timeout_sec=LMSTUDIO_TIMEOUT_SEC,
            messages=repair_messages,
            max_tokens=220,
            response_format=None,
        )
    if repair_status < 400:
        repaired = commentary_from_json(repaired_commentary, valid_symbols)
        if repaired:
            return repaired, repair_model

    if repair_status >= 400 and not raw_commentary:
        raise HTTPException(
            status_code=502,
            detail=f"LM Studio error: {repair_error or f'HTTP {repair_status}'}",
        )

    return fallback_commentary(raw_commentary, valid_symbols), used_model


async def build_watchlist_commentary_payload(
    hub: Any,
    symbols: list[str],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    sparkline_items = await hub.sparkline_payload(symbols, refresh=refresh)
    items_by_symbol: dict[str, dict[str, Any]] = {}
    for item in sparkline_items:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        items_by_symbol[symbol] = item

    metrics = [compute_watch_metrics(symbol, items_by_symbol.get(symbol)) for symbol in symbols]
    current_date = datetime.now(timezone.utc).astimezone().date().isoformat()
    prompt = build_watchlist_prompt(current_date=current_date, metrics=metrics)
    commentary, used_model = await _request_lmstudio_commentary(prompt, symbols)

    return {
        "symbols": symbols,
        "current_date": current_date,
        "model": used_model,
        "generated_at": utc_now_iso(),
        "comment": commentary,
        "prompt": prompt,
        "metrics": metrics_payload(metrics),
    }


__all__ = ["build_watchlist_commentary_payload"]
