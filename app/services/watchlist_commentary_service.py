"""High-level watchlist commentary orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ..config import settings
from ..utils import utc_now_iso
from .lmstudio_client import LmStudioClient
from .watchlist_commentary_metrics import compute_watch_metrics, metrics_payload
from .watchlist_commentary_parser import commentary_from_json, fallback_commentary
from .watchlist_commentary_prompt import (
    WATCHLIST_RESPONSE_FORMAT,
    build_base_messages,
    build_repair_messages,
    build_watchlist_prompt,
)


class WatchlistCommentaryService:
    def __init__(self) -> None:
        self.client = LmStudioClient(
            api_url=settings.lmstudio.lmstudio_chat_completions_url,
            api_key=settings.lmstudio.lmstudio_api_key,
            model=settings.lmstudio.lmstudio_model,
            timeout_sec=settings.lmstudio.lmstudio_timeout_sec,
        )

    async def build_payload(
        self,
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
        commentary, used_model = await self._request_commentary(prompt, symbols)

        return {
            "symbols": symbols,
            "current_date": current_date,
            "model": used_model,
            "generated_at": utc_now_iso(),
            "comment": commentary,
            "prompt": prompt,
            "metrics": metrics_payload(metrics),
        }

    async def _request_commentary(self, prompt: str, valid_symbols: list[str]) -> tuple[str, str]:
        base_messages = build_base_messages(prompt)
        initial_result = await self.client.chat(
            messages=base_messages,
            max_tokens=320,
            response_format=WATCHLIST_RESPONSE_FORMAT,
        )
        raw_commentary = initial_result.content
        used_model = initial_result.model
        if initial_result.status_code >= 400:
            fallback_result = initial_result
            if initial_result.status_code in {400, 404, 422}:
                fallback_result = await self.client.chat(
                    messages=base_messages,
                    max_tokens=320,
                    response_format=None,
                )
                raw_commentary = fallback_result.content
                used_model = fallback_result.model
            if fallback_result.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"LM Studio error: {fallback_result.error_detail or f'HTTP {fallback_result.status_code}'}",
                )

        if not raw_commentary:
            raise HTTPException(status_code=502, detail="LM Studio returned an empty commentary.")

        commentary = commentary_from_json(raw_commentary, valid_symbols)
        if commentary:
            return commentary, used_model

        repair_messages = build_repair_messages(raw_commentary, valid_symbols)
        repair_result = await self.client.chat(
            messages=repair_messages,
            max_tokens=220,
            response_format=WATCHLIST_RESPONSE_FORMAT,
        )
        if repair_result.status_code >= 400 and repair_result.status_code in {400, 404, 422}:
            repair_result = await self.client.chat(
                messages=repair_messages,
                max_tokens=220,
                response_format=None,
            )
        if repair_result.status_code < 400:
            repaired = commentary_from_json(repair_result.content, valid_symbols)
            if repaired:
                return repaired, repair_result.model

        if repair_result.status_code >= 400 and not raw_commentary:
            raise HTTPException(
                status_code=502,
                detail=f"LM Studio error: {repair_result.error_detail or f'HTTP {repair_result.status_code}'}",
            )

        return fallback_commentary(raw_commentary, valid_symbols), used_model
