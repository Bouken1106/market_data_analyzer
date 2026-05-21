"""Persistent store for named saved portfolio snapshots."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..services.portfolio_analysis import MAX_HOLDINGS_PER_REGION, normalize_region_holdings
from ..utils import utc_now_iso
from .json_state import JsonStateStore

_MAX_SAVED_PORTFOLIOS = 50
_MAX_PORTFOLIO_NAME_LEN = 120
_MAX_DRAFT_SYMBOL_LEN = 120
_MAX_DRAFT_QUANTITY_LEN = 40
_DEFAULT_LOOKBACK_DAYS = 252


class PortfolioAnalysisStore(JsonStateStore):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path, log_label="portfolio analysis cache")
        self._lock = asyncio.Lock()
        self._state = self._load_from_disk()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "portfolios": [],
            "draft": None,
            "updated_at": utc_now_iso(),
        }

    def _snapshot_state_no_lock(self) -> dict[str, Any]:
        return {
            "portfolios": [self._snapshot_portfolio(item) for item in self._state["portfolios"]],
            "draft": self._snapshot_draft(self._state["draft"]) if self._state["draft"] else None,
            "updated_at": str(self._state["updated_at"]),
        }

    @staticmethod
    def _snapshot_portfolio(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "portfolio_id": str(item["portfolio_id"]),
            "name": str(item["name"]),
            "jp_holdings": PortfolioAnalysisStore._snapshot_holdings(item["jp_holdings"]),
            "us_holdings": PortfolioAnalysisStore._snapshot_holdings(item["us_holdings"]),
            "created_at": str(item["created_at"]),
            "updated_at": str(item["updated_at"]),
        }

    @staticmethod
    def _snapshot_holdings(holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "symbol": str(holding["symbol"]),
                "quantity": float(holding["quantity"]),
            }
            for holding in holdings
        ]

    @staticmethod
    def _snapshot_draft(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "portfolio_id": str(item["portfolio_id"]),
            "name": str(item["name"]),
            "lookback_days": int(item["lookback_days"]),
            "jp_rows": PortfolioAnalysisStore._snapshot_draft_rows(item["jp_rows"]),
            "us_rows": PortfolioAnalysisStore._snapshot_draft_rows(item["us_rows"]),
            "updated_at": str(item["updated_at"]),
        }

    @staticmethod
    def _snapshot_draft_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "symbol": str(row["symbol"]),
                "quantity": str(row["quantity"]),
            }
            for row in rows
        ]

    def _load_from_disk(self) -> dict[str, Any]:
        payload = self._read_state_dict()
        if payload is None:
            return self._empty_state()

        portfolios_raw = payload.get("portfolios")
        if not isinstance(portfolios_raw, list):
            return self._empty_state()

        portfolios: list[dict[str, Any]] = []
        for item in portfolios_raw:
            normalized = self._normalize_portfolio(item)
            if normalized is not None:
                portfolios.append(normalized)
        portfolios.sort(key=lambda item: str(item["updated_at"]), reverse=True)
        draft = self._normalize_draft(payload.get("draft"))
        return {
            "portfolios": portfolios[:_MAX_SAVED_PORTFOLIOS],
            "draft": draft,
            "updated_at": str(payload.get("updated_at") or utc_now_iso()),
        }

    def _normalize_portfolio(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None

        portfolio_id = str(raw.get("portfolio_id") or "").strip()
        if not portfolio_id:
            portfolio_id = uuid4().hex

        name = self._normalize_name(raw.get("name"))
        if not name:
            return None

        try:
            jp_holdings = normalize_region_holdings(raw.get("jp_holdings"), region="jp")
            us_holdings = normalize_region_holdings(raw.get("us_holdings"), region="us")
        except ValueError:
            return None

        created_at = str(raw.get("created_at") or utc_now_iso())
        updated_at = str(raw.get("updated_at") or created_at)
        return {
            "portfolio_id": portfolio_id,
            "name": name,
            "jp_holdings": jp_holdings,
            "us_holdings": us_holdings,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    @staticmethod
    def _normalize_name(value: Any) -> str:
        name = " ".join(str(value or "").strip().split())
        return name[:_MAX_PORTFOLIO_NAME_LEN]

    @staticmethod
    def _normalize_lookback_days(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_LOOKBACK_DAYS
        return parsed if parsed > 0 else _DEFAULT_LOOKBACK_DAYS

    @staticmethod
    def _normalize_draft_row(raw: Any) -> dict[str, str] | None:
        if not isinstance(raw, dict):
            return None
        symbol = " ".join(str(raw.get("symbol") or "").strip().split())[:_MAX_DRAFT_SYMBOL_LEN]
        quantity = str(raw.get("quantity") or "").strip()[:_MAX_DRAFT_QUANTITY_LEN]
        if not symbol and not quantity:
            return None
        return {
            "symbol": symbol,
            "quantity": quantity,
        }

    @classmethod
    def _normalize_draft_rows(cls, raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        rows: list[dict[str, str]] = []
        for item in raw[:MAX_HOLDINGS_PER_REGION]:
            normalized_row = cls._normalize_draft_row(item)
            if normalized_row is not None:
                rows.append(normalized_row)
        return rows

    def _normalize_draft(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None

        portfolio_id = str(raw.get("portfolio_id") or "").strip()
        name = self._normalize_name(raw.get("name"))
        lookback_days = self._normalize_lookback_days(raw.get("lookback_days"))

        jp_rows_raw = raw.get("jp_rows")
        if not isinstance(jp_rows_raw, list):
            jp_rows_raw = raw.get("jp_holdings")
        us_rows_raw = raw.get("us_rows")
        if not isinstance(us_rows_raw, list):
            us_rows_raw = raw.get("us_holdings")

        jp_rows = self._normalize_draft_rows(jp_rows_raw)
        us_rows = self._normalize_draft_rows(us_rows_raw)

        if not portfolio_id and not name and not jp_rows and not us_rows and lookback_days == _DEFAULT_LOOKBACK_DAYS:
            return None

        return {
            "portfolio_id": portfolio_id,
            "name": name,
            "lookback_days": lookback_days,
            "jp_rows": jp_rows,
            "us_rows": us_rows,
            "updated_at": str(raw.get("updated_at") or utc_now_iso()),
        }

    def _write_no_lock(self) -> None:
        self._write_state(self._snapshot_state_no_lock())

    async def get_state(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_state_no_lock()

    async def list_portfolios(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._snapshot_portfolio(item) for item in self._state["portfolios"]]

    async def get_draft(self) -> dict[str, Any] | None:
        async with self._lock:
            draft = self._state.get("draft")
            return self._snapshot_draft(draft) if isinstance(draft, dict) else None

    async def save_draft(
        self,
        *,
        portfolio_id: str | None,
        name: str,
        lookback_days: int | None,
        jp_rows: list[dict[str, Any]],
        us_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized_draft = self._normalize_draft(
            {
                "portfolio_id": portfolio_id,
                "name": name,
                "lookback_days": lookback_days,
                "jp_rows": jp_rows,
                "us_rows": us_rows,
            }
        )
        timestamp = utc_now_iso()

        async with self._lock:
            self._state["draft"] = None
            if normalized_draft is not None:
                normalized_draft["updated_at"] = timestamp
                self._state["draft"] = normalized_draft
            self._state["updated_at"] = timestamp
            self._write_no_lock()
            draft = self._state.get("draft")
            return self._snapshot_draft(draft) if isinstance(draft, dict) else None

    async def clear_draft(self) -> None:
        async with self._lock:
            if self._state.get("draft") is None:
                return
            self._state["draft"] = None
            self._state["updated_at"] = utc_now_iso()
            self._write_no_lock()

    async def save_portfolio(
        self,
        *,
        portfolio_id: str | None,
        name: str,
        jp_holdings: list[dict[str, Any]],
        us_holdings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_name = self._normalize_name(name)
        if not normalized_name:
            raise ValueError("Portfolio name is required.")

        normalized_jp = normalize_region_holdings(jp_holdings, region="jp")
        normalized_us = normalize_region_holdings(us_holdings, region="us")
        timestamp = utc_now_iso()
        normalized_id = str(portfolio_id or "").strip()

        async with self._lock:
            portfolios = self._state["portfolios"]
            for item in portfolios:
                if normalized_id and item["portfolio_id"] == normalized_id:
                    item["name"] = normalized_name
                    item["jp_holdings"] = normalized_jp
                    item["us_holdings"] = normalized_us
                    item["updated_at"] = timestamp
                    portfolios.sort(key=lambda row: str(row["updated_at"]), reverse=True)
                    self._state["updated_at"] = timestamp
                    self._write_no_lock()
                    return self._snapshot_portfolio(item)

            created = {
                "portfolio_id": normalized_id or uuid4().hex,
                "name": normalized_name,
                "jp_holdings": normalized_jp,
                "us_holdings": normalized_us,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            portfolios.insert(0, created)
            if len(portfolios) > _MAX_SAVED_PORTFOLIOS:
                del portfolios[_MAX_SAVED_PORTFOLIOS:]
            self._state["updated_at"] = timestamp
            self._write_no_lock()
            return self._snapshot_portfolio(created)

    async def delete_portfolio(self, portfolio_id: str) -> bool:
        normalized_id = str(portfolio_id or "").strip()
        if not normalized_id:
            return False

        async with self._lock:
            portfolios = self._state["portfolios"]
            original_count = len(portfolios)
            portfolios[:] = [item for item in portfolios if item["portfolio_id"] != normalized_id]
            deleted = len(portfolios) != original_count
            if deleted:
                draft = self._state.get("draft")
                if isinstance(draft, dict) and str(draft.get("portfolio_id") or "").strip() == normalized_id:
                    self._state["draft"] = None
                self._state["updated_at"] = utc_now_iso()
                self._write_no_lock()
            return deleted
