"""Persistent store for named saved portfolio snapshots."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import LOGGER
from ..services.portfolio_analysis import normalize_region_holdings
from ..utils import read_json_file, utc_now_iso, write_json_file

_MAX_SAVED_PORTFOLIOS = 50
_MAX_PORTFOLIO_NAME_LEN = 120


class PortfolioAnalysisStore:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._lock = asyncio.Lock()
        self._state = self._load_from_disk()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "portfolios": [],
            "updated_at": utc_now_iso(),
        }

    def _snapshot_state_no_lock(self) -> dict[str, Any]:
        return {
            "portfolios": [self._snapshot_portfolio(item) for item in self._state["portfolios"]],
            "updated_at": str(self._state["updated_at"]),
        }

    @staticmethod
    def _snapshot_portfolio(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "portfolio_id": str(item["portfolio_id"]),
            "name": str(item["name"]),
            "jp_holdings": [
                {
                    "symbol": str(holding["symbol"]),
                    "quantity": float(holding["quantity"]),
                }
                for holding in item["jp_holdings"]
            ],
            "us_holdings": [
                {
                    "symbol": str(holding["symbol"]),
                    "quantity": float(holding["quantity"]),
                }
                for holding in item["us_holdings"]
            ],
            "created_at": str(item["created_at"]),
            "updated_at": str(item["updated_at"]),
        }

    def _load_from_disk(self) -> dict[str, Any]:
        payload = read_json_file(self.cache_path)
        if not isinstance(payload, dict):
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
        return {
            "portfolios": portfolios[:_MAX_SAVED_PORTFOLIOS],
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

    def _write_no_lock(self) -> None:
        try:
            write_json_file(self.cache_path, self._snapshot_state_no_lock())
        except Exception as exc:
            LOGGER.warning("Failed to write portfolio analysis cache: %s", exc)

    async def get_state(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_state_no_lock()

    async def list_portfolios(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._snapshot_portfolio(item) for item in self._state["portfolios"]]

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
                self._state["updated_at"] = utc_now_iso()
                self._write_no_lock()
            return deleted
