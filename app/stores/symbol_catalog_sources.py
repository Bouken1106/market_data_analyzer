"""Provider fetchers and row helpers for the symbol catalog store."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

import httpx
from fastapi import HTTPException

from ..config import (
    FMP_STOCK_LIST_LEGACY_URL,
    FMP_STOCK_LIST_URL,
    LOGGER,
    STOCKS_LIST_URL,
)
from ..services.market_data_provider_clients import FmpClient, TwelveDataClient
from ..services.payload_records import payload_row_list
from ..utils import is_valid_symbol, normalize_symbol

CatalogRow = dict[str, str]


def _pick_catalog_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_country_key(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


class SymbolCatalogRowNormalizer:
    def __init__(self, *, max_items: int, country: str = "") -> None:
        self.max_items = max_items
        self.country = _normalize_country_key(country)

    def _normalize_catalog_symbol(self, symbol: Any, exchange: Any = "") -> str:
        normalized_symbol = normalize_symbol(symbol)
        if (
            self.country == "JAPAN"
            and normalized_symbol.isdigit()
            and len(normalized_symbol) in {4, 5}
        ):
            return f"{normalized_symbol}.T"
        normalized_exchange = str(exchange or "").strip().upper()
        if (
            normalized_exchange in {"JPX", "TSE", "TYO"}
            and normalized_symbol.isdigit()
            and len(normalized_symbol) in {4, 5}
        ):
            return f"{normalized_symbol}.T"
        return normalized_symbol

    def build_row(
        self,
        *,
        symbol: Any,
        name: Any = "",
        exchange: Any = "",
        security_type: Any = "",
    ) -> CatalogRow | None:
        normalized_symbol = self._normalize_catalog_symbol(symbol, exchange=exchange)
        if not is_valid_symbol(normalized_symbol):
            return None
        return {
            "symbol": normalized_symbol,
            "name": str(name or "").strip(),
            "exchange": str(exchange or "").strip(),
            "type": str(security_type or "").strip(),
        }

    def dedupe_sorted(self, rows: Iterable[CatalogRow]) -> list[CatalogRow]:
        deduped: dict[str, CatalogRow] = {}
        for row in rows:
            normalized = self.build_row(
                symbol=row.get("symbol"),
                name=row.get("name"),
                exchange=row.get("exchange"),
                security_type=row.get("type"),
            )
            if normalized is None:
                continue
            deduped[normalized["symbol"]] = normalized
        return [deduped[key] for key in sorted(deduped.keys())][: self.max_items]

    def merge(self, primary_rows: list[CatalogRow], secondary_rows: list[CatalogRow]) -> list[CatalogRow]:
        merged: dict[str, CatalogRow] = {}
        for row in self.dedupe_sorted(secondary_rows):
            merged[row["symbol"]] = row

        for row in primary_rows:
            normalized_row = self.build_row(
                symbol=row.get("symbol"),
                name=row.get("name"),
                exchange=row.get("exchange"),
                security_type=row.get("type"),
            )
            if normalized_row is None:
                continue
            symbol = normalized_row["symbol"]
            base = merged.get(symbol, {})
            normalized = self.build_row(
                symbol=symbol,
                name=normalized_row.get("name") or base.get("name"),
                exchange=normalized_row.get("exchange") or base.get("exchange"),
                security_type=normalized_row.get("type") or base.get("type"),
            )
            if normalized is None:
                continue
            merged[symbol] = normalized

        return [merged[key] for key in sorted(merged.keys())][: self.max_items]


class TwelveDataSymbolCatalogFetcher:
    def __init__(
        self,
        *,
        api_key: str,
        country: str,
        normalizer: SymbolCatalogRowNormalizer,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.country = country
        self.normalizer = normalizer

    async def fetch(self) -> list[CatalogRow]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = (
                await TwelveDataClient(client, self.api_key).get_symbol_catalog(
                    STOCKS_LIST_URL,
                    country=self.country,
                )
            ).payload

        if isinstance(payload, dict) and payload.get("status") == "error":
            message = payload.get("message", "Failed to fetch symbol catalog.")
            raise HTTPException(status_code=400, detail=message)

        rows = payload_row_list(payload, "data")
        if rows is None:
            raise HTTPException(status_code=502, detail="Unexpected symbol catalog format from Twelve Data.")

        normalized_rows: list[CatalogRow] = []
        for item in rows:
            row = self.normalizer.build_row(
                symbol=item.get("symbol"),
                name=_pick_catalog_text(
                    item.get("name"),
                    item.get("instrument_name"),
                    item.get("company_name"),
                    item.get("companyName"),
                    item.get("description"),
                ),
                exchange=item.get("exchange"),
                security_type=item.get("type"),
            )
            if row is not None:
                normalized_rows.append(row)
        return self.normalizer.dedupe_sorted(normalized_rows)


class FmpSymbolCatalogFetcher:
    def __init__(
        self,
        *,
        api_key: str,
        normalizer: SymbolCatalogRowNormalizer,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.normalizer = normalizer

    async def fetch(self) -> list[CatalogRow]:
        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            fmp_client = FmpClient(client, self.api_key)
            payload = await fmp_client.get_symbol_catalog(FMP_STOCK_LIST_URL)
            if self._is_fmp_error_payload(payload):
                payload = await fmp_client.get_symbol_catalog(FMP_STOCK_LIST_LEGACY_URL)

        rows = payload_row_list(payload, "data")
        if rows is None:
            raise HTTPException(status_code=502, detail="Unexpected symbol catalog format from FMP.")

        normalized_rows: list[CatalogRow] = []
        for item in rows:
            row = self.normalizer.build_row(
                symbol=item.get("symbol"),
                name=_pick_catalog_text(
                    item.get("name"),
                    item.get("companyName"),
                    item.get("company_name"),
                    item.get("description"),
                ),
                exchange=item.get("exchangeShortName") or item.get("exchange"),
                security_type=item.get("type"),
            )
            if row is None or not self._is_us_equity_exchange(row["exchange"]):
                continue
            normalized_rows.append(row)
        return self.normalizer.dedupe_sorted(normalized_rows)

    @staticmethod
    def _is_fmp_error_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("status") == "error":
            return True
        message = str(payload.get("Error Message", "")).strip().lower()
        return bool(message)

    @staticmethod
    def _is_us_equity_exchange(exchange: str) -> bool:
        code = str(exchange or "").strip().upper()
        if not code:
            return False
        return code in {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}


class CombinedSymbolCatalogFetcher:
    def __init__(
        self,
        *,
        primary_name: str,
        primary_fetcher: TwelveDataSymbolCatalogFetcher | FmpSymbolCatalogFetcher,
        secondary_name: str,
        secondary_fetcher: TwelveDataSymbolCatalogFetcher | FmpSymbolCatalogFetcher,
        normalizer: SymbolCatalogRowNormalizer,
    ) -> None:
        self.primary_name = primary_name
        self.primary_fetcher = primary_fetcher
        self.secondary_name = secondary_name
        self.secondary_fetcher = secondary_fetcher
        self.normalizer = normalizer

    async def fetch(self) -> list[CatalogRow]:
        primary_result, secondary_result = await asyncio.gather(
            self.primary_fetcher.fetch(),
            self.secondary_fetcher.fetch(),
            return_exceptions=True,
        )
        if isinstance(primary_result, Exception):
            LOGGER.warning("Symbol catalog fetch failed (%s): %s", self.primary_name, primary_result)
        if isinstance(secondary_result, Exception):
            LOGGER.warning("Symbol catalog fetch failed (%s): %s", self.secondary_name, secondary_result)
        if isinstance(primary_result, Exception) and isinstance(secondary_result, Exception):
            if isinstance(primary_result, HTTPException):
                raise primary_result
            if isinstance(secondary_result, HTTPException):
                raise secondary_result
            raise HTTPException(status_code=502, detail="Failed to fetch symbol catalog from both providers.")

        primary_rows = primary_result if isinstance(primary_result, list) else []
        secondary_rows = secondary_result if isinstance(secondary_result, list) else []
        merged = self.normalizer.merge(primary_rows, secondary_rows)
        if merged:
            return merged
        raise HTTPException(status_code=502, detail="Failed to fetch symbol catalog from both providers.")


def build_symbol_catalog_fetcher(
    *,
    provider: str,
    twelvedata_api_key: str,
    fmp_api_key: str,
    country: str,
    max_items: int,
) -> TwelveDataSymbolCatalogFetcher | FmpSymbolCatalogFetcher | CombinedSymbolCatalogFetcher:
    normalizer = SymbolCatalogRowNormalizer(max_items=max_items, country=country)
    td_fetcher = TwelveDataSymbolCatalogFetcher(
        api_key=twelvedata_api_key,
        country=country,
        normalizer=normalizer,
    )
    fmp_fetcher = FmpSymbolCatalogFetcher(
        api_key=fmp_api_key,
        normalizer=normalizer,
    )
    resolved_provider = str(provider or "twelvedata").strip().lower()
    normalized_country = " ".join(str(country or "").strip().lower().split())
    is_us_catalog = normalized_country in {"united states", "united states of america", "us", "usa"}
    if not is_us_catalog:
        return td_fetcher
    if resolved_provider == "both":
        return CombinedSymbolCatalogFetcher(
            primary_name="TD",
            primary_fetcher=td_fetcher,
            secondary_name="FMP",
            secondary_fetcher=fmp_fetcher,
            normalizer=normalizer,
        )
    if resolved_provider == "fmp":
        return fmp_fetcher
    return td_fetcher
