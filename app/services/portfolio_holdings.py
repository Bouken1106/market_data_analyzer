"""Portfolio holding normalization and symbol-resolution helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from ..utils import finite_float_or_none, is_valid_symbol, normalize_symbol

MAX_HOLDINGS_PER_REGION = 50

REGION_LABELS = {
    "jp": "日本株",
    "us": "米国株",
}

REGION_CURRENCIES = {
    "jp": {"code": "JPY", "symbol": "¥"},
    "us": {"code": "USD", "symbol": "$"},
}

REGION_CATALOG_COUNTRIES = {
    "jp": "Japan",
    "us": "United States",
}

_JP_NUMERIC_SYMBOL_RE = re.compile(r"^\d{4,5}$")
_SEARCH_TOKEN_RE = re.compile(r"[A-Z0-9]+")
_COMPANY_SUFFIX_WORDS = frozenset(
    {
        "AND",
        "THE",
        "CO",
        "COMPANY",
        "CORP",
        "CORPORATION",
        "INC",
        "INCORPORATED",
        "LTD",
        "LIMITED",
        "HOLDINGS",
        "HOLDING",
        "GROUP",
        "PLC",
        "NV",
        "AG",
        "SA",
        "SE",
        "CLASS",
        "SERIES",
        "SHARES",
        "SHARE",
        "STOCK",
        "PREFERRED",
        "PREF",
        "REIT",
        "ETF",
        "TRUST",
        "FUND",
    }
)


def normalize_region(region: str) -> str:
    normalized = str(region or "").strip().lower()
    if normalized not in REGION_LABELS:
        raise ValueError("Unsupported portfolio region.")
    return normalized


def normalize_region_symbol(raw: Any, *, region: str) -> str:
    normalized_region = normalize_region(region)
    symbol = normalize_symbol(raw)
    if normalized_region == "jp" and _JP_NUMERIC_SYMBOL_RE.fullmatch(symbol):
        symbol = f"{symbol}.T"
    if not is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {raw!r}")
    return symbol


def _holding_mapping(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return item if isinstance(item, dict) else {}


def _iter_holding_inputs(raw: Any) -> list[tuple[str, Any]]:
    items = raw if isinstance(raw, list) else []
    inputs: list[tuple[str, Any]] = []
    for item in items:
        mapping = _holding_mapping(item)
        raw_symbol = str(mapping.get("symbol") or "").strip()
        raw_quantity = mapping.get("quantity")
        if not raw_symbol and (raw_quantity is None or str(raw_quantity).strip() == ""):
            continue
        inputs.append((raw_symbol, raw_quantity))
    return inputs


def _finalize_aggregated_holdings(aggregated: dict[str, float]) -> list[dict[str, float]]:
    holdings = [
        {
            "symbol": symbol,
            "quantity": quantity,
        }
        for symbol, quantity in aggregated.items()
        if quantity > 0
    ]
    if len(holdings) > MAX_HOLDINGS_PER_REGION:
        raise ValueError(f"You can save up to {MAX_HOLDINGS_PER_REGION} holdings per region.")
    holdings.sort(key=lambda item: item["symbol"])
    return holdings


def _aggregate_region_holdings(
    inputs: list[tuple[str, Any]],
    *,
    resolve_symbol: Callable[[str], str],
) -> list[dict[str, float]]:
    aggregated: dict[str, float] = {}
    for raw_symbol, raw_quantity in inputs:
        symbol = resolve_symbol(raw_symbol)
        quantity = finite_float_or_none(raw_quantity, minimum=0.0, strict_minimum=True)
        if quantity is None:
            raise ValueError(f"Quantity must be greater than 0 for {symbol or raw_symbol or 'holding'}.")
        aggregated[symbol] = aggregated.get(symbol, 0.0) + float(quantity)
    return _finalize_aggregated_holdings(aggregated)


def _normalize_search_text(raw: Any) -> str:
    text = unicodedata.normalize("NFKC", str(raw or ""))
    return " ".join(text.upper().strip().split())


def _search_tokens(raw: Any) -> list[str]:
    return _SEARCH_TOKEN_RE.findall(_normalize_search_text(raw))


def _build_name_initialism(name: Any) -> str:
    initials: list[str] = []
    for token in _search_tokens(name):
        if token in _COMPANY_SUFFIX_WORDS:
            continue
        initials.append(token[0])
    return "".join(initials)


def _rank_catalog_candidate(query: str, *, candidate_symbol: str, candidate_name: str) -> int | None:
    needle = _normalize_search_text(query).replace(" ", "")
    if not needle:
        return None

    symbol_text = _normalize_search_text(candidate_symbol).replace(" ", "")
    name_text = _normalize_search_text(candidate_name).replace(" ", "")
    initialism = _build_name_initialism(candidate_name)

    if symbol_text == needle:
        return 0
    if name_text == needle:
        return 1
    if initialism == needle and initialism:
        return 2
    if symbol_text.startswith(needle):
        return 3
    if name_text.startswith(needle):
        return 4
    if initialism.startswith(needle) and initialism:
        return 5
    if symbol_text.find(needle) >= 0:
        return 6
    if name_text.find(needle) >= 0:
        return 7
    return None


def _catalog_row_symbol(item: dict[str, Any], *, region: str) -> str | None:
    raw_symbol = item.get("symbol")
    try:
        return normalize_region_symbol(raw_symbol, region=region)
    except ValueError:
        return None


async def _load_region_catalog(symbol_catalog_store: Any, *, region: str) -> list[dict[str, Any]]:
    if symbol_catalog_store is None:
        return []
    normalized_region = normalize_region(region)
    country = REGION_CATALOG_COUNTRIES.get(normalized_region)
    try:
        payload = await symbol_catalog_store.get_catalog(refresh=False, cache_only=False, country=country)
    except Exception:
        return []
    rows = payload.get("symbols") if isinstance(payload, dict) else None
    return rows if isinstance(rows, list) else []


def _resolve_catalog_symbol(query: str, *, region: str, catalog_rows: list[dict[str, Any]]) -> str | None:
    normalized_region = normalize_region(region)
    ranked: list[tuple[int, str, str]] = []
    for item in catalog_rows:
        if not isinstance(item, dict):
            continue
        candidate_symbol = _catalog_row_symbol(item, region=normalized_region)
        if not candidate_symbol:
            continue
        rank = _rank_catalog_candidate(
            query,
            candidate_symbol=candidate_symbol,
            candidate_name=str(item.get("name") or ""),
        )
        if rank is None:
            continue
        ranked.append((rank, candidate_symbol, str(item.get("name") or "")))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][1]


async def resolve_region_holdings(
    raw: Any,
    *,
    region: str,
    symbol_catalog_store: Any | None = None,
) -> list[dict[str, float]]:
    normalized_region = normalize_region(region)
    catalog_rows = await _load_region_catalog(symbol_catalog_store, region=normalized_region)

    def resolve_symbol(raw_symbol: str) -> str:
        resolved_symbol = _resolve_catalog_symbol(raw_symbol, region=normalized_region, catalog_rows=catalog_rows)
        if resolved_symbol is None:
            resolved_symbol = normalize_region_symbol(raw_symbol, region=normalized_region)
        return resolved_symbol

    return _aggregate_region_holdings(_iter_holding_inputs(raw), resolve_symbol=resolve_symbol)


def normalize_region_holdings(raw: Any, *, region: str) -> list[dict[str, float]]:
    normalized_region = normalize_region(region)
    return _aggregate_region_holdings(
        _iter_holding_inputs(raw),
        resolve_symbol=lambda raw_symbol: normalize_region_symbol(raw_symbol, region=normalized_region),
    )
