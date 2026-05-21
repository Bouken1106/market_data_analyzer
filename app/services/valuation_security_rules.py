"""Security-type rules used by valuation models."""

from __future__ import annotations

from typing import Any


SECURITY_OPERATING = "operating"
SECURITY_BANK = "bank"
SECURITY_INSURANCE = "insurance"
SECURITY_REIT = "reit"

_SUPPORTED_SECURITY_TYPES = frozenset(
    {
        SECURITY_OPERATING,
        SECURITY_BANK,
        SECURITY_INSURANCE,
        SECURITY_REIT,
    }
)
_BLOCKED_METHOD_GROUPS_BY_SECURITY_TYPE = {
    SECURITY_BANK: frozenset({"sales", "ev", "ev_ebitda", "fcf"}),
    SECURITY_INSURANCE: frozenset({"sales", "ev", "ev_ebitda", "fcf"}),
    SECURITY_REIT: frozenset({"per", "sales", "ev", "ev_ebitda", "fcf"}),
}
_NO_BLOCKED_METHOD_GROUPS = frozenset()


def normalize_security_type(value: Any) -> str | None:
    security_type = _norm_text(value)
    return security_type if security_type in _SUPPORTED_SECURITY_TYPES else None


def infer_security_type(
    *,
    explicit_type: Any = None,
    sector: Any = None,
    industry: Any = None,
    company_name: Any = None,
) -> str:
    explicit = normalize_security_type(explicit_type)
    if explicit is not None:
        return explicit

    text = " ".join(item for item in (_norm_text(sector), _norm_text(industry), _norm_text(company_name)) if item)
    if any(token in text for token in ("reit", "不動産投資信託", "投資法人")):
        return SECURITY_REIT
    if any(token in text for token in ("bank", "banks", "銀行")):
        return SECURITY_BANK
    if any(token in text for token in ("insurance", "保険")):
        return SECURITY_INSURANCE
    return SECURITY_OPERATING


def method_blocked_for_security_type(security_type: str, method_group: str) -> bool:
    return method_group in _BLOCKED_METHOD_GROUPS_BY_SECURITY_TYPE.get(security_type, _NO_BLOCKED_METHOD_GROUPS)


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()
