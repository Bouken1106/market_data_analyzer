"""Fact-picking helpers for valuation data-source normalizers."""

from __future__ import annotations

from typing import Any

from .valuation_numeric import parse_float


class SecFactPicker:
    def __init__(self, facts: Any, *, annual_only: bool = False, instant_only: bool = False) -> None:
        self.facts = facts if isinstance(facts, dict) else {}
        self.annual_only = annual_only
        self.instant_only = instant_only

    def value(self, *concepts: str, unit: str, taxonomy: str = "us-gaap") -> float | None:
        fact = self.fact(*concepts, unit=unit, taxonomy=taxonomy)
        return parse_float(fact.get("val")) if fact else None

    def fact(self, *concepts: str, unit: str, taxonomy: str = "us-gaap") -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for concept in concepts:
            rows = self._rows(concept, unit=unit, taxonomy=taxonomy)
            candidates.extend(row for row in rows if self._matches_period(row))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (str(row.get("filed") or ""), str(row.get("end") or "")))
        return candidates[-1]

    def history(self, *concepts: str, unit: str, taxonomy: str = "us-gaap", max_items: int = 5) -> list[float]:
        rows: list[dict[str, Any]] = []
        for concept in concepts:
            rows.extend(row for row in self._rows(concept, unit=unit, taxonomy=taxonomy) if self._matches_period(row))
        rows.sort(key=lambda row: (int(row.get("fy") or 0), str(row.get("filed") or "")), reverse=True)
        seen_years: set[int] = set()
        values: list[float] = []
        for row in rows:
            year = int(row.get("fy") or 0)
            if not year or year in seen_years:
                continue
            value = parse_float(row.get("val"))
            if value is None:
                continue
            seen_years.add(year)
            values.append(value)
            if len(values) >= max_items:
                break
        return values

    def latest_end_date(self) -> str | None:
        rows: list[dict[str, Any]] = []
        taxonomy_rows = self.facts.get("us-gaap") if isinstance(self.facts, dict) else None
        if not isinstance(taxonomy_rows, dict):
            return None
        for concept_payload in taxonomy_rows.values():
            if not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit_rows in units.values():
                if isinstance(unit_rows, list):
                    rows.extend(row for row in unit_rows if isinstance(row, dict) and self._matches_period(row))
        rows.sort(key=lambda row: str(row.get("filed") or ""))
        return str(rows[-1].get("end") or "") if rows else None

    def _rows(self, concept: str, *, unit: str, taxonomy: str) -> list[dict[str, Any]]:
        taxonomy_facts = self.facts.get(taxonomy)
        if not isinstance(taxonomy_facts, dict):
            return []
        concept_payload = taxonomy_facts.get(concept)
        if not isinstance(concept_payload, dict):
            return []
        units = concept_payload.get("units")
        if not isinstance(units, dict):
            return []
        direct_rows = units.get(unit)
        if isinstance(direct_rows, list):
            return [dict(row) for row in direct_rows if isinstance(row, dict)]
        if unit == "USD/shares":
            for candidate_unit in ("USD/shares", "USD-per-shares"):
                candidate = units.get(candidate_unit)
                if isinstance(candidate, list):
                    return [dict(row) for row in candidate if isinstance(row, dict)]
        return []

    def _matches_period(self, row: dict[str, Any]) -> bool:
        if self.instant_only:
            return bool(row.get("end")) and not row.get("start")
        form = str(row.get("form") or "")
        if self.annual_only:
            return form == "10-K" or str(row.get("fp") or "").upper() == "FY"
        return form in {"10-K", "10-Q", "20-F", "40-F"} or not form


class EdinetFactPicker:
    def __init__(self, facts: dict[str, float]) -> None:
        self.facts = facts

    def value(self, *tokens: str) -> float | None:
        normalized_tokens = tuple(token.lower() for token in tokens)
        for key, value in self.facts.items():
            lower_key = key.lower()
            if any(token in lower_key for token in normalized_tokens):
                return value
        return None
