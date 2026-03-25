"""Shared parameter helpers for stock ML page routes and services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


def stock_ml_page_config_hash(
    *,
    prediction_date: str | None,
    universe_filter: str,
    model_family: str,
    feature_set: str,
    cost_buffer: float,
    train_window_months: int,
    gap_days: int,
    valid_window_months: int,
    random_seed: int,
) -> str:
    payload = {
        "prediction_date": prediction_date,
        "universe_filter": universe_filter,
        "model_family": model_family,
        "feature_set": feature_set,
        "cost_buffer": cost_buffer,
        "train_window_months": train_window_months,
        "gap_days": gap_days,
        "valid_window_months": valid_window_months,
        "random_seed": random_seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass(frozen=True)
class StockMlPageParams:
    prediction_date: str | None = None
    universe_filter: str = "jp_large_cap_stooq_v1"
    model_family: str = "LightGBM Classifier"
    feature_set: str = "base_v1"
    cost_buffer: float = 0.0
    train_window_months: int = 12
    gap_days: int = 5
    valid_window_months: int = 1
    random_seed: int = 42
    train_note: str = ""
    run_note: str = ""
    refresh: bool = False

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StockMlPageParams":
        return cls(**{name: mapping[name] for name in cls.field_names() if name in mapping})

    def service_kwargs(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return stock_ml_page_config_hash(
            prediction_date=self.prediction_date,
            universe_filter=self.universe_filter,
            model_family=self.model_family,
            feature_set=self.feature_set,
            cost_buffer=self.cost_buffer,
            train_window_months=self.train_window_months,
            gap_days=self.gap_days,
            valid_window_months=self.valid_window_months,
            random_seed=self.random_seed,
        )
