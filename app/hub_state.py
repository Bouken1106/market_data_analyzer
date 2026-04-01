"""Structured state objects used to initialize ``MarketDataHub``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class ProviderState:
    provider: str
    twelvedata_api_key: str
    fmp_api_key: str
    symbols: list[str]
    default_country_key: str
    symbol_country_map: dict[str, str]
    market_sessions: dict[str, Any]
    ui_state_store: Any | None


@dataclass
class RuntimeState:
    prices: dict[str, dict[str, Any]] = field(default_factory=dict)
    ws_connected: bool = False
    last_ws_message_at: float = 0.0
    mode: str = "starting"
    daily_credits_left: int | None = None
    daily_credits_used: int | None = None
    daily_credits_limit: int | None = None
    daily_credits_updated_at: str | None = None
    daily_credits_source: str | None = None
    daily_credits_is_estimated: bool = False
    minute_credits_left: int | None = None
    minute_credits_used: int | None = None
    _listeners: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    _worker_tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _restart_ws_event: asyncio.Event = field(default_factory=asyncio.Event)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _credits_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class CacheState:
    _historical_cache: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    _historical_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _sparkline_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _sparkline_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _overview_cache: dict[tuple[str, bool, bool, bool], dict[str, Any]] = field(default_factory=dict)
    _overview_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _fmp_reference_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _fmp_reference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def assign_state_fields(target: Any, state: Any) -> None:
    for item in fields(state):
        setattr(target, item.name, getattr(state, item.name))
