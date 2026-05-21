"""Shared helpers for in-memory TTL caches backed by async locks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, TypeVar

from ..utils import finite_float_or_none

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheLookup:
    payload: Any | None
    found: bool
    fresh: bool


def _copy_payload(payload: V, copy_fn: Callable[[V], V] | None) -> V:
    if copy_fn is None:
        return payload
    return copy_fn(payload)


def ttl_cache_is_fresh(cached_epoch: Any, ttl_sec: int, *, now_epoch: float | None = None) -> bool:
    reference_now = time.time() if now_epoch is None else finite_float_or_none(now_epoch)
    parsed_epoch = finite_float_or_none(cached_epoch)
    if reference_now is None or parsed_epoch is None:
        return False
    return (reference_now - parsed_epoch) <= ttl_sec


def build_cached_response(
    payload: Mapping[str, Any],
    *,
    source: str,
    ttl_sec: int | None = None,
    stale: bool = False,
) -> dict[str, Any]:
    response = dict(payload)
    response["source"] = source
    if ttl_sec is not None:
        response["cache_ttl_sec"] = ttl_sec
    response["cache_stale"] = stale
    return response


async def ttl_cache_lookup(
    cache: MutableMapping[K, dict[str, Any]],
    lock: asyncio.Lock,
    key: K,
    *,
    ttl_sec: int,
    copy_fn: Callable[[V], V] | None = None,
) -> CacheLookup:
    async with lock:
        entry = cache.get(key)
        if not isinstance(entry, dict):
            return CacheLookup(payload=None, found=False, fresh=False)
        payload = entry.get("payload")
        cached_epoch = entry.get("cached_epoch")
        fresh = ttl_cache_is_fresh(cached_epoch, ttl_sec)
        return CacheLookup(
            payload=_copy_payload(payload, copy_fn),
            found=True,
            fresh=fresh,
        )


async def ttl_cache_lookup_response(
    cache: MutableMapping[K, dict[str, Any]],
    lock: asyncio.Lock,
    key: K,
    *,
    ttl_sec: int,
    copy_fn: Callable[[V], V] | None = None,
    allow_stale: bool = False,
    source_fresh: str = "cache",
    source_stale: str = "cache-stale",
    include_cache_metadata: bool = False,
) -> dict[str, Any] | None:
    cached = await ttl_cache_lookup(
        cache,
        lock,
        key,
        ttl_sec=ttl_sec,
        copy_fn=copy_fn,
    )
    if not cached.found or not isinstance(cached.payload, dict):
        return None
    if not cached.fresh and not allow_stale:
        return None
    return build_cached_response(
        cached.payload,
        source=source_fresh if cached.fresh else source_stale,
        ttl_sec=ttl_sec if include_cache_metadata else None,
        stale=not cached.fresh,
    )


async def ttl_cache_store(
    cache: MutableMapping[K, dict[str, Any]],
    lock: asyncio.Lock,
    key: K,
    payload: V,
) -> None:
    async with lock:
        cache[key] = {
            "cached_epoch": time.time(),
            "payload": payload,
        }


async def ttl_cache_pop(
    cache: MutableMapping[K, dict[str, Any]],
    lock: asyncio.Lock,
    key: K,
) -> bool:
    async with lock:
        return cache.pop(key, None) is not None


async def ttl_cache_pop_matching(
    cache: MutableMapping[K, dict[str, Any]],
    lock: asyncio.Lock,
    predicate: Callable[[K], bool],
) -> int:
    async with lock:
        keys = [key for key in cache.keys() if predicate(key)]
        for key in keys:
            cache.pop(key, None)
        return len(keys)
