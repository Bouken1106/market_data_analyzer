"""Shared helpers for in-memory TTL caches backed by async locks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, TypeVar

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
        try:
            fresh = (time.time() - float(cached_epoch)) <= ttl_sec
        except (TypeError, ValueError):
            fresh = False
        return CacheLookup(
            payload=_copy_payload(payload, copy_fn),
            found=True,
            fresh=fresh,
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
