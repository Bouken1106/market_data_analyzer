"""Streaming helpers for market API routes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse


def build_market_stream_response(request: Request, hub: Any) -> StreamingResponse:
    queue = hub.register_listener()

    async def event_generator():
        initial_payload = await hub.snapshot_payload()
        yield f"data: {json.dumps(initial_payload)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.unregister_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
