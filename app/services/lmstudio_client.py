"""Small LM Studio chat-completions client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException


@dataclass(frozen=True)
class LmStudioChatResult:
    content: str
    status_code: int
    error_detail: str | None
    model: str


class LmStudioClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout_sec: float,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> LmStudioChatResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(self.timeout_sec, connect=min(10.0, self.timeout_sec))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LM Studio request failed: {exc}") from exc

        try:
            result = response.json()
        except ValueError:
            result = {}

        error_message: str | None = None
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, dict):
                error_message = str(error.get("message") or "").strip() or None
            elif isinstance(error, str):
                error_message = error.strip() or None
            if error_message is None:
                error_message = str(result.get("detail") or "").strip() or None

        if response.status_code >= 400:
            return LmStudioChatResult(
                content="",
                status_code=response.status_code,
                error_detail=error_message,
                model=self.model,
            )

        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="LM Studio returned an invalid response format.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise HTTPException(status_code=502, detail="LM Studio response does not include choices.")

        model_name = str(result.get("model") or "").strip() or self.model
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        return LmStudioChatResult(
            content=str(content or "").strip(),
            status_code=response.status_code,
            error_detail=None,
            model=model_name,
        )
