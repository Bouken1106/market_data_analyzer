#!/usr/bin/env python3
"""Minimal LM Studio chat CLI (OpenAI-compatible /v1/chat/completions)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _resolve_chat_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url

    env_chat = os.getenv("LMSTUDIO_CHAT_COMPLETIONS_URL", "").strip()
    if env_chat:
        return env_chat

    base = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def _post_chat(
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    api_key: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc

    try:
        parsed: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {body[:400]}") from exc

    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"No choices in response: {parsed}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    text = str(content or "").strip()
    if not text:
        raise RuntimeError(f"Empty content in response: {parsed}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to LM Studio via OpenAI-compatible chat API.")
    parser.add_argument("--url", default=None, help="Full chat completions URL.")
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_MODEL", "ministral-3-3b"))
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--system", default="", help="Optional system prompt.")
    parser.add_argument("--once", default="", help="Send one message and exit.")
    args = parser.parse_args()

    url = _resolve_chat_url(args.url)
    model = str(args.model).strip() or "ministral-3-3b"
    api_key = str(args.api_key or "").strip()

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.once:
        messages.append({"role": "user", "content": args.once})
        try:
            reply = _post_chat(
                url=url,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key=api_key,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print(reply)
        return 0

    print(f"LM Studio Chat CLI")
    print(f"- url: {url}")
    print(f"- model: {model}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        try:
            reply = _post_chat(
                url=url,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key=api_key,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")
            continue

        print(f"bot> {reply}")
        messages.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
