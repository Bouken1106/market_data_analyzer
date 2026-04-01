"""Parsing helpers for watchlist commentary generation."""

from __future__ import annotations

import json
import re
from typing import Any

_WATCHLIST_MAX_COMMENT_LEN = 80
_JSON_DECODER = json.JSONDecoder()


def extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    for idx, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, _ = _JSON_DECODER.raw_decode(raw_text[idx:])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_comment_line(text: str) -> str:
    compact = " ".join(str(text).replace("\r", "\n").split())
    compact = re.sub(r"[*_`>#]+", "", compact).strip()
    return compact[:_WATCHLIST_MAX_COMMENT_LEN].strip()


def commentary_from_json(raw_text: str, valid_symbols: list[str]) -> str | None:
    payload = extract_first_json_object(raw_text)
    if not isinstance(payload, dict):
        return None

    picks = payload.get("picks")
    if not isinstance(picks, list):
        return None

    symbol_set = {symbol.upper() for symbol in valid_symbols}
    selected: list[str] = []
    used_symbols: set[str] = set()
    for item in picks:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if symbol not in symbol_set or symbol in used_symbols:
            continue
        comment = normalize_comment_line(item.get("comment") or "")
        if not comment:
            continue
        selected.append(f"{symbol}: {comment}")
        used_symbols.add(symbol)
        if len(selected) >= 2:
            break

    if len(selected) < 2:
        return None
    return "\n".join(selected)


def fallback_commentary(raw_text: str, valid_symbols: list[str]) -> str:
    symbol_set = {symbol.upper() for symbol in valid_symbols}
    lines = [line.strip() for line in str(raw_text).replace("\r", "\n").split("\n")]

    accepted: list[str] = []
    used_symbols: set[str] = set()
    for raw_line in lines:
        if not raw_line:
            continue
        lowered = raw_line.lower()
        if lowered.startswith(("alright", "i need to", "let me", "first,", "first ", "second,", "third,")):
            continue

        tokens = re.findall(r"[A-Z]{1,6}(?:\.[A-Z]{1,5})?", raw_line.upper())
        symbol = ""
        for token in tokens:
            if token in symbol_set and token not in used_symbols:
                symbol = token
                break
        if not symbol:
            continue

        line = normalize_comment_line(raw_line)
        if not line:
            continue
        accepted.append(f"{symbol}: {line}")
        used_symbols.add(symbol)
        if len(accepted) >= 2:
            break

    if len(accepted) >= 2:
        return "\n".join(accepted)
    if accepted:
        return accepted[0]
    return "コメントの整形に失敗しました。再実行してください。"
