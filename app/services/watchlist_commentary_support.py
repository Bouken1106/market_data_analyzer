"""Helpers for watchlist commentary prompts, parsing, and LM Studio calls."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import httpx
from fastapi import HTTPException

from ..utils import finite_float_or_none

WATCHLIST_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "watchlist_commentary",
        "schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "comment": {"type": "string"},
                        },
                        "required": ["symbol", "comment"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["picks"],
            "additionalProperties": False,
        },
    },
}

_WATCHLIST_MAX_COMMENT_LEN = 80
_JSON_DECODER = json.JSONDecoder()


def safe_float(value: Any) -> float | None:
    return finite_float_or_none(value)


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def compute_watch_metrics(symbol: str, sparkline_item: dict[str, Any] | None) -> dict[str, Any]:
    latest_close = safe_float(sparkline_item.get("latest_close")) if isinstance(sparkline_item, dict) else None
    previous_close = safe_float(sparkline_item.get("previous_close")) if isinstance(sparkline_item, dict) else None

    trend_raw = sparkline_item.get("trend_30d") if isinstance(sparkline_item, dict) else []
    trend_closes: list[float] = []
    if isinstance(trend_raw, list):
        for raw_value in trend_raw:
            close_value = safe_float(raw_value)
            if close_value is None or close_value <= 0:
                continue
            trend_closes.append(close_value)

    day_change_pct: float | None = None
    if latest_close is not None and previous_close is not None and previous_close > 0:
        day_change_pct = ((latest_close - previous_close) / previous_close) * 100

    return_30d_pct: float | None = None
    if len(trend_closes) >= 2 and trend_closes[0] > 0:
        return_30d_pct = ((trend_closes[-1] - trend_closes[0]) / trend_closes[0]) * 100

    daily_returns: list[float] = []
    for idx in range(1, len(trend_closes)):
        prev_close = trend_closes[idx - 1]
        curr_close = trend_closes[idx]
        if prev_close <= 0:
            continue
        daily_returns.append((curr_close / prev_close) - 1.0)

    volatility_30d_pct: float | None = None
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((item - mean_return) ** 2 for item in daily_returns) / len(daily_returns)
        volatility_30d_pct = math.sqrt(max(variance, 0.0)) * 100

    return {
        "symbol": symbol,
        "day_change_pct": day_change_pct,
        "return_30d_pct": return_30d_pct,
        "volatility_30d_pct": volatility_30d_pct,
        "day_change_text": format_signed_percent(day_change_pct),
        "return_30d_text": format_signed_percent(return_30d_pct),
        "volatility_30d_text": format_percent(volatility_30d_pct),
    }


def build_watchlist_prompt(current_date: str, metrics: list[dict[str, Any]]) -> str:
    lines = [
        f"現在({current_date})の銘柄の情報は以下の通りです",
        "",
        "銘柄\t前日比\t30日リターン\t30日ボラティリティ",
    ]
    for item in metrics:
        lines.append(
            f"{item['symbol']}\t{item['day_change_text']}\t{item['return_30d_text']}\t{item['volatility_30d_text']}"
        )
    lines.extend(
        [
            "",
            "上記の銘柄だけを対象に、特徴的な2銘柄を選ぶ。",
            "必ずJSONのみで返すこと。形式:",
            '{"picks":[{"symbol":"AAPL","comment":"..."} , {"symbol":"NVDA","comment":"..."}]}',
            "制約: symbolは表内の銘柄のみ、commentは日本語1文、簡潔、余計な説明禁止。",
        ]
    )
    return "\n".join(lines)


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


def build_base_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "あなたは株式ウォッチリスト要約アシスタントです。"
                "思考過程や分析手順は一切出力しない。"
                "常にJSONのみを返す。"
                '形式は {"picks":[{"symbol":"...","comment":"..."},{"symbol":"...","comment":"..."}]} のみ。'
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def build_repair_messages(raw_commentary: str, valid_symbols: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "あなたはJSON整形器です。"
                "入力文を要約し、指定形式JSONのみを返す。"
                "説明文や前置きは出力禁止。"
            ),
        },
        {
            "role": "user",
            "content": (
                "有効な銘柄: "
                + ",".join(valid_symbols)
                + "\n次の文章を2銘柄の短評JSONへ整形してください。"
                + '\n形式: {"picks":[{"symbol":"...","comment":"..."},{"symbol":"...","comment":"..."}]}'
                + "\n文章:\n"
                + raw_commentary
            ),
        },
    ]


async def chat_lmstudio(
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout_sec: float,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> tuple[str, int, str | None, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(timeout_sec, connect=min(10.0, timeout_sec))
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, json=payload, headers=headers)
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
        return "", response.status_code, error_message, model

    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="LM Studio returned an invalid response format.")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="LM Studio response does not include choices.")

    model_name = str(result.get("model") or "").strip() or model
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    return str(content or "").strip(), response.status_code, None, model_name


def metrics_payload(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": item["symbol"],
            "day_change_pct": item["day_change_pct"],
            "return_30d_pct": item["return_30d_pct"],
            "volatility_30d_pct": item["volatility_30d_pct"],
            "day_change_text": item["day_change_text"],
            "return_30d_text": item["return_30d_text"],
            "volatility_30d_text": item["volatility_30d_text"],
        }
        for item in metrics
    ]
