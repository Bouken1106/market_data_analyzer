"""Prompt builders for watchlist commentary generation."""

from __future__ import annotations


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


def build_watchlist_prompt(current_date: str, metrics: list[dict[str, object]]) -> str:
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
