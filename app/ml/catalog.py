"""ML model catalog and related constants."""

from __future__ import annotations

from ..utils import normalize_symbols

ML_MODEL_CATALOG = [
    {
        "id": "quantile_lstm",
        "name": "Quantile LSTM",
        "short_description": "翌営業日の分位点分布を推定（現在利用可能）",
        "status": "ready",
        "status_label": "Ready",
        "run_label": "Run Quantile LSTM",
        "api_path": "/api/ml/quantile-lstm",
    },
    {
        "id": "patchtst_quantile",
        "name": "PatchTST Quantile",
        "short_description": "PatchTSTで翌営業日の分位点分布を推定（現在利用可能）",
        "status": "ready",
        "status_label": "Ready",
        "run_label": "Run PatchTST Quantile",
        "api_path": "/api/ml/patchtst",
    },
    {
        "id": "quantile_gru",
        "name": "Quantile GRU",
        "short_description": "LSTMより軽量な系列モデル（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run Quantile GRU",
        "api_path": "",
    },
    {
        "id": "temporal_transformer",
        "name": "Temporal Transformer",
        "short_description": "注意機構ベースの時系列モデル（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run Temporal Transformer",
        "api_path": "",
    },
    {
        "id": "xgboost_quantile",
        "name": "XGBoost Quantile",
        "short_description": "勾配ブースティングの分位点回帰（準備中）",
        "status": "coming_soon",
        "status_label": "Coming Soon",
        "run_label": "Run XGBoost Quantile",
        "api_path": "",
    },
]

ML_COMPARE_DEFAULT_SYMBOLS = normalize_symbols(
    "AAPL,MSFT,GOOG,JPM,XOM,UNH,WMT,META,LLY,BRK.B,NVDA,HD"
)

ML_COMPARE_ALLOWED_MODELS = {"quantile_lstm", "patchtst_quantile"}
