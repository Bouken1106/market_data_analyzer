# agent.md

## Purpose

`market_data_analyzer` is a local FastAPI application for:

- market data ingestion from Twelve Data / FMP
- historical analytics and security overview APIs
- strategy backtesting
- quantile-based ML forecasting

## Entry Points

- App startup: `app/main.py`
- Core hub: `app/hub.py`
- API routes: `app/api/`
- Strategy logic: `app/strategy_engine.py`
- ML models: `app/quantile_lstm.py`, `app/patchtst_quantile.py`

## Important Shared Modules

- OHLCV normalization / merge helpers: `app/ohlcv.py`
- Shared ML windowing / scaling helpers: `app/ml/array_utils.py`
- Persistent stores: `app/stores/`

## Working Rules

- Prefer extending shared helpers before duplicating OHLCV normalization or rolling-window logic.
- Keep provider-specific fetch code inside `app/services/market_data_queries.py` and `app/services/market_data_realtime.py`.
- Treat store-returned cached data as read-only unless you explicitly requested a copied structure.
- For strategy changes, preserve API response shape expected by `app/api/strategy.py` and the frontend.
- For ML changes, keep train/val/test chronological and avoid leakage from future rows.
- If the user gives a durable instruction, preference, or project-specific operating rule, treat it as project intent and update `agent.md` so the guidance persists for later work.
- For any help mark / tooltip / popover UI, do not rely on naive absolute positioning inside the local container. Verify viewport-fit, scroll/resize repositioning, and z-index/overflow behavior so the popup cannot be clipped or hidden behind other layers on smaller screens.

## Local Commands

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
python -m compileall app
```

## Refactoring Priorities

- Remove duplicated data-shaping code before adding new features.
- Prefer vectorized NumPy paths for rolling windows and aligned price matrices.
- Avoid extra copying for full-history caches; those payloads can be large.
- Keep external API usage conservative because provider quotas are tight.
