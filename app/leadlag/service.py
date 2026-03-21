"""High-level orchestration for the lead-lag PCA workflow."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_adapter import HistoricalPointBatch, HubHistoricalLeadLagAdapter
from .evaluation import evaluate_long_short
from .preprocessing import PreparedLeadLagDataset, prepare_leadlag_dataset
from .schemas import LeadLagConfig
from .signals import LeadLagSignalRun, SignalObservation, generate_leadlag_signals


class LeadLagService:
    """Facade that fetches data, runs the algorithm, and serializes the payload."""

    def __init__(self, hub: Any) -> None:
        self.hub = hub

    async def analyze(self, config: LeadLagConfig) -> dict[str, Any]:
        adapter = HubHistoricalLeadLagAdapter(self.hub, history_years=config.history_years)
        batch = await adapter.fetch_points(config.all_symbols, refresh=config.refresh)
        prepared = prepare_leadlag_dataset(config, batch)
        signal_run = generate_leadlag_signals(prepared, config)
        evaluation = (
            evaluate_long_short(signal_run.observations, quantile_q=config.quantile_q)
            if config.include_backtest
            else None
        )
        return self._serialize_payload(config, batch, prepared, signal_run, evaluation)

    @staticmethod
    def _serialize_signal(
        observation: SignalObservation,
        *,
        include_transfer_matrix: bool,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for symbol, signal_value in observation.predicted.sort_values(ascending=False).items():
            realized_value = observation.realized.get(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "signal": float(signal_value),
                    "realized_open_to_close": (
                        float(realized_value)
                        if realized_value is not None and np.isfinite(float(realized_value))
                        else None
                    ),
                }
            )

        payload = {
            "signal_date": observation.signal_date.date().isoformat(),
            "target_date": observation.target_date.date().isoformat(),
            "factors": [float(value) for value in observation.factors.tolist()],
            "eigenvalues": [float(value) for value in observation.eigenvalues.tolist()],
            "predicted_rows": rows,
            "us_loadings": observation.us_loadings.tolist(),
            "jp_loadings": observation.jp_loadings.tolist(),
        }
        if include_transfer_matrix:
            payload["transfer_matrix"] = observation.transfer_matrix.tolist()
        return payload

    @staticmethod
    def _recent_signal_summary(observation: SignalObservation) -> dict[str, Any]:
        ranked = observation.predicted.sort_values(ascending=False)
        return {
            "signal_date": observation.signal_date.date().isoformat(),
            "target_date": observation.target_date.date().isoformat(),
            "top_symbols": list(ranked.head(3).index),
            "bottom_symbols": list(ranked.tail(3).index),
            "factor_1": float(observation.factors[0]) if observation.factors.size else None,
        }

    def _serialize_payload(
        self,
        config: LeadLagConfig,
        batch: HistoricalPointBatch,
        prepared: PreparedLeadLagDataset,
        signal_run: LeadLagSignalRun,
        evaluation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        latest_signal = signal_run.observations[-1]
        excluded_rows = [
            {
                "symbol": symbol,
                "reason": reason,
            }
            for symbol, reason in sorted(prepared.excluded_symbols.items())
        ]
        recent_signals = [self._recent_signal_summary(item) for item in signal_run.observations[-10:]]

        overall_index = prepared.rcc_all.dropna(how="all").index
        data_range = {
            "from": overall_index.min().date().isoformat() if not overall_index.empty else None,
            "to": overall_index.max().date().isoformat() if not overall_index.empty else None,
            "cfull_observations": int(prepared.cfull_source.dropna(how="all").shape[0]),
            "candidate_signal_dates": len(prepared.candidate_signal_dates),
            "generated_signals": len(signal_run.observations),
        }

        return {
            "settings": {
                "us_symbols": list(config.us_symbols),
                "jp_symbols": list(config.jp_symbols),
                "rolling_window_days": config.rolling_window_days,
                "lambda_reg": config.lambda_reg,
                "n_components": config.n_components,
                "quantile_q": config.quantile_q,
                "cfull_start": config.cfull_start,
                "cfull_end": config.cfull_end,
                "cyclical_symbols": sorted(config.cyclical_symbols),
                "defensive_symbols": sorted(config.defensive_symbols),
                "refresh": config.refresh,
                "include_backtest": config.include_backtest,
                "include_transfer_matrix": config.include_transfer_matrix,
                "history_years": config.history_years,
            },
            "data_summary": {
                "requested_us_symbols": list(config.us_symbols),
                "requested_jp_symbols": list(config.jp_symbols),
                "included_us_symbols": list(prepared.us_symbols),
                "included_jp_symbols": list(prepared.jp_symbols),
                "excluded_symbols": excluded_rows,
                "fetch_failures": [
                    {"symbol": symbol, "reason": reason}
                    for symbol, reason in sorted(batch.failures.items())
                ],
                "point_counts": dict(sorted(batch.point_counts.items())),
                "range": data_range,
            },
            "regularization": {
                "symbol_order": list(prepared.combined_symbols),
                "d0": [float(value) for value in signal_run.d0.tolist()],
                "c0": signal_run.c0.tolist(),
                "cfull_corr": signal_run.cfull_corr.tolist(),
                "prior_subspace": {
                    "direction_names": [
                        "global_equal_weight",
                        "country_spread",
                        "cyclical_defensive",
                    ],
                    "matrix": signal_run.prior_subspace.tolist(),
                },
            },
            "latest_signal": self._serialize_signal(
                latest_signal,
                include_transfer_matrix=config.include_transfer_matrix,
            ),
            "recent_signals": recent_signals,
            "strategy": evaluation,
        }
