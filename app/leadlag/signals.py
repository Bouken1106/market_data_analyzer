"""Signal generation for the subspace-regularized lead-lag strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import PreparedLeadLagDataset
from .schemas import LeadLagConfig
from .subspace_pca import (
    build_pairwise_correlation,
    build_prior_subspace,
    build_target_c0,
    project_to_correlation_matrix,
    top_eigenpairs,
)


@dataclass(frozen=True)
class SignalObservation:
    """One daily prediction snapshot produced by the lead-lag model."""

    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    predicted: pd.Series
    realized: pd.Series
    factors: np.ndarray
    eigenvalues: np.ndarray
    us_loadings: np.ndarray
    jp_loadings: np.ndarray
    transfer_matrix: np.ndarray


@dataclass(frozen=True)
class LeadLagSignalRun:
    """Full output of the rolling signal generation stage."""

    observations: tuple[SignalObservation, ...]
    prior_subspace: np.ndarray
    d0: np.ndarray
    c0: np.ndarray
    cfull_corr: np.ndarray


def generate_leadlag_signals(dataset: PreparedLeadLagDataset, config: LeadLagConfig) -> LeadLagSignalRun:
    """Generate rolling signals using the paper's regularized PCA recipe."""

    symbols = dataset.combined_symbols
    cfull_corr = build_pairwise_correlation(
        dataset.cfull_source.dropna(how="all"),
        symbols,
        min_periods=max(3, min(10, dataset.cfull_source.dropna(how="all").shape[0])),
    )
    prior_subspace = build_prior_subspace(
        dataset.us_symbols,
        dataset.jp_symbols,
        cyclical_symbols=config.cyclical_symbols,
        defensive_symbols=config.defensive_symbols,
    )
    c0, d0 = build_target_c0(cfull_corr, prior_subspace)

    observations: list[SignalObservation] = []
    us_count = len(dataset.us_symbols)

    for signal_date in dataset.candidate_signal_dates:
        target_date = dataset.next_target_by_signal_date.get(signal_date)
        if target_date is None:
            continue

        history = dataset.z_all.loc[dataset.z_all.index < signal_date].dropna(how="all").tail(config.rolling_window_days)
        if history.shape[0] < config.rolling_window_days:
            continue

        ct = build_pairwise_correlation(
            history,
            symbols,
            fallback=c0,
            min_periods=max(3, min(10, history.shape[0])),
        )
        creg = project_to_correlation_matrix((1.0 - config.lambda_reg) * ct + (config.lambda_reg * c0))
        eigenvalues, eigenvectors = top_eigenpairs(creg, config.n_components)

        us_loadings = eigenvectors[:us_count, :]
        jp_loadings = eigenvectors[us_count:, :]
        z_u = dataset.z_us.loc[signal_date, list(dataset.us_symbols)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(z_u)):
            continue

        factors = us_loadings.T @ z_u
        predicted_values = jp_loadings @ factors
        realized = dataset.roc_jp.loc[target_date, list(dataset.jp_symbols)].astype(np.float64)
        transfer_matrix = jp_loadings @ us_loadings.T

        observations.append(
            SignalObservation(
                signal_date=signal_date,
                target_date=target_date,
                predicted=pd.Series(predicted_values, index=list(dataset.jp_symbols), dtype=np.float64),
                realized=realized,
                factors=factors,
                eigenvalues=eigenvalues,
                us_loadings=us_loadings,
                jp_loadings=jp_loadings,
                transfer_matrix=transfer_matrix,
            )
        )

    if not observations:
        raise ValueError(
            "No rolling signal could be produced. "
            "Try relaxing the universe, extending history, or shortening the rolling window."
        )

    return LeadLagSignalRun(
        observations=tuple(observations),
        prior_subspace=prior_subspace,
        d0=d0,
        c0=c0,
        cfull_corr=cfull_corr,
    )
