"""Subspace construction and PCA helpers for the lead-lag algorithm."""

from __future__ import annotations

import numpy as np
import pandas as pd


def project_to_correlation_matrix(matrix: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Project an arbitrary symmetric matrix to a valid correlation matrix."""

    arr = np.asarray(matrix, dtype=np.float64)
    symmetric = np.nan_to_num((arr + arr.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, eps, None)
    psd = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    diag = np.sqrt(np.clip(np.diag(psd), eps, None))
    correlation = psd / np.outer(diag, diag)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def build_pairwise_correlation(
    returns: pd.DataFrame,
    symbols: tuple[str, ...],
    *,
    fallback: np.ndarray | None = None,
    min_periods: int = 3,
) -> np.ndarray:
    """Build a stable pairwise correlation matrix with fallback filling."""

    raw = returns.corr(min_periods=max(1, int(min_periods)))
    corr = raw.reindex(index=list(symbols), columns=list(symbols)).to_numpy(dtype=np.float64)
    if fallback is None:
        fallback = np.eye(len(symbols), dtype=np.float64)
    mask = ~np.isfinite(corr)
    corr[mask] = fallback[mask]
    np.fill_diagonal(corr, 1.0)
    return project_to_correlation_matrix(corr)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Prior subspace direction collapsed to zero during orthogonalization.")
    return vector / norm


def _orthogonalize(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    result = vector.astype(np.float64, copy=True)
    for basis_vector in basis:
        result = result - basis_vector * float(basis_vector @ result)
    return result


def build_prior_subspace(
    us_symbols: tuple[str, ...],
    jp_symbols: tuple[str, ...],
    *,
    cyclical_symbols: frozenset[str],
    defensive_symbols: frozenset[str],
) -> np.ndarray:
    """Construct the three prior directions described in the paper."""

    combined_symbols = list(us_symbols) + list(jp_symbols)
    total_assets = len(combined_symbols)
    if total_assets == 0:
        raise ValueError("At least one symbol is required to build the prior subspace.")

    basis: list[np.ndarray] = []

    v1 = _normalize(np.ones(total_assets, dtype=np.float64))
    basis.append(v1)

    v2_seed = np.concatenate(
        [
            np.ones(len(us_symbols), dtype=np.float64),
            -np.ones(len(jp_symbols), dtype=np.float64),
        ]
    )
    v2 = _normalize(_orthogonalize(v2_seed, basis))
    basis.append(v2)

    v3_seed = np.zeros(total_assets, dtype=np.float64)
    for index, symbol in enumerate(combined_symbols):
        if symbol in cyclical_symbols:
            v3_seed[index] = 1.0
        elif symbol in defensive_symbols:
            v3_seed[index] = -1.0
    if not np.any(v3_seed):
        raise ValueError("The cyclical/defensive labels produced an all-zero v3 direction.")
    v3 = _normalize(_orthogonalize(v3_seed, basis))
    basis.append(v3)

    return np.column_stack(basis)


def build_target_c0(cfull_corr: np.ndarray, prior_subspace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the regularization target correlation matrix ``C0``."""

    d0 = np.diag(prior_subspace.T @ cfull_corr @ prior_subspace)
    raw = prior_subspace @ np.diag(d0) @ prior_subspace.T
    diagonal = np.sqrt(np.clip(np.diag(raw), 1e-8, None))
    c0 = raw / np.outer(diagonal, diagonal)
    c0 = project_to_correlation_matrix(c0)
    return c0, d0


def top_eigenpairs(corr: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading eigenpairs of a symmetric correlation matrix."""

    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    sorted_vectors = vectors[:, order]
    top_k = max(1, min(int(n_components), sorted_vectors.shape[1]))
    return sorted_values[:top_k], sorted_vectors[:, :top_k]
