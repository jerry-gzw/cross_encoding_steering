from __future__ import annotations

from typing import Iterable

import numpy as np


def paired_sign_flip_pvalue(
    values: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired randomization p-value for a zero mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    if len(values) <= 18:
        indices = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = ((indices >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
        permuted = np.abs((signs * values).mean(axis=1))
        return float(np.mean(permuted >= observed - 1e-12))
    if n_permutations < 1_000:
        raise ValueError("n_permutations must be at least 1000")
    exceed = 0
    remaining = int(n_permutations)
    while remaining:
        current = min(2_000, remaining)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-12))
        remaining -= current
    return float((exceed + 1) / (int(n_permutations) + 1))


def paired_weighted_sign_flip_pvalue(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = weights[finite]
    if len(values) == 0:
        return np.nan
    weights = weights / weights.sum()
    observed = abs(float(np.sum(weights * values)))
    if np.allclose(values, 0.0):
        return 1.0
    if n_permutations < 1_000:
        raise ValueError("n_permutations must be at least 1000")
    exceed = 0
    remaining = int(n_permutations)
    weighted_values = weights * values
    while remaining:
        current = min(2_000, remaining)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * weighted_values).sum(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-12))
        remaining -= current
    return float((exceed + 1) / (int(n_permutations) + 1))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm-adjust finite p-values while preserving missing entries."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
