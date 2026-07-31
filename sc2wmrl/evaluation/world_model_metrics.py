"""Standalone, dependency-light model prediction metrics."""

from __future__ import annotations

import numpy as np


def persistence_mse(observations: np.ndarray) -> float:
    """MSE of the persistence baseline for a ``[B,T,D]`` sequence."""
    if observations.ndim != 3 or observations.shape[1] < 2: raise ValueError("need [B,T,D] with T >= 2")
    return float(np.mean((observations[:, 1:] - observations[:, :-1]) ** 2))


def binary_f1(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.0) -> float:
    """Micro F1 for event predictions without a scikit-learn dependency."""
    predicted = np.asarray(logits) >= threshold; truth = np.asarray(targets).astype(bool)
    tp = np.logical_and(predicted, truth).sum(); fp = np.logical_and(predicted, ~truth).sum(); fn = np.logical_and(~predicted, truth).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))
