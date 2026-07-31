"""Display-only smoothing functions that never mutate source metric records."""

from __future__ import annotations

import numpy as np


def smooth(values: list[float] | np.ndarray, method: str = "raw", window: int = 10, factor: float = 0.95) -> np.ndarray:
    """Return raw, moving-average, or EMA values as a new float array."""
    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or not np.isfinite(source).all(): raise ValueError("values must be a finite one-dimensional series")
    if method == "raw": return source.copy()
    if method == "moving_average":
        if window <= 0: raise ValueError("window must be positive")
        return np.asarray([source[max(0, i - window + 1):i + 1].mean() for i in range(len(source))])
    if method == "ema":
        if not 0 <= factor < 1: raise ValueError("EMA factor must be in [0, 1)")
        result = source.copy()
        for i in range(1, len(result)): result[i] = factor * result[i - 1] + (1 - factor) * source[i]
        return result
    raise ValueError(f"unknown smoothing method {method}")
