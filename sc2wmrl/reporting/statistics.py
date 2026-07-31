"""Aligned multi-seed aggregate statistics and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregateSeries:
    """Interpolated mean, dispersion, and confidence band over true x-axis values."""
    steps: list[int]
    mean: list[float]
    std: list[float]
    lower: list[float]
    upper: list[float]
    run_count: int


def aggregate_series(series: list[tuple[np.ndarray, np.ndarray]], confidence: float = 1.96) -> AggregateSeries:
    """Interpolate each run onto common observed steps before aggregating it."""
    if not series: raise ValueError("at least one series is required")
    grid = np.unique(np.concatenate([np.asarray(x, dtype=int) for x, _ in series]))
    rows = []
    for steps, values in series:
        steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
        if len(steps) == 0 or len(steps) != len(values) or np.any(np.diff(steps) <= 0): raise ValueError("series steps must be non-empty and strictly increasing")
        valid = (grid >= steps[0]) & (grid <= steps[-1]); row = np.full(len(grid), np.nan); row[valid] = np.interp(grid[valid], steps, values); rows.append(row)
    matrix = np.asarray(rows); count = np.sum(~np.isnan(matrix), axis=0); mean = np.nanmean(matrix, axis=0); std = np.nanstd(matrix, axis=0)
    interval = confidence * std / np.sqrt(np.maximum(count, 1))
    return AggregateSeries(grid.tolist(), mean.tolist(), std.tolist(), (mean - interval).tolist(), (mean + interval).tolist(), len(series))
