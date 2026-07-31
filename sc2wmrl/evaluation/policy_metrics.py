"""Policy result aggregation without external analytics dependencies."""

from __future__ import annotations

import numpy as np


def summarize_episodes(returns: list[float], wins: list[bool]) -> dict[str, float]:
    """Produce reproducible return and win-rate summary statistics."""
    if not returns or len(returns) != len(wins): raise ValueError("returns and wins must be non-empty and aligned")
    values = np.asarray(returns, dtype=np.float64)
    return {"episodes": float(len(values)), "mean_return": float(values.mean()), "return_std": float(values.std()), "win_rate": float(np.mean(wins))}
