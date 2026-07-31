"""Deterministic Elo utilities for strategy snapshot ratings."""

from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return Elo expected score for player A."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float = 32.0) -> tuple[float, float]:
    """Update two ratings from score A in [0,1], preserving zero-sum delta."""
    if not 0 <= score_a <= 1 or k_factor <= 0: raise ValueError("invalid Elo update arguments")
    delta = k_factor * (score_a - expected_score(rating_a, rating_b))
    return rating_a + delta, rating_b - delta
