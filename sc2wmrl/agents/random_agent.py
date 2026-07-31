"""Mask-respecting random policy used solely as an evaluation control."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Uniformly samples only legal macro actions with a local deterministic RNG."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool = False) -> int:
        """Select a legal action; observation is accepted for policy interchangeability."""
        del observation, deterministic
        legal = np.flatnonzero(np.asarray(action_mask, dtype=np.bool_))
        if legal.size == 0:
            raise ValueError("cannot act with an empty action mask")
        return int(self._rng.choice(legal))
