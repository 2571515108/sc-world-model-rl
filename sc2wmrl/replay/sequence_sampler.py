"""Continuous-episode replay sampling for later RSSM training."""

from __future__ import annotations

import numpy as np

from .replay_buffer import ReplayBuffer
from .transition import MacroTransition


class SequenceSampler:
    """Samples only contiguous records with an optional burn-in prefix."""

    def __init__(self, replay: ReplayBuffer, *, seed: int = 0) -> None:
        self.replay = replay
        self._rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> list[list[MacroTransition]]:
        """Return fixed-length sequences that never cross episode boundaries."""
        if batch_size <= 0 or sequence_length <= 0 or burn_in_length < 0:
            raise ValueError("invalid sequence sampling parameters")
        total = sequence_length + burn_in_length
        items = self.replay.transitions()
        starts: list[int] = []
        for start in range(0, len(items) - total + 1):
            chunk = items[start:start + total]
            same_episode = all(item.episode_id == chunk[0].episode_id for item in chunk)
            contiguous = all(
                chunk[i + 1].game_loop > chunk[i].game_loop
                and np.array_equal(chunk[i].next_observation, chunk[i + 1].observation)
                for i in range(len(chunk) - 1)
            )
            if same_episode and contiguous:
                starts.append(start)
        if batch_size > len(starts):
            raise ValueError("not enough continuous replay sequences")
        choices = self._rng.choice(starts, size=batch_size, replace=False)
        return [list(items[int(start):int(start) + total]) for start in choices]
