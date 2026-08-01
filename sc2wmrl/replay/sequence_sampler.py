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


class MixedSequenceSampler:
    """Sample complete synthetic/real sequences at an explicit data ratio."""

    def __init__(self, synthetic: SequenceSampler, real: SequenceSampler, *, synthetic_ratio: float = 0.2) -> None:
        if not 0.0 <= synthetic_ratio <= 1.0:
            raise ValueError("synthetic ratio must be in [0, 1]")
        self.synthetic, self.real, self.synthetic_ratio = synthetic, real, synthetic_ratio

    def sample(self, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> list[list[MacroTransition]]:
        """Return a batch with deterministic per-source counts when possible."""
        synthetic_count = int(round(batch_size * self.synthetic_ratio))
        real_count = batch_size - synthetic_count
        sequences: list[list[MacroTransition]] = []
        if synthetic_count:
            sequences.extend(self.synthetic.sample(synthetic_count, sequence_length, burn_in_length))
        if real_count:
            sequences.extend(self.real.sample(real_count, sequence_length, burn_in_length))
        return sequences


def split_replay_by_episode(replay: ReplayBuffer, validation_fraction: float) -> tuple[ReplayBuffer, ReplayBuffer | None]:
    """Partition complete episodes deterministically without corrupting sequences."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if validation_fraction == 0.0:
        return replay, None
    episode_ids = sorted({item.episode_id for item in replay.transitions()})
    validation_count = max(1, int(round(len(episode_ids) * validation_fraction)))
    validation_ids = set(episode_ids[-validation_count:])
    train, validation = ReplayBuffer(replay.capacity, seed=0), ReplayBuffer(replay.capacity, seed=1)
    for item in replay.transitions():
        (validation if item.episode_id in validation_ids else train).append(item)
    if not len(train) or not len(validation):
        raise ValueError("validation split must leave non-empty train and validation replays")
    return train, validation
