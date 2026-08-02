"""Vectorized sequence batch sampling over :class:`ArrayReplay`."""

from __future__ import annotations

import numpy as np

from .array_replay import ArrayReplay


class ArrayBatchSampler:
    """Generate [B,T] array indexes and gather every model field in one pass."""

    def __init__(self, replay: ArrayReplay, *, seed: int = 0, validate_replay_on_index_build: bool = True) -> None:
        self.replay, self.rng, self.validate = replay, np.random.default_rng(seed), validate_replay_on_index_build

    def sample(self, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> dict[str, np.ndarray]:
        """Return contiguous training arrays without Python transition objects."""
        total = sequence_length + burn_in_length; starts = self.replay.valid_starts(total, validate=self.validate)
        if batch_size <= 0 or batch_size > len(starts): raise ValueError("not enough continuous replay sequences")
        indexes = self.rng.choice(starts, size=batch_size, replace=False)[:, None] + np.arange(total, dtype=np.int64)[None, :]
        arrays = self.replay.arrays
        return {"observations": arrays["observations"][indexes], "next_observations": arrays["next_observations"][indexes],
                "next_action_masks": self.replay.next_action_masks[indexes], "actions": arrays["actions"][indexes], "rewards": arrays["rewards"][indexes],
                "continues": (~(arrays["terminated"][indexes] | arrays["truncated"][indexes])).astype(np.float32), "events": arrays["events"][indexes],
                "opponent_actions": self.replay.opponent_actions[indexes], "opponent_ids": self.replay.opponent_ids[indexes[:, 0]],
                "environment_is_real": (self.replay.environment_types[indexes[:, 0]] == "real_sc2").astype(np.float32)}
