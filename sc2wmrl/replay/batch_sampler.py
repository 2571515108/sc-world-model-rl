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
        indexes = self._sample_indexes(starts, batch_size, total)
        return self._batch_from_indexes(indexes)

    def _sample_indexes(self, starts: np.ndarray, batch_size: int, total: int) -> np.ndarray:
        if batch_size <= 0 or batch_size > len(starts): raise ValueError("not enough continuous replay sequences")
        return self.rng.choice(starts, size=batch_size, replace=False)[:, None] + np.arange(total, dtype=np.int64)[None, :]

    def _batch_from_indexes(self, indexes: np.ndarray) -> dict[str, np.ndarray]:
        """Gather a complete model batch from precomputed [B,T] indexes."""
        arrays = self.replay.arrays
        batch = {"observations": arrays["observations"][indexes], "next_observations": arrays["next_observations"][indexes],
                "next_action_masks": self.replay.next_action_masks[indexes], "actions": arrays["actions"][indexes], "rewards": arrays["rewards"][indexes],
                "continues": (~(arrays["terminated"][indexes] | arrays["truncated"][indexes])).astype(np.float32), "events": arrays["events"][indexes],
                "opponent_actions": self.replay.opponent_actions[indexes], "opponent_action_valid": self.replay.opponent_action_valid[indexes],
                "opponent_ids": self.replay.opponent_ids[indexes[:, 0]],
                "environment_is_real": (self.replay.environment_types[indexes[:, 0]] == "real_sc2").astype(np.float32)}
        for name in ("feature_valid_masks", "next_feature_valid_masks", "next_action_mask_valid", "universal_intents", "universal_intent_valid"):
            if name in arrays:
                batch[name] = arrays[name][indexes]
        for name in ("player_races", "opponent_races"):
            if name in arrays:
                batch[name] = arrays[name][indexes[:, 0]]
        return batch


class RaceBalancedArrayBatchSampler(ArrayBatchSampler):
    """Sample episode-local sequences at explicit player-race ratios."""

    _RACE_IDS = {"Unknown": 0, "Terran": 1, "Protoss": 2, "Zerg": 3}

    def __init__(self, replay: ArrayReplay, *, race_sampling: dict[str, float], seed: int = 0,
                 validate_replay_on_index_build: bool = True) -> None:
        super().__init__(replay, seed=seed, validate_replay_on_index_build=validate_replay_on_index_build)
        if "player_races" not in replay.arrays:
            raise ValueError("race-balanced sampling requires player_races in the replay NPZ")
        requested = {self._RACE_IDS[str(name).title()]: float(weight) for name, weight in race_sampling.items() if str(name).title() in self._RACE_IDS}
        if not requested or any(value < 0 for value in requested.values()) or sum(requested.values()) <= 0:
            raise ValueError("race sampling must give a positive non-negative weight")
        self.race_sampling = {key: value / sum(requested.values()) for key, value in requested.items() if value > 0}

    def sample(self, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> dict[str, np.ndarray]:
        total = sequence_length + burn_in_length; starts = self.replay.valid_starts(total, validate=self.validate)
        available = {race: starts[self.replay.arrays["player_races"][starts] == race] for race in self.race_sampling}
        missing = [race for race, candidates in available.items() if not len(candidates)]
        if missing:
            raise ValueError(f"no valid sequence starts for requested race IDs: {missing}")
        ordered = list(self.race_sampling); raw = np.asarray([self.race_sampling[race] * batch_size for race in ordered]); counts = np.floor(raw).astype(int)
        for index in np.argsort(-(raw - counts))[:batch_size - int(counts.sum())]:
            counts[index] += 1
        chosen = [self.rng.choice(available[race], size=count, replace=count > len(available[race])) for race, count in zip(ordered, counts) if count]
        starts_batch = np.concatenate(chosen); self.rng.shuffle(starts_batch)
        indexes = starts_batch[:, None] + np.arange(total, dtype=np.int64)[None, :]
        return self._batch_from_indexes(indexes)
