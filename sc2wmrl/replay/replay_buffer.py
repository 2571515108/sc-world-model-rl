"""Bounded, persistent, shape-safe replay buffer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .transition import MacroTransition


class ReplayBuffer:
    """Stores complete macro transitions and serializes them without pickle."""

    FORMAT_VERSION = 2

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[MacroTransition] = []
        self._observation_shape: tuple[int, ...] | None = None
        self._rng = np.random.default_rng(seed)
        self._version = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def observation_shape(self) -> tuple[int, ...] | None:
        """Observation shape fixed by the first inserted transition."""
        return self._observation_shape

    @property
    def version(self) -> int:
        """Monotonic mutation counter used to invalidate sampler indexes."""
        return self._version

    def __getitem__(self, index: int) -> MacroTransition:
        """Read a transition without constructing a full replay snapshot."""
        return self._items[index]

    def items_view(self) -> tuple[MacroTransition, ...]:
        """Immutable sequence view retained for compatibility with old callers."""
        return tuple(self._items)

    def append(self, transition: MacroTransition) -> None:
        """Insert a transition while preserving fixed dimensionality."""
        shape = transition.observation.shape
        if self._observation_shape is None:
            self._observation_shape = shape
        if shape != self._observation_shape:
            raise ValueError(f"observation shape {shape} differs from replay schema {self._observation_shape}")
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(transition)
        self._version += 1

    def extend(self, transitions: Iterable[MacroTransition]) -> None:
        """Append transitions in order, preserving episode contiguity."""
        for transition in transitions:
            self.append(transition)

    def sample(self, batch_size: int) -> list[MacroTransition]:
        """Sample a deterministic RNG-controlled batch without replacement."""
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("batch size must be in [1, len(replay)]")
        indices = self._rng.choice(len(self._items), size=batch_size, replace=False)
        return [self._items[int(index)] for index in indices]

    def transitions(self) -> tuple[MacroTransition, ...]:
        """Read-only ordered transition snapshot."""
        return self.items_view()

    def save(self, path: str | Path) -> None:
        """Persist numeric tensors to NPZ and metadata to a JSON sidecar."""
        if not self._items:
            raise ValueError("cannot save an empty replay buffer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        observations = np.stack([item.observation for item in self._items])
        next_observations = np.stack([item.next_observation for item in self._items])
        masks = np.stack([item.action_mask for item in self._items])
        next_masks = np.stack([item.next_action_mask for item in self._items])
        events = np.stack([item.events for item in self._items])
        np.savez_compressed(path, observations=observations, next_observations=next_observations, action_masks=masks, next_action_masks=next_masks,
                             events=events, actions=np.asarray([item.action for item in self._items], dtype=np.int64),
                             opponent_actions=np.asarray([item.opponent_action for item in self._items], dtype=np.int64),
                             opponent_action_valid=np.asarray([item.opponent_action_valid for item in self._items], dtype=np.bool_),
                             rewards=np.asarray([item.reward for item in self._items], dtype=np.float32),
                            terminated=np.asarray([item.terminated for item in self._items], dtype=np.bool_),
                            truncated=np.asarray([item.truncated for item in self._items], dtype=np.bool_))
        metadata = {"format_version": self.FORMAT_VERSION, "capacity": self.capacity,
                    "episode_ids": [item.episode_id for item in self._items], "game_loops": [item.game_loop for item in self._items],
                    "opponent_ids": [item.opponent_id for item in self._items], "opponent_types": [item.opponent_type for item in self._items],
                    "policy_versions": [item.policy_version for item in self._items], "map_names": [item.map_name for item in self._items],
                    "environment_types": [item.environment_type for item in self._items], "infos": [item.info for item in self._items]}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> "ReplayBuffer":
        """Load a replay created by :meth:`save` and re-run all validations."""
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("format_version") not in {1, cls.FORMAT_VERSION}:
            raise ValueError("unsupported replay format version")
        with np.load(path, allow_pickle=False) as arrays:
            # Read each compressed member exactly once. Indexing an NpzFile inside
            # the loop repeatedly recreates a complete decompressed array; a row
            # view then keeps that whole array alive in every transition.
            names = ("observations", "next_observations", "action_masks", "events", "actions", "rewards", "terminated", "truncated")
            payload = {name: arrays[name] for name in names}
            if "next_action_masks" in arrays.files: payload["next_action_masks"] = arrays["next_action_masks"]
            if "opponent_actions" in arrays.files: payload["opponent_actions"] = arrays["opponent_actions"]
            if "opponent_action_valid" in arrays.files: payload["opponent_action_valid"] = arrays["opponent_action_valid"]
            count = len(payload["actions"])
            next_masks = payload.get("next_action_masks", payload["action_masks"])
            opponent_actions = payload.get("opponent_actions", np.zeros(count, dtype=np.int64))
            default_valid = np.asarray([value != "real_sc2" for value in metadata.get("environment_types", ["synthetic"] * count)], dtype=np.bool_)
            opponent_action_valid = payload.get("opponent_action_valid", default_valid)
            buffer = cls(max(int(metadata["capacity"]), count), seed=seed)
            for index in range(count):
                buffer.append(MacroTransition(
                    observation=payload["observations"][index].copy(), entity_observation=None, action=int(payload["actions"][index]),
                    action_mask=payload["action_masks"][index].copy(), reward=float(payload["rewards"][index]),
                    terminated=bool(payload["terminated"][index]), truncated=bool(payload["truncated"][index]),
                    next_observation=payload["next_observations"][index].copy(), opponent_id=metadata["opponent_ids"][index],
                    opponent_type=metadata["opponent_types"][index], policy_version=metadata["policy_versions"][index],
                    map_name=metadata["map_names"][index], game_loop=int(metadata["game_loops"][index]),
                    events=payload["events"][index].copy(), info=metadata["infos"][index], episode_id=int(metadata["episode_ids"][index]),
                    next_action_mask=next_masks[index].copy(), opponent_action=int(opponent_actions[index]),
                    opponent_action_valid=bool(opponent_action_valid[index]),
                    environment_type=metadata.get("environment_types", ["synthetic"] * count)[index],
                ))
        return buffer
