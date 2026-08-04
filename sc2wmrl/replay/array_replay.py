"""Array-backed offline replay for high-throughput world-model training."""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np


class ArrayReplay:
    """Keeps persisted numeric replay arrays without rebuilding transition objects."""

    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        self.arrays, self.metadata, self.version = arrays, metadata, 0
        self.size = len(arrays["actions"])
        self.observation_shape = tuple(arrays["observations"].shape[1:])
        self.episode_ids = np.asarray(metadata["episode_ids"], dtype=np.int64)
        self.game_loops = np.asarray(metadata["game_loops"], dtype=np.int64)
        self.opponent_ids = np.asarray([zlib.crc32(str(value).encode()) % 128 for value in metadata["opponent_ids"]], dtype=np.int64)
        self.environment_types = np.asarray(metadata.get("environment_types", ["synthetic"] * self.size))
        self.opponent_actions = arrays.get("opponent_actions", np.asarray([int(info.get("opponent_action", 0)) for info in metadata["infos"]], dtype=np.int64))
        self.opponent_action_valid = arrays.get(
            "opponent_action_valid", (self.environment_types != "real_sc2").astype(np.bool_)
        )
        self.next_action_masks = arrays.get("next_action_masks", arrays["action_masks"])
        self._start_cache: dict[tuple[int, bool], np.ndarray] = {}

    @classmethod
    def load(cls, path: str | Path) -> "ArrayReplay":
        """Load numeric NPZ members once; no ``MacroTransition`` allocation occurs."""
        path = Path(path); metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("format_version") not in {1, 2}:
            raise ValueError("unsupported replay format")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        return cls(arrays, metadata)

    def valid_starts(self, total_length: int, *, validate: bool = True) -> np.ndarray:
        """Cache episode-contiguous start indexes for O(1) repeated sampling."""
        if total_length <= 0: raise ValueError("total length must be positive")
        cache_key = (total_length, validate)
        if cache_key in self._start_cache: return self._start_cache[cache_key]
        candidates = np.arange(max(0, self.size - total_length + 1), dtype=np.int64)
        if validate and len(candidates):
            episode_ok = np.ones(len(candidates), dtype=np.bool_)
            loop_ok = np.ones(len(candidates), dtype=np.bool_)
            continuity_ok = np.ones(len(candidates), dtype=np.bool_)
            for offset in range(total_length - 1):
                episode_ok &= self.episode_ids[offset + 1:offset + 1 + len(candidates)] == self.episode_ids[offset:offset + len(candidates)]
                loop_ok &= self.game_loops[offset + 1:offset + 1 + len(candidates)] > self.game_loops[offset:offset + len(candidates)]
                continuity_ok &= np.all(self.arrays["next_observations"][offset:offset + len(candidates)] == self.arrays["observations"][offset + 1:offset + 1 + len(candidates)], axis=1)
            candidates = candidates[episode_ok & loop_ok & continuity_ok]
        self._start_cache[cache_key] = candidates
        return candidates
