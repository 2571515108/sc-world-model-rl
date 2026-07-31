"""Typed opponent records and configurable league opponent sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpponentRecord:
    """Immutable opponent descriptor retained independently of future checkpoints."""
    opponent_id: str
    opponent_type: str
    checkpoint: str | None
    league_role: str
    rating: float = 1000.0
    metadata: dict[str, Any] = field(default_factory=dict)


class OpponentPool:
    """Stores scripted, historical, recent, and exploiter opponents."""

    def __init__(self, seed: int = 0) -> None:
        self._records: dict[str, OpponentRecord] = {}; self._rng = np.random.default_rng(seed)

    def add(self, record: OpponentRecord) -> None:
        """Add an immutable ID exactly once to prevent historical overwrite."""
        if record.opponent_id in self._records: raise ValueError(f"opponent already exists: {record.opponent_id}")
        self._records[record.opponent_id] = record

    def get(self, opponent_id: str) -> OpponentRecord:
        """Retrieve a record by stable ID."""
        return self._records[opponent_id]

    def records(self) -> tuple[OpponentRecord, ...]:
        """Return deterministic ID-sorted records."""
        return tuple(self._records[key] for key in sorted(self._records))

    def sample(self, *, weights: dict[str, float] | None = None) -> OpponentRecord:
        """Sample role groups then uniformly sample member; empty groups are ignored."""
        if not self._records: raise ValueError("cannot sample an empty opponent pool")
        weights = weights or {"scripted": 0.4, "recent": 0.3, "historical": 0.2, "exploiter": 0.1}
        groups: dict[str, list[OpponentRecord]] = {}
        for record in self._records.values(): groups.setdefault(record.league_role, []).append(record)
        available = [(role, float(weight)) for role, weight in weights.items() if weight > 0 and groups.get(role)]
        if not available: return self._rng.choice(list(self._records.values()))
        roles, probabilities = zip(*available); probabilities = np.asarray(probabilities); probabilities /= probabilities.sum()
        role = str(self._rng.choice(roles, p=probabilities)); return self._rng.choice(groups[role])
