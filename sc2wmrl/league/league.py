"""Small league controller with snapshots, payoff matrix, and Elo updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .opponent_pool import OpponentPool, OpponentRecord
from .rating import update_elo


@dataclass(frozen=True)
class PolicySnapshot:
    """Persistent main/exploiter policy checkpoint metadata."""
    snapshot_id: str
    policy_checkpoint: str
    rating: float
    metadata: dict[str, Any] = field(default_factory=dict)


class League:
    """Owns opponent records and never mutates saved historical snapshots."""

    def __init__(self, seed: int = 0) -> None:
        self.pool = OpponentPool(seed); self.snapshots: dict[str, PolicySnapshot] = {}; self.payoffs: dict[tuple[str, str], float] = {}

    def add_scripted_opponents(self, names: tuple[str, ...] = ("rush_bot", "economy_bot", "defensive_bot", "ground_tech_bot", "air_tech_bot", "randomized_bot")) -> None:
        """Register the Phase 4 scripted strategic opponent suite."""
        for name in names:
            if name not in {record.opponent_id for record in self.pool.records()}:
                self.pool.add(OpponentRecord(name, "scripted", None, "scripted"))

    def add_snapshot(self, policy_checkpoint: str, rating: float, metadata: dict[str, Any]) -> PolicySnapshot:
        """Save a uniquely named policy snapshot and expose it as a recent opponent."""
        snapshot_id = f"snapshot-{len(self.snapshots):05d}"
        snapshot = PolicySnapshot(snapshot_id, policy_checkpoint, rating, dict(metadata)); self.snapshots[snapshot_id] = snapshot
        role = "recent" if len(self.snapshots) <= 5 else "historical"
        self.pool.add(OpponentRecord(snapshot_id, "policy_snapshot", policy_checkpoint, role, rating, dict(metadata)))
        return snapshot

    def add_exploiter(self, checkpoint: str, target_snapshot_id: str, rating: float = 1000.0) -> str:
        """Register an independently checkpointed exploiter against one main snapshot."""
        if target_snapshot_id not in self.snapshots: raise ValueError("exploiter target snapshot does not exist")
        exploiter_id = f"exploiter-{sum(record.league_role == 'exploiter' for record in self.pool.records()):05d}"
        self.pool.add(OpponentRecord(exploiter_id, "exploiter", checkpoint, "exploiter", rating, {"target_snapshot_id": target_snapshot_id}))
        return exploiter_id

    def record_result(self, player_id: str, opponent_id: str, score: float) -> tuple[float, float]:
        """Record payoff and Elo update for two registered opponents."""
        player, opponent = self.pool.get(player_id), self.pool.get(opponent_id); new_a, new_b = update_elo(player.rating, opponent.rating, score)
        self.pool._records[player_id] = OpponentRecord(**{**player.__dict__, "rating": new_a})
        self.pool._records[opponent_id] = OpponentRecord(**{**opponent.__dict__, "rating": new_b})
        self.payoffs[(player_id, opponent_id)] = score; self.payoffs[(opponent_id, player_id)] = 1.0 - score
        return new_a, new_b

    def payoff_matrix(self) -> tuple[tuple[str, ...], list[list[float | None]]]:
        """Return a complete labeled matrix with unknown matchups as ``None``."""
        labels = tuple(record.opponent_id for record in self.pool.records())
        return labels, [[self.payoffs.get((row, column)) for column in labels] for row in labels]
