"""Rating-aware opponent selection helpers."""

from __future__ import annotations

from .opponent_pool import OpponentPool, OpponentRecord


def hardest_opponent(pool: OpponentPool, main_rating: float) -> OpponentRecord:
    """Select closest stronger opponent, falling back to highest-rated record."""
    records = pool.records()
    if not records: raise ValueError("empty opponent pool")
    stronger = [record for record in records if record.rating >= main_rating]
    return min(stronger, key=lambda record: record.rating - main_rating) if stronger else max(records, key=lambda record: record.rating)
