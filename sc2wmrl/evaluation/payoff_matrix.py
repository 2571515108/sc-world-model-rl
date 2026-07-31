"""Pretty-print utility for a League payoff matrix."""

from __future__ import annotations

from sc2wmrl.league.league import League


def payoff_rows(league: League) -> list[dict[str, float | None | str]]:
    """Return serializable payoff rows keyed by opponent IDs."""
    labels, matrix = league.payoff_matrix()
    return [{"opponent": label, **dict(zip(labels, values))} for label, values in zip(labels, matrix)]
