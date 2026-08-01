"""Transparent reward shaping with independently recorded components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardConfig:
    """All shaping terms are intentionally small compared with terminal reward."""

    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_draw: float = 0.0
    army_advantage_scale: float = 0.02
    worker_advantage_scale: float = 0.005
    base_advantage_scale: float = 0.02
    technology_progress_scale: float = 0.01
    successful_scout_scale: float = 0.01
    map_control_scale: float = 0.005
    resource_efficiency_scale: float = 0.002
    enemy_damage_scale: float = 0.005
    own_losses_scale: float = 0.005
    clip_value: float | None = 1.0


class RewardTracker:
    """Computes delta-based shaping rewards from canonical scalar statistics."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.previous: dict[str, float] | None = None

    def reset(self, metrics: Mapping[str, float]) -> None:
        """Reset the delta baseline at the start of an episode."""
        self.previous = {key: float(value) for key, value in metrics.items()}

    def step(self, metrics: Mapping[str, float], outcome: str | None = None) -> tuple[float, dict[str, float]]:
        """Return scalar reward and every component, including zero-valued terms."""
        current = {key: float(value) for key, value in metrics.items()}
        if self.previous is None:
            self.reset(current)
        assert self.previous is not None
        delta = {key: current.get(key, 0.0) - self.previous.get(key, 0.0) for key in current}
        components = {
            "army_advantage": self.config.army_advantage_scale * delta.get("army_advantage", 0.0),
            "worker_advantage": self.config.worker_advantage_scale * delta.get("worker_advantage", 0.0),
            "base_advantage": self.config.base_advantage_scale * delta.get("base_advantage", 0.0),
            "technology_progress": self.config.technology_progress_scale * delta.get("technology_progress", 0.0),
            "successful_scout": self.config.successful_scout_scale * delta.get("successful_scout", 0.0),
            "map_control": self.config.map_control_scale * delta.get("map_control", 0.0),
            "resource_efficiency": self.config.resource_efficiency_scale * delta.get("resource_efficiency", 0.0),
            "enemy_damage": self.config.enemy_damage_scale * delta.get("enemy_damage", 0.0),
            "own_losses": -self.config.own_losses_scale * max(0.0, delta.get("own_losses", 0.0)),
            "terminal": 0.0,
        }
        if outcome == "win":
            components["terminal"] = self.config.terminal_win
        elif outcome == "loss":
            components["terminal"] = self.config.terminal_loss
        elif outcome == "draw":
            components["terminal"] = self.config.terminal_draw
        self.previous = current
        reward = float(sum(components.values()))
        if self.config.clip_value is not None:
            reward = max(-float(self.config.clip_value), min(float(self.config.clip_value), reward))
        if not all(abs(value) != float("inf") and value == value for value in components.values()):
            raise FloatingPointError("reward components must be finite")
        return reward, components


def reward_metrics_from_state(state: Mapping[str, Any]) -> dict[str, float]:
    """Derive shaping metrics from the public raw-state schema.

    Real environments only use visible enemy observations and their local
    estimate fields.  They must never replace these estimates with engine truth.
    """
    enemy = state.get("enemy", {})
    own_army = float(state.get("army_value", 0.0))
    enemy_army = float(state.get("enemy_army_value_estimate", enemy.get("estimated_army_value", 0.0)))
    own_workers = float(state.get("worker_count", 0.0))
    enemy_workers = float(enemy.get("estimated_worker_count", 0.0))
    own_bases = float(state.get("base_count", 0.0))
    enemy_bases = float(enemy.get("observed_buildings", {}).get("base", 0.0))
    map_control = state.get("map_control", {})
    controls = [float(item.get("control_score", 0.0)) for item in map_control.values()]
    resources = float(state.get("minerals", 0.0)) + float(state.get("vespene", 0.0))
    return {
        "army_advantage": (own_army - enemy_army) / 12000.0,
        "worker_advantage": (own_workers - enemy_workers) / 80.0,
        "base_advantage": (own_bases - enemy_bases) / 8.0,
        "technology_progress": sum(bool(value) for value in state.get("completed_upgrade_flags", [])) / 6.0,
        "successful_scout": float(enemy.get("last_seen_army_position") is not None),
        "map_control": sum(controls) / max(1, len(controls)),
        "resource_efficiency": min(resources / 7000.0, 1.0),
        "enemy_damage": float(state.get("enemy_damage", 0.0)) / 12000.0,
        "own_losses": float(state.get("lost_army_value", 0.0)) / 12000.0,
    }
