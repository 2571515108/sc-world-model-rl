"""Transparent reward shaping with independently recorded components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
            "terminal": 0.0,
        }
        if outcome == "win":
            components["terminal"] = self.config.terminal_win
        elif outcome == "loss":
            components["terminal"] = self.config.terminal_loss
        elif outcome == "draw":
            components["terminal"] = self.config.terminal_draw
        self.previous = current
        return float(sum(components.values())), components
