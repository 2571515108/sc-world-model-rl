"""Validated replay transition schema shared by data collection and PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sc2wmrl.envs.base_macro_env import ACTION_COUNT


@dataclass
class MacroTransition:
    """One macro interval, retaining all information needed by later phases."""

    observation: np.ndarray
    entity_observation: np.ndarray | None
    action: int
    action_mask: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    next_observation: np.ndarray
    opponent_id: str
    opponent_type: str
    policy_version: str
    map_name: str
    game_loop: int
    events: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    episode_id: int = 0
    next_action_mask: np.ndarray | None = None
    opponent_action: int = 0
    opponent_action_valid: bool | None = None
    environment_type: str = "synthetic"

    def __post_init__(self) -> None:
        """Normalize dtypes and reject corrupt/non-finite replay records early."""
        self.observation = np.asarray(self.observation, dtype=np.float32)
        self.next_observation = np.asarray(self.next_observation, dtype=np.float32)
        self.action_mask = np.asarray(self.action_mask, dtype=np.bool_)
        self.next_action_mask = self.action_mask.copy() if self.next_action_mask is None else np.asarray(self.next_action_mask, dtype=np.bool_)
        self.events = np.asarray(self.events, dtype=np.float32)
        if self.entity_observation is not None:
            self.entity_observation = np.asarray(self.entity_observation, dtype=np.float32)
        if self.observation.ndim != 1 or self.next_observation.shape != self.observation.shape:
            raise ValueError("observations must be matching one-dimensional vectors")
        if self.action_mask.shape != (ACTION_COUNT,) or not self.action_mask.any() or self.next_action_mask.shape != (ACTION_COUNT,) or not self.next_action_mask.any():
            raise ValueError("transition action masks are invalid")
        if isinstance(self.action, bool) or not 0 <= int(self.action) < ACTION_COUNT or not self.action_mask[int(self.action)]:
            raise ValueError("transition action must be legal under its action mask")
        if not np.isfinite(self.observation).all() or not np.isfinite(self.next_observation).all() or not np.isfinite(self.events).all():
            raise ValueError("transition arrays must be finite")
        if not np.isfinite(self.reward):
            raise ValueError("transition reward must be finite")
        if self.game_loop < 0 or self.episode_id < 0:
            raise ValueError("game loop and episode ID must be non-negative")
        if isinstance(self.opponent_action, bool) or not 0 <= int(self.opponent_action) < ACTION_COUNT:
            raise ValueError("opponent action must be a valid macro action")
        if self.environment_type not in {"synthetic", "real_sc2"}:
            raise ValueError("environment type must be synthetic or real_sc2")
        # A live/replay SC2 viewpoint contains no aligned opponent command.
        # Synthetic opponents expose their scripted action, so preserve it.
        if self.opponent_action_valid is None:
            self.opponent_action_valid = self.environment_type == "synthetic"
        else:
            self.opponent_action_valid = bool(self.opponent_action_valid)
