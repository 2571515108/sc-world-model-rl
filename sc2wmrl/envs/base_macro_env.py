"""Stable macro-environment contract and the Phase 0 action vocabulary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, TypeAlias

import numpy as np

Observation: TypeAlias = np.ndarray
InfoDict: TypeAlias = dict[str, Any]


class MacroAction(IntEnum):
    """Terran macro decisions; identifiers are checkpoint and replay stable."""

    NO_OP = 0
    TRAIN_WORKERS = 1
    BUILD_SUPPLY = 2
    BUILD_BARRACKS = 3
    BUILD_REFINERY = 4
    BUILD_FACTORY = 5
    BUILD_STARPORT = 6
    EXPAND = 7
    TRAIN_BASIC_ARMY = 8
    TRAIN_ANTI_GROUND = 9
    TRAIN_ANTI_AIR = 10
    RESEARCH_UPGRADE = 11
    SCOUT_ENEMY_MAIN = 12
    SCOUT_EXPANSION = 13
    DEFEND_MAIN = 14
    DEFEND_NATURAL = 15
    HARASS = 16
    ATTACK_ENEMY_NATURAL = 17
    ATTACK_ENEMY_MAIN = 18
    RETREAT = 19


ACTION_COUNT = len(MacroAction)


class MacroSC2Env(ABC):
    """Gymnasium-compatible, structured macro-RL environment interface."""

    observation_dim: int
    action_dim: int = ACTION_COUNT

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[Observation, InfoDict]:
        """Start an episode and return a fixed-shape, normalized observation."""

    @abstractmethod
    def step(
        self, macro_action: int
    ) -> tuple[Observation, float, bool, bool, InfoDict]:
        """Execute one macro interval and return Gymnasium's five-tuple."""

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """Return a boolean vector where ``True`` marks currently legal actions."""

    def close(self) -> None:
        """Release backend resources; environments without resources need no action."""

    @staticmethod
    def validate_action(action: int) -> MacroAction:
        """Convert a user action while rejecting bools and out-of-range values."""
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"macro action must be an integer, got {type(action)!r}")
        try:
            return MacroAction(int(action))
        except ValueError as exc:
            raise ValueError(f"unknown macro action {action}") from exc
