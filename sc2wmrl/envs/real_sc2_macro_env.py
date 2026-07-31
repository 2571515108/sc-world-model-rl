"""Production adapter contract for a real SC2 game/bot runtime.

The project intentionally does not invent unit-level commands.  A hosting bot
provides this small backend, allowing existing python-sc2/Ares controllers to
remain the authority for build locations, pathing, targeting, and game setup.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .base_macro_env import InfoDict, MacroAction, MacroSC2Env, Observation
from .feature_extractor import FeatureExtractor


class RealSC2UnavailableError(RuntimeError):
    """Raised only when the caller requests real SC2 without supplying a backend."""


class RealSC2Backend(Protocol):
    """Adapter implemented by the application that owns the SC2 client lifecycle."""

    def reset_game(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a match and return raw structured state plus metadata."""

    def execute_macro(self, action: MacroAction, duration: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run a macro skill and return raw next state and standard step values."""

    def get_action_mask(self) -> np.ndarray:
        """Return the skill controller's current action legality vector."""

    def close(self) -> None:
        """Release the SC2 process/session if the backend owns it."""


class RealSC2MacroEnv(MacroSC2Env):
    """Normalizes observations from a supplied real-game backend."""

    def __init__(self, backend: RealSC2Backend | None, *, macro_interval_game_loops: int = 32,
                 map_width: float = 200.0, map_height: float = 200.0) -> None:
        if backend is None:
            raise RealSC2UnavailableError(
                "RealSC2MacroEnv requires a RealSC2Backend from a configured python-sc2/Ares bot runtime; "
                "SyntheticMacroEnv remains fully available without SC2."
            )
        if macro_interval_game_loops <= 0:
            raise ValueError("macro interval must be positive")
        self.backend = backend
        self.macro_interval_game_loops = macro_interval_game_loops
        self.extractor = FeatureExtractor(map_width, map_height)
        self.observation_dim = self.extractor.dimension
        self._state: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None) -> tuple[Observation, InfoDict]:
        """Reset the delegated game and expose its normalized state."""
        state, info = self.backend.reset_game(seed)
        self._state = state
        return self.extractor.extract(state), self._validated_info(info)

    def step(self, macro_action: int) -> tuple[Observation, float, bool, bool, InfoDict]:
        """Validate then execute a macro action through the supplied backend."""
        if self._state is None:
            raise RuntimeError("reset must be called before step")
        action = self.validate_action(macro_action)
        if not self.get_action_mask()[action]:
            raise ValueError(f"illegal macro action {action.name}")
        state, reward, terminated, truncated, info = self.backend.execute_macro(action, self.macro_interval_game_loops)
        if not np.isfinite(reward):
            raise ValueError("real SC2 backend returned a non-finite reward")
        self._state = state
        return self.extractor.extract(state), float(reward), bool(terminated), bool(truncated), self._validated_info(info)

    def get_action_mask(self) -> np.ndarray:
        """Validate the backend's mask before policies consume it."""
        mask = np.asarray(self.backend.get_action_mask(), dtype=np.bool_)
        if mask.shape != (self.action_dim,) or not mask.any():
            raise ValueError("real SC2 backend returned an invalid action mask")
        return mask

    def close(self) -> None:
        """Close the delegated backend."""
        self.backend.close()

    def _validated_info(self, info: dict[str, Any]) -> InfoDict:
        copied = dict(info)
        copied["action_mask"] = self.get_action_mask().copy()
        return copied
