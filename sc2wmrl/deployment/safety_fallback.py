"""Safe deterministic fallback policy for invalid or slow real-time inference."""

from __future__ import annotations

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


class SafetyFallbackPolicy:
    """Prioritizes harmless recovery actions and always returns a legal action."""
    PRIORITY = (MacroAction.BUILD_SUPPLY, MacroAction.TRAIN_WORKERS, MacroAction.TRAIN_BASIC_ARMY, MacroAction.DEFEND_MAIN, MacroAction.NO_OP)
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, failure_context: dict[str, object] | None = None) -> int:
        """Select the first legal action from the documented safety priority order."""
        del observation, failure_context
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (len(MacroAction),) or not mask.any(): return int(MacroAction.NO_OP)
        for action in self.PRIORITY:
            if mask[int(action)]: return int(action)
        return int(np.flatnonzero(mask)[0])
