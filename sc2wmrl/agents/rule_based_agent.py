"""Conservative macro policy for collecting useful real-game trajectories."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


class RuleBasedAgent:
    """Terran opening/economy policy that always respects the supplied mask."""

    def act(self, observation: np.ndarray, action_mask: np.ndarray, raw_state: Mapping[str, object] | None = None) -> int:
        """Choose an available priority action; observations are kept for API parity."""
        del observation
        state = raw_state or {}
        buildings = state.get("buildings", {}) if isinstance(state.get("buildings", {}), Mapping) else {}
        supply = float(state.get("current_supply", 0.0)); cap = float(state.get("maximum_supply", 0.0))
        army = float(state.get("army_value", 0.0))
        priorities: list[MacroAction] = []
        if cap - supply <= 3: priorities.append(MacroAction.BUILD_SUPPLY)
        if float(buildings.get("barracks", 0)) < 1: priorities.append(MacroAction.BUILD_BARRACKS)
        if float(buildings.get("refinery", 0)) < 1: priorities.append(MacroAction.BUILD_REFINERY)
        if float(buildings.get("factory", 0)) < 1: priorities.append(MacroAction.BUILD_FACTORY)
        priorities += [MacroAction.TRAIN_WORKERS, MacroAction.TRAIN_BASIC_ARMY]
        if army >= 250: priorities.append(MacroAction.ATTACK_ENEMY_MAIN)
        if army > 0: priorities.append(MacroAction.DEFEND_MAIN)
        priorities += [MacroAction.SCOUT_ENEMY_MAIN, MacroAction.NO_OP]
        mask = np.asarray(action_mask, dtype=np.bool_)
        for action in priorities:
            if mask[action]: return int(action)
        return int(np.flatnonzero(mask)[0])
